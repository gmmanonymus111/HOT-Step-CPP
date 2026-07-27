// training/enhanceService.ts — the optional cloud steps
//
// Nothing here runs unless the user presses a button: Genius fetches real
// lyrics for a matched artist/title, and an LLM rewrites the local caption into
// Side-Step's structured 9-sentence form.
//
// Spec: docs/plans/2026-07-27-dataset-studio-implementation.md §4.9

import path from 'path';
import { searchSongLyrics } from '../lireek/geniusService.js';
import { getProvider } from '../lireek/llm/registry.js';
import { sanitizeHeaders } from './lyricsSanitizer.js';
import { buildUserPrompt, parseStructuredResponse, CAPTION_INSTRUCTIONS, CAPTION_TOP_P } from './captionPrompt.js';
import type { TrainingDatasetRow, TrainingSample } from './types.js';

const MAX_CAPTION_CHARS = 4000;
const MAX_LYRICS_CHARS = 20000;

/** Zero-width and control chars we never want inside a sidecar (§7.6). */
const ZERO_WIDTH = new RegExp('[\\u200B-\\u200D\\uFEFF]', 'g');
const CONTROL_CHARS = new RegExp('[\\u0000-\\u0008\\u000B\\u000C\\u000E-\\u001F\\u007F]', 'g');

function cleanText(raw: string, cap: number): string {
  return String(raw ?? '')
    .replace(/\r\n?/g, '\n')
    .replace(ZERO_WIDTH, '')
    .replace(CONTROL_CHARS, '')
    .trim()
    .slice(0, cap);
}

// ── Genius ───────────────────────────────────────────────────────────────

/**
 * Pull artist/title out of a filename. Handles `NN-artist-title`,
 * `artist - title` and `NN. title`, converting underscores to spaces and
 * stripping leading track numbers.
 */
export function parseArtistTitleFromFilename(filename: string): { artist: string; title: string } {
  const ext = path.extname(filename);
  let stem = (ext ? filename.slice(0, filename.length - ext.length) : filename).trim();
  stem = stem.replace(/_/g, ' ').trim();

  // `NN. title` / `NN - title` / `NN title`
  const numbered = /^\d{1,3}\s*[.\-–]?\s+(.*)$/.exec(stem);
  const numberedDash = /^\d{1,3}\s*-\s*(.*)$/.exec(stem);
  let body = stem;
  if (numberedDash) body = numberedDash[1].trim();
  else if (numbered) body = numbered[1].trim();

  // `artist - title`
  const dashed = /^(.+?)\s+-\s+(.+)$/.exec(body);
  if (dashed) return { artist: dashed[1].trim(), title: dashed[2].trim() };

  // `artist-title` (no spaces around the dash) — only when it splits cleanly in two
  const tight = body.split('-');
  if (tight.length === 2 && tight[0].trim() && tight[1].trim()) {
    return { artist: tight[0].trim(), title: tight[1].trim() };
  }

  return { artist: '', title: body };
}

/** Artist/title resolution order: embedded tags → filename → dataset defaults. */
export function resolveArtistTitle(
  sample: TrainingSample,
  ds: TrainingDatasetRow,
  opts: { artist?: string },
): { artist: string; title: string } {
  const fromName = parseArtistTitleFromFilename(sample.filename);
  const artist = sample.tagArtist || fromName.artist || opts.artist || ds.defaultArtist || '';
  const title = sample.tagTitle || fromName.title || '';
  return { artist: artist.trim(), title: title.trim() };
}

/**
 * Fetch lyrics from Genius for one sample.
 * `null` means "no confident match" — the caller logs a warning and moves on.
 */
export async function enhanceGenius(
  sample: TrainingSample,
  ds: TrainingDatasetRow,
  opts: { artist?: string; album?: string; sanitizeHeaders: boolean },
): Promise<{ lyrics: string; source: 'genius' } | null> {
  const { artist, title } = resolveArtistTitle(sample, ds, opts);
  if (!artist || !title) return null;

  const hit = await searchSongLyrics(artist, title);
  if (!hit || !hit.lyrics?.trim()) return null;

  let lyrics = cleanText(hit.lyrics, MAX_LYRICS_CHARS);
  if (opts.sanitizeHeaders) lyrics = sanitizeHeaders(lyrics);
  if (!lyrics.trim()) return null;

  return { lyrics, source: 'genius' };
}

// ── LLM caption ──────────────────────────────────────────────────────────

function isNetworkError(err: any): boolean {
  const msg = String(err?.message || err || '');
  return /fetch failed|ECONNRESET|ECONNREFUSED|ETIMEDOUT|ENOTFOUND|socket hang up|network|timed out/i.test(msg);
}

/**
 * Rewrite one sample's caption with an LLM. Returns the subset of sidecar
 * fields the model produced — `{}` when it produced nothing usable.
 */
export async function enhanceCaption(
  sample: TrainingSample,
  ds: TrainingDatasetRow,
  opts: { provider: string; model?: string; includeLyricsExcerpt: boolean; temperature: number },
): Promise<Record<string, string>> {
  const provider = getProvider(opts.provider);
  const { artist, title } = resolveArtistTitle(sample, ds, {});

  const userPrompt = buildUserPrompt({
    title: sample.tagTitle || title,
    artist,
    lyricsExcerpt: opts.includeLyricsExcerpt ? sample.lyrics.slice(0, 500) : undefined,
    audioAttached: false,
    localCaption: sample.caption,
    bpm: sample.bpm,
    key: sample.key,
    signature: sample.signature,
  });

  const model = opts.model || provider.defaultModel;

  let text = '';
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      let streamed = '';
      const result = await provider.call(
        CAPTION_INSTRUCTIONS,
        userPrompt,
        model,
        (chunk: string) => { streamed += chunk; },
        { temperature: opts.temperature, top_p: CAPTION_TOP_P },
      );
      // Defensive fallback (assistant.ts): some providers return '' and only stream.
      text = (result && result.trim()) ? result : streamed;
      break;
    } catch (err: any) {
      if (attempt === 0 && isNetworkError(err)) {
        await new Promise(r => setTimeout(r, 2000));
        continue;
      }
      throw err;
    }
  }

  if (!text.trim()) return {};

  const parsed = parseStructuredResponse(text);
  const out: Record<string, string> = {};
  if (parsed.caption) out.caption = cleanText(parsed.caption, MAX_CAPTION_CHARS);
  if (parsed.genre) out.genre = cleanText(parsed.genre, 300);
  if (parsed.bpm) out.bpm = cleanText(parsed.bpm, 16);
  if (parsed.key) out.key = cleanText(parsed.key, 64);
  if (parsed.signature) out.signature = cleanText(parsed.signature, 16);
  return out;
}
