// training/captionPrompt.ts — Side-Step's 5-line structured caption prompt
//
// CAPTION_INSTRUCTIONS is a character-for-character copy of Side-Step's
// `sidestep_engine/data/caption_config.py::_DEFAULT_PROMPT_INSTRUCTIONS`, so a
// caption produced here is indistinguishable from one produced there.
//
// No provider in our registry accepts audio (D14), so the user prompt carries
// `Audio attached to this request: no` plus the local analysis block — the LLM
// rewrites from local evidence rather than listening.
//
// Spec: docs/plans/2026-07-27-dataset-studio-implementation.md §4.10

import { parseLooseObject } from './sidecarIO.js';

/**
 * The nine caption sentences, in order, without their leading `- `.
 *
 * Lyric Studio's metadata planner reuses this so the captions it invents for a
 * new song describe the same nine things, in the same order, as the captions a
 * sound adapter was trained on (see `lireek/prompts.ts`). Keep the wording here
 * byte-identical to Side-Step — `CAPTION_INSTRUCTIONS` is rebuilt from it.
 */
export const CAPTION_SENTENCE_PLAN: readonly string[] = [
  'Sentence 1: identify the core genre/subgenre, tempo feel, groove character, and overall intensity without restating exact metadata fields.',
  'Sentence 2: describe drum design and groove, including kick, snare/clap, hats, percussion, swing, syncopation, and pulse.',
  'Sentence 3: describe the bass design and low-end behavior, including weight, tone, movement, rhythm, sustain, and kick interaction.',
  'Sentence 4: describe harmony and melody, including chords, tonal center, motifs, riffs, leads, pads, stabs, arps, or vocal hooks.',
  'Sentence 5: describe sound design and timbre, including synth character, source type, texture, brightness, distortion, saturation, envelopes, layering, and spectral character.',
  'Sentence 6: describe mix treatment and space, including reverb, delay, compression, transient shape, filtering, automation, density, and only clearly audible spatial placement if confidence is high.',
  'Sentence 7: describe the opening section and buildup in detail, including which elements are introduced first and how tension is created.',
  'Sentence 8: describe the drop and any break section in detail, including what the drop contains, what hits hardest, and which elements are removed or exposed during the break.',
  'Sentence 9: describe the late-song payoff, climax, or outro in detail, including how the track escalates, peaks, resolves, strips back, or closes.',
];

export const CAPTION_INSTRUCTIONS: string =
  "Write high-quality structured music dataset metadata grounded in the song's " +
  'audible content. If audio is attached, analyze the actual audio first and use ' +
  'title, artist, and lyrics only as weak secondary context.\n\n' +
  'Return EXACTLY 5 lines in plain text and nothing else. Each field must start at ' +
  'the beginning of its own new line. Never place two fields on the same line.\n\n' +
  'Use this exact output template:\n' +
  'caption: <single-line paragraph with EXACTLY 9 complete sentences>\n' +
  "genre: <comma-separated genre/style tags, e.g. 'bass house, electro house'>\n" +
  'bpm: <estimated BPM as integer, e.g. 120>\n' +
  "key: <musical key, e.g. 'C minor' or 'F# major'>\n" +
  "signature: <time signature, e.g. '4/4'>\n\n" +
  'Caption rules:\n' +
  '- The caption must be one line with exactly 9 complete sentences.\n' +
  '- Start `caption:` on line 1, `genre:` on line 2, `bpm:` on line 3, `key:` on line 4, and `signature:` on line 5.\n' +
  '- Do not merge fields together. For example, do not output `genre: ... bpm: ... key: ...` on one line.\n' +
  '- Do not use markdown, bullets, numbering, code fences, labels before the template, or commentary after the template.\n' +
  '- Write for machine-learning training metadata, not for reviews, marketing copy, or listener-facing blurbs.\n' +
  '- Prioritize musically useful descriptors: groove, drum pattern, bass design, harmonic density, melodic motifs, instrumentation, synthesis, timbre, texture, dynamics, stereo space, effects, and arrangement.\n' +
  '- Use concrete audio evidence such as syncopated hats, sidechained bass, plucked synth lead, saturated kick, washed reverb tails, filtered risers, wide stereo pads, chopped vocal textures, dry upfront drums, or distorted reese bass when applicable.\n' +
  "- Avoid vague or low-value phrases such as 'nice vibe', 'good energy', 'keeps you moving', 'hard to resist', 'captivating journey', 'atmospheric progression', 'listeners feel', or 'emotionally resonant' unless backed by specific sonic detail.\n" +
  "- Avoid generic openings like 'This track is' when more specific wording can be used immediately.\n" +
  '- Keep explicit metadata in the dedicated fields, not in the caption: do not state exact BPM numbers, key names, or time signatures in the caption unless absolutely necessary for a rare musically specific point.\n' +
  '- Mention stereo width, panning, depth, or imaging only when those traits are clearly audible with high confidence; if uncertain, prefer safer mix descriptors such as dry, wet, dense, open, compressed, bright, dark, upfront, distant, or saturated.\n' +
  '- If you mention a buildup, drop, break, climax, or outro, specify what changes musically: which layers enter, which layers drop out, which filters open, how percussion changes, how the bass changes, or how the energy is reshaped.\n' +
  CAPTION_SENTENCE_PLAN.map(s => `- ${s}\n`).join('') +
  '- Do not mention the artist name or song title in the caption.\n' +
  '- If audio is not attached or a field cannot be determined from available evidence, write N/A for that field instead of guessing.';

/** Gen params Side-Step uses for the caption call. */
export const CAPTION_TEMPERATURE = 0.45;
export const CAPTION_TOP_P = 0.9;

export interface UserPromptArgs {
  title: string;
  artist: string;
  lyricsExcerpt?: string;
  audioAttached: boolean;
  localCaption?: string;
  bpm?: number | null;
  key?: string;
  signature?: string;
}

/** Side-Step's user prompt, extended with the local-label block (D14). */
export function buildUserPrompt(args: UserPromptArgs): string {
  const lines: string[] = [];
  lines.push(CAPTION_INSTRUCTIONS);
  lines.push('');
  lines.push('Song metadata:');
  lines.push(`Audio attached to this request: ${args.audioAttached ? 'yes' : 'no'}`);
  lines.push(`Title: ${args.title || 'unknown'}`);
  lines.push(`Artist: ${args.artist || 'unknown'}`);
  lines.push(
    `Local analysis: bpm ${args.bpm ?? 'unknown'}, key ${args.key || 'unknown'}, ` +
    `signature ${args.signature || 'unknown'}`,
  );
  if (!args.audioAttached) {
    // Text-only mode leans on the local caption. With real audio attached it is
    // OMITTED entirely — a weak local caption anchors the model on wrong genres.
    lines.push('Existing caption (from local audio analysis — treat as primary audible evidence):');
    lines.push(args.localCaption || '(none)');
  }
  if (args.lyricsExcerpt && args.lyricsExcerpt.trim()) {
    lines.push(`Lyrics excerpt:\n${args.lyricsExcerpt.slice(0, 500)}`);
  }
  lines.push('');
  lines.push(args.audioAttached
    ? 'Reminder: describe what you HEAR in the attached audio; metadata and lyrics are secondary context only.'
    : 'Reminder: audible evidence comes first; metadata and lyrics are secondary.');
  return lines.join('\n');
}

// ── Response parsing ─────────────────────────────────────────────────────

type CaptionField = 'caption' | 'genre' | 'bpm' | 'key' | 'signature';
const FIELDS: readonly CaptionField[] = ['caption', 'genre', 'bpm', 'key', 'signature'];

type CaptionFields = Partial<Record<CaptionField, string>>;

function cleanValue(raw: string): string {
  let v = raw.trim();
  v = v.replace(/^["'`]+/, '').replace(/["'`,]+$/, '');
  return v.trim();
}

function isNA(v: string): boolean {
  return v.trim().toLowerCase() === 'n/a';
}

function fromMapping(obj: Record<string, unknown>): CaptionFields {
  const out: CaptionFields = {};
  for (const field of FIELDS) {
    const v = obj[field];
    if (v === null || v === undefined) continue;
    const s = typeof v === 'string' ? v : String(v);
    if (s.trim()) out[field] = cleanValue(s);
  }
  return out;
}

/** Regex hunt through a blob that would not parse as an object. */
function huntBlob(text: string): CaptionFields {
  const out: CaptionFields = {};
  for (const field of FIELDS) {
    const re = new RegExp(`["']?${field}["']?\\s*:\\s*["']([\\s\\S]*?)["']\\s*(?:,|\\}|$)`, 'i');
    const m = re.exec(text);
    if (m && m[1].trim()) out[field] = cleanValue(m[1]);
  }
  return out;
}

/** Plaintext label scan — the normal, well-behaved path. */
function scanLabels(text: string): CaptionFields {
  const out: CaptionFields = {};
  const re = /(?<!\w)(caption|genre|bpm|key|signature)\s*:/gi;
  const hits: Array<{ field: CaptionField; start: number; end: number }> = [];
  for (const m of text.matchAll(re)) {
    hits.push({
      field: m[1].toLowerCase() as CaptionField,
      start: m.index ?? 0,
      end: (m.index ?? 0) + m[0].length,
    });
  }
  if (hits.length === 0) return out;

  for (let i = 0; i < hits.length; i++) {
    const hit = hits[i];
    if (out[hit.field] !== undefined) continue;   // first occurrence wins
    const stop = i + 1 < hits.length ? hits[i + 1].start : text.length;
    const value = cleanValue(text.slice(hit.end, stop));
    if (value) out[hit.field] = value;
  }

  // Tail recovery: prose ahead of the first label is the caption when the
  // response never labelled one.
  if (!out.caption) {
    const head = cleanValue(text.slice(0, hits[0].start));
    if (head) out.caption = head;
  }
  return out;
}

function dropNA(fields: CaptionFields): CaptionFields {
  const out: CaptionFields = {};
  for (const field of FIELDS) {
    const v = fields[field];
    if (v === undefined) continue;
    if (!v.trim() || isNA(v)) continue;
    out[field] = v;
  }
  return out;
}

/**
 * Six-step resolution chain (§4.10): mapping input → JSON object string →
 * blob-regex hunt → plaintext label scan (+ tail recovery) → drop "n/a" →
 * total-failure fallback of `{ caption: raw }`.
 */
export function parseStructuredResponse(raw: string): CaptionFields {
  const input: unknown = raw;

  // 1. Already a mapping (some providers hand back parsed JSON)
  if (input && typeof input === 'object' && !Array.isArray(input)) {
    return dropNA(fromMapping(input as Record<string, unknown>));
  }
  if (typeof input !== 'string') return {};

  const text = input.trim();
  if (!text) return {};

  // 2. A `{...}`-shaped string
  if (text.startsWith('{')) {
    const obj = parseLooseObject(text);
    if (obj) {
      const mapped = dropNA(fromMapping(obj));
      if (Object.keys(mapped).length > 0) return mapped;
    }
    // 3. Blob-regex hunt when the object would not parse
    const hunted = dropNA(huntBlob(text));
    if (Object.keys(hunted).length > 0) return hunted;
  }

  // 4. Plaintext label scan
  const scanned = dropNA(scanLabels(text));
  if (Object.keys(scanned).length > 0) return scanned;

  // 6. Total failure — hand the whole response back as the caption
  return { caption: text };
}
