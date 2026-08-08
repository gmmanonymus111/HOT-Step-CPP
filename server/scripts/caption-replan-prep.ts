#!/usr/bin/env npx tsx
/**
 * Emit one work packet per lyrics set for the caption re-plan migration
 * (2026-08-08). Run with `npx tsx` from server/.
 *
 * Captions written before the format fix came out at 258-986 chars against
 * training captions of 1149-1537; this prepares the batch that rewrites them.
 *
 * A packet is a self-contained prompt for ONE set: its real training captions,
 * its audio-analysis block, and every generation of that set still needing a
 * rewrite. Batching per set rather than per song sends the (large, identical)
 * caption examples once per set instead of once per song.
 *
 * Sets whose own songs carry no captions borrow from another set BY THE SAME
 * ARTIST, preferring one whose album preset points at the same adapter — that
 * is the dataset the adapter actually saw. Nine sets need this: an old Genius
 * fetch and a Training Studio export coexist because findLyricsSetByAlbum
 * matches on album name and the export's detected name never equalled the old
 * one ("The Wall" vs "The Wall (MFSL UDCD-537) - Disc 2").
 *
 * IDEMPOTENT: a generation whose caption already validates is skipped, so a
 * re-run after a session limit only emits the shortfall.
 *
 *   npx tsx scripts/caption-replan-prep.ts --out <dir>
 *   npx tsx scripts/caption-replan-prep.ts --out <dir> --set 81
 *
 *   --out <dir>    where packets are written (required)
 *   --set <id>     only this lyrics set (pilot mode)
 *   --limit <n>    at most N sets
 *   --db <file>    SQLite db (default server/data/hotstep.db)
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import {
  buildCaptionReplanPrompt, CAPTION_REPLAN_SYSTEM_PROMPT, computeAlbumEnrichment,
  type CaptionReplanSong,
} from '../src/services/lireek/prompts.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(HERE, '..', 'package.json'));
const Database = require('better-sqlite3');

const argv = process.argv.slice(2);
const opt = (n: string, d?: string) => {
  const i = argv.indexOf(`--${n}`);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d;
};
const OUT = opt('out');
const ONLY_SET = opt('set') ? Number(opt('set')) : 0;
const LIMIT = opt('limit') ? Number(opt('limit')) : 0;
const DB_PATH = path.resolve(opt('db', path.join(HERE, '..', 'data', 'hotstep.db'))!);
if (!OUT) { console.error('usage: caption-replan-prep.ts --out <dir> [--set <id>] [--limit <n>]'); process.exit(2); }
fs.mkdirSync(OUT, { recursive: true });

/** A caption is DONE when it already looks like a training caption. */
export function captionIsAligned(c: string | null | undefined): boolean {
  const s = String(c ?? '').trim();
  if (s.length < 1100) return false;
  if (/\n/.test(s)) return false;
  const sentences = s.split(/(?<=[.!?])\s+/).filter(x => x.trim()).length;
  return sentences === 9;
}

const db = new Database(DB_PATH, { readonly: true });

type SetRow = { id: number; artist_id: number; album: string | null; songs: string };
const sets: SetRow[] = db.prepare('SELECT id, artist_id, album, songs FROM lyrics_sets').all();
const parsed = new Map<number, { artist_id: number; album: string | null; songs: any[] }>();
for (const s of sets) {
  let songs: any[] = [];
  try { songs = JSON.parse(s.songs); } catch { /* unreadable set */ }
  parsed.set(s.id, { artist_id: s.artist_id, album: s.album, songs });
}

/** adapter identity for a set, used to prefer the right donor. */
const adapterOf = new Map<number, string>();
for (const p of db.prepare('SELECT lyrics_set_id, adapter_path FROM album_presets').all() as any[]) {
  if (p.adapter_path) adapterOf.set(p.lyrics_set_id, String(p.adapter_path).toLowerCase());
}

const captionsOf = (id: number) => (parsed.get(id)?.songs ?? [])
  .map(s => (typeof s?.caption === 'string' ? s.caption.trim() : ''))
  .filter(Boolean);

/** Up to 3 verbatim training captions for a set, borrowing within the artist if needed. */
function examplesFor(setId: number): { examples: string[]; donor: number | null } {
  const own = captionsOf(setId);
  if (own.length) return { examples: [...new Set(own)].slice(0, 3), donor: null };

  const me = parsed.get(setId);
  if (!me) return { examples: [], donor: null };
  const siblings = [...parsed.entries()]
    .filter(([id, v]) => id !== setId && v.artist_id === me.artist_id && captionsOf(id).length > 0)
    .map(([id]) => id);
  if (!siblings.length) return { examples: [], donor: null };

  // Prefer a sibling trained into the same adapter as this set's preset.
  const mine = adapterOf.get(setId);
  const sameAdapter = mine ? siblings.find(id => adapterOf.get(id) === mine) : undefined;
  const donor = sameAdapter ?? siblings[0];
  return { examples: [...new Set(captionsOf(donor))].slice(0, 3), donor };
}

let packets = 0, songsTotal = 0, skipped = 0, noExamples = 0;
const index: any[] = [];

for (const [setId, info] of parsed) {
  if (ONLY_SET && setId !== ONLY_SET) continue;
  if (LIMIT && packets >= LIMIT) break;

  const profileRows = db.prepare('SELECT id, profile_data FROM profiles WHERE lyrics_set_id = ?').all(setId) as any[];
  if (!profileRows.length) continue;
  const profileIds = profileRows.map(p => p.id);

  const gens = db.prepare(
    `SELECT id, title, subject, bpm, key, duration, lyrics, caption
     FROM generations WHERE profile_id IN (${profileIds.map(() => '?').join(',')}) ORDER BY id`
  ).all(...profileIds) as any[];

  const todo: CaptionReplanSong[] = [];
  for (const g of gens) {
    if (captionIsAligned(g.caption)) { skipped++; continue; }
    if (!g.lyrics || !String(g.lyrics).trim()) { skipped++; continue; }
    todo.push({
      id: g.id, title: g.title, subject: g.subject,
      bpm: g.bpm, key: g.key, duration: g.duration, lyrics: g.lyrics,
    });
  }
  if (!todo.length) continue;

  const { examples, donor } = examplesFor(setId);
  if (!examples.length) noExamples++;

  // Enrichment is derived live from the set's own songs (its bpm/key/genre
  // block stays honest even when the caption examples are borrowed).
  const profile = JSON.parse(profileRows[0].profile_data);
  profile.audio_enrichment = computeAlbumEnrichment(info.songs);

  const prompt = buildCaptionReplanPrompt(profile, todo, examples);
  const file = path.join(OUT!, `set-${setId}.json`);
  fs.writeFileSync(file, JSON.stringify({
    setId, album: info.album, artistId: info.artist_id,
    exampleDonorSet: donor, exampleCount: examples.length,
    ids: todo.map(t => t.id),
    system: CAPTION_REPLAN_SYSTEM_PROMPT,
    prompt,
  }, null, 2));

  index.push({ setId, album: info.album, songs: todo.length, exampleDonorSet: donor, examples: examples.length, promptChars: prompt.length, file });
  packets++;
  songsTotal += todo.length;
}

fs.writeFileSync(path.join(OUT!, 'index.json'), JSON.stringify(index, null, 2));
console.log(`packets: ${packets}  songs: ${songsTotal}  already-aligned skipped: ${skipped}`);
if (noExamples) console.log(`WARN  ${noExamples} packet(s) have NO caption examples at all`);
const borrowed = index.filter(i => i.exampleDonorSet).length;
if (borrowed) console.log(`${borrowed} packet(s) borrowed examples from a sibling set`);
console.log(`avg prompt ${Math.round(index.reduce((s, i) => s + i.promptChars, 0) / Math.max(1, index.length))} chars`);
console.log(`out: ${OUT}`);
