#!/usr/bin/env npx tsx
/**
 * Strip empty [Intro*]/[Outro*] boundary tags from every generated lyric and
 * re-derive each song's duration from what remains (2026-08-09). Run with
 * `npx tsx` from server/.
 *
 * Ear-validated on the FFAF pair before this bulk pass (see d6bc62b): an
 * EMPTY boundary tag is an open invitation the model fills at its adapter's
 * preferred size — a bare [Intro] rendered 105s of riffing, a declared
 * [Outro - Instrumental] held ~60s regardless of duration. Real lyric
 * transcriptions almost never write them (Intro 5% empty, Outro 1%, against
 * 85%/11% in our output — our own prompts used to MANDATE the empty [Intro]).
 * Stripping both and re-deriving produced the pair Rob called perfect.
 *
 * Rules, identical to the validated pair:
 *   - An [Intro...] or [Outro...] tag (bare OR descriptored) whose section
 *     contains no lyric line is removed. Boundary tags WITH sung lines stay.
 *   - Mid-song instrumental tags ([Guitar Solo], [Instrumental], [Interlude],
 *     [Build], [Drop], [Breakdown]...) are untouched — in-distribution, and
 *     they sounded good.
 *   - Duration is re-derived from the stripped lyrics at the artist's rate
 *     floored to the model's measured sing rate (reconcileDurationToLyrics,
 *     VOCAL_FLOOR 1.25); kept when already within 15s.
 *
 * Songs whose derived duration falls below 0.6x their album's median real
 * duration are updated anyway (a short clean song beats a padded one — the
 * surplus WAS the filler) but listed as SPARSE for a future density pass.
 *
 *   npx tsx scripts/strip-boundary-tags.ts
 *   npx tsx scripts/strip-boundary-tags.ts --apply
 */
import path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import { computeAlbumEnrichment, reconcileDurationToLyrics } from '../src/services/lireek/prompts.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(HERE, '..', 'package.json'));
const Database = require('better-sqlite3');

const argv = process.argv.slice(2);
const APPLY = argv.includes('--apply');
const opt = (n: string, d?: string) => {
  const i = argv.indexOf(`--${n}`);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d;
};
const DB_PATH = path.resolve(opt('db', path.join(HERE, '..', 'data', 'hotstep.db'))!);

const TAG_LINE = /^[ \t]*\[([^\]]{1,60})\][ \t]*$/;
const BOUNDARY = /^(intro|outro)$/i;

/** Remove empty intro/outro sections; returns null when nothing changed. */
export function stripEmptyBoundaries(lyrics: string): { out: string; removed: string[] } | null {
  const lines = String(lyrics).split(/\r?\n/);
  const drop = new Set<number>();
  const removed: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    const m = TAG_LINE.exec(lines[i].trim());
    if (!m) continue;
    const head = m[1].split(/\s+[-–—]\s+/)[0].trim();
    if (!BOUNDARY.test(head)) continue;
    // section = lines until the next tag / EOF; empty when all blank
    let j = i + 1;
    let empty = true;
    for (; j < lines.length; j++) {
      const u = lines[j].trim();
      if (TAG_LINE.test(u)) break;
      if (u) { empty = false; break; }
    }
    if (!empty) continue;
    removed.push(m[1]);
    for (let k = i; k < j; k++) drop.add(k);   // tag + its blank lines
  }
  if (!removed.length) return null;
  const kept = lines.filter((_, i) => !drop.has(i));
  // tidy: collapse runs of blank lines left behind, trim boundary blanks
  const out: string[] = [];
  for (const l of kept) {
    if (l.trim() === '' && (out.length === 0 || out[out.length - 1].trim() === '')) continue;
    out.push(l);
  }
  while (out.length && out[out.length - 1].trim() === '') out.pop();
  return { out: out.join('\n'), removed };
}

const db = new Database(DB_PATH);

const setInfo = new Map<number, { rate: number; envLo: number }>();
for (const r of db.prepare('SELECT id, songs FROM lyrics_sets').all() as any[]) {
  let songs: any[] = []; try { songs = JSON.parse(r.songs); } catch { continue; }
  const rate = computeAlbumEnrichment(songs)?.wordsPerSec || 0;
  const durs = songs.map(s => Number(s?.duration)).filter(d => Number.isFinite(d) && d > 30).sort((a, b) => a - b);
  const med = durs.length ? durs[Math.floor(durs.length / 2)] : 0;
  setInfo.set(r.id, { rate, envLo: med ? Math.round(med * 0.6) : 0 });
}

const gens = db.prepare(`
  SELECT g.id, g.title, g.bpm, g.duration, g.lyrics, p.lyrics_set_id, a.name AS artist
  FROM generations g
  JOIN profiles p ON p.id = g.profile_id
  JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
  JOIN artists a ON a.id = ls.artist_id ORDER BY g.id`).all() as any[];

type Change = { id: number; title: string; artist: string; removed: string[]; from: number; to: number; sparse: boolean };
const changes: Change[] = [];
const tagTally = new Map<string, number>();
let durOnly = 0, untouched = 0;

for (const g of gens) {
  if (!g.lyrics || !String(g.lyrics).trim()) { untouched++; continue; }
  const info = setInfo.get(g.lyrics_set_id) ?? { rate: 0, envLo: 0 };
  const stripped = stripEmptyBoundaries(g.lyrics);
  const lyrics = stripped ? stripped.out : g.lyrics;
  const from = Number(g.duration) || 0;
  const to = reconcileDurationToLyrics(lyrics, Number(g.bpm) || 0, from, info.rate);
  if (!stripped && to === from) { untouched++; continue; }
  if (!stripped) durOnly++;
  for (const t of stripped?.removed ?? []) tagTally.set(t, (tagTally.get(t) || 0) + 1);
  changes.push({
    id: g.id, title: g.title, artist: g.artist,
    removed: stripped?.removed ?? [], from, to,
    sparse: !!(info.envLo && to < info.envLo),
  });
}

const tagsRemoved = [...tagTally.values()].reduce((a, b) => a + b, 0);
const sparse = changes.filter(c => c.sparse);
const durChanged = changes.filter(c => c.to !== c.from);
const deltas = durChanged.map(c => c.to - c.from).sort((a, b) => a - b);
const q = (p: number) => deltas.length ? deltas[Math.floor((deltas.length - 1) * p)] : 0;

console.log(`\nDB:   ${DB_PATH}`);
console.log(`Mode: ${APPLY ? 'APPLY' : 'DRY RUN'}\n`);
console.log(`generations: ${gens.length}   to update: ${changes.length}   untouched: ${untouched}`);
console.log(`empty boundary tags removed: ${tagsRemoved} across ${changes.length - durOnly} songs (+${durOnly} duration-only updates)`);
console.log(`duration changes: ${durChanged.length}  (delta p5 ${q(.05)}s  median ${q(.5)}s  p95 ${q(.95)}s)`);
console.log(`SPARSE after re-derive (below 0.6x album median — future density pass): ${sparse.length}\n`);
console.log('tags removed:');
for (const [t, n] of [...tagTally.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10)) {
  console.log(`  ${String(n).padStart(5)}  [${t}]`);
}
if (sparse.length) {
  console.log('\nsparse examples:');
  for (const c of sparse.slice(0, 8)) console.log(`  #${c.id}  ${c.from}s -> ${c.to}s  ${c.artist} — ${c.title}`);
}

if (!changes.length) { console.log('\nNothing to do.\n'); db.close(); process.exit(0); }
if (!APPLY) { console.log('\nDRY RUN — nothing written. Re-run with --apply.\n'); db.close(); process.exit(0); }

const stampNow = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
const bak = DB_PATH.replace(/\.db$/i, `_backup_striptags_${stampNow}.db`);
await db.backup(bak);
console.log(`\nBacked up to ${path.basename(bak)}`);

const upd = db.prepare('UPDATE generations SET lyrics = ?, duration = ? WHERE id = ?');
const getL = db.prepare('SELECT lyrics FROM generations WHERE id = ?');
const tx = db.transaction(() => {
  for (const c of changes) {
    const cur = String((getL.get(c.id) as any).lyrics);
    const s = stripEmptyBoundaries(cur);
    upd.run(s ? s.out : cur, c.to, c.id);
  }
});
tx();
db.close();
console.log(`Updated ${changes.length} generation(s).`);
