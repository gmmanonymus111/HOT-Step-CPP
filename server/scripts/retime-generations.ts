#!/usr/bin/env npx tsx
/**
 * Re-time the existing generation catalogue: recompute each song's duration
 * from its ACTUAL lyrics at its OWN ARTIST's measured vocal pacing
 * (2026-08-08). Run with `npx tsx` from server/.
 *
 * Pre-fix generations were planned with a line budget that priced every line
 * the same and scaled with BPM, so their durations disagree with their words —
 * the songs cut off mid-phrase (words > time) or pad with improvised filler
 * (time > words). This applies the same reconciliation that now runs at save
 * time for new generations (reconcileDurationToLyrics) to the back catalogue.
 *
 * PER-ARTIST RATES ONLY. Generations whose lyrics set has no measured rate
 * are SKIPPED, never retimed with the global constant: under the global rate
 * a sparse Pink Floyd song classifies as "shorten" when at Floyd's real
 * 0.67 w/s it needs "lengthen" — the wrong direction entirely. No LLM is
 * involved anywhere; this is arithmetic.
 *
 * Lyrics and captions are untouched. Songs already within 15s keep their
 * duration (the function returns the planned value unchanged). Online backup
 * before any write. DRY RUN by default.
 *
 *   npx tsx scripts/retime-generations.ts
 *   npx tsx scripts/retime-generations.ts --apply
 *
 *   --db <file>   SQLite db (default server/data/hotstep.db)
 */
import path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import { computeAlbumEnrichment, lyricsDurationSeconds } from '../src/services/lireek/prompts.js';

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

const db = new Database(DB_PATH);

// Per set: the artist's measured rate AND the album's real duration envelope.
// The envelope is the guard against "consistent but absurd": a derived
// duration far outside anything the artist actually records means the LYRICS
// are the wrong density for this artist, and no duration can fix that.
const setInfo = new Map<number, { rate: number; envLo: number; envHi: number }>();
for (const r of db.prepare('SELECT id, songs FROM lyrics_sets').all() as any[]) {
  let songs: any[] = []; try { songs = JSON.parse(r.songs); } catch { continue; }
  const rate = computeAlbumEnrichment(songs)?.wordsPerSec || 0;
  // Envelope from the MEDIAN album duration, not min/max: real albums carry
  // 30s skits and 8-minute epics that make a min/max envelope accept a 106s
  // Eminem fragment or a 484s Pendulum. The median is what a typical song by
  // this artist runs; ±(40%..50%) around it is generous but sane.
  const durs = songs.map(s => Number(s?.duration)).filter(d => Number.isFinite(d) && d > 30).sort((a, b) => a - b);
  const med = durs.length ? durs[Math.floor(durs.length / 2)] : 0;
  const envLo = med ? Math.max(90, Math.round(med * 0.6)) : 0;
  const envHi = med ? Math.round(med * 1.5) : 0;
  setInfo.set(r.id, { rate, envLo, envHi });
}

const gens = db.prepare(`
  SELECT g.id, g.title, g.bpm, g.duration, g.lyrics, p.lyrics_set_id, a.name AS artist,
         EXISTS(SELECT 1 FROM audio_generations ag WHERE ag.generation_id = g.id) AS has_audio
  FROM generations g
  JOIN profiles p ON p.id = g.profile_id
  JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
  JOIN artists a ON a.id = ls.artist_id
  ORDER BY g.id`).all() as any[];

type Change = { id: number; artist: string; title: string; from: number; to: number; rate: number; hasAudio: boolean };
const changes: Change[] = [];
const regen: Array<Change & { envLo: number; envHi: number }> = [];
let unchanged = 0, skippedNoRate = 0, skippedNoLyrics = 0;

for (const g of gens) {
  const info = setInfo.get(g.lyrics_set_id);
  if (!info || !info.rate) { skippedNoRate++; continue; }
  if (!g.lyrics || !String(g.lyrics).trim()) { skippedNoLyrics++; continue; }
  const from = Number(g.duration) || 0;
  const derived = lyricsDurationSeconds(g.lyrics, Number(g.bpm) || 0, info.rate);
  if (!derived) { skippedNoLyrics++; continue; }
  if (from > 0 && Math.abs(derived - from) <= 15) { unchanged++; continue; }
  const c = { id: g.id, artist: g.artist, title: g.title, from, to: derived, rate: info.rate, hasAudio: !!g.has_audio };
  // Envelope guard: outside the artist's real recorded range (±25%), the
  // lyrics themselves are mis-densified for this artist — flag for lyric
  // regeneration instead of stamping an absurd length on them.
  if (info.envLo && (derived < info.envLo || derived > info.envHi)) {
    regen.push({ ...c, envLo: info.envLo, envHi: info.envHi });
    continue;
  }
  changes.push(c);
}

const longer = changes.filter(c => c.to > c.from);
const shorter = changes.filter(c => c.to < c.from);
const deltas = changes.map(c => c.to - c.from).sort((a, b) => a - b);
const q = (p: number) => deltas.length ? deltas[Math.floor((deltas.length - 1) * p)] : 0;

console.log(`\nDB:   ${DB_PATH}`);
console.log(`Mode: ${APPLY ? 'APPLY' : 'DRY RUN'}\n`);
console.log(`generations: ${gens.length}`);
console.log(`  to re-time            : ${changes.length}  (${longer.length} longer — words didn't fit; ${shorter.length} shorter — empty time removed)`);
console.log(`  already consistent    : ${unchanged}  (within 15s at the artist's rate)`);
console.log(`  NEEDS LYRIC REGEN     : ${regen.length}  (derived duration outside the artist's real album range — wrong word density, not fixable by timing)`);
console.log(`  skipped, no artist rate: ${skippedNoRate}   skipped, no lyrics: ${skippedNoLyrics}`);
console.log(`  of the re-timed, already rendered to audio: ${changes.filter(c => c.hasAudio).length}\n`);
console.log(`delta seconds: p5 ${q(.05)}  p25 ${q(.25)}  median ${q(.5)}  p75 ${q(.75)}  p95 ${q(.95)}\n`);

const show = (list: Change[], label: string) => {
  console.log(label);
  for (const c of list.slice(0, 8)) {
    console.log(`  #${c.id}  ${String(c.from).padStart(3)}s -> ${String(c.to).padStart(3)}s  (${(c.to - c.from > 0 ? '+' : '')}${c.to - c.from}s @ ${c.rate.toFixed(2)}w/s)${c.hasAudio ? '  [has audio]' : ''}  ${c.artist} — ${c.title}`);
  }
};
show([...changes].sort((a, b) => (b.to - b.from) - (a.to - a.from)), 'biggest lengthens:');
console.log('');
show([...changes].sort((a, b) => (a.to - a.from) - (b.to - b.from)), 'biggest shortens:');
if (regen.length) {
  console.log('\nneeds lyric regeneration (worst first):');
  for (const c of [...regen].sort((a, b) => Math.abs(b.to - b.from) - Math.abs(a.to - a.from)).slice(0, 12)) {
    console.log(`  #${c.id}  ${String(c.from).padStart(3)}s, words need ${String(c.to).padStart(3)}s, album range ${c.envLo}-${c.envHi}s  ${c.artist} — ${c.title}`);
  }
  if (regen.length > 12) console.log(`  … and ${regen.length - 12} more`);
}

if (!changes.length) { console.log('\nNothing to do.\n'); db.close(); process.exit(0); }
if (!APPLY) { console.log('\nDRY RUN — nothing written. Re-run with --apply.\n'); db.close(); process.exit(0); }

const stampNow = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
const bak = DB_PATH.replace(/\.db$/i, `_backup_retime_${stampNow}.db`);
await db.backup(bak);
console.log(`\nBacked up to ${path.basename(bak)}`);

const upd = db.prepare('UPDATE generations SET duration = ? WHERE id = ?');
const tx = db.transaction(() => { for (const c of changes) upd.run(c.to, c.id); });
tx();
db.close();
console.log(`Re-timed ${changes.length} generation(s).`);
