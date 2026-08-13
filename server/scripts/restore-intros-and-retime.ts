#!/usr/bin/env npx tsx
/**
 * Restore instrumental intros and re-time the whole generation catalogue
 * (2026-08-13). Run with `npx tsx` from server/.
 *
 * Two corrections in one atomic pass, in this order because the first feeds
 * the second (a declared intro is instrumental time the duration counts):
 *
 * 1. INTROS. The 2026-08-09 boundary-tag sweep stripped every empty [Intro]
 *    because a BARE one is an open invitation the model fills at whatever size
 *    its adapter prefers (105s of riffing on one FFAF song). Right about the
 *    bare tag, wrong about the conclusion — with no tag at all the vocal now
 *    starts at 0s on essentially every song. This restores the DESCRIPTORED
 *    form, [Intro - Instrumental], on ~80% of songs (deterministic per song,
 *    seeded from artist+title, so a re-run makes the same choices). Songs that
 *    open on a SUNG intro are left alone; the ~20% that slam into [Verse 1]
 *    stay that way.
 *
 * 2. DURATIONS. `words / max(rate, 1.25)` treated the model's SING rate as the
 *    rate the song RUNS at, which priced every artist's instrumental time at
 *    zero. Measured against 2009 real recordings it reproduced the length of
 *    artists at/above 1.25 w/s (median 1.00x) and shortened everyone below it
 *    (median 0.77x, worst 0.26x) — 90 of 160 albums measure below it. That is
 *    where the 1:38 Muse and 1:47 Pink Floyd generations came from. The new
 *    derivation takes the artist's own total-duration rate, floored at
 *    DURATION_RATE_FLOOR (= 1.25 / 1.5, i.e. no song more than a third
 *    instrumental), and honours declared sections when they exceed it.
 *
 * PER-ARTIST RATES ONLY, as with every pass before it: generations whose
 * lyrics set has no measured rate get their intro but keep their duration —
 * the global constant misprices sparse artists in the wrong direction.
 *
 * Lyrics are otherwise untouched: no words are added, removed or rewritten.
 * Online backup before any write. DRY RUN by default.
 *
 *   npx tsx scripts/restore-intros-and-retime.ts
 *   npx tsx scripts/restore-intros-and-retime.ts --apply
 *
 *   --db <file>       SQLite db (default server/data/hotstep.db)
 *   --no-intros       re-time only, leave lyrics alone
 *   --no-retime       add intros only, leave durations alone
 */
import path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import {
  computeAlbumEnrichment, reconcileDurationToLyrics, ensureInstrumentalIntro,
  countLyricWords, DURATION_RATE_FLOOR,
} from '../src/services/lireek/prompts.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(HERE, '..', 'package.json'));
const Database = require('better-sqlite3');

const argv = process.argv.slice(2);
const APPLY = argv.includes('--apply');
const DO_INTROS = !argv.includes('--no-intros');
const DO_RETIME = !argv.includes('--no-retime');
const opt = (n: string, d?: string) => {
  const i = argv.indexOf(`--${n}`);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d;
};
const DB_PATH = path.resolve(opt('db', path.join(HERE, '..', 'data', 'hotstep.db'))!);

const db = new Database(DB_PATH);

// Per set: the artist's rate and their real album duration envelope. The
// envelope does not gate the write — this pass exists BECAUSE the old
// durations were wrong — but a song landing outside it still means its word
// density is wrong for the artist, which no amount of retiming fixes.
const setInfo = new Map<number, { rate: number; envLo: number; envHi: number; med: number }>();
for (const r of db.prepare('SELECT id, songs FROM lyrics_sets').all() as any[]) {
  let songs: any[] = []; try { songs = JSON.parse(r.songs); } catch { continue; }
  const rate = computeAlbumEnrichment(songs)?.wordsPerSec || 0;
  const durs = songs.map(s => Number(s?.duration)).filter(d => Number.isFinite(d) && d > 30).sort((a, b) => a - b);
  const med = durs.length ? durs[Math.floor(durs.length / 2)] : 0;
  setInfo.set(r.id, {
    rate, med,
    envLo: med ? Math.max(90, Math.round(med * 0.6)) : 0,
    envHi: med ? Math.round(med * 1.5) : 0,
  });
}

const gens = db.prepare(`
  SELECT g.id, g.title, g.bpm, g.duration, g.lyrics, p.lyrics_set_id, a.name AS artist,
         EXISTS(SELECT 1 FROM audio_generations ag WHERE ag.generation_id = g.id) AS has_audio
  FROM generations g
  JOIN profiles p ON p.id = g.profile_id
  JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
  JOIN artists a ON a.id = ls.artist_id
  ORDER BY g.id`).all() as any[];

type Change = {
  id: number; artist: string; title: string;
  lyrics: string | null;            // null = lyrics unchanged
  from: number; to: number; rate: number; hasAudio: boolean;
  intro: boolean; sparse: boolean; longer: boolean;
};

const changes: Change[] = [];
let noLyrics = 0, noRate = 0, introOnly = 0, durOnly = 0, untouched = 0, hadIntro = 0;

for (const g of gens) {
  if (!g.lyrics || !String(g.lyrics).trim()) { noLyrics++; continue; }
  const info = setInfo.get(g.lyrics_set_id) ?? { rate: 0, envLo: 0, envHi: 0, med: 0 };

  const withIntro = DO_INTROS
    ? ensureInstrumentalIntro(String(g.lyrics), `${g.artist}|${g.title}`)
    : null;
  if (DO_INTROS && withIntro === null) hadIntro++;
  const lyrics = withIntro ?? String(g.lyrics);

  const from = Number(g.duration) || 0;
  let to = from;
  if (DO_RETIME) {
    if (!info.rate) noRate++;
    else to = reconcileDurationToLyrics(lyrics, Number(g.bpm) || 0, from, info.rate);
  }

  if (withIntro === null && to === from) { untouched++; continue; }
  if (withIntro !== null && to === from) introOnly++;
  if (withIntro === null && to !== from) durOnly++;

  changes.push({
    id: g.id, artist: g.artist, title: g.title,
    lyrics: withIntro, from, to, rate: info.rate, hasAudio: !!g.has_audio,
    intro: withIntro !== null,
    sparse: !!(info.envLo && to < info.envLo),
    longer: !!(info.envHi && to > info.envHi),
  });
}

const durChanged = changes.filter(c => c.to !== c.from);
const deltas = durChanged.map(c => c.to - c.from).sort((a, b) => a - b);
const q = (p: number) => deltas.length ? deltas[Math.floor((deltas.length - 1) * p)] : 0;
const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;
const durs = (list: Change[], pick: (c: Change) => number) => {
  const s = list.map(pick).sort((a, b) => a - b);
  const at = (p: number) => s.length ? s[Math.floor((s.length - 1) * p)] : 0;
  return `p10 ${fmt(at(.1))}  p25 ${fmt(at(.25))}  median ${fmt(at(.5))}  p75 ${fmt(at(.75))}  p90 ${fmt(at(.9))}`;
};

console.log(`\nDB:   ${DB_PATH}`);
console.log(`Mode: ${APPLY ? 'APPLY' : 'DRY RUN'}${DO_INTROS ? '' : '  (--no-intros)'}${DO_RETIME ? '' : '  (--no-retime)'}`);
console.log(`Duration rate floor: ${DURATION_RATE_FLOOR.toFixed(3)} w/s (max ~33% instrumental)\n`);
console.log(`generations: ${gens.length}`);
console.log(`  to update              : ${changes.length}  (${introOnly} intro only, ${durOnly} duration only, ${changes.length - introOnly - durOnly} both)`);
console.log(`  intros added/upgraded  : ${changes.filter(c => c.intro).length}`);
console.log(`  left without an intro  : ${hadIntro}  (already had one, sung intro, or in the ~20% that open on the vocal)`);
console.log(`  durations changed      : ${durChanged.length}  (${durChanged.filter(c => c.to > c.from).length} longer, ${durChanged.filter(c => c.to < c.from).length} shorter)`);
console.log(`  unchanged              : ${untouched}   no lyrics: ${noLyrics}   no artist rate (kept duration): ${noRate}`);
console.log(`  already rendered to audio: ${changes.filter(c => c.hasAudio).length}\n`);
console.log(`delta seconds: p5 ${q(.05)}  p25 ${q(.25)}  median ${q(.5)}  p75 ${q(.75)}  p95 ${q(.95)}`);
if (durChanged.length) {
  console.log(`duration before: ${durs(durChanged, c => c.from)}`);
  console.log(`duration after : ${durs(durChanged, c => c.to)}`);
}

const vsAlbum = changes.filter(c => setInfo.get(gens.find(g => g.id === c.id)!.lyrics_set_id)?.med);
const sparse = changes.filter(c => c.sparse);
const longer = changes.filter(c => c.longer);
console.log(`\nstill SPARSE after retime (< 0.6x album median — lyrics too thin for the artist): ${sparse.length}`);
console.log(`now OVER 1.5x album median (lyrics too dense — a density pass, not a timing one): ${longer.length}`);
void vsAlbum;

const show = (list: Change[], label: string, n = 8) => {
  if (!list.length) return;
  console.log(`\n${label}`);
  for (const c of list.slice(0, n)) {
    console.log(`  #${c.id}  ${fmt(c.from)} -> ${fmt(c.to)}  (${c.to - c.from > 0 ? '+' : ''}${c.to - c.from}s @ ${c.rate.toFixed(2)}w/s)${c.intro ? ' +intro' : ''}${c.hasAudio ? '  [has audio]' : ''}  ${c.artist} — ${c.title}`);
  }
};
show([...durChanged].sort((a, b) => (b.to - b.from) - (a.to - a.from)), 'biggest changes:');
show([...durChanged].sort((a, b) => Math.abs(a.to - a.from) - Math.abs(b.to - b.from)), 'smallest changes:');
show(sparse, 'still sparse — lyrics too thin for the artist (density pass, not timing):');
show(longer, 'now over 1.5x the album median — check these:', 12);

if (!changes.length) { console.log('\nNothing to do.\n'); db.close(); process.exit(0); }
if (!APPLY) { console.log('\nDRY RUN — nothing written. Re-run with --apply.\n'); db.close(); process.exit(0); }

const stampNow = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
const bak = DB_PATH.replace(/\.db$/i, `_backup_introsretime_${stampNow}.db`);
await db.backup(bak);
console.log(`\nBacked up to ${path.basename(bak)}`);

const updBoth = db.prepare('UPDATE generations SET lyrics = ?, duration = ? WHERE id = ?');
const updDur = db.prepare('UPDATE generations SET duration = ? WHERE id = ?');
const tx = db.transaction(() => {
  for (const c of changes) {
    if (c.lyrics !== null) updBoth.run(c.lyrics, c.to, c.id);
    else updDur.run(c.to, c.id);
  }
});
tx();

// Post-write sanity: every updated row must still parse to the duration we
// intended, and no lyric may have lost words.
const check = db.prepare('SELECT lyrics, duration FROM generations WHERE id = ?');
let bad = 0;
for (const c of changes) {
  const row = check.get(c.id) as any;
  if (Number(row.duration) !== c.to) bad++;
  if (c.lyrics !== null && countLyricWords(row.lyrics) !== countLyricWords(c.lyrics)) bad++;
}
db.close();
console.log(`Updated ${changes.length} generation(s). Post-write mismatches: ${bad}.`);
