#!/usr/bin/env npx tsx
/**
 * Emit per-set work packets for the lyric DENSITY REGENERATION batch
 * (2026-08-08). Run with `npx tsx` from server/.
 *
 * Targets the songs retime-generations.ts flagged NEEDS LYRIC REGEN: their
 * lyrics are the wrong word density for their artist (written before
 * per-artist pacing existed), so the duration their words need falls outside
 * 0.6-1.5x of the album's median real duration. Retiming can't fix them; the
 * lyrics must be expanded (fast artists, too sparse) or condensed (slow
 * artists, too wordy) to targetWords = targetDuration x artistRate.
 *
 * targetDuration = the song's current duration clamped into the album
 * envelope (or the album median when it has none). Idempotent: once a song's
 * rewritten lyrics land inside the envelope, it stops being emitted.
 *
 *   npx tsx scripts/lyric-regen-prep.ts --out <dir>
 *   npx tsx scripts/lyric-regen-prep.ts --out <dir> --set 81
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import {
  buildDensityRewritePrompt, computeAlbumEnrichment, countLyricWords,
  lyricsDurationSeconds, LYRIC_DENSITY_REWRITE_SYSTEM_PROMPT,
  type DensityRewriteRef, type DensityRewriteSong,
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
const ONLY_SET = Number(opt('set', '0'));
const DB_PATH = path.resolve(opt('db', path.join(HERE, '..', 'data', 'hotstep.db'))!);
if (!OUT) { console.error('usage: lyric-regen-prep.ts --out <dir> [--set <id>]'); process.exit(2); }
fs.mkdirSync(OUT, { recursive: true });

const db = new Database(DB_PATH, { readonly: true });

interface SetCtx {
  rate: number; med: number; envLo: number; envHi: number;
  artist: string; ref: DensityRewriteRef | null;
}
const setCtx = new Map<number, SetCtx>();
for (const r of db.prepare('SELECT ls.id, ls.songs, a.name artist FROM lyrics_sets ls JOIN artists a ON a.id=ls.artist_id').all() as any[]) {
  let songs: any[] = []; try { songs = JSON.parse(r.songs); } catch { continue; }
  const rate = computeAlbumEnrichment(songs)?.wordsPerSec || 0;
  const durs = songs.map(s => Number(s?.duration)).filter(d => Number.isFinite(d) && d > 30).sort((a, b) => a - b);
  const med = durs.length ? durs[Math.floor(durs.length / 2)] : 0;
  // Reference: the real song nearest the album median with a substantial lyric.
  let ref: DensityRewriteRef | null = null;
  let best = Infinity;
  for (const s of songs) {
    const d = Number(s?.duration);
    if (!Number.isFinite(d) || d <= 30 || typeof s.lyrics !== 'string') continue;
    const w = countLyricWords(s.lyrics);
    if (w < 80) continue;
    const dist = Math.abs(d - med);
    if (dist < best) {
      best = dist;
      ref = { title: s.title, duration: Math.round(d), words: w, lyrics: s.lyrics.slice(0, 3000) };
    }
  }
  setCtx.set(r.id, {
    rate, med,
    envLo: med ? Math.max(90, Math.round(med * 0.6)) : 0,
    envHi: med ? Math.round(med * 1.5) : 0,
    artist: r.artist, ref,
  });
}

const bySet = new Map<number, DensityRewriteSong[]>();
let expand = 0, condense = 0;
for (const g of db.prepare(`
  SELECT g.id, g.title, g.subject, g.bpm, g.key, g.duration, g.lyrics, p.lyrics_set_id
  FROM generations g JOIN profiles p ON p.id = g.profile_id ORDER BY g.id`).all() as any[]) {
  if (ONLY_SET && g.lyrics_set_id !== ONLY_SET) continue;
  const ctx = setCtx.get(g.lyrics_set_id);
  if (!ctx || !ctx.rate || !ctx.envLo || !g.lyrics) continue;
  const derived = lyricsDurationSeconds(g.lyrics, g.bpm || 0, ctx.rate);
  if (!derived) continue;
  if (g.duration > 0 && Math.abs(derived - g.duration) <= 15) continue;            // consistent
  if (derived >= ctx.envLo && derived <= ctx.envHi) continue;                       // retimable, not regen
  const targetDuration = Math.min(ctx.envHi, Math.max(ctx.envLo, g.duration > 0 ? g.duration : ctx.med));
  const targetWords = Math.round(targetDuration * ctx.rate);
  const currentWords = countLyricWords(g.lyrics);
  if (targetWords > currentWords) expand++; else condense++;
  if (!bySet.has(g.lyrics_set_id)) bySet.set(g.lyrics_set_id, []);
  bySet.get(g.lyrics_set_id)!.push({
    id: g.id, title: g.title, subject: g.subject, bpm: g.bpm, key: g.key,
    targetDuration, targetWords, currentWords, lyrics: g.lyrics,
  });
}

const index: any[] = [];
for (const [setId, songs] of bySet) {
  const ctx = setCtx.get(setId)!;
  const prompt = buildDensityRewritePrompt(ctx.artist, ctx.rate, ctx.med, ctx.ref, songs);
  const file = path.join(OUT!, `set-${setId}.json`);
  fs.writeFileSync(file, JSON.stringify({
    setId, artist: ctx.artist, rate: ctx.rate,
    ids: songs.map(s => s.id),
    targets: Object.fromEntries(songs.map(s => [s.id, { words: s.targetWords, duration: s.targetDuration }])),
    system: LYRIC_DENSITY_REWRITE_SYSTEM_PROMPT,
    prompt,
  }, null, 2));
  index.push({ setId, artist: ctx.artist, songs: songs.length, promptChars: prompt.length, hasRef: !!ctx.ref, file });
}
fs.writeFileSync(path.join(OUT!, 'index.json'), JSON.stringify(index, null, 2));

const total = [...bySet.values()].reduce((s, v) => s + v.length, 0);
console.log(`packets: ${bySet.size}  songs: ${total}  (expand ${expand}, condense ${condense})`);
console.log(`packets without a reference lyric: ${index.filter(i => !i.hasRef).length}`);
console.log(`avg prompt ${Math.round(index.reduce((s, i) => s + i.promptChars, 0) / Math.max(1, index.length))} chars`);
console.log(`out: ${OUT}`);
