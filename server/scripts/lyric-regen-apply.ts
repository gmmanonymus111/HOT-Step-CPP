#!/usr/bin/env npx tsx
/**
 * Validate and apply density-rewritten lyrics from the regen batch agents
 * (2026-08-08). Run with `npx tsx` from server/.
 *
 * Reads out-set-*.json ({generationId: lyrics}) against the packets'
 * per-song targets. A rewrite is accepted only when it hits the target word
 * count, keeps the section vocabulary, and passes the pipeline's formatting
 * rules; anything else is reported and NOT written, so a bad batch degrades
 * to "re-run those sets". Accepted rewrites update BOTH lyrics and duration —
 * the duration is re-derived from the new words at the artist's rate, which
 * is the whole point of the exercise.
 *
 *   npx tsx scripts/lyric-regen-apply.ts --work <dir>
 *   npx tsx scripts/lyric-regen-apply.ts --work <dir> --apply
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import { countLyricWords, reconcileDurationToLyrics } from '../src/services/lireek/prompts.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(HERE, '..', 'package.json'));
const Database = require('better-sqlite3');

const argv = process.argv.slice(2);
const APPLY = argv.includes('--apply');
const opt = (n: string, d?: string) => {
  const i = argv.indexOf(`--${n}`);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d;
};
const WORK = opt('work');
const DB_PATH = path.resolve(opt('db', path.join(HERE, '..', 'data', 'hotstep.db'))!);
if (!WORK || !fs.existsSync(WORK)) { console.error('usage: lyric-regen-apply.ts --work <dir> [--apply]'); process.exit(2); }

const TAG_LINE = /^[ \t]*\[([^\]]{1,60})\][ \t]*$/;
/** Documented base vocabulary + the two training-corpus extras. */
const ALLOWED_HEADS = /^(intro|verse(\s+\d+)?|pre-chorus|chorus|post-chorus|bridge|outro|build|drop|breakdown|instrumental|guitar solo|piano interlude|fade out|silence|interlude|refrain)$/i;
const PUNCT_END = /[.,!?;…\-—")'\]]$/;

function validateLyrics(text: unknown, targetWords: number): string[] {
  const f: string[] = [];
  if (typeof text !== 'string' || !text.trim()) return ['missing'];
  const s = text.trim();
  if (/^Title:/im.test(s)) f.push('title-line');
  const lines = s.split(/\r?\n/).map(l => l.trim());
  let chorus = 0, lyricLines = 0, punctOk = 0;
  for (const l of lines) {
    if (!l) continue;
    const m = TAG_LINE.exec(l);
    if (m) {
      const head = m[1].split(/\s+[-–—]\s+/)[0].trim();
      if (m[1].includes(':')) f.push(`colon-tag[${m[1]}]`);
      else if (!ALLOWED_HEADS.test(head)) f.push(`bad-tag[${m[1]}]`);
      if (/^chorus$/i.test(head)) chorus++;
      continue;
    }
    lyricLines++;
    if (PUNCT_END.test(l)) punctOk++;
  }
  if (!chorus) f.push('no-chorus');
  if (!lyricLines) return ['no-lyrics'];
  if (punctOk / lyricLines < 0.9) f.push(`punctuation=${Math.round(100 * punctOk / lyricLines)}%`);
  const words = countLyricWords(s);
  const dev = Math.abs(words - targetWords) / targetWords;
  if (dev > 0.12) f.push(`words=${words}/target=${targetWords}`);
  return f;
}

const db = new Database(DB_PATH);
const genRow = db.prepare('SELECT id, title, bpm, duration, lyrics FROM generations WHERE id = ?');

// targets come from the packets in the same work dir
const targets = new Map<number, { words: number; duration: number; rate: number }>();
for (const f of fs.readdirSync(WORK).filter(x => /^set-\d+\.json$/.test(x))) {
  const p = JSON.parse(fs.readFileSync(path.join(WORK, f), 'utf8'));
  for (const [id, t] of Object.entries(p.targets)) {
    targets.set(Number(id), { ...(t as any), rate: p.rate });
  }
}

const good: Array<{ id: number; lyrics: string; duration: number }> = [];
const bad: Array<{ id: number; file: string; why: string[] }> = [];
let unknown = 0;

for (const f of fs.readdirSync(WORK).filter(x => /^out-set-\d+\.json$/.test(x))) {
  let obj: Record<string, unknown>;
  try {
    obj = JSON.parse(fs.readFileSync(path.join(WORK, f), 'utf8').replace(/^```(json)?|```$/gm, '').trim());
  } catch (e: any) { console.log(`  ${f}: UNPARSEABLE — ${e.message}`); continue; }
  for (const [k, v] of Object.entries(obj)) {
    const id = Number(k);
    const t = targets.get(id);
    const g = genRow.get(id);
    if (!t || !g) { unknown++; continue; }
    const why = validateLyrics(v, t.words);
    if (typeof v === 'string' && v.trim() === String(g.lyrics).trim()) why.push('unchanged');
    if (why.length) { bad.push({ id, file: f, why }); continue; }
    const lyrics = (v as string).trim();
    const duration = reconcileDurationToLyrics(lyrics, g.bpm || 0, t.duration, t.rate);
    good.push({ id, lyrics, duration });
  }
}

const dedup = new Map<number, { lyrics: string; duration: number }>();
for (const g of good) if (!dedup.has(g.id)) dedup.set(g.id, g);

console.log(`\nDB:   ${DB_PATH}`);
console.log(`work: ${WORK}`);
console.log(`Mode: ${APPLY ? 'APPLY' : 'DRY RUN'}\n`);
console.log(`valid:    ${dedup.size}`);
console.log(`rejected: ${bad.length}`);
if (unknown) console.log(`unknown ids: ${unknown}`);
for (const b of bad.slice(0, 20)) {
  const g = genRow.get(b.id);
  console.log(`  REJECT #${b.id} [${b.file}] ${b.why.join(',')}  ${g ? g.title : ''}`);
}
if (bad.length > 20) console.log(`  … and ${bad.length - 20} more`);

if (!dedup.size) { console.log('\nNothing valid to write.\n'); process.exit(1); }
if (!APPLY) { console.log('\nDRY RUN — nothing written. Re-run with --apply.\n'); process.exit(0); }

const stampNow = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
const bak = DB_PATH.replace(/\.db$/i, `_backup_lyricregen_${stampNow}.db`);
await db.backup(bak);
console.log(`\nBacked up to ${path.basename(bak)}`);

const upd = db.prepare('UPDATE generations SET lyrics = ?, duration = ? WHERE id = ?');
const tx = db.transaction(() => { for (const [id, g] of dedup) upd.run(g.lyrics, g.duration, id); });
tx();
db.close();
console.log(`Rewrote ${dedup.size} generation(s) (lyrics + duration).`);
