#!/usr/bin/env npx tsx
/**
 * Validate and apply re-planned captions produced by the batch agents
 * (2026-08-08). Run with `npx tsx` from server/.
 *
 * Reads every out-*.json in the work dir — each a {generationId: caption} map —
 * validates each caption against the training-caption format, and writes the
 * survivors. Anything that fails is reported and NOT written, so a bad batch
 * degrades to "some sets still need a re-run" rather than corrupting captions.
 *
 * The upper length bound is load-bearing, not cosmetic. The Haiku arm of the
 * set-81 pilot returned 2313-2689 chars against a training band of 1184-1553:
 * nine sentences, but nearly twice the size, and describing the LYRICS ("as the
 * tour bus pulls onto the old block") rather than the sound. A lower bound
 * alone passed all of it. The tag-leak and quote checks catch the same failure
 * from the other end — that arm also emitted "the bridge-sparse-and-quiet" as
 * prose and quoted lyrics back verbatim.
 *
 *   npx tsx scripts/caption-replan-apply.ts --work <dir>
 *   npx tsx scripts/caption-replan-apply.ts --work <dir> --apply
 *
 *   --work <dir>   dir holding out-*.json (required)
 *   --db <file>    SQLite db (default server/data/hotstep.db)
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';

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
if (!WORK || !fs.existsSync(WORK)) { console.error('usage: caption-replan-apply.ts --work <dir> [--apply]'); process.exit(2); }

/** Training captions run 1149-1553 chars; allow a little headroom either side. */
const MIN_CHARS = 1100, MAX_CHARS = 1700;
const KEY_IN_PROSE = /\b[A-G][#b]?\s+(Major|Minor)\b/;          // case-sensitive: "a minor-key riff" is fine
const BPM_IN_PROSE = /\b\d+\s*bpm\b|\b\d+\/\d\b/i;
// Tag leakage means a SECTION TAG pasted in as prose ("the bridge-sparse-and-quiet"),
// which is what the Haiku arm did. It does NOT mean any hyphenated compound
// containing a section word: "chorus-pedal shimmer" and "chorus-treated tone"
// are the guitar effect, and "verse-chorus cycles" / "pre-chorus-into-chorus
// lift" are exactly the arrangement prose sentences 7-9 are supposed to use.
// So: a literal bracket, or a section word hyphenated to a TAG DESCRIPTOR.
const SECTION_TAG_LEAK =
  /\[|\b(intro|verse|pre-?chorus|chorus|bridge|outro|breakdown)-(sparse|heavy|high|quiet|screamed|whispered|menacing|building|final|instrumental|spoken)\b/i;
// "listeners feel" is the banned vagueness; "drums sitting close to the
// listener" is a normal mix descriptor, so the bare noun must not trip this.
const SLOP = /\b(captivating|emotionally resonant|nice vibe|good energy|keeps you moving|hard to resist|a captivating journey|listeners? (?:feel|will feel|are left))\b/i;

const sentenceCount = (s: string) => s.split(/(?<=[.!?])\s+/).filter(x => x.trim()).length;

export function validateCaption(c: unknown, artist: string, title: string): string[] {
  const f: string[] = [];
  if (typeof c !== 'string' || !c.trim()) return ['missing'];
  const s = c.trim();
  const n = sentenceCount(s);
  if (n !== 9) f.push(`sentences=${n}`);
  if (s.length < MIN_CHARS) f.push(`too-short=${s.length}`);
  if (s.length > MAX_CHARS) f.push(`too-long=${s.length}`);
  if (/\n/.test(s)) f.push('multiline');
  if (KEY_IN_PROSE.test(s)) f.push('key-in-prose');
  if (BPM_IN_PROSE.test(s)) f.push('bpm-in-prose');
  if (SECTION_TAG_LEAK.test(s)) f.push('section-tag-leak');
  if (SLOP.test(s)) f.push('slop-language');
  if (/["“”]/.test(s)) f.push('quotes-lyrics');
  const esc = (x: string) => x.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  if (artist && new RegExp(`\\b${esc(artist)}\\b`, 'i').test(s)) f.push('artist-named');
  // Only multi-word titles are checked. A one-word title is usually a common
  // noun the caption may legitimately need — "hymn-like, almost chorale motion"
  // for a song called Hymn, "the lead vocal delivered dry" for one called
  // Delivered — and flagging those rejects good captions for no reason.
  const bare = title.replace(/ - (Fable|Opus|Sonnet|Haiku|Claude|Nemotron)[^-]*$/i, '').trim();
  if (bare.includes(' ') && new RegExp(`\\b${esc(bare)}\\b`, 'i').test(s)) f.push('title-named');
  return f;
}

const db = new Database(DB_PATH);
const meta = db.prepare(`
  SELECT g.id, g.title, g.caption, a.name AS artist
  FROM generations g
  JOIN profiles p ON p.id = g.profile_id
  JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
  JOIN artists a ON a.id = ls.artist_id
`).all() as any[];
const byId = new Map(meta.map(m => [m.id, m]));

const files = fs.readdirSync(WORK).filter(f => /^out-.*\.json$/i.test(f));
const good: Array<{ id: number; caption: string }> = [];
const bad: Array<{ id: number; file: string; why: string[]; len: number }> = [];
let unknown = 0;

for (const f of files) {
  let obj: Record<string, unknown>;
  try {
    obj = JSON.parse(fs.readFileSync(path.join(WORK, f), 'utf8').replace(/^```(json)?|```$/gm, '').trim());
  } catch (e: any) { console.log(`  ${f}: UNPARSEABLE — ${e.message}`); continue; }

  for (const [k, v] of Object.entries(obj)) {
    const id = Number(k);
    const m = byId.get(id);
    if (!m) { unknown++; continue; }
    const why = validateCaption(v, m.artist, m.title);
    if (why.length) bad.push({ id, file: f, why, len: typeof v === 'string' ? v.length : 0 });
    else good.push({ id, caption: (v as string).trim() });
  }
}

// Last writer wins is not good enough when two arms produced the same id.
const dedup = new Map<number, string>();
for (const g of good) if (!dedup.has(g.id)) dedup.set(g.id, g.caption);

console.log(`\nDB:    ${DB_PATH}`);
console.log(`work:  ${WORK}  (${files.length} output file(s))`);
console.log(`Mode:  ${APPLY ? 'APPLY' : 'DRY RUN'}\n`);
console.log(`valid:    ${dedup.size} caption(s)`);
console.log(`rejected: ${bad.length}`);
if (unknown) console.log(`unknown ids (not in DB): ${unknown}`);
for (const b of bad.slice(0, 20)) {
  const m = byId.get(b.id);
  console.log(`  REJECT #${b.id} ${String(b.len).padStart(4)}ch [${b.file}] ${b.why.join(',')}  ${m ? m.title : ''}`);
}
if (bad.length > 20) console.log(`  … and ${bad.length - 20} more`);

if (!dedup.size) { console.log('\nNothing valid to write.\n'); process.exit(1); }
if (!APPLY) { console.log('\nDRY RUN — nothing written. Re-run with --apply.\n'); process.exit(0); }

const stampNow = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
const bak = DB_PATH.replace(/\.db$/i, `_backup_replan_${stampNow}.db`);
await db.backup(bak);
console.log(`\nBacked up to ${path.basename(bak)}`);

const upd = db.prepare('UPDATE generations SET caption = ? WHERE id = ?');
const tx = db.transaction(() => { for (const [id, c] of dedup) upd.run(c, id); });
tx();
db.close();
console.log(`Wrote ${dedup.size} caption(s).`);
