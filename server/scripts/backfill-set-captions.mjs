#!/usr/bin/env node
/**
 * Backfill audio-grounded metadata onto a plain-Genius lyrics set from the
 * dataset sidecars of the same artist (2026-08-08).
 *
 * A lyrics set exported from the Training Studio carries per-song `caption`,
 * `genre`, `bpm`, `key`, `signature` and `language`. A set fetched straight
 * from Genius carries only `title`, `album` and `lyrics`, so
 * computeAlbumEnrichment() returns null for it and the metadata planner falls
 * back to inventing a 1-3 sentence tag-list caption — a format the sound
 * adapter was never trained on. The sidecars next to the training audio hold
 * exactly the missing fields, so copy them across by title.
 *
 * Only EMPTY fields are filled; an existing value is never overwritten unless
 * --force is given. `lyrics` is never touched — the Genius text stays canonical.
 * The DB is backed up via SQLite's online backup API before any write.
 * DRY RUN by default.
 *
 *   node server/scripts/backfill-set-captions.mjs --set 39 --dataset "M:\HOT-Step-CPP\Datasets\electriccallboy"
 *   node server/scripts/backfill-set-captions.mjs --set 39 --dataset "…\electriccallboy" --apply
 *
 *   --set <id>         lyrics_sets.id to fill (required)
 *   --dataset <dir>    dataset folder holding <stem>.txt sidecars (required)
 *   --force            overwrite fields that already have a value
 *   --db <file>        SQLite db (default server/data/hotstep.db)
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
const FORCE = argv.includes('--force');
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d; };
const SET_ID = Number(opt('set', ''));
const DATASET = opt('dataset', '');
const DB_PATH = path.resolve(opt('db', path.join(HERE, '..', 'data', 'hotstep.db')));

if (!Number.isInteger(SET_ID) || SET_ID <= 0 || !DATASET) {
  console.error('usage: backfill-set-captions.mjs --set <id> --dataset <dir> [--apply] [--force]');
  process.exit(2);
}
if (!fs.existsSync(DATASET)) { console.error(`dataset dir not found: ${DATASET}`); process.exit(2); }

/** Fields a sidecar may contribute, in the order the Training Studio writes them.
 *  `duration` feeds the per-artist vocal-pacing rate (words/sec) — without it
 *  the lyric planner falls back to a global constant. */
const FIELDS = ['caption', 'genre', 'bpm', 'key', 'signature', 'language', 'duration'];
const NUMERIC_FIELDS = new Set(['bpm', 'duration']);

/** Normalised title for matching: case/punctuation/spacing insensitive. */
const norm = (s) => String(s ?? '').toLowerCase()
  .replace(/[\u2018\u2019\u201c\u201d]/g, "'")
  .replace(/[^a-z0-9']+/g, ' ')
  .trim();

/** Same, with a trailing "(feat. …)" / "(with …)" credit removed. */
const normNoFeat = (s) => norm(String(s ?? '').replace(/\s*[([]\s*(feat|ft|with)\b[^)\]]*[)\]]\s*$/i, ''));

/**
 * Song title from a sidecar filename: "Artist - Album - 01-02 Title.txt".
 * Falls back to the whole stem when there is no " - " separator.
 */
function titleFromSidecar(file) {
  const stem = path.basename(file).replace(/\.txt$/i, '');
  const parts = stem.split(' - ');
  const last = parts.length > 1 ? parts[parts.length - 1] : stem;
  return last.replace(/^\d{1,3}([-.]\d{1,3})?\s+/, '').trim();
}

/** Parse the scalar `key: value` head of an Option-A sidecar (stops at `lyrics:`). */
function readSidecarHead(file) {
  const out = {};
  for (const raw of fs.readFileSync(file, 'utf8').split(/\r?\n/)) {
    if (/^lyrics\s*:/i.test(raw)) break;
    const m = /^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$/.exec(raw);
    if (m) out[m[1].toLowerCase()] = m[2].trim();
  }
  return out;
}

// ── collect sidecars ────────────────────────────────────────────────────────
const sidecars = fs.readdirSync(DATASET)
  .filter(f => /\.txt$/i.test(f) && !/\.bak$/i.test(f))
  .map(f => path.join(DATASET, f));

const byTitle = new Map();   // normalised title -> parsed fields
const byTitleNoFeat = new Map();
for (const file of sidecars) {
  const head = readSidecarHead(file);
  const vals = {};
  for (const f of FIELDS) {
    const v = head[f];
    if (v === undefined || v === '' || v.toUpperCase() === 'N/A') continue;
    vals[f] = NUMERIC_FIELDS.has(f) ? Number(v) : v;
  }
  for (const nf of NUMERIC_FIELDS) if (Number.isNaN(vals[nf])) delete vals[nf];
  if (!Object.keys(vals).length) continue;
  const t = titleFromSidecar(file);
  if (!byTitle.has(norm(t))) byTitle.set(norm(t), { title: t, vals });
  if (!byTitleNoFeat.has(normNoFeat(t))) byTitleNoFeat.set(normNoFeat(t), { title: t, vals });
}

// ── match against the set ───────────────────────────────────────────────────
const db = new Database(DB_PATH);
const row = db.prepare('SELECT id, artist_id, album, songs FROM lyrics_sets WHERE id = ?').get(SET_ID);
if (!row) { console.error(`lyrics set ${SET_ID} not found in ${DB_PATH}`); process.exit(2); }

const songs = JSON.parse(row.songs);
const filled = [];
const unmatched = [];
const usedSidecars = new Set();

for (const song of songs) {
  const hit = byTitle.get(norm(song.title)) ?? byTitleNoFeat.get(normNoFeat(song.title));
  if (!hit) { unmatched.push(song.title); continue; }
  usedSidecars.add(norm(hit.title));

  const added = [];
  for (const f of FIELDS) {
    if (!(f in hit.vals)) continue;
    const cur = song[f];
    const empty = cur === undefined || cur === null || cur === '' || (NUMERIC_FIELDS.has(f) && !(Number(cur) > 0));
    if (!empty && !FORCE) continue;
    song[f] = hit.vals[f];
    added.push(f);
  }
  if (added.length) filled.push({ title: song.title, from: hit.title, added });
}

const spareSidecars = [...byTitle.entries()].filter(([k]) => !usedSidecars.has(k)).map(([, v]) => v.title);

console.log(`\nDB:      ${DB_PATH}`);
console.log(`Set:     ${SET_ID} — ${row.album || '(all songs)'} (artist ${row.artist_id}), ${songs.length} song(s)`);
console.log(`Dataset: ${DATASET} — ${sidecars.length} sidecar(s), ${byTitle.size} with usable fields`);
console.log(`Mode:    ${APPLY ? 'APPLY' : 'DRY RUN'}${FORCE ? ' (--force: overwriting existing values)' : ''}\n`);
console.log(`${filled.length} song(s) to fill, ${unmatched.length} unmatched, ${spareSidecars.length} sidecar(s) unused.\n`);

for (const f of filled) {
  const via = norm(f.title) === norm(f.from) ? '' : `  ← "${f.from}"`;
  console.log(`  + ${f.title}${via}   [${f.added.join(', ')}]`);
}
if (unmatched.length) console.log(`\n  no sidecar: ${unmatched.join(', ')}`);
if (spareSidecars.length) console.log(`  not in set: ${spareSidecars.join(', ')}`);

if (!filled.length) { console.log('\nNothing to do.\n'); db.close(); process.exit(0); }
if (!APPLY) { console.log('\nDRY RUN — nothing written. Re-run with --apply.\n'); db.close(); process.exit(0); }

// Online backup first — safe against the running server's open handle.
const stampNow = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
const bak = DB_PATH.replace(/\.db$/i, `_backup_setcaptions_${stampNow}.db`);
await db.backup(bak);
console.log(`\nBacked up to ${path.basename(bak)}`);

db.prepare('UPDATE lyrics_sets SET songs = ? WHERE id = ?').run(JSON.stringify(songs), SET_ID);
db.close();
console.log(`Filled ${filled.length} song(s) in set ${SET_ID}.`);
