#!/usr/bin/env node
/**
 * Repoint every album preset's adapter paths at the per-base adapter layout
 * (2026-07-28). The artist stem is taken from the currently stored path's
 * filename and looked up in the new hierarchy:
 *
 *   adapter_path     ->  <adapters>/dit-xl-thirds/<stem>       when trained there
 *                        <adapters>/dit-xl-base-turbo/<stem>   otherwise (archive)
 *   lm_adapter_path  ->  <adapters>/lm-4b/<stem>
 *
 * A stem with no folder in the new layout is left untouched and warned about.
 * Rows already pointing inside <adapters> keep their setting (hand-fixed).
 * The DB is backed up via SQLite's online backup API before any write.
 * DRY RUN by default.
 *
 *   node server/scripts/retarget-album-presets.mjs
 *   node server/scripts/retarget-album-presets.mjs --apply
 *
 *   --adapters <dir>   adapters root (default M:\HOT-Step-CPP\Adapters)
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
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d; };
const ROOT = path.resolve(opt('adapters', 'M:\\HOT-Step-CPP\\Adapters'));
const DB_PATH = path.resolve(opt('db', path.join(HERE, '..', 'data', 'hotstep.db')));

const stemOf = (p) => {
  if (!p) return '';
  const base = String(p).split(/[\\/]/).filter(Boolean).pop() || '';
  // Bare file, canonical weights file (stem = its folder), or already a folder.
  if (/^adapter_model\.safetensors$/i.test(base)) {
    const parts = String(p).split(/[\\/]/).filter(Boolean);
    return parts[parts.length - 2] || '';
  }
  return base.replace(/\.safetensors$/i, '').replace(/-(0\.6B|1\.7B|4B)$/i, '');
};

const hasAdapter = (dir) => fs.existsSync(path.join(dir, 'adapter_model.safetensors'));

const ditTarget = (stem) => {
  for (const basedir of ['dit-xl-thirds', 'dit-xl-base-turbo']) {
    const d = path.join(ROOT, basedir, stem);
    if (hasAdapter(d)) return d;
  }
  return '';
};
const lmTarget = (stem) => {
  const d = path.join(ROOT, 'lm-4b', stem);
  return hasAdapter(d) ? d : '';
};

const db = new Database(DB_PATH);
const rows = db.prepare(
  `SELECT ap.lyrics_set_id AS id, ls.album, ap.adapter_path, ap.lm_adapter_path
   FROM album_presets ap LEFT JOIN lyrics_sets ls ON ls.id = ap.lyrics_set_id`).all();

const updates = [];
const warns = [];
for (const r of rows) {
  const upd = { id: r.id, album: r.album || `(set ${r.id})` };

  for (const [field, target] of [['adapter_path', ditTarget], ['lm_adapter_path', lmTarget]]) {
    const cur = r[field] || '';
    if (!cur) continue;
    if (path.resolve(cur).toLowerCase().startsWith(ROOT.toLowerCase())) continue;  // hand-fixed already
    const stem = stemOf(cur);
    const dest = stem ? target(stem) : '';
    if (dest) upd[field] = dest;
    else warns.push(`${upd.album}: ${field} stem "${stem}" has no folder in the new layout — left as ${cur}`);
  }
  if (upd.adapter_path || upd.lm_adapter_path) updates.push(upd);
}

console.log(`\nDB: ${DB_PATH}\nAdapters: ${ROOT}\nMode: ${APPLY ? 'APPLY' : 'DRY RUN'}\n`);
console.log(`${rows.length} preset(s), ${updates.length} to update, ${warns.length} warning(s).\n`);
for (const u of updates.slice(0, 6)) {
  console.log(`${u.album}`);
  if (u.adapter_path) console.log(`   dit -> ${path.relative(ROOT, u.adapter_path)}`);
  if (u.lm_adapter_path) console.log(`   lm  -> ${path.relative(ROOT, u.lm_adapter_path)}`);
}
if (updates.length > 6) console.log(`… and ${updates.length - 6} more, same shape`);
for (const w of warns) console.log(`WARN  ${w}`);

if (!APPLY) { console.log('\nDRY RUN — nothing written. Re-run with --apply.\n'); db.close(); process.exit(0); }

// Online backup first — safe against the running server's open handle.
const stampNow = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
const bak = DB_PATH.replace(/\.db$/i, `_backup_presets_${stampNow}.db`);
await db.backup(bak);
console.log(`\nBacked up to ${path.basename(bak)}`);

const setBoth = db.prepare('UPDATE album_presets SET adapter_path = ?, lm_adapter_path = ? WHERE lyrics_set_id = ?');
const cur = db.prepare('SELECT adapter_path, lm_adapter_path FROM album_presets WHERE lyrics_set_id = ?');
let n = 0;
for (const u of updates) {
  const c = cur.get(u.id);
  setBoth.run(u.adapter_path ?? c.adapter_path, u.lm_adapter_path ?? c.lm_adapter_path, u.id);
  n++;
}
db.close();
console.log(`Updated ${n} preset(s).`);
