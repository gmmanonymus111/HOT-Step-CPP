#!/usr/bin/env node
/**
 * Merge one artist's lyrics set into another and retire the source (2026-08-08).
 *
 * Written for the Electric Callboy tidy-up: a Training Studio export named
 * itself after the majority album tag (lyricStudioExport.ts detectAlbum) and
 * landed as a second "TEKKNO" set alongside the original Genius fetch, so the
 * artist had two overlapping albums with generations split across both.
 *
 * What it does, in order:
 *   1. appends songs present in --from but missing from --into (matched by
 *      title, feat-credit tolerant), carrying every field across
 *   2. repoints --from's generations onto --into's profile
 *   3. deletes --from's profiles, album presets and the set itself
 *
 * Nothing is merged into a song that already exists in --into; the target set's
 * own rows always win. The DB is backed up via SQLite's online backup API
 * before any write. DRY RUN by default.
 *
 *   node server/scripts/merge-lyrics-sets.mjs --from 215 --into 39
 *   node server/scripts/merge-lyrics-sets.mjs --from 215 --into 39 --apply
 *
 *   --from <id>        lyrics set to retire (required)
 *   --into <id>        lyrics set to keep (required)
 *   --album <name>     also set the surviving set's album label
 *   --keep-source      migrate everything but do NOT delete the source set
 *   --db <file>        SQLite db (default server/data/hotstep.db)
 */
import path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(HERE, '..', 'package.json'));
const Database = require('better-sqlite3');

const argv = process.argv.slice(2);
const APPLY = argv.includes('--apply');
const KEEP_SOURCE = argv.includes('--keep-source');
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d; };
const FROM = Number(opt('from', ''));
const INTO = Number(opt('into', ''));
const ALBUM = opt('album', null);
const DB_PATH = path.resolve(opt('db', path.join(HERE, '..', 'data', 'hotstep.db')));

if (!Number.isInteger(FROM) || !Number.isInteger(INTO) || FROM <= 0 || INTO <= 0 || FROM === INTO) {
  console.error('usage: merge-lyrics-sets.mjs --from <id> --into <id> [--album <name>] [--apply] [--keep-source]');
  process.exit(2);
}

const norm = (s) => String(s ?? '').toLowerCase()
  .replace(/[‘’“”]/g, "'")
  .replace(/\s*[([]\s*(feat|ft|with)\b[^)\]]*[)\]]\s*/gi, ' ')
  .replace(/[^a-z0-9']+/g, ' ')
  .trim();

const db = new Database(DB_PATH);
const setRow = (id) => db.prepare('SELECT id, artist_id, album, songs FROM lyrics_sets WHERE id = ?').get(id);
const src = setRow(FROM);
const dst = setRow(INTO);
if (!src) { console.error(`lyrics set ${FROM} not found`); process.exit(2); }
if (!dst) { console.error(`lyrics set ${INTO} not found`); process.exit(2); }
if (src.artist_id !== dst.artist_id) {
  console.error(`refusing to merge across artists (${FROM} is artist ${src.artist_id}, ${INTO} is artist ${dst.artist_id})`);
  process.exit(2);
}

const srcSongs = JSON.parse(src.songs);
const dstSongs = JSON.parse(dst.songs);
const have = new Set(dstSongs.map(s => norm(s.title)));
const toAdd = srcSongs.filter(s => !have.has(norm(s.title)));

const srcProfiles = db.prepare('SELECT id FROM profiles WHERE lyrics_set_id = ?').all(FROM).map(r => r.id);
const dstProfiles = db.prepare('SELECT id FROM profiles WHERE lyrics_set_id = ? ORDER BY created_at DESC').all(INTO).map(r => r.id);
const srcPresets = db.prepare('SELECT id FROM album_presets WHERE lyrics_set_id = ?').all(FROM).map(r => r.id);
const target = dstProfiles[0];

const gens = srcProfiles.length
  ? db.prepare(`SELECT id, title, profile_id FROM generations WHERE profile_id IN (${srcProfiles.map(() => '?').join(',')})`).all(...srcProfiles)
  : [];

console.log(`\nDB:   ${DB_PATH}`);
console.log(`From: set ${FROM} "${src.album ?? '(none)'}" — ${srcSongs.length} song(s), profile(s) ${srcProfiles.join(', ') || '—'}, preset(s) ${srcPresets.join(', ') || '—'}`);
console.log(`Into: set ${INTO} "${dst.album ?? '(none)'}" — ${dstSongs.length} song(s), profile(s) ${dstProfiles.join(', ') || '—'}`);
console.log(`Mode: ${APPLY ? 'APPLY' : 'DRY RUN'}${KEEP_SOURCE ? ' (--keep-source)' : ''}\n`);

if (!target && gens.length) {
  console.error(`set ${INTO} has no profile to receive ${gens.length} generation(s) — aborting.`);
  process.exit(2);
}

console.log(`songs to append: ${toAdd.length}`);
for (const s of toAdd) console.log(`  + ${s.title}${s.caption ? '  [enriched]' : ''}`);
console.log(`\ngenerations to repoint onto profile ${target ?? '—'}: ${gens.length}`);
for (const g of gens) console.log(`  ~ #${g.id} "${g.title}" (profile ${g.profile_id})`);
if (ALBUM !== null) console.log(`\nalbum label: "${dst.album ?? '(none)'}" -> "${ALBUM}"`);
if (!KEEP_SOURCE) console.log(`\nwill delete: set ${FROM}, profile(s) ${srcProfiles.join(', ') || '—'}, preset(s) ${srcPresets.join(', ') || '—'}`);

if (!APPLY) { console.log('\nDRY RUN — nothing written. Re-run with --apply.\n'); db.close(); process.exit(0); }

// Online backup first — safe against the running server's open handle.
const stampNow = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
const bak = DB_PATH.replace(/\.db$/i, `_backup_mergesets_${stampNow}.db`);
await db.backup(bak);
console.log(`\nBacked up to ${path.basename(bak)}`);

const merged = [...dstSongs, ...toAdd];
const run = db.transaction(() => {
  db.prepare('UPDATE lyrics_sets SET songs = ?, max_songs = ? WHERE id = ?')
    .run(JSON.stringify(merged), merged.length, INTO);
  if (ALBUM !== null) db.prepare('UPDATE lyrics_sets SET album = ? WHERE id = ?').run(ALBUM, INTO);

  for (const g of gens) db.prepare('UPDATE generations SET profile_id = ? WHERE id = ?').run(target, g.id);

  if (!KEEP_SOURCE) {
    for (const id of srcPresets) db.prepare('DELETE FROM album_presets WHERE id = ?').run(id);
    for (const id of srcProfiles) db.prepare('DELETE FROM profiles WHERE id = ?').run(id);
    db.prepare('DELETE FROM lyrics_sets WHERE id = ?').run(FROM);
  }
});
run();
db.close();

console.log(`Appended ${toAdd.length} song(s) (set ${INTO} now ${merged.length}), repointed ${gens.length} generation(s)` +
            (KEEP_SOURCE ? '.' : `, deleted set ${FROM}.`));
