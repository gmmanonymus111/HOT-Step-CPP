#!/usr/bin/env node
/**
 * Backfill per-song `duration` onto EVERY lyrics set from its dataset's
 * sidecars, so computeAlbumEnrichment can measure each artist's real vocal
 * pacing (words/sec) (2026-08-08).
 *
 * Why: the lyric duration budget and post-generation reconciliation both need
 * the ARTIST's pacing — per-artist medians span 0.51 w/s (Muse) to 3.29
 * (Eminem), a 6.5x spread, so the global 1.20 fallback misprices most artists.
 * Durations exist in the dataset sidecars but were dropped by the Lyric Studio
 * export until today.
 *
 * Set → dataset mapping goes through the album preset's adapter paths: an
 * adapter lives at <Adapters>/<base>/<stem>/<run> and its dataset at
 * <Datasets>/<stem>. DiT path first, LM fallback. Sets with no preset or no
 * matching dataset folder are reported and skipped.
 *
 * Title matching and fill discipline are identical to backfill-set-captions:
 * feat-credit tolerant, fills EMPTY duration fields only, lyrics untouched.
 * Online backup before any write. DRY RUN by default.
 *
 *   node server/scripts/backfill-set-durations.mjs
 *   node server/scripts/backfill-set-durations.mjs --apply
 *
 *   --datasets <dir>  dataset root (default M:\HOT-Step-CPP\Datasets)
 *   --set <id>        only this set
 *   --db <file>       SQLite db (default server/data/hotstep.db)
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
const DS_ROOT = path.resolve(opt('datasets', 'M:\\HOT-Step-CPP\\Datasets'));
const ONLY_SET = Number(opt('set', '0'));
const DB_PATH = path.resolve(opt('db', path.join(HERE, '..', 'data', 'hotstep.db')));

const norm = (s) => String(s ?? '').toLowerCase()
  .replace(/[\u2018\u2019\u201c\u201d]/g, "'")
  .replace(/[^a-z0-9']+/g, ' ')
  .trim();
const normNoFeat = (s) => norm(String(s ?? '').replace(/\s*[([]\s*(feat|ft|with)\b[^)\]]*[)\]]\s*$/i, ''));

/**
 * Candidate titles from a sidecar filename. Dataset naming varies by ripper:
 *   "Artist - Album - 01-01 Title.txt"   (dashes + disc-track)
 *   "A1. American Idiot.txt"             (vinyl side+track)
 *   "07 Title.txt" / "07. Title.txt"     (plain track number)
 * Returns every plausible reading — raw and prefix-stripped — because
 * stripping is ambiguous for titles that ARE numbers ("21 Guns", "1985"): the
 * caller registers all candidates and first-seen wins, so a genuine number
 * title still matches via its raw form.
 */
function titleCandidates(file) {
  const stem = path.basename(file).replace(/\.txt$/i, '');
  const parts = stem.split(' - ');
  const last = (parts.length > 1 ? parts[parts.length - 1] : stem).trim();
  const out = [last];
  // vinyl side+track ("A1." / "B12 ") or plain/disc track number ("07 ", "1-01 ")
  const stripped = last.replace(/^(?:[A-Da-d]\d{1,2}|\d{1,3}(?:[-.]\d{1,3})?)[.)\-]?\s+/, '').trim();
  if (stripped && stripped !== last) out.push(stripped);
  return out;
}

/** Dataset stem from an adapter path: <...>/Adapters/<base>/<stem>/<run>. */
function stemFromAdapterPath(p) {
  if (!p) return '';
  const parts = String(p).split(/[\\/]/).filter(Boolean);
  const i = parts.findIndex(x => x.toLowerCase() === 'adapters');
  return i >= 0 && parts.length > i + 2 ? parts[i + 2] : '';
}

/** stem -> Map(normTitle -> duration) for one dataset dir, cached. */
const dsCache = new Map();
function datasetDurations(stem) {
  if (dsCache.has(stem)) return dsCache.get(stem);
  const dir = path.join(DS_ROOT, stem);
  let entry = null;
  if (stem && fs.existsSync(dir)) {
    const byTitle = new Map(), byTitleNoFeat = new Map();
    const stems = [];   // for the suffix fallback
    for (const f of fs.readdirSync(dir)) {
      if (!/\.txt$/i.test(f) || /\.bak$/i.test(f)) continue;
      let txt; try { txt = fs.readFileSync(path.join(dir, f), 'utf8'); } catch { continue; }
      const dm = /^duration\s*:\s*([0-9.]+)/mi.exec(txt);
      if (!dm) continue;
      const dur = Math.round(Number(dm[1]) * 100) / 100;
      if (!(dur > 0)) continue;
      for (const t of titleCandidates(f)) {
        if (!byTitle.has(norm(t))) byTitle.set(norm(t), dur);
        if (!byTitleNoFeat.has(normNoFeat(t))) byTitleNoFeat.set(normNoFeat(t), dur);
      }
      const rawStem = path.basename(f).replace(/\.txt$/i, '');
      stems.push({ normStem: norm(rawStem), normStemNoFeat: normNoFeat(rawStem), dur });
    }
    if (byTitle.size) entry = { byTitle, byTitleNoFeat, stems };
  }
  dsCache.set(stem, entry);
  return entry;
}

/**
 * Fallback for filename conventions the candidate parser doesn't know
 * ("01.Lycanthrope", "02-blink-182-dont_leave_me"): after norm() flattens
 * every separator to spaces, the true title survives as the SUFFIX of the
 * stem. Accepted only when exactly ONE file matches — ambiguity ("Home" vs
 * "Coming Home") skips rather than guesses — and never for very short titles.
 */
function suffixLookup(ds, title) {
  const t = norm(title), tNf = normNoFeat(title);
  if (t.length < 4) return 0;
  const hits = ds.stems.filter(x =>
    x.normStem === t || x.normStem.endsWith(' ' + t) ||
    x.normStemNoFeat === tNf || x.normStemNoFeat.endsWith(' ' + tNf));
  return hits.length === 1 ? hits[0].dur : 0;
}

const db = new Database(DB_PATH);
const sets = db.prepare(`
  SELECT ls.id, ls.album, ls.songs, a.name AS artist,
         ap.adapter_path, ap.lm_adapter_path
  FROM lyrics_sets ls
  JOIN artists a ON a.id = ls.artist_id
  LEFT JOIN album_presets ap ON ap.lyrics_set_id = ls.id
  ORDER BY ls.id`).all();

const updates = [];
let songsFilled = 0, alreadyHad = 0, unmatched = 0;
const noDataset = [];

for (const s of sets) {
  if (ONLY_SET && s.id !== ONLY_SET) continue;
  const stems = [...new Set([stemFromAdapterPath(s.adapter_path), stemFromAdapterPath(s.lm_adapter_path)].filter(Boolean))];
  const ds = stems.map(datasetDurations).find(Boolean);
  if (!ds) { noDataset.push(`set ${s.id} ${s.artist} — ${s.album ?? '(all)'} [stems: ${stems.join(', ') || 'none'}]`); continue; }

  let songs; try { songs = JSON.parse(s.songs); } catch { continue; }
  let filled = 0;
  for (const song of songs) {
    if (Number(song.duration) > 0) { alreadyHad++; continue; }
    const dur = ds.byTitle.get(norm(song.title))
      ?? ds.byTitleNoFeat.get(normNoFeat(song.title))
      ?? suffixLookup(ds, song.title);
    if (!dur) { unmatched++; continue; }
    song.duration = dur;
    filled++;
  }
  if (filled) {
    updates.push({ id: s.id, artist: s.artist, album: s.album, filled, total: songs.length, songs });
    songsFilled += filled;
  }
}

console.log(`\nDB:       ${DB_PATH}`);
console.log(`Datasets: ${DS_ROOT}`);
console.log(`Mode:     ${APPLY ? 'APPLY' : 'DRY RUN'}\n`);
console.log(`sets to update: ${updates.length}   songs filled: ${songsFilled}   already had duration: ${alreadyHad}   no title match: ${unmatched}`);
console.log(`sets with no usable dataset: ${noDataset.length}\n`);
for (const u of updates.slice(0, 20)) {
  console.log(`  set ${String(u.id).padStart(3)}  +${String(u.filled).padStart(2)}/${String(u.total).padEnd(2)}  ${u.artist} — ${u.album ?? '(all)'}`);
}
if (updates.length > 20) console.log(`  … and ${updates.length - 20} more sets, same shape`);
if (noDataset.length) {
  console.log('\nno dataset found for:');
  for (const n of noDataset.slice(0, 12)) console.log(`  ${n}`);
  if (noDataset.length > 12) console.log(`  … and ${noDataset.length - 12} more`);
}

if (!updates.length) { console.log('\nNothing to do.\n'); db.close(); process.exit(0); }
if (!APPLY) { console.log('\nDRY RUN — nothing written. Re-run with --apply.\n'); db.close(); process.exit(0); }

const stampNow = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
const bak = DB_PATH.replace(/\.db$/i, `_backup_setdurations_${stampNow}.db`);
await db.backup(bak);
console.log(`\nBacked up to ${path.basename(bak)}`);

const upd = db.prepare('UPDATE lyrics_sets SET songs = ? WHERE id = ?');
const tx = db.transaction(() => { for (const u of updates) upd.run(JSON.stringify(u.songs), u.id); });
tx();
db.close();
console.log(`Updated ${updates.length} set(s), ${songsFilled} song duration(s).`);
