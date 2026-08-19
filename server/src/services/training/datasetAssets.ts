// training/datasetAssets.ts — "what has this dataset actually got?" off disk
//
// The dataset row says whether a dataset was BUILT and nothing else, so the
// list used to be silent about the three stages that matter most: preprocess,
// train-dit, train-lm. Every one of those leaves an artefact on disk, and disk
// stays the source of truth (D3) — so the flags are re-read on every request
// rather than cached in a column. Deleting a tensors folder or an adapter shows
// up on the next refresh.
//
// The friendly ALBUM name is the one exception: detecting it means reading
// embedded tags, which the list cannot afford per request. It is detected once
// (label store first, then a bounded probe of the audio files) and cached in
// training_datasets.album_name, then refreshed for free whenever a caller
// already holds the sample list.
//
// Never throws — an unreadable adapter root or tensors dir degrades to
// "nothing trained", never to a 500.

import fs from 'fs';
import path from 'path';
import * as audioMeta from './audioMeta.js';
import { latestRunDir, lmSizeFromSlug } from './adapterLayout.js';
import { scanAudioFiles } from './datasetScan.js';
import * as repo from './datasetsRepo.js';
import { readAllLabels } from './labelStore.js';
import { countPreprocessedVariants } from './preprocessStatus.js';
import { readTrainDitStatus } from './trainDitStatus.js';
import { adapterLmRoot, lmArtistDirFor, safeAdapterName } from './trainLmStatus.js';
import type {
  DatasetAssets, LmSize, TrainingAdapterHit, TrainingDatasetRow, TrainingSample,
} from './types.js';

// ── Trained-adapter lookup ───────────────────────────────────────────────
//
// Both finders live here rather than in lyricStudioExport.ts (their original
// home) because the dataset list, the batch wizard and the Lyric Studio export
// must all agree on what "this dataset has an adapter" means.

const LM_SIZES: LmSize[] = ['4B', '1.7B', '0.6B'];

function weightsMtime(dir: string): number {
  for (const f of ['adapter_model.safetensors', 'lokr_weights.safetensors']) {
    try { return fs.statSync(path.join(dir, f)).mtimeMs; } catch { /* next */ }
  }
  return 0;
}

/** Newest trained planner-LM adapter for this dataset across every size root
 *  (plus the legacy flat `lm/<name>-<size>` dir). null when none trained. */
export function findLmAdapter(ds: TrainingDatasetRow): TrainingAdapterHit | null {
  let best: { hit: TrainingAdapterHit; mtime: number } | null = null;
  for (const size of LM_SIZES) {
    const candidates = [
      latestRunDir(lmArtistDirFor(ds.slug, size)),
      path.join(adapterLmRoot(), `${safeAdapterName(ds.slug)}-${size}`),
    ];
    for (const dir of candidates) {
      if (!dir) continue;
      const mtime = weightsMtime(dir);
      if (!mtime || (best && mtime <= best.mtime)) continue;
      // A legacy flat dir's parent is `lm/`, whose slug lookup fails → keep the
      // loop's size; a per-size run dir confirms it from the folder itself.
      const sizeOfDir = lmSizeFromSlug(path.basename(path.dirname(path.dirname(dir)))) || size;
      best = {
        mtime,
        hit: { path: dir, kind: 'lm', detail: sizeOfDir, trainedAt: new Date(mtime).toISOString() },
      };
    }
  }
  return best ? best.hit : null;
}

/** The dataset's newest trained DiT adapter, resolved exactly the way the
 *  Train panel resolves it (newest preprocess variant → base → latest run). */
export function findDitAdapter(ds: TrainingDatasetRow): TrainingAdapterHit | null {
  return ditHitFrom(readTrainDitStatus(ds, {}));
}

/** The hit for an already-read DiT status — so readDatasetAssets pays for the
 *  variant scan once instead of twice. */
function ditHitFrom(status: ReturnType<typeof readTrainDitStatus>): TrainingAdapterHit | null {
  if (!status.adapterExists) return null;
  return {
    path: status.adapterDir,
    kind: 'dit',
    // The per-base folder the run lives in (…/dit-<base>/<artist>/<stamp>).
    detail: path.basename(path.dirname(path.dirname(status.adapterDir))),
    trainedAt: status.trainedAt || new Date(weightsMtime(status.adapterDir) || Date.now()).toISOString(),
  };
}

// ── Assets ───────────────────────────────────────────────────────────────

/**
 * Every pipeline artefact this dataset has on disk. Cheap enough for the list
 * endpoint: one readdir of the tensors root, one of the newest variant, a
 * couple of stats per adapter candidate, and one small JSON parse.
 *
 * `labeled` comes off the row's cached counter, not a fresh scan — the list
 * already shows that counter, and re-scanning N source folders to confirm it
 * would cost far more than the flag is worth.
 */
export function readDatasetAssets(ds: TrainingDatasetRow): DatasetAssets {
  const assets: DatasetAssets = {
    labeled: ds.labeledCount > 0,
    built: ds.status === 'built' || !!ds.builtAt,
    tensorVariants: 0,
    tensorVariantKey: '',
    tensorSamples: 0,
    ditBase: '',
    lm: null,
    dit: null,
  };

  // A row that says 'built' but whose dataset.json has since been deleted is
  // not built any more — the file is what the trainers actually read.
  if (assets.built && ds.datasetJsonPath) {
    try { assets.built = fs.existsSync(ds.datasetJsonPath); } catch { /* keep the row's word */ }
  }

  try { assets.tensorVariants = countPreprocessedVariants(ds.slug); } catch { /* stays 0 */ }

  try {
    const dit = readTrainDitStatus(ds, {});
    assets.tensorVariantKey = dit.variantKey;
    assets.tensorSamples = dit.sampleCount;
    assets.ditBase = dit.ditModel;
    assets.dit = ditHitFrom(dit);
  } catch { /* stays null/0 */ }

  try { assets.lm = findLmAdapter(ds); } catch { /* stays null */ }

  return assets;
}

// ── Album detection ──────────────────────────────────────────────────────

/** Most common non-empty value, case-insensitively grouped, first-seen casing
 *  returned. '' when nothing usable at all. */
function majority(values: string[]): string {
  const counts = new Map<string, { display: string; n: number }>();
  for (const raw of values) {
    const v = String(raw ?? '').trim();
    if (!v) continue;
    const k = v.toLowerCase();
    const hit = counts.get(k);
    if (hit) hit.n++;
    else counts.set(k, { display: v, n: 1 });
  }
  let best: { display: string; n: number } | null = null;
  for (const c of counts.values()) if (!best || c.n > best.n) best = c;
  return best ? best.display : '';
}

/**
 * Album name from a sample list already in hand — tag majority first, then the
 * dataset's own default.
 *
 * The SOURCE FOLDER NAME is deliberately not a fallback here (unlike the Lyric
 * Studio export's detectAlbum): this value is displayed as "the album these
 * tracks say they are", and a folder name dressed up as a tag read is exactly
 * the confusion that published "4yearstrong_someway" as an album (2026-07-31).
 * '' means "unknown", and the UI shows the dataset name instead.
 */
export function albumFromSamples(ds: TrainingDatasetRow, samples: TrainingSample[]): string {
  const included = samples.filter(s => !s.excluded && !s.fileMissing);
  return majority(included.map(s => s.tagAlbum)) || ds.defaultAlbum.trim();
}

/** Persist a freshly detected album name when it actually changed. */
function cacheAlbum(ds: TrainingDatasetRow, album: string): void {
  if (!album || album === ds.albumName) return;
  ds.albumName = album;
  try { repo.updateDataset(ds.id, { albumName: album }); } catch { /* display-only */ }
}

/** Refresh the cached album from a scan that already has the samples — free,
 *  and it upgrades a probe-derived guess the moment the Label job writes tags. */
export function syncAlbumName(ds: TrainingDatasetRow, samples: TrainingSample[]): void {
  cacheAlbum(ds, albumFromSamples(ds, samples));
}

/** Datasets whose files have already been probed this process. A probe that
 *  finds nothing must not re-read the same audio on every list request; a
 *  server restart is a cheap enough retry. */
const probed = new Set<string>();

/** How many files a cold dataset is worth reading tags out of — an album only
 *  needs a majority, not a census. Same bound the Lyric Studio export uses. */
const TAG_PROBE_LIMIT = 12;

/**
 * The album name for a dataset that has never been opened: label-store tags
 * first (no audio parsing at all), then a bounded read of the files themselves.
 * Whatever it finds is cached, so this costs something exactly once.
 */
export async function ensureAlbumName(ds: TrainingDatasetRow): Promise<string> {
  if (ds.albumName.trim()) return ds.albumName;
  if (probed.has(ds.id)) return '';
  probed.add(ds.id);

  try {
    const labels = [...readAllLabels(ds.slug).values()];
    const fromLabels = majority(labels.map(l => l.tags?.album ?? ''));
    if (fromLabels) { cacheAlbum(ds, fromLabels); return fromLabels; }
  } catch { /* no labels dir — fall through to the file probe */ }

  let files: string[] = [];
  try {
    files = scanAudioFiles(ds.sourceDir, ds.recursive).slice(0, TAG_PROBE_LIMIT).map(f => f.absPath);
  } catch { /* folder gone or too large — the default below still applies */ }

  const albums: string[] = [];
  for (const file of files) {
    try {
      const md = await audioMeta.read(file);
      if (md.album) albums.push(md.album);
    } catch { /* unreadable file — the rest still vote */ }
  }

  const album = majority(albums) || ds.defaultAlbum.trim();
  cacheAlbum(ds, album);
  return album;
}

/** How long ONE list request will spend detecting albums before handing the
 *  rest off to a background pass. A corpus of never-opened datasets must not
 *  turn the first list of the session into a multi-second stall; whatever the
 *  background finishes is cached and shows up on the next refresh. */
const ALBUM_BUDGET_MS = 1500;

function probeInBackground(rows: TrainingDatasetRow[]): void {
  void (async () => {
    for (const ds of rows) {
      try { await ensureAlbumName(ds); } catch { /* album stays '' */ }
    }
  })();
}

/** The list payload: every row with its album filled in and its disk artefacts
 *  attached. Album detection is bounded and once-per-dataset; the asset read is
 *  fresh every time. */
export async function listDatasetsWithAssets(): Promise<TrainingDatasetRow[]> {
  const rows = repo.listDatasets();
  const deadline = Date.now() + ALBUM_BUDGET_MS;
  const deferred: TrainingDatasetRow[] = [];

  for (const ds of rows) {
    if (!ds.albumName.trim() && !probed.has(ds.id)) {
      if (Date.now() < deadline) {
        try { await ensureAlbumName(ds); } catch { /* album stays '' */ }
      } else {
        deferred.push(ds);
      }
    }
    ds.assets = readDatasetAssets(ds);
  }

  if (deferred.length) probeInBackground(deferred);
  return rows;
}
