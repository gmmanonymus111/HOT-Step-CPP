// training/preprocessStatus.ts — reads the tensor-cache state off disk
//
// Disk is the source of truth here exactly as it is for datasets: every read
// re-scans data/training/tensors/<slug>/ and re-stats the source audio, so a
// cache goes stale the moment the user re-encodes or replaces a file.
//
// Never throws — a corrupt or unreadable variant degrades to zeroed counters.
//
// Spec: docs/plans/2026-07-27-preprocess-implementation.md §4.4

import fs from 'fs';
import path from 'path';
import { tensorsRoot } from './aceTrain.js';
import { datasetSampleId } from './paths.js';
import type {
  PreprocessStatus, PreprocessVariantStatus, TrainingDatasetRow, TrainingSample,
} from './types.js';

interface MetaSample {
  id?: unknown;
  file?: unknown;
  relPath?: unknown;
  audioPath?: unknown;
  srcBytes?: unknown;
  srcMtimeMs?: unknown;
  ok?: unknown;
  skipped?: unknown;
}

interface MetaFile {
  created_at?: unknown;
  model_variant?: unknown;
  compat?: unknown;
  dtype?: unknown;
  total?: unknown;
  processed?: unknown;
  failed?: unknown;
  samples?: unknown;
}

function num(value: unknown): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function str(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/** Total size of every regular file directly inside `dir`, plus the count of
 *  `.safetensors` files. One readdir, one stat per entry. */
function scanVariantDir(dir: string): { bytes: number; cached: number } {
  let bytes = 0;
  let cached = 0;
  try {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (!entry.isFile()) continue;
      if (entry.name.toLowerCase().endsWith('.safetensors')) cached++;
      try { bytes += fs.statSync(path.join(dir, entry.name)).size; } catch { /* raced */ }
    }
  } catch { /* unreadable — zeros are honest */ }
  return { bytes, cached };
}

function readVariant(
  root: string,
  variantKey: string,
  includableIds: Set<string>,
): PreprocessVariantStatus | null {
  const dir = path.join(root, variantKey);
  const metaPath = path.join(dir, 'preprocess_meta.json');
  if (!fs.existsSync(metaPath)) return null;

  const { bytes, cached } = scanVariantDir(dir);
  const status: PreprocessVariantStatus = {
    variantKey,
    modelVariant: '',
    outputDir: dir,
    createdAt: '',
    compat: '',
    dtype: '',
    total: 0,
    processed: 0,
    failed: 0,
    cachedCount: cached,
    staleCount: 0,
    missingCount: includableIds.size,
    bytes,
    hasChannelStats: fs.existsSync(path.join(dir, 'channel_stats.json')),
  };

  let meta: MetaFile;
  try {
    meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8')) as MetaFile;
  } catch {
    return status;   // corrupt meta: the directory still exists and is listable
  }

  status.modelVariant = str(meta.model_variant);
  status.createdAt = str(meta.created_at);
  status.compat = str(meta.compat);
  status.dtype = str(meta.dtype);
  status.total = num(meta.total);
  status.processed = num(meta.processed);
  status.failed = num(meta.failed);

  const rows: MetaSample[] = Array.isArray(meta.samples) ? meta.samples as MetaSample[] : [];
  const cachedIds = new Set<string>();
  let stale = 0;

  for (const row of rows) {
    if (!row || typeof row !== 'object') continue;
    if (row.ok !== true) continue;
    const id = str(row.id);
    if (id) cachedIds.add(id);

    // A cache entry whose source audio changed size/mtime — or vanished — no
    // longer describes what is on disk. Same idiom as labelStore's durationCache.
    const audioPath = str(row.audioPath);
    if (!audioPath) continue;
    try {
      const st = fs.statSync(audioPath);
      if (st.size !== num(row.srcBytes) || Math.round(st.mtimeMs) !== Math.round(num(row.srcMtimeMs))) stale++;
    } catch {
      stale++;
    }
  }

  status.staleCount = stale;
  let missing = 0;
  for (const id of includableIds) if (!cachedIds.has(id)) missing++;
  status.missingCount = missing;

  return status;
}

export function readPreprocessStatus(
  ds: TrainingDatasetRow,
  samples: TrainingSample[],
): PreprocessStatus {
  const root = tensorsRoot(ds.userId, ds.slug);
  const variants: PreprocessVariantStatus[] = [];

  // dataset.json ids (8 hex) are what preprocess_meta.json records.
  const includableIds = new Set<string>();
  for (const s of samples) {
    if (s.excluded || s.fileMissing) continue;
    includableIds.add(datasetSampleId(s.relPath));
  }

  try {
    for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      try {
        const v = readVariant(root, entry.name, includableIds);
        if (v) variants.push(v);
      } catch { /* one bad variant never sinks the list */ }
    }
  } catch { /* no tensors root yet — nothing has been preprocessed */ }

  variants.sort((a, b) => (a.createdAt < b.createdAt ? 1 : a.createdAt > b.createdAt ? -1 : 0));
  return { tensorsRoot: root, variants };
}

/** Cheap counter for the dataset detail payload: subdirs holding a meta file. */
export function countPreprocessedVariants(userId: string, slug: string): number {
  try {
    const root = tensorsRoot(userId, slug);
    let n = 0;
    for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      if (fs.existsSync(path.join(root, entry.name, 'preprocess_meta.json'))) n++;
    }
    return n;
  } catch {
    return 0;
  }
}
