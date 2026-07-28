// training/trainLmStatus.ts — reads LM-trainer state off disk
//
// A training run has no database row: its whole state lives in the preprocess
// variant's lm_codes.jsonl and in the PEFT adapter directory, exactly as the
// preprocess phase's state lives in preprocess_meta.json. Every GET re-scans.
//
// Never throws — an unreadable codes file or a legacy/Side-Step-produced adapter
// dir degrades to zeroed counters and '' / -1 / [] defaults.
//
// Spec: docs/plans/2026-07-27-lm-trainer-implementation.md §4.3

import fs from 'fs';
import path from 'path';
import { config } from '../../config.js';
import { tensorsRoot } from './aceTrain.js';
import type {
  LmSize, TrainLmEpoch, TrainLmStatus, TrainingDatasetRow, TrainingSample,
} from './types.js';

const CODES_FILE = 'lm_codes.jsonl';

function num(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function str(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/** `<config.aceServer.adapters>/lm` — the dir GET /api/adapters/lm scans, which
 *  is why a freshly trained adapter needs no engine restart (L12). */
export function adapterLmRoot(): string {
  return path.join(config.aceServer.adapters, 'lm');
}

/** Strip anything outside the route's `/^[A-Za-z0-9._-]{1,64}$/` alphabet so a
 *  query-string name can never build a path outside adapterLmRoot(). */
export function safeAdapterName(name: string): string {
  const cleaned = String(name ?? '').replace(/[^A-Za-z0-9._-]/g, '-').replace(/^\.+/, '').slice(0, 64);
  return cleaned || 'adapter';
}

/** `<adapters>/lm/<name>-<size>` — matches the naming of the existing dirs. */
export function adapterDirFor(name: string, size: LmSize): string {
  return path.join(adapterLmRoot(), `${safeAdapterName(name)}-${size}`);
}

/**
 * Newest preprocess variant of a dataset, '' when nothing is preprocessed.
 *
 * Reads only `created_at` out of each variant's preprocess_meta.json. §4.3 says
 * "reuse preprocessStatus.ts's reader", but readPreprocessStatus() needs a full
 * dataset row plus its sample list and re-stats every source audio file to
 * compute staleness — none of which this answer depends on — and its module is
 * outside this phase's editable file set, so a slug-only reader cannot be added
 * there. The parse below reads one small field per variant and nothing else.
 */
export function newestVariantKey(slug: string): string {
  let bestKey = '';
  let bestAt = '';
  try {
    const root = tensorsRoot(slug);
    for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const metaPath = path.join(root, entry.name, 'preprocess_meta.json');
      if (!fs.existsSync(metaPath)) continue;
      let createdAt = '';
      try {
        createdAt = str((JSON.parse(fs.readFileSync(metaPath, 'utf-8')) as { created_at?: unknown }).created_at);
      } catch { /* corrupt meta — the variant still counts, just with no date */ }
      // ISO strings sort lexicographically. A variant with no date loses to any
      // dated one but still wins over nothing at all.
      if (!bestKey || createdAt > bestAt) { bestKey = entry.name; bestAt = createdAt; }
    }
  } catch { /* no tensors root yet — nothing has been preprocessed */ }
  return bestKey;
}

/** The variant's own DiT file name, '' if unknown — the FSQ tokenizer must be
 *  the one the latents were made against (§4.5 step 6). */
export function variantDitModel(slug: string, variantKey: string): string {
  if (!isSafeVariantKey(variantKey)) return '';
  try {
    const metaPath = path.join(tensorsRoot(slug), variantKey, 'preprocess_meta.json');
    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8')) as { model_variant?: unknown };
    return str(meta.model_variant);
  } catch {
    return '';
  }
}

/**
 * A variant key is ONE directory name under `tensorsRoot(slug)` and nothing else.
 *
 * Without this, `path.join(tensorsRoot(slug), '../../otherslug/…')` resolves
 * outside the root and `isInside()` returns false — i.e. a request body could
 * point `--tensors` (which ace-train reads every safetensors from) and `--codes`
 * (which it WRITES) at an arbitrary directory. Both sibling routes already apply
 * this rule (§7.8): POST /preprocess uses isInside, DELETE uses path.basename.
 */
export function isSafeVariantKey(key: string): boolean {
  if (!key || key !== key.trim()) return false;
  if (key === '.' || key === '..') return false;
  if (key.includes('/') || key.includes('\\')) return false;
  if (path.isAbsolute(key)) return false;
  return path.basename(key) === key;
}

/** Does `<tensorsRoot(slug)>/<variantKey>` hold a preprocess_meta.json? */
export function variantExists(slug: string, variantKey: string): boolean {
  if (!isSafeVariantKey(variantKey)) return false;
  try {
    return fs.existsSync(path.join(tensorsRoot(slug), variantKey, 'preprocess_meta.json'));
  } catch {
    return false;
  }
}

interface CodesCounts {
  exists: boolean;
  count: number;
  stale: number;
  missing: number;
}

/**
 * Row counts for lm_codes.jsonl against what is actually in the variant dir.
 *
 * `stale`   — a row whose tensor file changed size/mtime (or vanished) since the
 *             extract; the engine re-encodes exactly these on the next run (L15).
 * `missing` — a .safetensors in the variant dir that has no row at all.
 */
function countCodes(tensorsDirPath: string, codesPath: string): CodesCounts {
  const out: CodesCounts = { exists: false, count: 0, stale: 0, missing: 0 };

  // Tensor files present on disk, plus the 8-hex id embedded in each name.
  const files = new Set<string>();
  const idsOnDisk = new Map<string, string>();   // id -> filename
  try {
    for (const entry of fs.readdirSync(tensorsDirPath, { withFileTypes: true })) {
      if (!entry.isFile() || !entry.name.toLowerCase().endsWith('.safetensors')) continue;
      files.add(entry.name);
      const m = /-([0-9a-f]{8})\.safetensors$/i.exec(entry.name);
      if (m) idsOnDisk.set(m[1].toLowerCase(), entry.name);
    }
  } catch { /* unreadable variant dir — zeros are honest */ }

  let raw = '';
  try {
    if (!fs.existsSync(codesPath)) return { ...out, missing: files.size };
    raw = fs.readFileSync(codesPath, 'utf-8');
    out.exists = true;
  } catch {
    return { ...out, missing: files.size };
  }

  const seenFiles = new Set<string>();
  const seenIds = new Set<string>();

  for (const line of raw.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let row: Record<string, unknown>;
    try {
      row = JSON.parse(trimmed) as Record<string, unknown>;
    } catch {
      continue;   // a torn line is not a row
    }
    if (!row || typeof row !== 'object') continue;
    out.count++;

    const file = str(row.file);
    const id = str(row._id).toLowerCase();
    if (file) seenFiles.add(file);
    if (id) seenIds.add(id);

    // Freshness is decided against the tensor file the row names.
    const target = file || (id ? idsOnDisk.get(id) ?? '' : '');
    if (!target) { out.stale++; continue; }
    try {
      const st = fs.statSync(path.join(tensorsDirPath, target));
      // WHOLE-SECOND comparison, deliberately. `_st_mtime_ms` is written by the
      // ENGINE (pm_stat_file, preprocess-io.h:104) as `st.st_mtime * 1000` —
      // always a multiple of 1000 — while Node's mtimeMs carries NTFS sub-second
      // precision. Rounding to ms can therefore NEVER match, which reported every
      // row stale on every GET while the engine's own L15 check (lm-extract.h:343,
      // both sides second-truncated) called them all cached. Not the same as the
      // preprocess idiom it was copied from: there the mtime originates server-side.
      if (st.size !== num(row._st_bytes, -1)
        || Math.floor(st.mtimeMs / 1000) !== Math.floor(num(row._st_mtime_ms, -1) / 1000)) out.stale++;
    } catch {
      out.stale++;   // the tensor the row describes is gone
    }
  }

  let missing = 0;
  for (const name of files) {
    if (seenFiles.has(name)) continue;
    const m = /-([0-9a-f]{8})\.safetensors$/i.exec(name);
    if (m && seenIds.has(m[1].toLowerCase())) continue;
    missing++;
  }
  out.missing = missing;
  return out;
}

interface TrainLogFile {
  created_at?: unknown;
  config?: { lm_size?: unknown; target_loss?: unknown } | unknown;
  epochs?: unknown;
  target_loss_stop?: unknown;
  final_loss?: unknown;
  best_loss?: unknown;
  epochs_run?: unknown;
  milestones?: unknown;
}

/** Read `<adapterDir>/lm_train_log.json` into the status fields it owns.
 *  Every field is individually defaulted so a Side-Step-produced or legacy
 *  adapter directory reports honestly instead of throwing (§4.3). */
function applyTrainLog(dir: string, status: TrainLmStatus): void {
  let log: TrainLogFile;
  try {
    log = JSON.parse(fs.readFileSync(path.join(dir, 'lm_train_log.json'), 'utf-8')) as TrainLogFile;
  } catch {
    return;
  }
  if (!log || typeof log !== 'object') return;

  status.trainedAt = str(log.created_at);
  status.finalLoss = num(log.final_loss, -1);
  status.bestLoss = num(log.best_loss, -1);
  status.epochsRun = num(log.epochs_run, 0);

  const cfg = (log.config && typeof log.config === 'object') ? log.config as Record<string, unknown> : {};
  status.lmSize = str(cfg.lm_size);
  status.targetLoss = num(cfg.target_loss, -1);

  const stop = log.target_loss_stop;
  status.stoppedOnTarget = !!stop && typeof stop === 'object';

  if (Array.isArray(log.epochs)) {
    const epochs: TrainLmEpoch[] = [];
    for (const e of log.epochs) {
      if (!e || typeof e !== 'object') continue;
      const row = e as Record<string, unknown>;
      epochs.push({
        epoch: num(row.epoch),
        loss: num(row.loss),
        lr: num(row.lr),
        gradNorm: num(row.grad_norm),
        ms: num(row.ms),
      });
    }
    status.epochs = epochs;
  }

  if (Array.isArray(log.milestones)) {
    const milestones: TrainLmStatus['milestones'] = [];
    for (const m of log.milestones) {
      if (!m || typeof m !== 'object') continue;
      const row = m as Record<string, unknown>;
      milestones.push({ loss: num(row.loss), epoch: num(row.epoch), path: str(row.path) });
    }
    status.milestones = milestones;
  }
}

/**
 * Everything the Train panel needs about one dataset's LM adapter, read fresh
 * off disk. `samples` is accepted for signature parity with the other status
 * readers and to keep the door open for sample-derived counters; the codes
 * counters are deliberately measured against the tensor cache, not the dataset,
 * because that cache is what `extract` actually consumes (L13).
 */
export function readTrainLmStatus(
  ds: TrainingDatasetRow,
  _samples: TrainingSample[],
  q: { variantKey?: string; adapterName?: string; lmSize?: LmSize },
): TrainLmStatus {
  const lmSize: LmSize = q.lmSize === '1.7B' ? '1.7B' : '0.6B';
  const requested = typeof q.variantKey === 'string' ? q.variantKey.trim() : '';
  // A requested key that does not exist falls back to nothing rather than
  // reporting the newest variant's codes under the wrong name.
  const variantKey = requested
    ? (variantExists(ds.slug, requested) ? requested : '')
    : newestVariantKey(ds.slug);

  const dir = variantKey ? path.join(tensorsRoot(ds.slug), variantKey) : '';
  const codesPath = dir ? path.join(dir, CODES_FILE) : '';

  const rawName = typeof q.adapterName === 'string' ? q.adapterName.trim() : '';
  const adapterName = rawName || ds.slug;
  const adapterDir = adapterDirFor(adapterName, lmSize);

  const status: TrainLmStatus = {
    variantKey,
    tensorsDir: dir,
    codesPath,
    codesExists: false,
    codesCount: 0,
    codesStale: 0,
    codesMissing: 0,
    adapterName,
    adapterDir,
    adapterExists: false,
    adapterBytes: 0,
    lmSize: '',
    trainedAt: '',
    finalLoss: -1,
    bestLoss: -1,
    epochsRun: 0,
    targetLoss: -1,
    stoppedOnTarget: false,
    epochs: [],
    milestones: [],
  };

  if (dir) {
    try {
      const counts = countCodes(dir, codesPath);
      status.codesExists = counts.exists;
      status.codesCount = counts.count;
      status.codesStale = counts.stale;
      status.codesMissing = counts.missing;
    } catch { /* counters stay zero */ }
  }

  try {
    const model = path.join(adapterDir, 'adapter_model.safetensors');
    const st = fs.statSync(model);
    status.adapterExists = st.isFile();
    status.adapterBytes = st.size;
  } catch { /* not trained yet */ }

  try { applyTrainLog(adapterDir, status); } catch { /* defaults stand */ }

  return status;
}
