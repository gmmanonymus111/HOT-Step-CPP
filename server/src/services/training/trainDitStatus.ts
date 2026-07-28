// training/trainDitStatus.ts — reads DiT-trainer state off disk
//
// A DiT training run has no database row either: its whole state lives in the
// preprocess variant it trains from and in the PEFT adapter directory it
// writes, exactly as the LM trainer's does. Every GET re-scans.
//
// Never throws — an unreadable variant dir or a Side-Step/ComfyUI-produced
// adapter directory degrades to zeroed counters and '' / -1 / [] / false
// defaults rather than a 500 (§4.3).
//
// Spec: docs/plans/2026-07-28-dit-trainer-implementation.md §4.3

import fs from 'fs';
import path from 'path';
import { config } from '../../config.js';
import { ditShorthand, latestRunDir, newRunDir } from './adapterLayout.js';
import { tensorsRoot } from './aceTrain.js';
import {
  newestVariantKey, safeAdapterName, variantDitModel, variantExists,
} from './trainLmStatus.js';
import type { TrainDitEpoch, TrainDitStatus, TrainingDatasetRow } from './types.js';

function num(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function str(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/**
 * `<config.aceServer.adapters>` — the DiT adapter root itself, NOT an `lm/`
 * style subdirectory.
 *
 * This is the folder the Create view's Adapter dropdown scans (POST
 * /api/adapters/scan), which is the whole reason a freshly trained DiT adapter
 * needs no engine restart (D17). The LM trainer's `lm/` subdir exists because
 * planner-LM adapters are a different consumer; DiT adapters belong at the top
 * level, beside every hand-installed one.
 */
export function adapterDitRoot(): string {
  return config.aceServer.adapters;
}

/** `<adapters>/dit-<shorthand>/<name>` — the ARTIST folder of the per-base
 *  layout (2026-07-28). `ditModel` is the base the variant was preprocessed
 *  against; adapters trained on different bases are not interchangeable, so
 *  each base gets its own folder (adapterLayout.ts). Runs live in stamped
 *  subfolders beneath it. */
export function ditArtistDirFor(name: string, ditModel: string): string {
  return path.join(adapterDitRoot(), ditShorthand(ditModel), safeAdapterName(name));
}

/** A fresh `<artist>/<YYYY-MM-DD_HH-MM-SS>` dir for a NEW training run to
 *  write into — retraining never overwrites an earlier adapter. */
export function ditRunDirFor(name: string, ditModel: string): string {
  return newRunDir(ditArtistDirFor(name, ditModel));
}

/** The newest trained adapter for status READS: latest stamped run, else an
 *  unversioned adapter directly in the artist folder, else the legacy
 *  top-level `<adapters>/<name>` dir — so pre-migration adapters keep
 *  reporting as trained. */
export function adapterDitDirFor(name: string, ditModel: string): string {
  const latest = latestRunDir(ditArtistDirFor(name, ditModel));
  if (latest) return latest;
  const legacy = path.join(adapterDitRoot(), safeAdapterName(name));
  if (fs.existsSync(path.join(legacy, 'adapter_config.json'))) return legacy;
  return ditArtistDirFor(name, ditModel);  // nothing trained yet
}

/** `.safetensors` files in a variant dir. The sidecars (preprocess_meta.json,
 *  channel_stats.json, lm_codes.jsonl) are excluded by the extension test, not
 *  by name, so a future sidecar cannot silently inflate the count. */
function countTensorFiles(dir: string): number {
  let n = 0;
  try {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.isFile() && entry.name.toLowerCase().endsWith('.safetensors')) n++;
    }
  } catch { /* unreadable variant dir — 0 is honest */ }
  return n;
}

interface DitTrainLogFile {
  created_at?: unknown;
  config?: unknown;
  epochs?: unknown;
  target_loss_stop?: unknown;
  final_loss?: unknown;
  best_loss?: unknown;
  epochs_run?: unknown;
  partial_depth?: unknown;
  milestones?: unknown;
}

/**
 * Read `<adapterDir>/dit_train_log.json` into the status fields it owns.
 *
 * Every field is individually defaulted (§4.3): an adapter directory produced
 * by Side-Step, ComfyUI or an older ace-train has no such file at all, and one
 * produced by a crashed run may have a truncated one. Neither may throw.
 */
function applyTrainLog(dir: string, status: TrainDitStatus): void {
  let log: DitTrainLogFile;
  try {
    log = JSON.parse(fs.readFileSync(path.join(dir, 'dit_train_log.json'), 'utf-8')) as DitTrainLogFile;
  } catch {
    return;
  }
  if (!log || typeof log !== 'object') return;

  status.trainedAt = str(log.created_at);
  status.finalLoss = num(log.final_loss, -1);
  status.bestLoss = num(log.best_loss, -1);
  status.epochsRun = num(log.epochs_run, 0);

  const cfg = (log.config && typeof log.config === 'object') ? log.config as Record<string, unknown> : {};
  status.targetLoss = num(cfg.target_loss, -1);
  status.crop = num(cfg.crop, 0);
  status.layers = num(cfg.layers, 0);
  status.adapterType = str(cfg.adapter_type);

  // `partial_depth` is the engine's own verdict (layers < n_layers_total, §2.4)
  // and wins. The derived comparison is only a fallback for a log that predates
  // the field — never a re-computation that could disagree with the artefact.
  if (typeof log.partial_depth === 'boolean') {
    status.partialDepth = log.partial_depth;
  } else {
    const total = num(cfg.n_layers_total, 0);
    status.partialDepth = total > 0 && status.layers > 0 && status.layers < total;
  }

  const stop = log.target_loss_stop;
  status.stoppedOnTarget = !!stop && typeof stop === 'object';

  if (Array.isArray(log.epochs)) {
    const epochs: TrainDitEpoch[] = [];
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
    const milestones: TrainDitStatus['milestones'] = [];
    for (const m of log.milestones) {
      if (!m || typeof m !== 'object') continue;
      const row = m as Record<string, unknown>;
      milestones.push({ loss: num(row.loss), epoch: num(row.epoch), path: str(row.path) });
    }
    status.milestones = milestones;
  }
}

/**
 * Everything the Train panel's DiT card needs about one dataset, read fresh off
 * disk.
 *
 * The variant is resolved the same way the LM trainer resolves it: a requested
 * key that does not exist falls back to nothing rather than silently reporting
 * the newest variant's numbers under the wrong name.
 */
export function readTrainDitStatus(
  ds: TrainingDatasetRow,
  q: { variantKey?: string; adapterName?: string },
): TrainDitStatus {
  const requested = typeof q.variantKey === 'string' ? q.variantKey.trim() : '';
  const variantKey = requested
    ? (variantExists(ds.slug, requested) ? requested : '')
    : newestVariantKey(ds.slug);

  const dir = variantKey ? path.join(tensorsRoot(ds.slug), variantKey) : '';

  const rawName = typeof q.adapterName === 'string' ? q.adapterName.trim() : '';
  const adapterName = rawName || ds.slug;
  // The base decides which dit-<shorthand> folder the adapter lives in, so it
  // must be resolved before the dir (adapterLayout.ts).
  let ditModel = '';
  if (dir) {
    try { ditModel = variantDitModel(ds.slug, variantKey); } catch { /* stays '' */ }
  }
  const adapterDir = adapterDitDirFor(adapterName, ditModel);

  const status: TrainDitStatus = {
    variantKey,
    tensorsDir: dir,
    ditModel,
    sampleCount: 0,
    channelStats: false,
    adapterName,
    adapterDir,
    adapterExists: false,
    adapterBytes: 0,
    adapterType: '',
    trainedAt: '',
    finalLoss: -1,
    bestLoss: -1,
    epochsRun: 0,
    targetLoss: -1,
    stoppedOnTarget: false,
    crop: 0,
    layers: 0,
    partialDepth: false,
    epochs: [],
    milestones: [],
  };

  if (dir) {
    try { status.sampleCount = countTensorFiles(dir); } catch { /* stays 0 */ }
    try {
      status.channelStats = fs.existsSync(path.join(dir, 'channel_stats.json'));
    } catch { /* stays false */ }
  }

  // K8: LoKR exports write `lokr_weights.safetensors` instead of the PEFT
  // `adapter_model.safetensors` — check both, PEFT first (lora is still the
  // engine's own CLI default).
  try {
    const model = path.join(adapterDir, 'adapter_model.safetensors');
    const st = fs.statSync(model);
    status.adapterExists = st.isFile();
    status.adapterBytes = st.size;
  } catch { /* not a lora adapter */ }
  if (!status.adapterExists) {
    try {
      const model = path.join(adapterDir, 'lokr_weights.safetensors');
      const st = fs.statSync(model);
      status.adapterExists = st.isFile();
      status.adapterBytes = st.size;
    } catch { /* not trained yet */ }
  }

  try { applyTrainLog(adapterDir, status); } catch { /* defaults stand */ }

  return status;
}
