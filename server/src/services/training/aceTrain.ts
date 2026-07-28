// training/aceTrain.ts — ace-train binary discovery, tensor-cache paths and
// the cached engine model snapshot.
//
// The preprocess job stops ace-server to free VRAM, so the model picker cannot
// depend on the engine being reachable at the moment it is rendered. Every
// successful /props read is cached here and served while the engine is down
// (P28).
//
// Spec: docs/plans/2026-07-27-preprocess-implementation.md §4.2

import fs from 'fs';
import path from 'path';
import { config } from '../../config.js';
import { aceClient } from '../aceClient.js';
import { slugify, trainingBaseDir } from './paths.js';
import type {
  DitAdapterType, LmSize, PreprocessCompat, PreprocessDtype, PreprocessNormalize,
  PreprocessOptions, TrainDitStage, TrainLmStage,
} from './types.js';

/** Model-file extensions stripped when deriving a variant key. */
const MODEL_EXTENSIONS = ['.gguf', '.safetensors', '.bin', '.pt', '.pth', '.onnx'];

/** Absolute path to ace-train, or null. Sibling of ace-server in both the
 *  CMake and portable layouts — same pattern as aceMidiExe(). */
export function aceTrainExe(): string | null {
  const dir = path.dirname(config.aceServer.exe);
  const exe = path.join(dir, process.platform === 'win32' ? 'ace-train.exe' : 'ace-train');
  return fs.existsSync(exe) ? exe : null;
}

/** DiT model name → filesystem-safe variant key (extension stripped). */
export function variantKeyFor(ditModel: string): string {
  const raw = String(ditModel ?? '').replace(/\\/g, '/');
  let base = raw.slice(raw.lastIndexOf('/') + 1);
  // Only KNOWN model extensions are stripped: names like
  // `acestep-v15-merge-base-turbo-xl-ta-0.5` must not lose their `.5`.
  const lower = base.toLowerCase();
  for (const ext of MODEL_EXTENSIONS) {
    if (lower.endsWith(ext)) { base = base.slice(0, base.length - ext.length); break; }
  }
  const safe = base.replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 96);
  return safe || 'default';
}

/** data/training/tensors/<slug> */
export function tensorsRoot(slug: string): string {
  return path.join(trainingBaseDir, 'tensors', slugify(slug));
}

/** data/training/tensors/<slug>/<variantKey> */
export function tensorsDir(slug: string, variantKey: string): string {
  return path.join(tensorsRoot(slug), variantKeyFor(variantKey));
}

// ── Cached /props model snapshot (P28) ───────────────────────────────────

export interface ModelSnapshot {
  dit: string[]; vae: string[]; textEnc: string[]; lm: string[]; cachedAt: number;
}

let snapshot: ModelSnapshot = { dit: [], vae: [], textEnc: [], lm: [], cachedAt: 0 };

/** Last successful /props read. Survives the engine being stopped mid-job. */
export function getModelSnapshot(): ModelSnapshot {
  return snapshot;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((v): v is string => typeof v === 'string') : [];
}

/** Re-probe /props. Never throws — returns the previous snapshot on any failure. */
export async function refreshModelSnapshot(): Promise<ModelSnapshot> {
  try {
    const props = await aceClient.props();
    const dit = stringList(props?.models?.dit);
    const vae = stringList(props?.models?.vae);
    // `models.embedding` IS the text-encoder bucket (Qwen3-Embedding).
    const textEnc = stringList(props?.models?.embedding);
    const lm = stringList(props?.models?.lm);
    // §4.2 says the previous snapshot survives a FAILURE, not emptiness. A
    // successful /props that honestly reports three empty buckets (models
    // deleted, ACESTEPCPP_MODELS repointed) must replace the cache, or the
    // picker offers names that no longer exist and the POST validation accepts
    // them — the user only finds out as an ace-train exit 2, after the engine
    // has already been stopped. A malformed response (no `models` object at
    // all) is still treated as a failure.
    if (props && typeof props === 'object' && props.models && typeof props.models === 'object') {
      snapshot = { dit, vae, textEnc, lm, cachedAt: Date.now() };
    }
  } catch {
    // Engine down or stopped for a job — the cache is exactly what we want.
  }
  return snapshot;
}

/** First name matching /bf16/i, else ''. */
export function pickBf16(names: string[]): string {
  return names.find(n => /bf16/i.test(n)) ?? '';
}

/**
 * Preferred BF16 LM for a size, e.g. '0.6B' -> 'acestep-5Hz-lm-0.6B-BF16.gguf'.
 *
 * Match rule (§4.2): the name contains `-<size>-` (case-insensitive) AND /bf16/i.
 * Falls back to the first name containing `-<size>-` at all; '' when none.
 * The fallback is deliberately kept: the trainer refuses a genuinely quantized
 * base at mirror time with a clear message, which beats offering nothing.
 */
export function pickLmFor(size: LmSize, names: string[]): string {
  // '0.6B' carries a literal '.', which must not become a regex wildcard.
  const token = new RegExp(`-${size.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}-`, 'i');
  const matches = names.filter(n => token.test(n));
  return matches.find(n => /bf16/i.test(n)) ?? matches[0] ?? '';
}

// ── Argv construction ────────────────────────────────────────────────────

/** Every PreprocessOptions field the runner needs, with defaults already applied
 *  by the route. `job.opts` carries exactly this shape for a preprocess job. */
export interface ResolvedPreprocessOptions {
  ditModel: string;
  vaeModel: string;
  textEncoder: string;
  maxDuration: number;
  normalize: PreprocessNormalize;
  targetDb: number;
  dtype: PreprocessDtype;
  compat: PreprocessCompat;
  maxCaptionTokens: number;
  maxLyricTokens: number;
  vaeChunk: number;
  vaeOverlap: number;
  overwrite: boolean;
  stopEngine: boolean;
  /** Absolute; data/training/tensors/<slug>/<variantKey> unless overridden. */
  outputDir: string;
  variantKey: string;
}

type PreprocessArgOpts = Required<Pick<PreprocessOptions,
  'maxDuration' | 'normalize' | 'targetDb' | 'dtype' | 'compat' |
  'maxCaptionTokens' | 'maxLyricTokens' | 'vaeChunk' | 'vaeOverlap' | 'overwrite'>>;

/** Build the full argv for `ace-train preprocess` (§2.1 order). */
export function buildPreprocessArgs(input: {
  manifestPath: string; outDir: string; modelsDir: string;
  dit: string; vae: string; textEnc: string;
  opts: PreprocessArgOpts;
  ffmpeg: string | null;
}): string[] {
  const o = input.opts;
  const args = [
    'preprocess',
    '--manifest', input.manifestPath,
    '--out', input.outDir,
    '--models', input.modelsDir,
    '--dit', input.dit,
    '--vae', input.vae,
    '--text-enc', input.textEnc,
    '--max-duration', String(o.maxDuration),
    '--normalize', o.normalize,
    '--target-db', String(o.targetDb),
    '--dtype', o.dtype,
    '--compat', o.compat,
    '--max-caption-tokens', String(o.maxCaptionTokens),
    '--max-lyric-tokens', String(o.maxLyricTokens),
    '--vae-chunk', String(o.vaeChunk),
    '--vae-overlap', String(o.vaeOverlap),
  ];
  if (input.ffmpeg) args.push('--ffmpeg', input.ffmpeg);
  if (o.overwrite) args.push('--overwrite');
  args.push('--jsonl');
  return args;
}

// ── LM LoRA training (phase 3) ───────────────────────────────────────────

/** Every TrainLmOptions field the runner needs, with defaults and clamps
 *  already applied by the route. `job.opts` carries exactly this shape for a
 *  train-lm job. Spec §4.2. */
export interface ResolvedTrainLmOptions {
  lmSize: LmSize;
  lmModel: string;
  ditModel: string;
  variantKey: string;
  /** Absolute preprocess variant dir the codes are extracted from. */
  tensorsDir: string;
  /** Absolute <tensorsDir>/lm_codes.jsonl. */
  codesPath: string;
  adapterName: string;
  /** Absolute <adapters>/lm/<adapterName>-<lmSize>. */
  adapterDir: string;
  targetLoss: number;
  epochs: number;
  rank: number;
  alpha: number;
  learningRate: number;
  gradAccum: number;
  gradClip: number;
  warmupRatio: number;
  weightDecay: number;
  maxLen: number;
  seed: number;
  lossOnCot: boolean;
  order: 'shuffle' | 'fixed';
  milestoneStep: number;
  milestoneKeep: number;
  stages: TrainLmStage[];
  overwrite: boolean;
  stopEngine: boolean;
}

/** Full argv for `ace-train train-lm` (§2.1 order). */
export function buildTrainLmArgs(input: {
  opts: ResolvedTrainLmOptions; modelsDir: string;
}): string[] {
  const o = input.opts;
  const args = [
    'train-lm',
    '--stages', o.stages.join(','),
    '--tensors', o.tensorsDir,
    '--codes', o.codesPath,
    '--out', o.adapterDir,
    '--models', input.modelsDir,
    // `--dit` DEFAULTS to preprocess_meta.json's dit_path (§2.1). Omitting the
    // pair is not the same as passing an empty value: resolve_model('') fails and
    // the sibling cmd_preprocess exits 2 on it — which here would happen only
    // AFTER the runner stopped ace-server, so the user would pay a full engine
    // stop/restart cycle for a resolvable-by-default condition. ditModel is ''
    // whenever the variant's meta is missing/unparseable or lacks model_variant.
    ...(o.ditModel ? ['--dit', o.ditModel] : []),
    '--lm', o.lmModel,
    '--lm-size', o.lmSize,
    '--rank', String(o.rank),
    '--alpha', String(o.alpha),
    '--lr', String(o.learningRate),
    '--epochs', String(o.epochs),
    '--grad-accum', String(o.gradAccum),
    '--warmup-ratio', String(o.warmupRatio),
    '--grad-clip', String(o.gradClip),
    '--weight-decay', String(o.weightDecay),
    '--seed', String(o.seed),
    '--target-loss', String(o.targetLoss),
    '--order', o.order,
    '--max-len', String(o.maxLen),
    '--milestone-step', String(o.milestoneStep),
    '--milestone-keep', String(o.milestoneKeep),
  ];
  // `--loss-on-cot` is the CLI default; only the negation needs emitting.
  if (!o.lossOnCot) args.push('--no-loss-on-cot');
  if (o.overwrite) args.push('--overwrite');
  args.push('--jsonl');
  return args;
}

// ── DiT LoRA training (phase 4) ──────────────────────────────────────────

/** Every TrainDitOptions field the runner needs, with defaults and clamps
 *  already applied by the route. `job.opts` carries exactly this shape for a
 *  train-dit job. Spec §4.2. */
export interface ResolvedTrainDitOptions {
  variantKey: string; tensorsDir: string;
  ditModel: string; ditPath: string;
  adapterName: string; adapterDir: string;
  adapterType: DitAdapterType; rank: number; alpha: number; targetMlp: boolean;
  layers: number; crop: number; cropMin: number; cropMax: number;
  targetLoss: number; epochs: number; learningRate: number;
  gradAccum: number; gradClip: number; warmupRatio: number; weightDecay: number;
  lossWeighting: 'none' | 'flow_snr'; snrGamma: number; tBias: number;
  channelBalance: boolean; timestepMu: number; timestepSigma: number;
  tMin: number; tMax: number; cfgRatio: number; genreRatio: number;
  seed: number; order: 'shuffle' | 'fixed';
  milestoneStep: number; milestoneKeep: number; vramReserveMb: number;
  stages: TrainDitStage[]; overwrite: boolean; stopEngine: boolean;
}

/**
 * The variant's own base DiT, as an ABSOLUTE path, '' when it is no longer on
 * disk (§4.2 base-match guard).
 *
 * The encoder states and context latents in the cache are that exact model's
 * outputs, so training against anything else is silently wrong — which is why
 * this reads the variant's record and never user input. Three sources, in
 * descending order of trust:
 *   1. `dit_path`      — the absolute path the preprocess run actually loaded
 *   2. `model_variant` — the file name, resolved against the models dir
 *   3. `variantKey`    — the same name with its extension stripped (§ variantKeyFor)
 * Returning '' is the caller's cue to answer 400 rather than stop the engine
 * for a run that cannot load its base.
 */
export function pickDitBaseFor(variantKey: string, tensorsDirPath: string): string {
  const exists = (p: string): boolean => {
    try { return !!p && fs.existsSync(p); } catch { return false; }
  };

  let ditPath = '';
  let modelVariant = '';
  try {
    const meta = JSON.parse(
      fs.readFileSync(path.join(tensorsDirPath, 'preprocess_meta.json'), 'utf-8'),
    ) as { dit_path?: unknown; model_variant?: unknown };
    if (typeof meta.dit_path === 'string') ditPath = meta.dit_path.trim();
    if (typeof meta.model_variant === 'string') modelVariant = meta.model_variant.trim();
  } catch {
    return '';   // no meta = not a real variant
  }

  if (exists(ditPath)) return ditPath;

  // The models dir may have moved since the preprocess run (portable release,
  // ACESTEPCPP_MODELS repointed) while the same file is still installed.
  const modelsDir = config.aceServer.models;
  if (modelVariant) {
    const byName = path.join(modelsDir, modelVariant);
    if (exists(byName)) return byName;
  }
  if (variantKey) {
    for (const ext of ['.gguf', '']) {
      const byKey = path.join(modelsDir, `${variantKey}${ext}`);
      if (exists(byKey)) return byKey;
    }
  }
  return '';
}

/** Full argv for `ace-train train-dit` (§2.1 order). */
export function buildTrainDitArgs(input: {
  opts: ResolvedTrainDitOptions; modelsDir: string;
}): string[] {
  const o = input.opts;
  const args = [
    'train-dit',
    '--stages', o.stages.join(','),
    '--tensors', o.tensorsDir,
    '--out', o.adapterDir,
    '--models', input.modelsDir,
    // Always present in practice — the route refuses the request when
    // pickDitBaseFor() came back empty, so the engine never has to fall back to
    // preprocess_meta.json's own default. Guarded anyway: passing an empty value
    // would make resolve_model('') exit 2, AFTER the runner already stopped the
    // engine.
    ...(o.ditPath ? ['--dit', o.ditPath] : []),
    '--adapter-type', o.adapterType,
    '--rank', String(o.rank),
    '--alpha', String(o.alpha),
    '--layers', String(o.layers),
    '--crop', String(o.crop),
    '--crop-min', String(o.cropMin),
    '--crop-max', String(o.cropMax),
    '--loss-weighting', o.lossWeighting,
    '--snr-gamma', String(o.snrGamma),
    '--t-bias', String(o.tBias),
    '--timestep-mu', String(o.timestepMu),
    '--timestep-sigma', String(o.timestepSigma),
    '--t-min', String(o.tMin),
    '--t-max', String(o.tMax),
    '--cfg-ratio', String(o.cfgRatio),
    '--genre-ratio', String(o.genreRatio),
    '--lr', String(o.learningRate),
    '--epochs', String(o.epochs),
    '--grad-accum', String(o.gradAccum),
    '--warmup-ratio', String(o.warmupRatio),
    '--grad-clip', String(o.gradClip),
    '--weight-decay', String(o.weightDecay),
    '--seed', String(o.seed),
    '--target-loss', String(o.targetLoss),
    '--order', o.order,
    '--vram-reserve-mb', String(o.vramReserveMb),
    '--milestone-step', String(o.milestoneStep),
    '--milestone-keep', String(o.milestoneKeep),
  ];
  // Flag-shaped options: only the non-default side is emitted (§2.1).
  if (o.targetMlp) args.push('--target-mlp');
  if (!o.channelBalance) args.push('--no-channel-balance');
  if (o.overwrite) args.push('--overwrite');
  args.push('--jsonl');
  return args;
}
