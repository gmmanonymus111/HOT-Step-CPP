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
  PreprocessCompat, PreprocessDtype, PreprocessNormalize, PreprocessOptions,
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
  dit: string[]; vae: string[]; textEnc: string[]; cachedAt: number;
}

let snapshot: ModelSnapshot = { dit: [], vae: [], textEnc: [], cachedAt: 0 };

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
    // §4.2 says the previous snapshot survives a FAILURE, not emptiness. A
    // successful /props that honestly reports three empty buckets (models
    // deleted, ACESTEPCPP_MODELS repointed) must replace the cache, or the
    // picker offers names that no longer exist and the POST validation accepts
    // them — the user only finds out as an ace-train exit 2, after the engine
    // has already been stopped. A malformed response (no `models` object at
    // all) is still treated as a failure.
    if (props && typeof props === 'object' && props.models && typeof props.models === 'object') {
      snapshot = { dit, vae, textEnc, cachedAt: Date.now() };
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
