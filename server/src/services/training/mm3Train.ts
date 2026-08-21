// training/mm3Train.ts — MiniMax-Music3 training: paths, defaults, arg building.
//
// The MM3 analogue of aceTrain.ts. Kept apart from it deliberately: the two
// share no CLI surface (`mm3-codes` / `mm3-lm-train` take a manifest, a
// captions directory and a codes directory; `train-lm` takes a tensor cache),
// and folding them together would mean a file where half the fields are always
// unused.
//
// Spec: docs/plans/2026-08-20-mm3-training-server-design.md §2, §3.

import fs from 'fs';
import path from 'path';

import { config } from '../../config.js';
import { datasetDir } from './paths.js';

/** The MM3 model files training needs, resolved under <models>/mm3/.
 *
 *  CORRECTION (2026-08-21): this comment used to say a quantized base "cannot
 *  be trained" because out_prod is F32-only. That was true of the code, not of
 *  the maths. The default lm_linear path emits `mul_mat(qwen3_f32(w), x)` — an
 *  in-graph cast that gallocr frees with the segment — so the backward never
 *  sees the quantized tensor, only the cast's F32 output. That is QLoRA's
 *  dequantize-per-matmul, and it now works.
 *
 *  The base is a CHOICE. It used to be a real trade; since the
 *  cpy-q-occupancy patch (engine/patches/) it is not (5090, production recipe):
 *
 *                     base    VRAM used        free      s/step
 *      f16       16.0 GB   31.0/32.6 GB    1.5 GB    3.7   (idle card)
 *      q8_0       8.5 GB   22.6/32.6 GB   10.0 GB    3.75
 *
 *  q8_0 was 11.1 s/step before that patch, which is the ONLY reason f16 used to
 *  be the default. Now they are the same speed, so the headroom decides it:
 *  f16's 1.5 GB does not survive a desktop with a browser open. Measured on a
 *  card holding ~3.2 GB of other work, the SAME 12 steps ran 3.75 s/step on
 *  q8_0 and 12-14 s/step on f16, because f16 was paging over WDDM.
 *
 *  Accuracy is not the trade-off it sounds like: the trained thing is the F32
 *  LoRA, the base is frozen, and a same-seed 12-step A/B agreed to ~4 s.f. at
 *  every step (max relative loss deviation 5.7e-4). */
export interface Mm3TrainModels {
  lm: string;
  depth: string;
  /** Audio -> RVQ codes encoder. Any mm3-rvq-*.gguf; newest wins. */
  rvq: string;
  /** DAV encoder (audio -> latents), the mm3-codes input stage. */
  enc: string;
}

function mm3ModelDir(): string {
  return path.join(config.aceServer.models, 'mm3');
}

/** Newest file matching a prefix, or '' — used for the RVQ encoder, whose
 *  filename carries the checkpoint name rather than a fixed quant token. */
function newestMatching(dir: string, prefix: string): string {
  try {
    const hits = fs.readdirSync(dir)
      .filter(f => f.startsWith(prefix) && f.endsWith('.gguf'))
      .map(f => ({ f, m: fs.statSync(path.join(dir, f)).mtimeMs }))
      .sort((a, b) => b.m - a.m);
    return hits.length ? path.join(dir, hits[0].f) : '';
  } catch {
    return '';
  }
}

export type Mm3BasePrecision = 'f16' | 'q8_0';

export function resolveMm3TrainModels(base: Mm3BasePrecision = 'f16'): Mm3TrainModels {
  const dir = mm3ModelDir();
  return {
    lm:    path.join(dir, base === 'q8_0' ? 'mm3-lm-q8_0.gguf' : 'mm3-lm-f16.gguf'),
    depth: path.join(dir, 'mm3-depth-f16.gguf'),
    rvq:   newestMatching(dir, 'mm3-rvq-'),
    enc:   newestMatching(dir, 'mm3-enc-'),
  };
}

/** Which of the required files are missing, as user-facing names. Empty = ready.
 *  `need` narrows the check: the codes job does not need the LM. */
export function missingMm3TrainModels(need: 'codes' | 'train',
                                      base: Mm3BasePrecision = 'f16'): string[] {
  const m = resolveMm3TrainModels(base);
  const wanted: Array<[string, string]> = need === 'codes'
    ? [[m.rvq, 'an RVQ encoder (mm3-rvq-*.gguf)'], [m.enc, 'the DAV encoder (mm3-enc-*.gguf)']]
    : [[m.lm, path.basename(m.lm)], [m.depth, 'mm3-depth-f16.gguf']];
  return wanted.filter(([p]) => !p || !fs.existsSync(p)).map(([, label]) => label);
}

/** Which base precisions are actually installed, so the picker only offers
 *  what exists rather than failing at spawn time. */
export function availableMm3Bases(): Mm3BasePrecision[] {
  return (['f16', 'q8_0'] as Mm3BasePrecision[])
    .filter(b => fs.existsSync(resolveMm3TrainModels(b).lm));
}

// ── Layout ──────────────────────────────────────────────────────────────────

/** Codes live beside the dataset, not in a shared pool.
 *
 *  The design doc floated a shared `<training>/mm3-codes/<encoder>/<dataset>/`
 *  so one corpus could be re-encoded by several encoders for a shoot-out. That
 *  is a research workflow; the product one is "this dataset's codes", and
 *  keeping them under the dataset means deleting a dataset takes its derived
 *  data with it. Re-encoding with a different encoder overwrites, and
 *  `codes.json` records which encoder produced what. */
export function mm3CodesDir(slug: string): string {
  return path.join(datasetDir(slug), 'mm3-codes');
}

/** Adapters go straight where the picker looks. Writing anywhere else would
 *  add an install step for no reason — the shipped lister scans two directory
 *  levels under this root and reads `<file>.json` as the sidecar, which is
 *  exactly the layout `ace-train mm3-lm-train` writes. */
export function mm3AdapterRunDir(runName: string): string {
  return path.join(config.aceServer.adapters, 'mm3-lm-adapters', runName);
}

/** `<dataset>-YYYY-MM-DD_HH-MM-SS`, the logs/ convention: name-sorted is
 *  time-sorted, and retraining never overwrites an earlier run. */
export function mm3RunName(slug: string): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, '0');
  return `${slug}-${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_`
       + `${p(d.getHours())}-${p(d.getMinutes())}-${p(d.getSeconds())}`;
}

// ── Defaults: the validated recipe, in one place ────────────────────────────
//
// Every number here is from docs/plans/2026-08-20-mm3-training-studio.md's
// recipe table or from a measurement recorded there. The Jobs form is a VIEW
// of this object; it must never carry a second copy of these numbers.
export const MM3_LM_DEFAULTS = {
  rank: 256,
  alpha: 256,
  lr: 8e-5,
  steps: 800,
  saveEvery: 100,
  warmup: 0,
  gradAccum: 1,
  seed: 42,
  /** Requires random crops — with `beginning` this trains intros only, which
   *  is exactly what went wrong in the lm2 run. */
  maxFrames: 1500,
  cropMode: 'random' as 'random' | 'beginning',
  /** Muon, not AdamW, for two measured reasons: AdamW's second momentum buffer
   *  does not FIT at rank 256 on a 32 GB card, and Muon's normalised update
   *  makes an AdamW learning rate meaningless. 64 is the best of {1,4,16,64}
   *  measured over 50 steps — not a tuned optimum. */
  optimizer: 'muon' as 'muon' | 'adamw',
  muonLrScale: 64,
  /** q8_0, not f16 — see the note on Mm3TrainModels. Same step time since the
   *  cpy-q-occupancy patch, ~8.5 GB less resident, and therefore the only one
   *  of the two that survives a GPU shared with a desktop session. Pick f16
   *  only to reproduce a pre-patch run exactly. */
  basePrecision: 'q8_0' as Mm3BasePrecision,
  holdout: 0.15,
  evalEvery: 50,
} as const;

// ── Arg building ────────────────────────────────────────────────────────────

export interface Mm3CodesArgs {
  datasetJson: string;
  outDir: string;
  maxDuration?: number;
}

export function buildMm3CodesArgs(a: Mm3CodesArgs): string[] {
  const m = resolveMm3TrainModels();
  const args = [
    'mm3-codes', '--jsonl',
    '--dataset', a.datasetJson,
    '--rvq', m.rvq,
    '--enc', m.enc,
    '--out', a.outDir,
  ];
  if (a.maxDuration && a.maxDuration > 0) args.push('--max-duration', String(Math.round(a.maxDuration)));
  return args;
}

export interface ResolvedMm3TrainLmOptions {
  manifest: string;
  captionsDir: string;
  codesDir: string;
  outDir: string;
  rank: number;
  alpha: number;
  lr: number;
  steps: number;
  saveEvery: number;
  warmup: number;
  gradAccum: number;
  seed: number;
  maxFrames: number;
  cropMode: 'random' | 'beginning';
  optimizer: 'muon' | 'adamw';
  muonLrScale: number;
  holdout: number;
  evalEvery: number;
  trigger: string;
  datasetName: string;
  basePrecision: Mm3BasePrecision;
}

export function buildMm3TrainLmArgs(o: ResolvedMm3TrainLmOptions): string[] {
  const m = resolveMm3TrainModels(o.basePrecision);
  const args = [
    'mm3-lm-train', '--jsonl',
    '--lm', m.lm,
    '--depth', m.depth,
    '--manifest', o.manifest,
    '--captions', o.captionsDir,
    '--codes', o.codesDir,
    '--out', o.outDir,
    '--rank', String(o.rank),
    '--alpha', String(o.alpha),
    '--lr', String(o.lr),
    '--steps', String(o.steps),
    '--save-every', String(o.saveEvery),
    '--warmup', String(o.warmup),
    '--grad-accum', String(o.gradAccum),
    '--seed', String(o.seed),
    '--max-frames', String(o.maxFrames),
    '--crop-mode', o.cropMode,
    '--optimizer', o.optimizer,
  ];
  if (o.optimizer === 'muon') args.push('--muon-lr-scale', String(o.muonLrScale));
  args.push('--holdout', String(o.holdout));
  args.push('--eval-every', String(o.evalEvery));
  if (o.trigger) args.push('--trigger', o.trigger);
  if (o.datasetName) args.push('--dataset-name', o.datasetName);
  return args;
}
