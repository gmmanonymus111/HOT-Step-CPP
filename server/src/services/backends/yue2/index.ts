// backends/yue2/index.ts — YuE2 backend
//
// The third registered generation backend (docs/plans/yue2/06-engine-port-
// plan.md §7, docs/plans/yue2/01-seam-design.md §4.2). Thin by design, same
// shape as backends/minimax/index.ts: the heavy lifting is C++/GGML inside
// the SAME ace-server process, so this module is a capability manifest + a
// client wrapper, with the generation path in ./generate.ts.
//
// LICENSE NOTE: YuE2 weights are CC BY-NC 4.0 (non-commercial). This backend
// must never be presented as unconditionally licensed for a commercial
// product — the picker and Model Manager carry a plain notice plus the
// upstream commercial-licence contact (gezhang@umich.edu). `licenseNotice`
// below is additive on BackendCapabilities so the UI has one place to read it
// from rather than hardcoding the string a second time.

import { engineReady } from '../../../engineState.js';
import { isEngineSuspended } from '../../aceEngineProcess.js';
import { getSetting, setSetting } from '../../../db/lireekDb.js';
import { runYue2Generation } from './generate.js';
import {
  yue2Props, yue2PropsCached, yue2SelectModel, yue2Unload,
} from './client.js';
import type { Yue2Selection } from './client.js';
import type {
  EngineBackend,
  BackendCapabilities,
  BackendModels,
  BackendLifecycleStatus,
  GenerationArtifact,
  GenerationContext,
  GenerationOperation,
  GenerationOutcome,
  ResolvedRequest,
} from '../types.js';
import type { GenerationJob } from '../../generation/jobTypes.js';

/** Mirrors capabilities().core.duration.max — the plan/semantic stages end on
 *  their own terminator; this is the honest v1 ceiling (§7 request fields:
 *  "duration ... max: 360, auto: true"). */
const YUE2_MAX_DURATION_SEC = 360;

/** CC BY-NC 4.0 — non-commercial only. See the header note above and the
 *  Model Manager entry for the same text. */
export const YUE2_LICENSE_NOTICE =
  'YuE2 weights are licensed CC BY-NC 4.0 (non-commercial use only). '
  + 'For a commercial license, contact the upstream authors at gezhang@umich.edu.';

const LM_TYPE_SETTING = 'yue2_lm_type';
const VAE_VARIANT_SETTING = 'yue2_vae_variant';

/** The persisted selection. '' = auto (engine best-first / standard). */
export function yue2PersistedSelection(): { lm: string; vae_variant: string } {
  return {
    lm: getSetting(LM_TYPE_SETTING, ''),
    vae_variant: getSetting(VAE_VARIANT_SETTING, ''),
  };
}

const YUE2_OPERATIONS: readonly GenerationOperation[] = ['text2music'];

function firstText(submission: Readonly<Record<string, unknown>>, keys: string[]): string {
  for (const key of keys) {
    const value = submission[key];
    if (typeof value === 'string' && value) return value;
  }
  return '';
}

/** Pure descriptive snapshot. The live mapper remains mapYue2Params(). */
function resolveRequest(submission: Readonly<Record<string, unknown>>): ResolvedRequest {
  const instrumental = typeof submission.instrumental === 'boolean'
    ? submission.instrumental : undefined;
  const options: Record<string, unknown> = {};
  for (const key of Object.keys(submission)) {
    if (key.startsWith('yue2')) options[key] = submission[key];
  }
  const common = {
    caption: firstText(submission, ['prompt', 'songDescription', 'caption', 'style']),
    lyrics: instrumental === true ? '' : typeof submission.lyrics === 'string' ? submission.lyrics : '',
    ...(instrumental === undefined ? {} : { instrumental }),
    ...(typeof submission.seed === 'number' && Number.isFinite(submission.seed) ? { seed: submission.seed } : {}),
    ...(typeof submission.randomSeed === 'boolean' ? { randomSeed: submission.randomSeed } : {}),
    ...(typeof submission.title === 'string' ? { title: submission.title } : {}),
  };
  return {
    operation: 'text2music',
    common,
    models: yue2PersistedSelection(),
    options,
    policy: { retry: { maxAttempts: 2, reseedOnRetry: true } },
  };
}

function outcomeFromJob(job: GenerationJob): GenerationOutcome {
  const result = job.result;
  const artifacts: GenerationArtifact[] = (result?.audioUrls ?? []).map((url, trackIndex) => ({
    kind: 'audio', trackIndex, url,
  }));
  return {
    endReason: job.status === 'succeeded' ? 'completed' : job.status === 'cancelled' ? 'cancelled' : 'failed',
    stages: result?.timing ?? [],
    artifacts,
    songIds: result?.songIds ?? [],
    result,
    error: job.error,
  };
}

function status(): BackendLifecycleStatus {
  if (isEngineSuspended()) return 'suspended';
  if (!engineReady) return 'down';
  const props = yue2PropsCached();
  return props?.synth_ready ? 'ready' : 'down';
}

async function capabilities(): Promise<BackendCapabilities> {
  const { props, stale } = await yue2Props();
  const synthReady = props?.synth_ready === true;
  const up = engineReady && !isEngineSuspended() && synthReady;

  // modelsMissing: honest, narrower than `!up` — true only when the engine
  // is reachable but the weight files themselves weren't found. Fail open
  // (false) on a stale/never-fetched manifest, same contract as MM3's own.
  const modelsMissing = !stale && props != null && !synthReady &&
    (props.files?.lm?.found === false
      || (props.files?.vae_standard?.found === false && props.files?.vae_legacy?.found === false));

  return {
    backend: 'yue2',
    up,
    core: {
      // auto: the AR plan/semantic stages end on their own terminator; a
      // requested length has no wire slot at all (see generate.ts's duration
      // note), so `editable: false` hides the UI control the same way MM3's
      // does.
      duration: { max: YUE2_MAX_DURATION_SEC, auto: true, editable: false },
      bpm: false,
      keyscale: false,
      negativePrompt: false,
      batch: { max: 1 },
      seed: true,
      captionFormat: 'freeform',
      timeSignature: false,
      languageMeans: 'lyrics',
      modelsMissing,
      modelsMissingHint: 'yue2-lm + yue2-vae GGUFs (~8 GB)',
      propsStale: stale,
    },
    features: {
      models: true,
      lm: false,
      plugins: false,
      samplerPlugins: false,
      adapters: false,
      lmAdapters: false,
      postProcess: true,
      stableStep: true,
      whisper: true,
      // YuE2's DiT (the NAR/flow stack) has no lyric cross-attention or
      // decode-alignment head yet — no route to per-line timestamps in v1.
      lyricTimestamps: false,
      cover: false,
      repaint: false,
      lego: false,
      extract: false,
      streaming: false,
      training: false,
      midi: false,
      stems: false,
      understand: false,
      conceptSteering: false,
      captionDatasetSource: false,
      timbreReference: false,
    },
    license: YUE2_LICENSE_NOTICE,
    extensions: [
      {
        key: 'yue2Cot',
        type: 'select',
        label: 'Chain of Thought',
        hint: 'How much of the song structure the planner writes out before generating audio. '
            + '"full" is the reference default; "off" skips the ABC plan entirely and is the '
            + 'fastest, least-structured mode (its CFG is on by default, unlike the other two).',
        default: 'full',
        options: [
          { value: 'full', label: 'Full (structure + melody)' },
          { value: 'melody', label: 'Melody only' },
          { value: 'off', label: 'Off (fastest, least structure)' },
        ],
      },
      {
        key: 'yue2CfgScale',
        type: 'text',
        label: 'CFG Scale',
        hint: 'Blank = the checkpoint\'s own default for the selected Chain of Thought mode '
            + '(1.0 for melody/full, 1.01 for off). Set a number to override it.',
        default: '',
      },
      {
        key: 'yue2OdeSteps',
        type: 'slider',
        label: 'ODE Steps',
        hint: 'Midpoint-solver steps per NAR chunk (the checkpoint default is 32; each step is '
            + 'two network evaluations). Lower is faster and less refined.',
        default: 32,
        min: 8,
        max: 64,
        step: 1,
      },
      {
        key: 'yue2VaeVariant',
        type: 'select',
        label: 'VAE Variant',
        hint: 'Which of the two shipped decoder checkpoints renders the final audio.',
        default: 'standard',
        options: [
          { value: 'standard', label: 'Standard' },
          { value: 'legacy', label: 'Legacy' },
        ],
      },
    ],
  };
}

async function models(): Promise<BackendModels> {
  const { props } = await yue2Props();
  const v = props?.variants;

  const meta: NonNullable<BackendModels['meta']> = {};
  if (v?.lm?.available?.length) {
    meta.lm = {};
    for (const f of v.lm.available) meta.lm[f.type] = { label: f.filename, bytes: f.bytes };
  }

  return {
    buckets: {
      lm: (v?.lm?.available ?? []).map(f => f.type),
      // Fixed two-choice enum, not a disk scan — matches the extension's own
      // yue2VaeVariant select. modelsMissing/props.files is what tells the
      // user whether a given variant is actually installed.
      vae: ['standard', 'legacy'],
    },
    adapters: [],
    lmAdapters: [],
    defaults: {
      lm: v?.lm?.selected ?? '',
      vae: props?.files?.vae_standard?.found ? 'standard' : (props?.files?.vae_legacy?.found ? 'legacy' : ''),
    },
    meta,
  };
}

async function selectModel(selection: Record<string, string>) {
  const persisted = yue2PersistedSelection();
  const sel: Yue2Selection = {
    lm: selection.lm ?? persisted.lm,
    vae_variant: (selection.vae as Yue2Selection['vae_variant']) ?? (persisted.vae_variant as Yue2Selection['vae_variant']),
  };
  // `changed` isn't part of the engine's own response (yue2_handle_select_model
  // returns {selected, vae_variant, lm_type_want, lm_file, lm_found} — no
  // `changed`/`lm` field, unlike mm3SelectModel's shape); EngineBackend's
  // interface requires it, so it's derived here from the persisted values.
  const changed = (sel.lm ?? '') !== persisted.lm || (sel.vae_variant ?? '') !== persisted.vae_variant;
  const result = await yue2SelectModel(sel);
  setSetting(LM_TYPE_SETTING, sel.lm ?? '');
  setSetting(VAE_VARIANT_SETTING, sel.vae_variant ?? '');
  if (changed) {
    console.log(`[Backends] YuE2 models: lm_type=${result.lm_type_want || '(auto)'} vae_variant=${result.vae_variant} lm_found=${result.lm_found}`);
  }
  return { ...result, changed };
}

export const yue2Backend: EngineBackend = {
  id: 'yue2',
  displayName: 'YuE2',
  resourcePool: 'gpu',
  lifecycle: {
    // No process of its own — a model family inside ace-server, same as MM3.
    async start() { /* no separate process — ace-server owns the lifecycle */ },
    async stop() { await yue2Unload(); },
    status,
  },
  capabilities,
  models,
  selectModel,
  operations: YUE2_OPERATIONS,
  resolveRequest,
  async generate(job: GenerationJob, ctx: GenerationContext): Promise<GenerationOutcome> {
    await runYue2Generation(job, {
      attempt: ctx.attempt,
      pollUntilDone: ctx.pollUntilDone,
      signal: ctx.signal,
    });
    return outcomeFromJob(job);
  },
  // v1 choice per docs/plans/yue2/06-engine-port-plan.md §7 Residency: YuE2
  // has no engine-side cross-family arbitration code (unlike MM3's
  // mm3_arbitrate_vram), so the shared Node eviction runner handles it —
  // same value and same reasoning as ACE's.
  arbitratesResidencyInEngine: false,
  /** Model-residency arbitration: frees YuE2's ~8 GB when the active backend
   *  switches away, so it never sits in VRAM next to another family. No
   *  keepCaches concept to preserve (no AR-cache analog in v1). */
  async releaseVram() {
    const r = await yue2Unload();
    if (r?.unloaded) {
      console.log(`[Backends] YuE2 unloaded (${(r.freed_mb ?? 0).toFixed(0)} MB freed)`);
    }
  },
};
