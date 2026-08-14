// backends/minimax/index.ts — MiniMax-Music3 backend
//
// The second registered generation backend (plan §4.3). Thin by design: the
// heavy lifting is C++/GGML inside the SAME ace-server process as the ACE
// pipeline, so this module is a capability manifest + a client wrapper, with
// the generation path in ./generate.ts.
//
// LICENSE NOTE (plan §1.5): the MiniMax-Music3 Community License requires a
// commercial product to prominently display the string "MiniMax-Music3" in its
// UI. `displayName` below IS that display — the backend toggle renders it
// verbatim. Do not shorten, prettify, or localize it.

import { engineReady } from '../../../engineState.js';
import { isEngineSuspended } from '../../aceEngineProcess.js';
import { mm3Props, mm3PropsCached, mm3SelectModel, mm3Unload } from './client.js';
import type {
  EngineBackend,
  BackendCapabilities,
  BackendModels,
  BackendLifecycleStatus,
} from '../types.js';

/** Duration ceiling we expose. The checkpoint's own cap is 9,000 frames
 *  (= 360 s @ 25 fps) and the engine clamps to it; 300 s is the honest
 *  user-facing limit for v1 (the model card's "5 minutes"). */
const MM3_MAX_DURATION_SEC = 300;

function status(): BackendLifecycleStatus {
  if (isEngineSuspended()) return 'suspended';
  if (!engineReady) return 'down';
  // Cached-only: status() is called from route handlers and must never block
  // behind an in-flight MM3 generation (GET /mm3/props holds the MM3 mutex).
  const props = mm3PropsCached();
  return props?.synth_ready ? 'ready' : 'down';
}

async function capabilities(): Promise<BackendCapabilities> {
  const { props, stale } = await mm3Props();
  // stale === true means the props probe timed out — nearly always "an MM3
  // generation is running and holds the engine mutex", which is the opposite
  // of down. Keep the last-known-good answer rather than flapping to false.
  const synthReady = props?.synth_ready === true;
  const up = engineReady && !isEngineSuspended() && synthReady;

  // modelsMissing: honest, narrower than `!up` — true only when the engine IS
  // reachable but the weight files themselves weren't found (as opposed to
  // the engine being down/suspended, or files found-but-corrupt). This is
  // what the UI needs to decide "point the user at the Model Manager" vs
  // "the engine hasn't started yet". `found` is per-file (see Mm3PropsFile);
  // fail open (false) on a stale/never-fetched manifest — a probe timeout
  // must never be misread as "go nag the user to download 24 GB again".
  const modelsMissing = !stale && props != null && !synthReady &&
    (props.files?.lm?.found === false || props.files?.synth?.found === false);

  return {
    backend: 'minimax-m3',
    up,
    core: {
      duration: { max: MM3_MAX_DURATION_SEC },
      // MM3 takes no structured musical metadata — tempo/key live inside the
      // Structured Caption prose, not as fields.
      bpm: false,
      keyscale: false,
      // No negative/uncond prompt on the wire: the flow stage's uncond pass is
      // zeros-conditioning, fixed by the checkpoint.
      negativePrompt: false,
      // v1 generates one take per job (the engine has no batch axis exposed).
      batch: { max: 1 },
      seed: true,
      // Non-standard extras, honestly reported (BackendCoreCapabilities is an
      // open type). The UI ignores what it doesn't know.
      promptTokenLimit: props?.prompt_token_limit ?? 5000,
      maxAudioFrames: props?.max_audio_frames_limit ?? 9000,
      sampleRate: 44100,
      propsStale: stale,
      modelsMissing,
    },
    // Everything except `models` is false. This is the honest v1 manifest:
    // none of these subsystems exist for MM3 — they are not "coming soon"
    // flags, they gate UI regions. `models` IS true: MM3 ships a quant ladder
    // (mm3-{lm,synth}-<quant>.gguf) and the picker is live.
    features: {
      models: true,
      lm: false,
      plugins: false,
      adapters: false,
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
    },
    // cfg_flow / steps exist on the wire but are checkpoint-fixed sampling
    // contract, not user knobs (plan §1.5). Nothing to render yet.
    extensions: [],
  };
}

async function models(): Promise<BackendModels> {
  // MM3 ships as exactly two GGUFs (mm3-lm + mm3-synth), so there is no
  // lm/dit/vae split to show (plan §4.5) — but each of the two exists at
  // several quant levels, and THAT is the user-facing choice. Buckets are
  // therefore the two roles, with the quant tokens as options.
  const { props } = await mm3Props();
  const lm = props?.variants?.lm;
  const synth = props?.variants?.synth;

  const meta: NonNullable<BackendModels['meta']> = {};
  const bucketOf = (v: typeof lm, key: string): string[] => {
    if (!v?.available?.length) return [];
    meta[key] = {};
    for (const f of v.available) {
      meta[key][f.quant] = { label: f.filename, bytes: f.bytes };
    }
    return v.available.map(f => f.quant);
  };

  return {
    buckets: { lm: bucketOf(lm, 'lm'), synth: bucketOf(synth, 'synth') },
    // No adapter or planner-adapter subsystem for MM3 yet — the UI renders
    // these clusters as empty placeholders (features.adapters is false).
    adapters: [],
    lmAdapters: [],
    // The quant actually in force, not what was requested: if a selected file
    // is deleted the engine falls back and the UI must show the truth.
    defaults: { lm: lm?.selected ?? '', synth: synth?.selected ?? '' },
    meta,
  };
}

/** Switch which quant of each role runs. The engine unloads on a real change
 *  (the resident weights ARE the outgoing quant), so the next generation pays
 *  a warm — that is the honest cost of the switch, not a bug. */
async function selectModel(selection: Record<string, string>) {
  const result = await mm3SelectModel({
    lm: selection.lm ?? '',
    synth: selection.synth ?? '',
  });
  if (result.changed) {
    console.log(`[Backends] MiniMax-Music3 models: ${result.lm} + ${result.synth}` +
                (result.unloaded ? ' (evicted previous weights)' : ''));
  }
  return { ...result };
}

export const minimaxBackend: EngineBackend = {
  id: 'minimax-m3',
  displayName: 'MiniMax-Music3',   // license-mandated exact string — see header
  resourcePool: 'gpu',
  lifecycle: {
    // MM3 has no process of its own: it is a model family inside ace-server.
    // "Start" is therefore whatever ace-server's own lifecycle already did —
    // weights load lazily on the first job (the job's own VRAM arbitration
    // evicts the ACE side if needed).
    async start() { /* no separate process — ace-server owns the lifecycle */ },
    // "Clean shutdown (frees VRAM/resources)" for a residency-only backend is
    // exactly the unload.
    async stop() { await mm3Unload(); },
    status,
  },
  capabilities,
  models,
  selectModel,
  /** Model-residency arbitration (plan §4.4): switching away from MM3 frees
   *  its ~13 GB rather than leaving it parked next to the ACE pipeline. */
  async releaseVram() {
    const r = await mm3Unload();
    if (r?.unloaded) {
      console.log(`[Backends] MiniMax-Music3 unloaded (${(r.freed_mb ?? 0).toFixed(0)} MB freed)`);
    }
  },
};
