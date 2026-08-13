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
import { mm3Props, mm3PropsCached, mm3Unload } from './client.js';
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
    },
    // ALL false. This is the honest v1 manifest: none of these subsystems
    // exist for MM3 — they are not "coming soon" flags, they gate UI regions.
    features: {
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
  // MM3 ships as exactly two GGUFs (mm3-lm + mm3-synth) with no user-facing
  // model split — plan §4.5: "Music 3 has no lm/dit/vae split to show".
  // Reporting empty buckets is the honest answer, not a degraded one.
  return { buckets: {}, adapters: [], lmAdapters: [], defaults: {} };
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
  /** Model-residency arbitration (plan §4.4): switching away from MM3 frees
   *  its ~13 GB rather than leaving it parked next to the ACE pipeline. */
  async releaseVram() {
    const r = await mm3Unload();
    if (r?.unloaded) {
      console.log(`[Backends] MiniMax-Music3 unloaded (${(r.freed_mb ?? 0).toFixed(0)} MB freed)`);
    }
  },
};
