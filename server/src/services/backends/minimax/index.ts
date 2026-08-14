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
import { getSetting, setSetting } from '../../../db/lireekDb.js';
import { mm3Props, mm3PropsCached, mm3SelectModel, mm3Unload } from './client.js';
import type { Mm3Props } from './client.js';
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

// ── Persisted quant selection ───────────────────────────────────────────────
//
// The engine holds the chosen quant in memory only, so a restart (or a
// crash-respawn) silently reverts to best-first — f16 — and the user's pick is
// lost. ACE doesn't have this problem because its model names ride on every
// generate call; MM3's selection is engine STATE, so somebody has to remember
// it and put it back.
//
// Stored in the same generic settings table backends/registry.ts uses for the
// active backend id: no new mechanism, no schema change. Server-side rather
// than in localStorage deliberately — the engine must be restored even when no
// browser has been opened.
const LM_SETTING    = 'mm3_lm_quant';
const SYNTH_SETTING = 'mm3_synth_quant';

/** Push the persisted selection back into the engine if it has drifted.
 *
 *  `variants.<role>.requested` is the engine's own record of what it was last
 *  asked for, and it resets to '' when the process restarts — which makes it an
 *  exact drift signal rather than a guess. Idempotent and self-healing: safe to
 *  call on a timer, and it repairs a crash-respawn as well as a cold boot. */
async function reconcileSelection(props: Mm3Props | null, stale: boolean): Promise<void> {
  // A stale manifest means the props probe timed out, which nearly always means
  // a generation is holding the engine mutex. Re-selecting then would block and
  // then evict weights out from under the very next job — never reconcile on a
  // guess.
  if (stale || !props?.variants) return;

  const wantLm    = getSetting(LM_SETTING, '');
  const wantSynth = getSetting(SYNTH_SETTING, '');
  if (!wantLm && !wantSynth) return;   // never chosen — engine default is right

  const lm    = props.variants.lm;
  const synth = props.variants.synth;
  if (!lm || !synth) return;

  // Only ask for a quant the engine can actually see; a file deleted since the
  // choice was made must fall back, not wedge every capability poll on a 400.
  const has = (v: typeof lm, q: string) => !q || v.available?.some(f => f.quant === q);
  const lmTarget    = has(lm, wantLm) ? wantLm : '';
  const synthTarget = has(synth, wantSynth) ? wantSynth : '';

  if (lmTarget === (lm.requested ?? '') && synthTarget === (synth.requested ?? '')) return;

  try {
    const r = await mm3SelectModel({ lm: lmTarget, synth: synthTarget });
    if (r.changed) {
      console.log(`[Backends] MiniMax-Music3 restored persisted models: ${r.lm} + ${r.synth}`);
    }
  } catch (err: any) {
    // Advisory: a failed restore leaves the engine on its default, which still
    // generates. Logging beats throwing out of a capability poll.
    console.warn('[Backends] MiniMax-Music3 selection restore failed:', err?.message || err);
  }
}

/** Called once when the engine reports ready (server/src/index.ts), so the
 *  persisted choice is in force before the first generation rather than after
 *  the UI happens to poll. */
export async function restoreMm3Selection(): Promise<void> {
  const { props, stale } = await mm3Props();
  await reconcileSelection(props, stale);
}

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
  // Self-healing restore: the UI polls this, so an engine crash-respawn (which
  // resets the in-memory selection) is repaired without anyone touching the
  // dropdown. Fire-and-forget — capabilities must stay fast and never fail
  // because of a residency concern.
  void reconcileSelection(props, stale);
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
    // cfg_flow / steps were held back as "checkpoint-fixed sampling contract".
    // That was the right call for a bring-up, but the flow stage is ~56% of
    // wall time on a full-length track, and `steps` is a direct linear dial on
    // it — the single most useful speed/quality control MM3 has. Rendered
    // generically by the existing PluginControls schema renderer.
    extensions: [
      {
        key: 'mm3Steps',
        type: 'slider',
        label: 'Flow Steps',
        hint: 'Euler steps per window. The checkpoint default is 30; lower is '
            + 'proportionally faster (the flow stage dominates long renders) at '
            + 'some cost in detail.',
        default: 30,
        min: 8,
        max: 60,
        step: 1,
      },
      {
        key: 'mm3CfgFlow',
        type: 'slider',
        label: 'Flow Guidance (CFG)',
        hint: 'Checkpoint default 1.7. Higher follows the caption harder; '
            + 'lower is looser. 1.0 disables guidance.',
        default: 1.7,
        min: 1.0,
        max: 5.0,
        step: 0.1,
      },
    ],
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
  const lm    = selection.lm ?? '';
  const synth = selection.synth ?? '';
  const result = await mm3SelectModel({ lm, synth });
  // Persist only after the engine accepted it, so a rejected quant can never
  // be written back and then replayed on every subsequent boot.
  setSetting(LM_SETTING, lm);
  setSetting(SYNTH_SETTING, synth);
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
