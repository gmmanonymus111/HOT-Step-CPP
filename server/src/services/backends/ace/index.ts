// backends/ace/index.ts — ACE-Step 1.5 backend
//
// Phase 1 scaffolding: implements EngineBackend by DELEGATING to the existing
// aceClient / aceEngineProcess / engineState modules — no logic moves, no
// behavior changes. This is the proof-of-shape backend; translateParams and
// the LM-echo generation path stay where they are (routes/generate.ts) until
// a later phase migrates them behind this interface (plan §4.1, §5 Phase 1).

import { aceClient } from '../../aceClient.js';
import { isEngineSuspended, restartAceServer, stopAceServer } from '../../aceEngineProcess.js';
import { engineReady } from '../../../engineState.js';
import type {
  EngineBackend,
  BackendCapabilities,
  BackendModels,
  BackendLifecycleStatus,
} from '../types.js';

function status(): BackendLifecycleStatus {
  if (isEngineSuspended()) return 'suspended';
  if (engineReady) return 'ready';
  return 'down';
}

async function capabilities(): Promise<BackendCapabilities> {
  const up = engineReady && !isEngineSuspended();
  return {
    backend: 'ace',
    up,
    core: {
      // 240 matches the UI's long-standing ACE duration ceiling — the slider
      // max is now capability-driven, so this value IS the user-visible limit.
      duration: { max: 240 },
      bpm: true,
      keyscale: true,
      negativePrompt: true,
      batch: { max: 8 },
      seed: true,
    },
    features: {
      lm: true,
      plugins: true,
      adapters: true,
      cover: true,
      repaint: true,
      lego: true,
      extract: true,
      streaming: true,
      training: true,
      midi: true,
      stems: true,
      understand: true,
      conceptSteering: true,
    },
    // ACE's solver/scheduler/guidance/adapter knobs are still surfaced via
    // the existing /api/plugins registry, not through this manifest yet —
    // Phase 1 leaves this empty rather than half-duplicating that route.
    extensions: [],
  };
}

async function models(): Promise<BackendModels> {
  try {
    const props = await aceClient.props();
    return {
      buckets: {
        lm: props.models.lm,
        dit: props.models.dit,
        vae: props.models.vae,
        embedding: props.models.embedding,
      },
      adapters: props.adapters,
      lmAdapters: props.lm_adapters ?? [],
      defaults: props.default,
    };
  } catch {
    // ace-server unreachable — same degrade-empty contract as routes/models.ts
    return {
      buckets: { lm: [], dit: [], vae: [], embedding: [] },
      adapters: [],
      lmAdapters: [],
      defaults: {},
    };
  }
}

export const aceBackend: EngineBackend = {
  id: 'ace',
  displayName: 'ACE-Step 1.5',
  resourcePool: 'gpu',
  lifecycle: {
    // Phase 1: the engine process is bootstrapped and managed by
    // index.ts/aceEngineProcess.ts directly (respawn, crash budget,
    // suspension). Routing start/stop through here is a later-phase move —
    // for now these delegate to the same restart/stop primitives so the
    // interface shape is exercised without duplicating lifecycle ownership.
    async start() {
      await restartAceServer();
    },
    async stop() {
      await stopAceServer('Stopped via backend abstraction');
    },
    status,
  },
  capabilities,
  models,
};
