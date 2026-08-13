// routes/backends.ts — backend registry + capabilities API
//
// Phase 1 scaffolding (docs/plans/multi-backend-architecture.md §4.1/§4.2).
// GET  /api/backends            — list registered backends + which is active
// GET  /api/capabilities        — active (or ?backend=<id>) backend's manifest
// POST /api/backends/active     — switch the active backend { id }
//
// Style follows routes/plugins.ts (minimal route, short TTL cache so the UI
// can poll cheaply) and generalises the capabilities-endpoint pattern in
// routes/training.ts (~176-265): every probe degrades independently, the
// route itself never throws upward.
//
// NOTE on mounting: /api/capabilities is a top-level path per the plan
// (§4.2), sitting alongside /api/backends rather than nested under it — so
// this router is mounted at '/api' (not '/api/backends') in index.ts, and
// every route below spells its own full sub-path.

import { Router } from 'express';
import {
  getBackend,
  listBackends,
  getActiveBackendId,
  setActiveBackendId,
} from '../services/backends/registry.js';
import type { BackendCapabilities } from '../services/backends/types.js';

const router = Router();

// GET /api/backends — { id, displayName, active }[]
router.get('/backends', (_req, res) => {
  const activeId = getActiveBackendId();
  res.json({
    backends: listBackends().map(b => ({
      id: b.id,
      displayName: b.displayName,
      resourcePool: b.resourcePool,
      active: b.id === activeId,
    })),
    activeId,
  });
});

// POST /api/backends/active { id } — switch the active backend
router.post('/backends/active', (req, res) => {
  const id = req.body?.id;
  if (typeof id !== 'string' || !id.trim()) {
    res.status(400).json({ error: 'Missing "id" in request body' });
    return;
  }
  const ok = setActiveBackendId(id);
  if (!ok) {
    res.status(404).json({ error: `Unknown backend: ${id}` });
    return;
  }
  res.json({ activeId: getActiveBackendId() });
});

// GET /api/capabilities?backend=<id> — defaults to the active backend.
// Cached ~10s per backend id (mirrors the 60s /api/plugins cache, shorter
// because `up` should track engine state fairly closely).
const CACHE_TTL = 10_000;
const cache = new Map<string, { data: BackendCapabilities; ts: number }>();

router.get('/capabilities', async (req, res) => {
  const id = (req.query.backend as string) || getActiveBackendId();
  const backend = getBackend(id);
  if (!backend) {
    res.status(404).json({ error: `Unknown backend: ${id}` });
    return;
  }

  const cached = cache.get(id);
  if (cached && Date.now() - cached.ts < CACHE_TTL) {
    res.json(cached.data);
    return;
  }

  try {
    const data = await backend.capabilities();
    cache.set(id, { data, ts: Date.now() });
    res.json(data);
  } catch (err: any) {
    console.error(`[backends] capabilities probe failed for '${id}':`, err.message);
    // Degrade to an honest "down" manifest rather than a 500 — mirrors the
    // never-throw-upward contract of routes/training.ts's /capabilities.
    res.json({
      backend: id,
      up: false,
      core: { duration: { max: 0 }, bpm: false, keyscale: false, negativePrompt: false, batch: { max: 1 }, seed: false },
      features: {
        lm: false, plugins: false, adapters: false, cover: false, repaint: false,
        lego: false, extract: false, streaming: false, training: false, midi: false,
        stems: false, understand: false, conceptSteering: false,
      },
      extensions: [],
    } satisfies BackendCapabilities);
  }
});

export default router;
