// routes/plugins.ts — Proxy to ace-server GET /plugins
//
// Exposes the Lua plugin registry (solvers, schedulers, guidance modes)
// with their metadata and parameter schemas to the frontend.

import { Router } from 'express';
import { aceClient } from '../services/aceClient.js';

const router = Router();

// GET /api/plugins — fetch plugin registry from ace-server
// Cached for 60s to avoid hammering the engine on every UI render.
let cache: { data: unknown; ts: number } | null = null;
const CACHE_TTL = 60_000;

router.get('/', async (_req, res) => {
  try {
    if (cache && Date.now() - cache.ts < CACHE_TTL) {
      return res.json(cache.data);
    }
    const registry = await aceClient.plugins();
    cache = { data: registry, ts: Date.now() };
    res.json(registry);
  } catch (err) {
    console.error('[plugins] Failed to fetch registry:', err);
    // Return empty registry on error so UI still works with fallback lists
    res.json({ solvers: [], schedulers: [], guidance: [] });
  }
});

// POST /api/plugins/reload — clear cache to force re-fetch
router.post('/reload', (_req, res) => {
  cache = null;
  res.json({ ok: true });
});

export default router;
