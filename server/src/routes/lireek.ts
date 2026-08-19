// lireek.ts — Express routes for Lyric Studio / Lireek
//
// All endpoints under /api/lireek/*
// Route handlers are split into focused modules:
//   - lireek/crudRoutes.ts: Artists, Lyrics Sets, Profiles, Generations, Presets
//   - lireek/llmRoutes.ts: LLM-powered generation, profiling, refinement
// Small utility routes (slop, purge, prompts, recent) remain here.
//
// USER ISOLATION: every route that accesses Lireek data extracts userId
// from req.user and passes it to all db functions.

import { Router, type Request, type Response } from 'express';
import * as db from '../db/lireekDb.js';
import { scanForSlop, BLACKLISTED_WORDS, BLACKLISTED_PHRASES } from '../services/lireek/slopDetector.js';
import {
  GENERATION_SYSTEM_PROMPT,
  SONG_METADATA_SYSTEM_PROMPT,
  PROFILE_PROMPT_1, PROFILE_PROMPT_2, PROFILE_PROMPT_3,
  REFINEMENT_SYSTEM_PROMPT,
} from '../services/lireek/prompts.js';
import { registerCrudRoutes } from './lireek/crudRoutes.js';
import { registerLlmRoutes } from './lireek/llmRoutes.js';
import { getUserId } from './auth.js';

const router = Router();

/** Safely extract a route param as string (Express 5 types params as string | string[]) */
function param(req: Request, name: string): string {
  const v = req.params[name];
  return Array.isArray(v) ? v[0] : v;
}

/** Require auth and return userId, or respond 401 */
function requireUserId(req: Request, res: Response): string | null {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return null; }
  return userId;
}

// ── Register modular route groups ────────────────────────────────────────────
registerCrudRoutes(router);
registerLlmRoutes(router);

// ── Slop Scanner ────────────────────────────────────────────────────────────

router.post('/slop-scan', (req: Request, res: Response) => {
  try {
    const { text, fingerprint, statistical_weight } = req.body;
    if (!text) { res.status(400).json({ error: 'text required' }); return; }
    const result = scanForSlop(text, fingerprint, statistical_weight);
    res.json(result);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── Purge ───────────────────────────────────────────────────────────────────

router.post('/purge', (req: Request, res: Response) => {
  const userId = requireUserId(req, res);
  if (!userId) return;
  try {
    const result = db.purgeProfilesAndGenerations(userId);
    res.json(result);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/purge-generations', (req: Request, res: Response) => {
  const userId = requireUserId(req, res);
  if (!userId) return;
  try {
    const result = db.purgeGenerationsOnly(userId);
    res.json(result);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/purge-profiles', (req: Request, res: Response) => {
  const userId = requireUserId(req, res);
  if (!userId) return;
  try {
    const result = db.purgeProfilesOnly(userId);
    res.json(result);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── Settings / Prompts ──────────────────────────────────────────────────────

router.get('/prompts', (req: Request, res: Response) => {
  const userId = requireUserId(req, res);
  if (!userId) return;
  const defaults: Record<string, string> = {
    generation_system: GENERATION_SYSTEM_PROMPT,
    metadata_system: SONG_METADATA_SYSTEM_PROMPT,
    profile_system: [PROFILE_PROMPT_1, PROFILE_PROMPT_2, PROFILE_PROMPT_3].join('\n\n---\n\n'),
    refine_system: REFINEMENT_SYSTEM_PROMPT,
  };
  const names = Object.keys(defaults);
  const prompts = names.map(name => ({
    name,
    default_content: defaults[name],
    custom: db.getSetting(userId, `prompt_${name}`) || null,
  }));
  res.json({ prompts });
});

router.put('/prompts/:name', (req: Request, res: Response) => {
  const userId = requireUserId(req, res);
  if (!userId) return;
  try {
    const promptName = param(req, 'name');
    const { value } = req.body;
    if (!value) { res.status(400).json({ error: 'value required' }); return; }
    db.setSetting(userId, `prompt_${promptName}`, value);
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.delete('/prompts/:name', (req: Request, res: Response) => {
  const userId = requireUserId(req, res);
  if (!userId) return;
  try {
    const promptName = param(req, 'name');
    db.setSetting(userId, `prompt_${promptName}`, '');
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── Recent Songs ────────────────────────────────────────────────────────────

router.get('/recent-songs', (req: Request, res: Response) => {
  const userId = requireUserId(req, res);
  if (!userId) return;
  try {
    const limit = parseInt(req.query.limit as string, 10) || 50;
    const rows = db.getRecentGenerationsWithAudio(userId, limit);
    res.json({ songs: rows });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

export default router;