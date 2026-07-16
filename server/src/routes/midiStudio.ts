// midiStudio.ts — MIDI Studio audio→MIDI transcription route
//
// Transcription itself is being ported to a native GGML binary (`ace-midi`) —
// see docs/plans/muscriptor-cpp-port.md. The old Python-venv runner was
// removed (2026-07-16). Until the engine port lands, POST /transcribe returns
// 501; job history, piano-roll notes, .mid downloads and the Hugging Face
// token store (needed to download the gated weights) all remain live.
//
// Mounts at: /api/midi-studio
// Routes:
//   GET    /api/midi-studio/status            — feature/engine status
//   POST   /api/midi-studio/hf-token          — save/clear HF token (gated weights)
//   POST   /api/midi-studio/transcribe        — 501 until the ace-midi port lands
//   GET    /api/midi-studio/jobs              — list past jobs (disk-backed)
//   GET    /api/midi-studio/:jobId/notes      — parsed notes for piano roll
//   GET    /api/midi-studio/:jobId/file       — download the .mid
//   DELETE /api/midi-studio/:jobId            — delete a job

import { Router, Request, Response } from 'express';
import path from 'path';
import fs from 'fs';
import { config } from '../config.js';
import { getHfToken, setHfToken, MUSCRIPTOR_MODELS } from '../services/muscriptor.js';
import { parseMidiFile } from '../services/midiParser.js';

const router = Router();

const midiBaseDir = path.join(config.data.dir, 'midi');
fs.mkdirSync(midiBaseDir, { recursive: true });

function jobDir(id: string): string { return path.join(midiBaseDir, id); }
function midPath(id: string): string { return path.join(jobDir(id), 'out.mid'); }

// ── Routes ───────────────────────────────────────────────────────────────

/** GET /status — feature status (engine port pending) */
router.get('/status', (_req: Request, res: Response) => {
  res.json({
    // The native ace-midi engine is not built yet; the UI shows a
    // "port in progress" banner while enginePending is true.
    enginePending: true,
    hfTokenSet: getHfToken() !== null,
    models: MUSCRIPTOR_MODELS,
  });
});

/**
 * POST /hf-token — store (or clear, with empty string) the Hugging Face
 * read token used to download the gated MuScriptor weights. Never echoed back.
 */
router.post('/hf-token', (req: Request, res: Response) => {
  try {
    const token = typeof req.body?.token === 'string' ? req.body.token : '';
    setHfToken(token);
    console.log(`[MidiStudio] HF token ${token.trim() ? 'saved' : 'cleared'}`);
    res.json({ ok: true, hfTokenSet: !!token.trim() });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

/** POST /transcribe — not available until the ace-midi engine port lands */
router.post('/transcribe', (_req: Request, res: Response) => {
  res.status(501).json({
    error: 'Transcription is being ported to the native C++ engine and is temporarily unavailable.',
  });
});

/** GET /jobs — list completed jobs from disk */
router.get('/jobs', (_req: Request, res: Response) => {
  try {
    const summaries: any[] = [];
    if (fs.existsSync(midiBaseDir)) {
      for (const entry of fs.readdirSync(midiBaseDir, { withFileTypes: true })) {
        if (!entry.isDirectory()) continue;
        const metaPath = path.join(midiBaseDir, entry.name, '_meta.json');
        if (!fs.existsSync(metaPath)) continue;
        try {
          const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
          summaries.push({
            id: meta.id || entry.name,
            status: 'done',
            sourceFileName: meta.sourceFileName || 'unknown',
            songId: meta.songId,
            model: meta.model || 'small',
            noteCount: meta.noteCount || 0,
            durationSec: meta.durationSec || 0,
            createdAt: meta.createdAt || '',
          });
        } catch { /* skip corrupted meta */ }
      }
    }
    summaries.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
    res.json(summaries);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

/** GET /:jobId/notes — parsed note data for the piano roll */
router.get('/:jobId/notes', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const notesPath = path.join(jobDir(jobId), 'notes.json');
  try {
    if (fs.existsSync(notesPath)) {
      res.setHeader('Content-Type', 'application/json');
      fs.createReadStream(notesPath).pipe(res);
      return;
    }
    // Fall back to parsing the .mid on demand
    if (fs.existsSync(midPath(jobId))) {
      const parsed = parseMidiFile(fs.readFileSync(midPath(jobId)));
      fs.writeFileSync(notesPath, JSON.stringify(parsed));
      res.json(parsed);
      return;
    }
    res.status(404).json({ error: 'No MIDI data for this job' });
  } catch (err: any) {
    res.status(500).json({ error: `MIDI parse failed: ${err.message}` });
  }
});

/** GET /:jobId/file — download the .mid */
router.get('/:jobId/file', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const p = midPath(jobId);
  if (!fs.existsSync(p)) {
    res.status(404).json({ error: 'MIDI file not found' });
    return;
  }
  let base = 'transcription';
  try {
    const meta = JSON.parse(fs.readFileSync(path.join(jobDir(jobId), '_meta.json'), 'utf-8'));
    base = (meta.sourceFileName || base).replace(/\.[^.]+$/, '').replace(/[^\w\s.-]/g, '_');
  } catch { /* keep default */ }
  res.download(p, `${base}.mid`);
});

/** DELETE /:jobId — delete a job's files */
router.delete('/:jobId', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const dir = jobDir(jobId);
  if (fs.existsSync(dir)) {
    fs.rmSync(dir, { recursive: true, force: true });
    console.log(`[MidiStudio] Deleted job ${jobId}`);
  }
  res.json({ ok: true });
});

export default router;
