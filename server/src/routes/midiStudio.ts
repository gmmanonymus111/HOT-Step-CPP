// midiStudio.ts — MIDI Studio audio→MIDI transcription route
//
// Transcribes library tracks (or any /audio, /references URL) to multi-track
// MIDI via MuScriptor (Kyutai & Mirelo, arXiv:2607.08168), running in a
// server-managed Python venv (services/muscriptor.ts). Results persist to
// data/midi/<jobId>/ (out.mid + notes.json + _meta.json).
//
// Mounts at: /api/midi-studio
// Routes:
//   GET    /api/midi-studio/status            — MuScriptor install status
//   POST   /api/midi-studio/setup             — begin venv install (poll /status)
//   POST   /api/midi-studio/transcribe        — start a transcription job
//   GET    /api/midi-studio/jobs              — list past jobs (disk-backed)
//   GET    /api/midi-studio/:jobId/progress   — poll a running job
//   GET    /api/midi-studio/:jobId/notes      — parsed notes for piano roll
//   GET    /api/midi-studio/:jobId/file       — download the .mid
//   DELETE /api/midi-studio/:jobId            — cancel/delete a job

import { Router, Request, Response } from 'express';
import path from 'path';
import fs from 'fs';
import { randomUUID } from 'crypto';
import { config } from '../config.js';
import {
  getStatus, startInstall, isInstalled, transcribe,
  MUSCRIPTOR_MODELS, type MuscriptorModel, type TranscribeHandle,
} from '../services/muscriptor.js';
import { parseMidiFile } from '../services/midiParser.js';

const router = Router();

const midiBaseDir = path.join(config.data.dir, 'midi');
fs.mkdirSync(midiBaseDir, { recursive: true });

// One transcription at a time — MuScriptor is CPU/GPU heavy and the box is
// usually busy generating music already.
const TRANSCRIBE_TIMEOUT_MS = 60 * 60 * 1000;

interface MidiJob {
  id: string;
  status: 'queued' | 'transcribing' | 'done' | 'failed' | 'cancelled';
  sourceAudioUrl: string;
  sourceFileName: string;
  songId?: string;
  model: MuscriptorModel;
  progressLine: string;
  error?: string;
  createdAt: number;
  handle?: TranscribeHandle;
}

const jobs = new Map<string, MidiJob>();
let queueTail: Promise<void> = Promise.resolve();

/** Resolve a URL-style audio path to an absolute filesystem path */
function resolveAudioPath(audioUrl: string): string {
  if (audioUrl.startsWith('/references/')) {
    return path.join(config.data.dir, 'references', path.basename(audioUrl));
  }
  if (audioUrl.startsWith('/audio/')) {
    return path.join(config.data.audioDir, path.basename(audioUrl));
  }
  if (path.isAbsolute(audioUrl)) {
    return audioUrl;
  }
  return path.join(config.data.dir, 'references', path.basename(audioUrl));
}

function jobDir(id: string): string { return path.join(midiBaseDir, id); }
function midPath(id: string): string { return path.join(jobDir(id), 'out.mid'); }

async function runTranscription(job: MidiJob): Promise<void> {
  if ((job.status as string) === 'cancelled') return;
  const dir = jobDir(job.id);
  fs.mkdirSync(dir, { recursive: true });

  try {
    const srcPath = resolveAudioPath(job.sourceAudioUrl);
    if (!fs.existsSync(srcPath)) throw new Error(`Source audio not found: ${srcPath}`);
    if (!isInstalled()) throw new Error('MuScriptor is not installed — run setup first');

    job.status = 'transcribing';
    job.progressLine = 'Starting MuScriptor…';
    console.log(`[MidiStudio] Job ${job.id}: transcribing ${path.basename(srcPath)} (model=${job.model})`);

    const handle = transcribe(srcPath, midPath(job.id), job.model, (line) => {
      job.progressLine = line;
    });
    job.handle = handle;
    const killer = setTimeout(() => {
      console.error(`[MidiStudio] Job ${job.id}: timed out — killing muscriptor`);
      handle.child.kill();
    }, TRANSCRIBE_TIMEOUT_MS);
    try {
      await handle.done;
    } finally {
      clearTimeout(killer);
      job.handle = undefined;
    }
    if ((job.status as string) === 'cancelled') return;
    if (!fs.existsSync(midPath(job.id))) throw new Error('muscriptor finished but produced no MIDI file');

    // Parse once for the piano-roll preview; preview failure is non-fatal.
    let noteCount = 0;
    let durationSec = 0;
    try {
      const parsed = parseMidiFile(fs.readFileSync(midPath(job.id)));
      noteCount = parsed.noteCount;
      durationSec = parsed.durationSec;
      fs.writeFileSync(path.join(dir, 'notes.json'), JSON.stringify(parsed));
    } catch (err: any) {
      console.warn(`[MidiStudio] Job ${job.id}: MIDI parse for preview failed (${err.message}) — download still available`);
    }

    fs.writeFileSync(path.join(dir, '_meta.json'), JSON.stringify({
      id: job.id,
      sourceAudioUrl: job.sourceAudioUrl,
      sourceFileName: job.sourceFileName,
      songId: job.songId,
      model: job.model,
      noteCount,
      durationSec,
      createdAt: new Date(job.createdAt).toISOString(),
    }, null, 2));

    job.status = 'done';
    console.log(`[MidiStudio] Job ${job.id}: done (${noteCount} notes, ${durationSec.toFixed(1)}s)`);
  } catch (err: any) {
    if ((job.status as string) !== 'cancelled') {
      job.status = 'failed';
      job.error = err.message || 'Unknown error';
      console.error(`[MidiStudio] Job ${job.id}: FAILED — ${job.error}`);
    }
  }
}

// ── Routes ───────────────────────────────────────────────────────────────

/** GET /status — MuScriptor environment status */
router.get('/status', (_req: Request, res: Response) => {
  try {
    res.json({ ...getStatus(), models: MUSCRIPTOR_MODELS });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

/** POST /setup — start venv bootstrap + pip install; poll /status for progress */
router.post('/setup', (_req: Request, res: Response) => {
  const r = startInstall();
  if (!r.started) {
    res.status(409).json({ error: r.error });
    return;
  }
  res.json({ ok: true });
});

/** POST /transcribe — queue a transcription job */
router.post('/transcribe', (req: Request, res: Response) => {
  const { sourceAudioUrl, sourceFileName, songId, model } = req.body || {};
  if (!sourceAudioUrl || typeof sourceAudioUrl !== 'string') {
    res.status(400).json({ error: 'sourceAudioUrl is required' });
    return;
  }
  const m: MuscriptorModel = MUSCRIPTOR_MODELS.includes(model) ? model : 'small';
  if (!isInstalled()) {
    res.status(409).json({ error: 'MuScriptor is not installed — run setup first' });
    return;
  }

  const job: MidiJob = {
    id: randomUUID(),
    status: 'queued',
    sourceAudioUrl,
    sourceFileName: sourceFileName || path.basename(sourceAudioUrl),
    songId: songId || undefined,
    model: m,
    progressLine: 'Queued…',
    createdAt: Date.now(),
  };
  jobs.set(job.id, job);
  queueTail = queueTail.then(() => runTranscription(job));

  console.log(`[MidiStudio] Job ${job.id} queued: ${job.sourceFileName} (model=${m})`);
  res.json({ id: job.id });
});

/** GET /jobs — list completed jobs from disk + any in-flight jobs */
router.get('/jobs', (_req: Request, res: Response) => {
  try {
    const summaries: any[] = [];
    const onDisk = new Set<string>();
    if (fs.existsSync(midiBaseDir)) {
      for (const entry of fs.readdirSync(midiBaseDir, { withFileTypes: true })) {
        if (!entry.isDirectory()) continue;
        const metaPath = path.join(midiBaseDir, entry.name, '_meta.json');
        if (!fs.existsSync(metaPath)) continue;
        try {
          const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
          onDisk.add(meta.id || entry.name);
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
    // In-flight / failed jobs only exist in memory
    for (const [, job] of jobs) {
      if (job.status === 'done' || onDisk.has(job.id)) continue;
      summaries.push({
        id: job.id,
        status: job.status,
        sourceFileName: job.sourceFileName,
        songId: job.songId,
        model: job.model,
        noteCount: 0,
        durationSec: 0,
        createdAt: new Date(job.createdAt).toISOString(),
        error: job.error,
      });
    }
    summaries.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
    res.json(summaries);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

/** GET /:jobId/progress — poll job progress */
router.get('/:jobId/progress', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const job = jobs.get(jobId);
  if (!job) {
    // Completed job surviving a server restart
    if (fs.existsSync(path.join(jobDir(jobId), '_meta.json'))) {
      res.json({ status: 'done', progressLine: '' });
      return;
    }
    res.status(404).json({ error: 'Job not found' });
    return;
  }
  res.json({ status: job.status, progressLine: job.progressLine, error: job.error });
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
    // Fall back to parsing on demand (e.g. notes.json write failed earlier)
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

/** DELETE /:jobId — cancel a running job and/or delete its files */
router.delete('/:jobId', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const job = jobs.get(jobId);
  if (job && (job.status === 'queued' || job.status === 'transcribing')) {
    job.status = 'cancelled';
    job.handle?.child.kill();
    console.log(`[MidiStudio] Job ${jobId} cancelled`);
  }
  jobs.delete(jobId);

  const dir = jobDir(jobId);
  if (fs.existsSync(dir)) {
    fs.rmSync(dir, { recursive: true, force: true });
    console.log(`[MidiStudio] Deleted job ${jobId}`);
  }
  res.json({ ok: true });
});

export default router;
