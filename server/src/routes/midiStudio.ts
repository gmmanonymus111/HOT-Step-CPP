// midiStudio.ts — MIDI Studio audio→MIDI transcription route
//
// Transcribes library tracks to multi-instrument MIDI via the NATIVE ace-midi
// engine binary (MuScriptor GGML port — docs/plans/muscriptor-cpp-port.md).
// ace-midi reads WAV/MP3 directly, streams note events as JSONL on stdout
// (relayed live over SSE for the piano roll), and writes the .mid.
// Weights are gated on Hugging Face; the download endpoints fetch them with
// the user's stored read token. Results persist to data/midi/<userId>/<jobId>/.
//
// Mounts at: /api/midi-studio
// Routes:
//   GET    /api/midi-studio/status                — engine/models/token status
//   POST   /api/midi-studio/hf-token              — save/clear HF token
//   POST   /api/midi-studio/models/:size/download — download gated weights
//   POST   /api/midi-studio/transcribe            — queue a transcription job
//   GET    /api/midi-studio/jobs                  — list jobs (disk + active)
//   GET    /api/midi-studio/:jobId/progress       — poll a job
//   GET    /api/midi-studio/:jobId/stream         — SSE: live note events
//   GET    /api/midi-studio/:jobId/notes          — parsed notes (piano roll)
//   GET    /api/midi-studio/:jobId/file           — download the .mid
//   DELETE /api/midi-studio/:jobId                — cancel/delete a job

import { Router, Request, Response } from 'express';
import path from 'path';
import fs from 'fs';
import { spawn, ChildProcess } from 'child_process';
import { randomUUID } from 'crypto';
import readline from 'readline';
import { config } from '../config.js';
import {
  getHfToken, setHfToken, looksLikeGatedError, aceMidiExe,
  modelDir, isModelDownloaded, startModelDownload, getModelStates,
  MUSCRIPTOR_MODELS, type MuscriptorModel,
} from '../services/muscriptor.js';
import { parseMidiFile } from '../services/midiParser.js';

const router = Router();

const midiBaseDir = path.join(config.data.dir, 'midi');
fs.mkdirSync(midiBaseDir, { recursive: true });

const TRANSCRIBE_TIMEOUT_MS = 60 * 60 * 1000;

interface MidiJob {
  id: string;
  userId: string;
  status: 'queued' | 'transcribing' | 'done' | 'failed' | 'cancelled';
  sourceAudioUrl: string;
  sourceFileName: string;
  songId?: string;
  model: MuscriptorModel;
  // live event stream (JSONL objects from ace-midi, replayed to SSE clients)
  events: any[];
  chunksDone: number;
  chunksTotal: number;
  noteCount: number;
  error?: string;
  gated?: boolean;
  createdAt: number;
  child?: ChildProcess;
  listeners: Set<Response>;
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

/** User-scoped job directory */
function jobDir(userId: string, id: string): string { return path.join(midiBaseDir, userId, id); }
function midPath(userId: string, id: string): string { return path.join(jobDir(userId, id), 'out.mid'); }

/** Check if the caller is the job owner or admin */
function isOwnerOrAdmin(req: Request, job: MidiJob): boolean {
  const caller = req.user;
  if (!caller) return false;
  return caller.userId === job.userId || caller.role === 'admin';
}

/** Check ownership for a job looked up by ID */
function checkJobAccess(req: Request, jobId: string, userId?: string): { job?: MidiJob; error?: string; status?: number } {
  const job = jobs.get(jobId);
  if (!job) return {};
  if (!isOwnerOrAdmin(req, job)) {
    return { error: 'Forbidden', status: 403 };
  }
  return { job };
}

/** Read job metadata from disk (for completed jobs) */
function readJobMeta(userId: string, jobId: string): any | null {
  try {
    const metaPath = path.join(jobDir(userId, jobId), '_meta.json');
    if (fs.existsSync(metaPath)) {
      return JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
    }
  } catch { /* ignore */ }
  return null;
}

function broadcast(job: MidiJob, ev: any): void {
  const line = `data: ${JSON.stringify(ev)}\n\n`;
  for (const res of job.listeners) {
    try { res.write(line); } catch { /* client gone; cleanup on close */ }
  }
}

function pushEvent(job: MidiJob, ev: any): void {
  job.events.push(ev);
  if (ev.type === 'progress') {
    job.chunksDone = ev.completed ?? job.chunksDone;
    job.chunksTotal = ev.total ?? job.chunksTotal;
  } else if (ev.type === 'note_start') {
    job.noteCount++;
  }
  broadcast(job, ev);
}

function endStream(job: MidiJob): void {
  broadcast(job, { type: 'status', status: job.status, error: job.error, noteCount: job.noteCount });
  for (const res of job.listeners) {
    try { res.end(); } catch { /* ignore */ }
  }
  job.listeners.clear();
}

async function runTranscription(job: MidiJob): Promise<void> {
  if ((job.status as string) === 'cancelled') return;
  const dir = jobDir(job.userId, job.id);
  fs.mkdirSync(dir, { recursive: true });

  try {
    const exe = aceMidiExe();
    if (!exe) throw new Error('ace-midi engine binary not found — rebuild the engine or reinstall');
    if (!isModelDownloaded(job.model)) throw new Error(`Model '${job.model}' is not downloaded`);
    const srcPath = resolveAudioPath(job.sourceAudioUrl);
    if (!fs.existsSync(srcPath)) throw new Error(`Source audio not found: ${srcPath}`);

    job.status = 'transcribing';
    console.log(`[MidiStudio] Job ${job.id}: ace-midi ${path.basename(srcPath)} (model=${job.model})`);

    const child = spawn(exe, [
      '--model', modelDir(job.model),
      '--transcribe', srcPath,
      '--out', midPath(job.userId, job.id),
      '--jsonl',
    ], { windowsHide: true });
    job.child = child;

    const stderrTail: string[] = [];
    child.stderr?.on('data', (buf: Buffer) => {
      for (const raw of buf.toString('utf-8').split(/[\r\n]+/)) {
        const line = raw.trim();
        if (!line) continue;
        stderrTail.push(line);
        if (stderrTail.length > 30) stderrTail.shift();
      }
    });

    const rl = readline.createInterface({ input: child.stdout! });
    rl.on('line', (line) => {
      try { pushEvent(job, JSON.parse(line)); } catch { /* non-JSON noise */ }
    });

    const killer = setTimeout(() => {
      console.error(`[MidiStudio] Job ${job.id}: timed out — killing ace-midi`);
      child.kill();
    }, TRANSCRIBE_TIMEOUT_MS);

    const code: number | null = await new Promise((resolve, reject) => {
      child.on('error', (err) => reject(new Error(`Failed to launch ace-midi: ${err.message}`)));
      child.on('close', (c, signal) => resolve(signal ? null : c));
    }).finally(() => {
      clearTimeout(killer);
      job.child = undefined;
    }) as number | null;

    if ((job.status as string) === 'cancelled') return;
    if (code !== 0) {
      throw new Error(`ace-midi exited with code ${code}: ${stderrTail.slice(-5).join(' | ')}`);
    }
    if (!fs.existsSync(midPath(job.userId, job.id))) throw new Error('ace-midi finished but produced no MIDI file');

    // Parse for the piano-roll preview (preview failure is non-fatal)
    let noteCount = job.noteCount;
    let durationSec = 0;
    try {
      const parsed = parseMidiFile(fs.readFileSync(midPath(job.userId, job.id)));
      noteCount = parsed.noteCount;
      durationSec = parsed.durationSec;
      fs.writeFileSync(path.join(dir, 'notes.json'), JSON.stringify(parsed));
    } catch (err: any) {
      console.warn(`[MidiStudio] Job ${job.id}: MIDI parse for preview failed (${err.message})`);
    }

    fs.writeFileSync(path.join(dir, '_meta.json'), JSON.stringify({
      id: job.id,
      userId: job.userId,
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
      job.gated = looksLikeGatedError(job.error || '');
      console.error(`[MidiStudio] Job ${job.id}: FAILED — ${job.error}`);
    }
  } finally {
    endStream(job);
  }
}

// ── Routes ───────────────────────────────────────────────────────────────

/** GET /status — engine, models, token */
router.get('/status', (req: Request, res: Response) => {
  try {
    const userId = req.user?.userId;
    res.json({
      engineAvailable: aceMidiExe() !== null,
      hfTokenSet: userId ? getHfToken(userId) !== null : false,
      models: getModelStates(),
    });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

/** POST /hf-token — store (or clear, with empty string) the HF read token */
router.post('/hf-token', (req: Request, res: Response) => {
  try {
    const userId = req.user?.userId;
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    const token = typeof req.body?.token === 'string' ? req.body.token : '';
    setHfToken(token, userId);
    console.log(`[MidiStudio] HF token ${token.trim() ? 'saved' : 'cleared'} for user ${userId}`);
    res.json({ ok: true, hfTokenSet: !!token.trim() });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

/** POST /models/:size/download — begin downloading gated weights */
router.post('/models/:size/download', (req: Request, res: Response) => {
  const size = req.params.size as MuscriptorModel;
  if (!MUSCRIPTOR_MODELS.includes(size)) {
    res.status(400).json({ error: `Unknown model '${size}'` });
    return;
  }
  const userId = req.user?.userId;
  const r = startModelDownload(size, userId);
  if (!r.started) {
    res.status(409).json({ error: r.error });
    return;
  }
  res.json({ ok: true });
});

/** POST /transcribe — queue a transcription job */
router.post('/transcribe', (req: Request, res: Response) => {
  const userId = req.user?.userId;
  if (!userId) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }
  const { sourceAudioUrl, sourceFileName, songId, model } = req.body || {};
  if (!sourceAudioUrl || typeof sourceAudioUrl !== 'string') {
    res.status(400).json({ error: 'sourceAudioUrl is required' });
    return;
  }
  const m: MuscriptorModel = MUSCRIPTOR_MODELS.includes(model) ? model : 'small';
  if (!aceMidiExe()) {
    res.status(503).json({ error: 'ace-midi engine binary not found' });
    return;
  }
  if (!isModelDownloaded(m)) {
    res.status(409).json({ error: `Model '${m}' is not downloaded — download it first` });
    return;
  }

  const job: MidiJob = {
    id: randomUUID(),
    userId,
    status: 'queued',
    sourceAudioUrl,
    sourceFileName: sourceFileName || path.basename(sourceAudioUrl),
    songId: songId || undefined,
    model: m,
    events: [],
    chunksDone: 0,
    chunksTotal: 0,
    noteCount: 0,
    createdAt: Date.now(),
    listeners: new Set(),
  };
  jobs.set(job.id, job);
  queueTail = queueTail.then(() => runTranscription(job));

  console.log(`[MidiStudio] Job ${job.id} queued: ${job.sourceFileName} (model=${m})`);
  res.json({ id: job.id });
});

/** GET /jobs — completed jobs from disk + in-flight jobs (user-scoped) */
router.get('/jobs', (req: Request, res: Response) => {
  try {
    const userId = req.user?.userId;
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    const isAdmin = req.user?.role === 'admin';
    const summaries: any[] = [];
    const onDisk = new Set<string>();

    // Read from user's directory
    const userMidiDir = path.join(midiBaseDir, userId);
    if (fs.existsSync(userMidiDir)) {
      for (const entry of fs.readdirSync(userMidiDir, { withFileTypes: true })) {
        if (!entry.isDirectory()) continue;
        const metaPath = path.join(userMidiDir, entry.name, '_meta.json');
        if (!fs.existsSync(metaPath)) continue;
        try {
          const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
          onDisk.add(meta.id || entry.name);
          summaries.push({
            id: meta.id || entry.name,
            status: 'done',
            sourceFileName: meta.sourceFileName || 'unknown',
            sourceAudioUrl: meta.sourceAudioUrl,
            songId: meta.songId,
            model: meta.model || 'small',
            noteCount: meta.noteCount || 0,
            durationSec: meta.durationSec || 0,
            createdAt: meta.createdAt || '',
          });
        } catch { /* skip corrupted meta */ }
      }
    }

    // If admin, also include jobs from other users
    if (isAdmin) {
      for (const entry of fs.readdirSync(midiBaseDir, { withFileTypes: true })) {
        if (!entry.isDirectory() || entry.name === userId) continue;
        const otherUserDir = path.join(midiBaseDir, entry.name);
        for (const jobEntry of fs.readdirSync(otherUserDir, { withFileTypes: true })) {
          if (!jobEntry.isDirectory()) continue;
          const metaPath = path.join(otherUserDir, jobEntry.name, '_meta.json');
          if (!fs.existsSync(metaPath)) continue;
          try {
            const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
            const jobId = meta.id || jobEntry.name;
            if (onDisk.has(jobId)) continue;
            onDisk.add(jobId);
            summaries.push({
              id: jobId,
              status: 'done',
              sourceFileName: meta.sourceFileName || 'unknown',
              sourceAudioUrl: meta.sourceAudioUrl,
              songId: meta.songId,
              model: meta.model || 'small',
              noteCount: meta.noteCount || 0,
              durationSec: meta.durationSec || 0,
              createdAt: meta.createdAt || '',
              userId: entry.name,
            });
          } catch { /* skip corrupted meta */ }
        }
      }
    }

    // Add in-flight jobs
    for (const [, job] of jobs) {
      // Skip done jobs already on disk
      if (job.status === 'done' || onDisk.has(job.id)) continue;
      // Skip jobs from other users (unless admin)
      if (job.userId !== userId && !isAdmin) continue;
      summaries.push({
        id: job.id,
        status: job.status,
        sourceFileName: job.sourceFileName,
        sourceAudioUrl: job.sourceAudioUrl,
        songId: job.songId,
        model: job.model,
        noteCount: job.noteCount,
        durationSec: 0,
        chunksDone: job.chunksDone,
        chunksTotal: job.chunksTotal,
        createdAt: new Date(job.createdAt).toISOString(),
        error: job.error,
        gated: job.gated,
        userId: job.userId,
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
  const userId = req.user?.userId;
  if (!userId) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }

  // Check in-memory job
  const job = jobs.get(jobId);
  if (job) {
    if (!isOwnerOrAdmin(req, job)) {
      res.status(403).json({ error: 'Forbidden' });
      return;
    }
    res.json({
      status: job.status,
      chunksDone: job.chunksDone,
      chunksTotal: job.chunksTotal,
      noteCount: job.noteCount,
      error: job.error,
      gated: job.gated,
    });
    return;
  }

  // Check disk for completed job
  const meta = readJobMeta(userId, jobId);
  if (meta && (meta.userId === userId || req.user?.role === 'admin')) {
    res.json({ status: 'done' });
    return;
  }

  res.status(404).json({ error: 'Job not found' });
});

/** GET /:jobId/stream — SSE: replay buffered events then tail live ones */
router.get('/:jobId/stream', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const job = jobs.get(jobId);
  if (!job) {
    res.status(404).json({ error: 'Job not found or already finished (use /notes)' });
    return;
  }
  if (!isOwnerOrAdmin(req, job)) {
    res.status(403).json({ error: 'Forbidden' });
    return;
  }
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  for (const ev of job.events) {
    res.write(`data: ${JSON.stringify(ev)}\n\n`);
  }
  if (job.status === 'done' || job.status === 'failed' || job.status === 'cancelled') {
    res.write(`data: ${JSON.stringify({ type: 'status', status: job.status, error: job.error, noteCount: job.noteCount })}\n\n`);
    res.end();
    return;
  }
  job.listeners.add(res);
  req.on('close', () => job.listeners.delete(res));
});

/** GET /:jobId/notes — parsed note data for the piano roll */
router.get('/:jobId/notes', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const userId = req.user?.userId;
  if (!userId) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }

  const notesPath = path.join(jobDir(userId, jobId), 'notes.json');
  try {
    if (fs.existsSync(notesPath)) {
      res.setHeader('Content-Type', 'application/json');
      fs.createReadStream(notesPath).pipe(res);
      return;
    }
    const midFile = midPath(userId, jobId);
    if (fs.existsSync(midFile)) {
      const parsed = parseMidiFile(fs.readFileSync(midFile));
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
  const userId = req.user?.userId;
  if (!userId) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }

  const p = midPath(userId, jobId);
  if (!fs.existsSync(p)) {
    res.status(404).json({ error: 'MIDI file not found' });
    return;
  }

  // Check ownership
  const meta = readJobMeta(userId, jobId);
  if (meta && meta.userId !== userId && req.user?.role !== 'admin') {
    res.status(403).json({ error: 'Forbidden' });
    return;
  }

  let base = 'transcription';
  try {
    if (meta) {
      base = (meta.sourceFileName || base).replace(/\.[^.]+$/, '').replace(/[^\w\s.-]/g, '_');
    }
  } catch { /* keep default */ }
  res.download(p, `${base}.mid`);
});

/** DELETE /:jobId — cancel a running job and/or delete its files */
router.delete('/:jobId', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const userId = req.user?.userId;
  if (!userId) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }

  const job = jobs.get(jobId);
  if (job) {
    if (!isOwnerOrAdmin(req, job)) {
      res.status(403).json({ error: 'Forbidden' });
      return;
    }
    if (job.status === 'queued' || job.status === 'transcribing') {
      job.status = 'cancelled';
      job.child?.kill();
      endStream(job);
      console.log(`[MidiStudio] Job ${jobId} cancelled`);
    }
    jobs.delete(jobId);

    const dir = jobDir(job.userId, jobId);
    if (fs.existsSync(dir)) {
      fs.rmSync(dir, { recursive: true, force: true });
      console.log(`[MidiStudio] Deleted job ${jobId}`);
    }
    res.json({ ok: true });
    return;
  }

  // Check disk for completed job
  const meta = readJobMeta(userId, jobId);
  if (meta && (meta.userId === userId || req.user?.role === 'admin')) {
    const dir = jobDir(meta.userId, jobId);
    if (fs.existsSync(dir)) {
      fs.rmSync(dir, { recursive: true, force: true });
      console.log(`[MidiStudio] Deleted job ${jobId} from disk`);
    }
    res.json({ ok: true });
    return;
  }

  res.status(404).json({ error: 'Job not found' });
});

export default router;