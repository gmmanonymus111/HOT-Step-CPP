// training/understandClient.ts — one /understand call, start to finish
//
// The engine has a SINGLE FIFO GPU worker shared by /lm, /synth, /understand,
// /warm and /vae. Labeling therefore queues behind whatever the user is
// generating, which is why the poll interval is deliberately slow (every poll
// competes with the GPU worker), the per-file cap is 20 minutes, and the queue
// depth is reported so the UI can say "waiting behind 2 generations".
//
// Spec: docs/plans/2026-07-27-dataset-studio-implementation.md §4.6, §7.1, §7.2

import fs from 'fs';
import os from 'os';
import path from 'path';
import crypto from 'crypto';
import { execFile } from 'child_process';
import { aceClient, type AceRequest } from '../aceClient.js';
import { config, getFFmpegPath } from '../../config.js';

export interface UnderstandResult {
  caption: string;
  lyrics: string;
  bpm: number | null;
  keyscale: string;
  timesignature: string;
  vocalLanguage: string;
  duration: number | null;
  seed: number | null;
  audioCodes: string | null;
}

export interface UnderstandHooks {
  onQueue?(depth: number): void;
  onPhase?(phase: string): void;
  signal: AbortSignal;
}

const POLL_INTERVAL_MS = 2000;   // NOT 1000 — every poll competes with the GPU worker
const QUEUE_CHECK_EVERY = 5;     // polls

/** An aborted job throws this; the caller treats it as "cancelled", not failed. */
export class AbortError extends Error {
  constructor(message = 'Cancelled') {
    super(message);
    this.name = 'AbortError';
  }
}

/** Are the LM / DiT / VAE registries all populated? Also returns the LM list. */
export async function engineUnderstandReady(): Promise<{ ok: boolean; missing: string[]; lmModels: string[] }> {
  const props = await aceClient.props();
  const missing: string[] = [];
  if (!props.models?.lm?.length) missing.push('lm');
  if (!props.models?.dit?.length) missing.push('dit');
  if (!props.models?.vae?.length) missing.push('vae');
  return { ok: missing.length === 0, missing, lmModels: props.models?.lm ?? [] };
}

/**
 * Best LM for understand. Without an explicit request the engine falls back to
 * "whatever is loaded, else FIRST REGISTRY ENTRY" — alphabetically the 0.6B,
 * whose captions/lyrics are far below the 4B's. Prefer the biggest model in a
 * fast near-lossless quant.
 */
export function pickBestLm(names: string[]): string {
  const score = (n: string): number => {
    let s = 0;
    if (/4B/i.test(n)) s += 300;
    else if (/1\.7B/i.test(n)) s += 200;
    else if (/0\.6B/i.test(n)) s += 100;
    if (/Q8_0/i.test(n)) s += 50;        // near-lossless, fastest of the high-quality quants
    else if (/Q6_K/i.test(n)) s += 45;
    else if (/Q5/i.test(n)) s += 40;
    else if (/BF16/i.test(n)) s += 30;   // best quality but slow + heavy
    else if (/NVFP4/i.test(n)) s += 10;
    return s;
  };
  return [...names].sort((a, b) => score(b) - score(a))[0] ?? '';
}

/**
 * How many engine jobs are still in flight. 0 when unknown.
 * `excludeJobId` drops our own understand job from the count so the number the
 * UI shows really is "jobs ahead of ours" (§2.0), not "including ours".
 */
export async function engineQueueDepth(excludeJobId?: string): Promise<number> {
  try {
    const jobs = await aceClient.listJobs();
    // The engine's job table keeps finished rows; only unfinished ones are
    // actually ahead of us in the FIFO worker.
    return jobs.filter(j => j.status === 'running' && j.id !== excludeJobId).length;
  } catch {
    return 0;
  }
}

/** Formats the engine can decode from an upload buffer ("[Audio] No audio decoded from buffer" for anything else). */
const ENGINE_DECODABLE = new Set(['.wav', '.mp3']);

/**
 * Read the file as engine-decodable audio: WAV/MP3 pass through untouched,
 * everything else (.flac/.ogg/.opus/.m4a/.aac) is transcoded to a temp
 * 48 kHz stereo WAV via ffmpeg first.
 */
async function readEngineDecodable(audioPath: string, signal: AbortSignal): Promise<Buffer> {
  const ext = path.extname(audioPath).toLowerCase();
  if (ENGINE_DECODABLE.has(ext)) return fs.readFileSync(audioPath);

  const ffmpeg = getFFmpegPath();
  if (!ffmpeg) {
    throw new Error(`Cannot label ${ext} audio — ffmpeg not found to convert it to WAV`);
  }
  const tmpWav = path.join(os.tmpdir(), `hs_training_understand_${crypto.randomBytes(6).toString('hex')}.wav`);
  try {
    await new Promise<void>((resolve, reject) => {
      execFile(
        ffmpeg,
        ['-y', '-i', audioPath, '-ar', '48000', '-ac', '2', '-c:a', 'pcm_s16le', tmpWav],
        { timeout: 120_000, maxBuffer: 10 * 1024 * 1024, signal },
        (error) => {
          if (error && (error as NodeJS.ErrnoException).name === 'AbortError') { reject(error); return; }
          resolve();  // caller checks the output file
        },
      );
    });
    if (!fs.existsSync(tmpWav)) {
      throw new Error(`ffmpeg could not convert ${path.basename(audioPath)} to WAV`);
    }
    return fs.readFileSync(tmpWav);
  } finally {
    try { fs.unlinkSync(tmpWav); } catch { /* never existed */ }
  }
}

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function str(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

/**
 * Submit one file to /understand and wait for the result.
 * Every field except `audio_codes` is best-effort — the LM may omit any of them.
 */
export async function runUnderstand(
  audioPath: string,
  params: Partial<AceRequest> | undefined,
  hooks: UnderstandHooks,
): Promise<UnderstandResult> {
  if (hooks.signal.aborted) throw new AbortError();

  const buf = await readEngineDecodable(audioPath, hooks.signal);
  if (hooks.signal.aborted) throw new AbortError();
  hooks.onPhase?.('understand');
  const jobId = await aceClient.submitUnderstand(buf, params);

  const deadline = Date.now() + config.training.understandTimeoutMs;
  let polls = 0;
  let lastPhase = '';

  for (;;) {
    if (hooks.signal.aborted) {
      await aceClient.cancelJob(jobId).catch(() => { /* engine may already be gone */ });
      throw new AbortError();
    }
    if (Date.now() > deadline) {
      await aceClient.cancelJob(jobId).catch(() => { /* best effort */ });
      throw new Error('Understand timed out after 20 min');
    }

    await new Promise(r => setTimeout(r, POLL_INTERVAL_MS));
    polls++;

    const status = await aceClient.pollJob(jobId);

    if (polls % QUEUE_CHECK_EVERY === 0) {
      // Our own job is `running` for the whole call — exclude it, or the UI
      // reports "1 job ahead" for the entire labeling run.
      const depth = await engineQueueDepth(jobId);
      hooks.onQueue?.(depth);
      if (depth > 0 && status.status === 'running' && (status.phase ?? '') === lastPhase) {
        hooks.onPhase?.('waiting-for-engine');
      }
    }
    lastPhase = status.phase ?? '';

    if (status.status === 'failed') throw new Error('Engine reported the understand job failed');
    if (status.status === 'cancelled') throw new AbortError('Understand job was cancelled');
    if (status.status !== 'done') continue;

    const res = await aceClient.getJobResult(jobId);
    if (!res.ok) throw new Error(`Understand result fetch failed (${res.status})`);
    const parsed = await res.json() as unknown;
    const row = (Array.isArray(parsed) ? parsed[0] : parsed) as Record<string, unknown> | undefined;
    if (!row || typeof row !== 'object') throw new Error('Understand returned an empty result');

    return {
      caption: str(row.caption),
      lyrics: str(row.lyrics),
      bpm: num(row.bpm),
      keyscale: str(row.keyscale),
      timesignature: str(row.timesignature),
      // The engine field is `vocal_language`, NOT `language`.
      vocalLanguage: str(row.vocal_language),
      duration: num(row.duration),
      seed: num(row.seed),
      audioCodes: str(row.audio_codes) || null,
    };
  }
}
