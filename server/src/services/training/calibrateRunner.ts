// training/calibrateRunner.ts — the lm-calibrate job (2026-08-10).
//
// Runs AFTER a train-lm job (same GPU lane, so strictly ordered): spawns
// `npx tsx scripts/lm-adapter-rollout.ts calibrate` — the exact code path the
// 182-artist rollout validated — relays its output into the job log, parses
// the final CALIBRATE_RESULT line, and repoints this artist's album preset(s)
// at the served adapter.
//
// Why a spawned script and not an in-process service: the calibration flow
// (candidate evals via /lm, JS-divergence analysis, safetensors bake) lives in
// server/scripts/ where the standalone eval/rollout tools share it. src/ can't
// import from scripts/ (rootDir), and duplicating the logic here is how the
// two copies drift. The child is cheap (node + tsx), runs on the same machine,
// and its stdout is the job log — the same relay pattern as ace-train runners.
//
// PRESET RULE: trainLmRunner skips its refreshPresetsForNewRun when a
// calibration is queued — THIS job has the final say, and it points presets at
// the SERVED dir (which is the OLD adapter when the new one lost). Without
// that ordering a fresh-but-worse run would capture the preset.

import { spawn } from 'child_process';
import path from 'path';
import readline from 'readline';
import { fileURLToPath } from 'url';
import { pushLog } from '../../routes/logs.js';
import {
  emitJob, finishJob, isCancelled, pushEvent, type LmCalibrateOptions, type TrainingJob,
} from './labelingQueue.js';
import { getDataset } from './datasetsRepo.js';
import { refreshPresetsForNewRun } from './lyricStudioExport.js';

const CALIBRATE_TIMEOUT_MS = 40 * 60_000;   // lean protocol measures ~6-9 min; 4x headroom

/** server/ — works from both src/ (tsx dev) and dist/ (prod build), which
 *  mirror the same directory depth. scripts/ lives beside them. */
const SERVER_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..');

interface CalibrateOutcome {
  artist: string;
  status: 'repointed' | 'kept-old' | 'baked-old' | 'failed' | 'skipped';
  servedDir: string;
  oldScore: number | null;
  newScore: number | null;
  winner: string;
  detail: string;
  minutes: number;
}

function log(job: TrainingJob, level: 'info' | 'warn' | 'error', message: string): void {
  pushEvent(job, { type: 'log', level, message, ts: Date.now() });
}

export async function runLmCalibrateJob(job: TrainingJob): Promise<void> {
  return runCalibrateJob(job, 'lm');
}

export async function runDitCalibrateJob(job: TrainingJob): Promise<void> {
  return runCalibrateJob(job, 'dit');
}

async function runCalibrateJob(job: TrainingJob, kind: 'lm' | 'dit'): Promise<void> {
  if (isCancelled(job)) return;
  const opts = job.opts as LmCalibrateOptions;
  const script = kind === 'lm' ? 'scripts/lm-adapter-rollout.ts' : 'scripts/dit-adapter-rollout.ts';

  try {
    const ds = getDataset(job.datasetId);
    if (!ds) { finishJob(job, 'failed', 'Dataset not found'); return; }

    job.status = 'running';
    job.startedAt = Date.now();
    job.phase = 'calibrate';
    emitJob(job);
    pushLog(`[Training] ${kind}-calibrate job ${job.id}: ${ds.slug} around ${opts.newRunDir}`);

    const args = [
      'tsx', script, 'calibrate',
      '--dataset', ds.slug,
      '--new-run', opts.newRunDir,
      ...(opts.oldRunDir ? ['--old-run', opts.oldRunDir] : []),
      ...(opts.variantKey ? ['--variant', opts.variantKey] : []),
    ];

    const child = spawn(process.platform === 'win32' ? 'npx.cmd' : 'npx', args, {
      cwd: SERVER_DIR, windowsHide: true, shell: process.platform === 'win32',
    });
    job.child = child;

    let outcome: CalibrateOutcome | null = null;
    const tail: string[] = [];
    const relay = (line: string) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      if (trimmed.startsWith('CALIBRATE_RESULT ')) {
        try { outcome = JSON.parse(trimmed.slice('CALIBRATE_RESULT '.length)) as CalibrateOutcome; } catch { /* torn */ }
        return;
      }
      tail.push(trimmed);
      if (tail.length > 8) tail.shift();
      log(job, 'info', trimmed);
    };
    readline.createInterface({ input: child.stdout }).on('line', relay);
    readline.createInterface({ input: child.stderr }).on('line', relay);

    const timer = setTimeout(() => {
      log(job, 'error', `calibration exceeded ${CALIBRATE_TIMEOUT_MS / 60_000} min — killing`);
      try { child.kill(); } catch { /* already gone */ }
    }, CALIBRATE_TIMEOUT_MS);

    const code: number | null = await new Promise(resolve => {
      child.on('close', c => { clearTimeout(timer); resolve(c); });
      child.on('error', () => { clearTimeout(timer); resolve(-1); });
    });
    job.child = undefined;
    if (isCancelled(job)) return;

    if (code !== 0 || !outcome) {
      throw new Error(outcome && (outcome as CalibrateOutcome).detail
        ? (outcome as CalibrateOutcome).detail
        : `calibrate exited ${code}: ${tail.slice(-3).join(' | ')}`);
    }
    const done: CalibrateOutcome = outcome;

    // Preset repoint — the final say (see header). refreshPresetsForNewRun
    // moves every preset pointing at any run of this artist folder, which is
    // exactly right for all three outcomes: served may be the new run, a
    // -calibrated bake, or the old run itself (then this is a no-op for
    // presets already there).
    if (opts.repoint && done.servedDir) {
      const touched = refreshPresetsForNewRun(done.servedDir, kind);
      if (touched) log(job, 'info', `${touched} album preset(s) now point at ${path.basename(done.servedDir)}`);
    }

    const scoreTxt = done.newScore !== null && done.oldScore !== null
      ? ` (new ${done.newScore} vs old ${done.oldScore})`
      : '';
    log(job, 'info', `calibration ${done.status}: ${done.winner || 'n/a'}${scoreTxt} — ${done.detail}`);
    pushLog(`[Training] ${kind}-calibrate job ${job.id}: ${done.status}${scoreTxt} → ${done.servedDir || 'unchanged'}`);
    finishJob(job, 'done');
  } catch (err) {
    if (isCancelled(job)) return;
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[Training] ${kind}-calibrate job ${job.id} FAILED — ${message}`);
    finishJob(job, 'failed', message);
  }
}
