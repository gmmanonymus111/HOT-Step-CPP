// training/mm3TrainRunner.ts — runs one `ace-train mm3-codes` or
// `ace-train mm3-lm-train` job.
//
// Structural clone of trainLmRunner.ts, which is the point: because the MM3
// binaries emit the SAME JSONL vocabulary as `train-lm` (init / step /
// milestone / progress / export / fatal / done), the relay below is the same
// relay and the Monitor's loss chart works with no new event types.
//
// Both kinds live in the GPU lane (labelingQueue laneFor) and both stop the
// engine first. That is not caution, it is the only configuration that fits:
// an MM3 LM training step peaks at 31.7 GB of a 32 GB card, so anything else
// resident means WDDM shared-memory thrash — measured at 38 s/step for work
// that takes 3.9 s when it fits.
//
// Spec: docs/plans/2026-08-20-mm3-training-server-design.md §2.3.

import fs from 'fs';
import path from 'path';
import readline from 'readline';
import { spawn } from 'child_process';

import { pushLog } from '../../routes/logs.js';
import { restartAceServer, stopAceServer } from '../aceEngineProcess.js';
import { aceTrainExe } from './aceTrain.js';
import { getDataset } from './datasetsRepo.js';
import {
  buildMm3CodesArgs, buildMm3TrainLmArgs, missingMm3TrainModels,
  type Mm3CodesArgs, type ResolvedMm3TrainLmOptions,
} from './mm3Train.js';
import {
  emitJob, emitProgress, finishJob, isCancelled, killJobChild, pushEvent, type TrainingJob,
} from './labelingQueue.js';

function log(job: TrainingJob, level: 'info' | 'warn' | 'error', message: string): void {
  pushEvent(job, { type: 'log', level, message, ts: Date.now() });
}

function int(v: unknown): number {
  const n = Number(v);
  return Number.isFinite(n) ? Math.trunc(n) : 0;
}

function optNum(ev: Record<string, unknown>, key: string): number | undefined {
  const v = ev[key];
  if (v === undefined || v === null) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function text(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

interface RelayState { fatalMessage: string; doneSeen: boolean }

/** ace-train JSONL -> the TrainingStreamEvent union. Deliberately defensive:
 *  every field access guarded, unknown `type` ignored. `step` emits no log line
 *  — one per optimizer step would flood the pane, and the chart reads metrics. */
function relay(job: TrainingJob, ev: Record<string, unknown>, st: RelayState): void {
  switch (text(ev.type)) {
    case 'init': {
      // MM3 trains in shuffled passes, so it has REAL epochs: stepsPerEpoch is
      // the training-song count. Publishing it is what puts the step layer and
      // the epoch line on one x axis (fractional epochs), exactly as ACE does —
      // no MM3 special case in the chart at all.
      const perEpoch = int(ev.stepsPerEpoch);
      const totalSteps = int(ev.totalSteps);
      pushEvent(job, {
        type: 'metric', metric: 'data', ts: Date.now(),
        stepsPerEpoch: perEpoch || undefined,
        epochs: perEpoch > 0 && totalSteps > 0 ? Math.ceil(totalSteps / perEpoch) : undefined,
        samples: optNum(ev, 'samples'),
        totalSteps: totalSteps || undefined,
      });
      const hold = int(ev.holdout);
      log(job, 'info',
        `${int(ev.samples)} training songs${hold ? ` (+${hold} held out)` : ''}, prompt up to `
        + `${int(ev.maxPrompt)} tok, crops to ${int(ev.maxFrames)} frames `
        + `(sequence up to ${int(ev.seqMax)}), rank ${int(ev.rank)}`);
      if (!hold) {
        log(job, 'warn',
          'No held-out songs, so there is no evaluation — the training loss is the only signal, and it '
          + 'cannot tell learning from memorising.');
      }
      break;
    }
    case 'epoch':
      pushEvent(job, {
        type: 'metric', metric: 'epoch', ts: Date.now(),
        epoch: optNum(ev, 'epoch'), loss: optNum(ev, 'loss'), lr: optNum(ev, 'lr'),
        ms: optNum(ev, 'ms'), step: optNum(ev, 'step'),
      });
      break;
    case 'eval':
      // The line that actually answers "is it still learning?".
      pushEvent(job, {
        type: 'metric', metric: 'eval', ts: Date.now(),
        step: optNum(ev, 'step'), loss: optNum(ev, 'loss'), crops: optNum(ev, 'crops'),
      });
      log(job, 'info', `Held-out loss ${(optNum(ev, 'loss') ?? 0).toFixed(4)} at step ${int(ev.step)}`);
      break;
    case 'best':
      log(job, 'info',
        `Best held-out loss ${(optNum(ev, 'loss') ?? 0).toFixed(4)} at step ${int(ev.step)} — `
        + 'start the ear test at that checkpoint.');
      break;
    case 'vram': {
      const usedMb = int(ev.usedMb), totalMb = int(ev.totalMb);
      pushEvent(job, {
        type: 'metric', metric: 'vram', ts: Date.now(),
        usedMb: optNum(ev, 'usedMb'), totalMb: optNum(ev, 'totalMb'), freeMb: optNum(ev, 'freeMb'),
      });
      // The one failure this run can suffer that looks like "it works, slowly":
      // over the card, WDDM pages to host memory and a 4 s step becomes 40.
      // Say it plainly rather than leaving the user to wonder.
      if (totalMb > 0 && usedMb > totalMb - 512) {
        log(job, 'warn',
          `VRAM is at ${usedMb}/${totalMb} MB — this run is on the edge of the card. If steps are far `
          + 'slower than expected, it is spilling into shared memory: lower Crop (frames) or close other '
          + 'GPU users.');
      }
      break;
    }
    case 'optimizer': {
      const muon = int(ev.muon), tensors = int(ev.tensors);
      log(job, 'info',
        `Optimizer ${text(ev.name)}: ${muon}/${tensors} tensors on Muon in ${int(ev.buckets)} buckets`
        + (text(ev.name) === 'muon' ? ` at lr-scale ${optNum(ev, 'lrScale') ?? '?'}` : ''));
      // A Muon run that classified NOTHING onto Muon trains as AdamW and is
      // otherwise indistinguishable. Say so rather than let it pass silently.
      if (text(ev.name) === 'muon' && muon === 0) {
        log(job, 'warn',
          'Muon was selected but no parameter qualified — this run is training as AdamW. '
          + 'Rank is probably below the Muon minimum dimension.');
      }
      break;
    }
    case 'step':
      pushEvent(job, {
        type: 'metric', metric: 'step', ts: Date.now(),
        step: optNum(ev, 'step'), loss: optNum(ev, 'loss'), lr: optNum(ev, 'lr'),
        gradNorm: optNum(ev, 'gradNorm'), clipScale: optNum(ev, 'clipScale'),
        totalSteps: optNum(ev, 'totalSteps'), stepMs: optNum(ev, 'stepMs'),
      });
      break;
    case 'milestone':
      pushEvent(job, {
        type: 'metric', metric: 'milestone', ts: Date.now(),
        step: optNum(ev, 'step'), loss: optNum(ev, 'loss'), path: text(ev.path) || undefined,
      });
      log(job, 'info', `Checkpoint at step ${int(ev.step)} (loss ${(optNum(ev, 'loss') ?? 0).toFixed(4)})`);
      break;
    case 'progress': {
      job.done = int(ev.completed);
      const total = int(ev.total);
      if (total > 0) job.total = total;
      const phase = text(ev.phase);
      if (phase) job.phase = phase;
      if (ev.failed !== undefined) job.failed = int(ev.failed);
      emitProgress(job);
      break;
    }
    case 'export':
      log(job, 'info', `Exported ${int(ev.tensors)} tensors → ${text(ev.path)}`);
      break;
    case 'fatal':
      st.fatalMessage = text(ev.message) || 'ace-train reported a fatal error';
      log(job, 'error', st.fatalMessage);
      break;
    case 'done':
      st.doneSeen = true;
      break;
    default:
      break;
  }
}

/** Shared spawn + relay + engine restore. `kind` only shapes the log lines. */
async function runMm3AceTrain(
  job: TrainingJob,
  kind: 'mm3-codes' | 'mm3-lm-train',
  args: string[],
  timeoutMs: number,
  verifyOutput: () => string | null,
): Promise<void> {
  const exe = aceTrainExe();
  if (!exe) {
    finishJob(job, 'failed', 'ace-train is not in this build — rebuild the engine');
    return;
  }

  // `enqueue()` does NOT mark a job running — every runner does it itself
  // (trainLmRunner:301, preprocessRunner:171). Without this the UI shows
  // "Queued…" for the whole run and, because startedAt stays unset, elapsed
  // time and every ETA derived from it never start.
  job.status = 'running';
  job.startedAt = Date.now();
  job.phase = 'engine-stop';
  emitJob(job);
  emitProgress(job);
  log(job, 'info', 'Stopping the engine to free VRAM…');
  // Not gated on a live child: a crashed engine leaves a respawn scheduled, and
  // stopAceServer is what cancels it. An engine spawning into the middle of an
  // MM3 training run would compete for a card that is already at 97 %.
  const engineExited = await stopAceServer(`Paused for ${kind}`);

  try {
    if (!engineExited) {
      throw new Error('The engine did not shut down — MM3 training needs its VRAM. Restart the app and try again.');
    }
    if (isCancelled(job)) return;

    job.phase = 'loading-models';
    emitProgress(job);
    pushLog(`[Training] ${kind} job ${job.id}: ${exe} ${args.slice(0, 2).join(' ')}`);

    const child = spawn(exe, args, { windowsHide: true });
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

    const st: RelayState = { fatalMessage: '', doneSeen: false };
    const rl = readline.createInterface({ input: child.stdout! });
    rl.on('line', (line) => {
      try {
        const ev = JSON.parse(line) as Record<string, unknown>;
        if (ev && typeof ev === 'object') relay(job, ev, st);
      } catch { /* the human log also goes to stdout on some paths — ignore */ }
    });

    let timedOut = false;
    const killer = setTimeout(() => {
      timedOut = true;
      log(job, 'error', `${kind} exceeded its ${Math.round(timeoutMs / 60000)} min budget — stopping.`);
      killJobChild(job);
    }, timeoutMs);

    const code: number | null = await new Promise<number | null>((resolve, reject) => {
      child.on('error', err => reject(new Error(`Failed to launch ace-train: ${err.message}`)));
      child.on('close', (c, signal) => resolve(signal ? null : c));
    }).finally(() => {
      clearTimeout(killer);
      try { rl.close(); } catch { /* already closed */ }
      job.child = undefined;
    });

    if (isCancelled(job)) return;
    if (timedOut) {
      // Before the exit-code branch: a killed child yields null, which would
      // otherwise surface as "exited with code null". Checkpoints already
      // written survive this.
      throw new Error(`${kind} timed out after ${Math.round(timeoutMs / 60000)} min and was stopped`
        + (stderrTail.length ? `: ${stderrTail.slice(-3).join(' | ')}` : ''));
    }
    if (code !== 0) {
      throw new Error(st.fatalMessage
        || `ace-train exited with code ${code === null ? 'null (killed)' : code}: ${stderrTail.slice(-5).join(' | ')}`);
    }
    if (st.fatalMessage) throw new Error(st.fatalMessage);

    const problem = verifyOutput();
    if (problem) throw new Error(problem);
  } finally {
    // ALWAYS restore the engine — success, failure, cancel, timeout.
    job.phase = 'engine-restart';
    emitProgress(job);
    log(job, 'info', 'Restarting the engine…');
    pushLog(`[Training] ${kind} job ${job.id}: restarting the engine…`);
    const back = await restartAceServer();
    if (!back) {
      log(job, 'warn', 'Engine did not answer /health within 90 s — restart the app if generation fails');
    }
  }
}

// ── mm3-codes ───────────────────────────────────────────────────────────────

export async function runMm3CodesJob(job: TrainingJob): Promise<void> {
  const opts = job.opts as (Mm3CodesArgs & { datasetSlug: string }) | undefined;
  if (!opts?.datasetJson || !opts.outDir) {
    finishJob(job, 'failed', 'mm3-codes job is missing its dataset or output path');
    return;
  }
  const missing = missingMm3TrainModels('codes');
  if (missing.length) {
    finishJob(job, 'failed', `MiniMax-Music3 training models are missing: ${missing.join(', ')}`);
    return;
  }

  const args = buildMm3CodesArgs(opts);
  // Encoding is DAV + a 169M encoder per track: minutes, not hours. The floor
  // covers model load on a cold cache.
  const ds = getDataset(job.datasetId);
  const songs = Math.max(1, ds?.sampleCount ?? 1);
  const timeoutMs = Math.max(30 * 60 * 1000, songs * 60 * 1000);

  try {
    await runMm3AceTrain(job, 'mm3-codes', args, timeoutMs, () => {
      const dir = path.join(opts.outDir, 'codes');
      const n = fs.existsSync(dir) ? fs.readdirSync(dir).filter(f => f.endsWith('.codes')).length : 0;
      return n > 0 ? null : 'mm3-codes finished but wrote no .codes files';
    });
    if (!isCancelled(job)) finishJob(job, 'done');
  } catch (err: any) {
    if (!isCancelled(job)) finishJob(job, 'failed', err?.message || String(err));
  }
}

// ── mm3-lm-train ────────────────────────────────────────────────────────────

export async function runMm3TrainLmJob(job: TrainingJob): Promise<void> {
  const opts = job.opts as ResolvedMm3TrainLmOptions | undefined;
  if (!opts?.manifest || !opts.codesDir || !opts.outDir) {
    finishJob(job, 'failed', 'mm3-lm-train job is missing its manifest, codes or output path');
    return;
  }
  const missing = missingMm3TrainModels('train');
  if (missing.length) {
    finishJob(job, 'failed',
      `MiniMax-Music3 training models are missing: ${missing.join(', ')}. `
      + 'Training needs the F16 files specifically — a quantized base cannot be trained.');
    return;
  }

  const args = buildMm3TrainLmArgs(opts);
  // 3.9 s/step measured at the production recipe; budget 20 s/step so a slower
  // card or a longer crop does not trip the killer, with a 6 h floor.
  const timeoutMs = Math.max(6 * 60 * 60 * 1000, opts.steps * 20 * 1000);

  try {
    await runMm3AceTrain(job, 'mm3-lm-train', args, timeoutMs, () => {
      // saveEvery 0 means "never checkpoint" — a legitimate ask for a probe
      // run, and demanding one anyway failed a run that did exactly what it
      // was told. Only a run that was SUPPOSED to write one is judged on it.
      if (opts.saveEvery <= 0) return null;
      // A checkpoint is a directory holding the PEFT pair. Finding none means
      // the run produced nothing usable however cleanly it exited.
      const found = fs.existsSync(opts.outDir)
        && fs.readdirSync(opts.outDir).some(d =>
          fs.existsSync(path.join(opts.outDir, d, 'adapter_model.safetensors')));
      return found ? null : 'mm3-lm-train finished but wrote no checkpoint';
    });
    if (!isCancelled(job)) {
      log(job, 'info',
        'Checkpoints are in the MM3 adapter folder — they appear in the adapter picker with no install step.');
      finishJob(job, 'done');
    }
  } catch (err: any) {
    if (!isCancelled(job)) finishJob(job, 'failed', err?.message || String(err));
  }
}
