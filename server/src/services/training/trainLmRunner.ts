// training/trainLmRunner.ts — runs one `ace-train train-lm` job
//
// Structural clone of preprocessRunner.ts: engine stop → spawn → readline over
// stdout JSONL → relay into the SSE union → engine restart in a `finally`.
// Success, failure, cancel and timeout all restore the engine.
//
// The difference from preprocess is scale. LM training owns the whole card for
// the length of the run (L8: ace-train auto-fits its sequence length to the free
// VRAM it finds AFTER the engine is gone), so `stopEngine` defaults to true and
// the timeout budget is hours, not minutes. The adapter is exported every epoch
// (§3.5.4), so a cancelled or timed-out run still leaves a usable directory.
//
// Spec: docs/plans/2026-07-27-lm-trainer-implementation.md §2.9, §4.4

import fs from 'fs';
import path from 'path';
import readline from 'readline';
import { spawn } from 'child_process';
import { config } from '../../config.js';
import { pushLog } from '../../routes/logs.js';
import { restartAceServer, stopAceServer } from '../aceEngineProcess.js';
import { aceTrainExe, buildTrainLmArgs, type ResolvedTrainLmOptions } from './aceTrain.js';
import { getDataset } from './datasetsRepo.js';
import {
  emitJob, emitProgress, finishJob, isCancelled, killJobChild, pushEvent, type TrainingJob,
} from './labelingQueue.js';

/** 6 h floor, 1 min per song-epoch (§4.4 item 6). LM training is the longest
 *  job the studio runs — a 17-song, 16-epoch 1.7B run is hours of wall time. */
function trainLmTimeoutMs(sampleCount: number, epochs: number): number {
  return Math.max(6 * 60 * 60 * 1000, Math.max(0, sampleCount) * Math.max(1, epochs) * 60 * 1000);
}

function log(job: TrainingJob, level: 'info' | 'warn' | 'error', message: string): void {
  pushEvent(job, { type: 'log', level, message, ts: Date.now() });
}

function int(value: unknown): number {
  const n = Number(value);
  return Number.isFinite(n) ? Math.trunc(n) : 0;
}

function flt(value: unknown): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function text(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/** Only forward numbers the engine actually sent — a metric field defaulted to
 *  0 is indistinguishable from a real 0 in the loss curve. */
function optNum(ev: Record<string, unknown>, key: string): number | undefined {
  const v = ev[key];
  if (v === undefined || v === null) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

/** Cross-event state the relay accumulates for the close handler. */
interface RelayState {
  /** Message from a `fatal` line — the reason the user must see instead of
   *  "ace-train exited with code 1". */
  fatalMessage: string;
  /** A `done` line arrived, i.e. the run reached its own clean end. */
  doneSeen: boolean;
}

function mb(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(1);
}

/**
 * ace-train train-lm JSONL → the TrainingStreamEvent union (§2.9).
 *
 * Defensive by contract: every field access guarded, unknown `type` ignored,
 * every `metric` event stamped with `ts: Date.now()`. `step` deliberately emits
 * no log line — one per optimizer step would flood the log pane.
 */
function relay(job: TrainingJob, ev: Record<string, unknown>, state: RelayState): void {
  switch (ev.type) {
    case 'start': {
      job.done = 0;
      job.failed = 0;
      job.currentSampleId = null;
      job.engineQueueDepth = 0;
      log(job, 'info', `Training ${text(ev.lmSize) || 'LM'} LoRA → ${text(ev.out)}`);
      break;
    }
    case 'stage': {
      if (ev.state !== 'begin') break;
      const stage = text(ev.stage);
      if (stage) job.phase = stage;
      const total = int(ev.total);
      if (total > 0) job.total = total;
      job.done = 0;
      emitProgress(job);
      log(job, 'info', `Stage: ${stage || 'unknown'}`);
      break;
    }
    case 'model': {
      job.phase = 'loading-models';
      log(job, 'info', `Loaded ${text(ev.stage)} (${int(ev.ms)} ms)`);
      break;
    }
    case 'vram': {
      pushEvent(job, {
        type: 'metric',
        metric: 'vram',
        ts: Date.now(),
        freeMb: optNum(ev, 'freeMb'),
        totalMb: optNum(ev, 'totalMb'),
        estMb: optNum(ev, 'estMb'),
        maxLen: optNum(ev, 'maxLen'),
      });
      log(job, 'info', `VRAM ${int(ev.freeMb)} MB free — max sequence ${int(ev.maxLen)} tokens`);
      break;
    }
    case 'data': {
      pushEvent(job, {
        type: 'metric',
        metric: 'data',
        ts: Date.now(),
        samples: optNum(ev, 'samples'),
        skippedLong: optNum(ev, 'skippedLong'),
        stepsPerEpoch: optNum(ev, 'stepsPerEpoch'),
        totalSteps: optNum(ev, 'totalSteps'),
        loraParams: optNum(ev, 'loraParams'),
        maxLen: optNum(ev, 'maxLen'),
      });
      log(job, 'info',
        `${int(ev.samples)} songs, ${int(ev.stepsPerEpoch)} steps/epoch, ${int(ev.loraParams)} LoRA parameters`);
      if (int(ev.skippedLong) > 0) {
        log(job, 'warn',
          `${int(ev.skippedLong)} song(s) skipped — longer than the ${int(ev.maxLenCap) || int(ev.maxLen)}-token limit`);
      }
      break;
    }
    case 'extract_song': {
      log(job, 'info', `${text(ev.file)} → ${int(ev.nCodes)} codes (${int(ev.ms)} ms)`);
      break;
    }
    case 'extract_skip': {
      log(job, 'info', `${text(ev.file)} — cached`);
      break;
    }
    case 'extract_fail': {
      log(job, 'error', `${text(ev.file)} — ${text(ev.error)}`);
      break;
    }
    case 'step': {
      // Metric only — no log line, a per-optimizer-step log would flood the pane.
      pushEvent(job, {
        type: 'metric',
        metric: 'step',
        ts: Date.now(),
        epoch: optNum(ev, 'epoch'),
        step: optNum(ev, 'step'),
        totalSteps: optNum(ev, 'totalSteps'),
        loss: optNum(ev, 'loss'),
        lr: optNum(ev, 'lr'),
        gradNorm: optNum(ev, 'gradNorm'),
        clipScale: optNum(ev, 'clipScale'),
        ms: optNum(ev, 'ms'),
        vramMb: optNum(ev, 'vramMb'),
      });
      break;
    }
    case 'epoch': {
      pushEvent(job, {
        type: 'metric',
        metric: 'epoch',
        ts: Date.now(),
        epoch: optNum(ev, 'epoch'),
        epochs: optNum(ev, 'epochs'),
        loss: optNum(ev, 'loss'),
        lr: optNum(ev, 'lr'),
        gradNorm: optNum(ev, 'gradNorm'),
        etaMs: optNum(ev, 'etaMs'),
        ms: optNum(ev, 'ms'),
        best: ev.best === true ? true : undefined,
      });
      log(job, 'info',
        `Epoch ${int(ev.epoch)}/${int(ev.epochs)} loss ${flt(ev.loss).toFixed(6)} lr ${flt(ev.lr).toExponential(3)}`);
      break;
    }
    case 'milestone': {
      pushEvent(job, {
        type: 'metric',
        metric: 'milestone',
        ts: Date.now(),
        loss: optNum(ev, 'loss'),
        epoch: optNum(ev, 'epoch'),
        path: text(ev.path) || undefined,
      });
      log(job, 'info', `Milestone: loss ${flt(ev.loss).toFixed(1)} at epoch ${int(ev.epoch)} → ${text(ev.path)}`);
      break;
    }
    case 'progress': {
      job.done = int(ev.completed);
      const total = int(ev.total);
      if (total > 0) job.total = total;
      const phase = text(ev.phase);
      if (phase) job.phase = phase;
      // The extract progress line carries `failed` (§2.2: completed = processed
      // + skipped + failed) and preprocessRunner relays it. Without this a run
      // where songs hit extract_fail finishes reporting `failed: 0`, so both the
      // JobProgress counter and the persisted _meta.json claim a clean run.
      if (ev.failed !== undefined) job.failed = int(ev.failed);
      emitProgress(job);
      break;
    }
    case 'target_stop': {
      log(job, 'info',
        `Target loss reached (${flt(ev.loss).toFixed(6)} ≤ ${flt(ev.targetLoss)}) at epoch ${int(ev.epoch)} — stopping`);
      break;
    }
    case 'export': {
      log(job, 'info',
        `Wrote ${int(ev.tensors)} tensors (${mb(int(ev.bytes))} MB) → ${text(ev.path)}`);
      break;
    }
    case 'log': {
      const level = ev.level === 'warn' || ev.level === 'error' ? ev.level : 'info';
      log(job, level, text(ev.message));
      break;
    }
    case 'fatal': {
      // Recorded, not thrown from here — the close handler turns it into the
      // job's failure message so the user sees "needs ~54 GB", not "exit 1".
      const reason = text(ev.reason) || 'unknown';
      state.fatalMessage = text(ev.message) || `ace-train failed (${reason})`;
      job.error = state.fatalMessage;
      log(job, 'error', state.fatalMessage);
      break;
    }
    case 'done': {
      // Recorded only — finishJob() emits the terminal status after the engine
      // restart completes, so the UI never sees "done" with the engine still down.
      state.doneSeen = true;
      log(job, 'info',
        `ace-train finished — ${int(ev.epochsRun)} epoch(s), final loss ${flt(ev.finalLoss).toFixed(6)} (${int(ev.ms)} ms)`);
      break;
    }
    default:
      break;   // unknown type — ignore, never fatal
  }
}

/** How many songs the run will see — used only to size the timeout budget. */
function countTensorFiles(dir: string): number {
  try {
    let n = 0;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.isFile() && entry.name.toLowerCase().endsWith('.safetensors')) n++;
    }
    return n;
  } catch {
    return 0;
  }
}

export async function runTrainLmJob(job: TrainingJob): Promise<void> {
  if (isCancelled(job)) return;

  const opts = job.opts as ResolvedTrainLmOptions;

  try {
    const ds = getDataset(job.datasetId);
    if (!ds) { finishJob(job, 'failed', 'Dataset not found'); return; }

    job.status = 'running';
    job.startedAt = Date.now();
    job.phase = opts.stages[0] ?? 'train';
    emitJob(job);

    const exe = aceTrainExe();
    if (!exe) throw new Error('ace-train was not found next to ace-server — rebuild the engine');

    // ── 1. output dir ────────────────────────────────────────────────────
    // Created up front so a permission problem fails before the engine is
    // stopped rather than after.
    fs.mkdirSync(path.dirname(opts.adapterDir), { recursive: true });
    fs.mkdirSync(opts.adapterDir, { recursive: true });

    const songCount = countTensorFiles(opts.tensorsDir);
    if (songCount > 0) job.total = songCount;
    emitProgress(job);

    // §4.2 quantized-base guard: an unusual BF16 filename is allowed through,
    // but a genuinely quantized base cannot be trained against (the engine
    // refuses at mirror time) so the warning has to be visible before the wait.
    if (opts.lmModel && !/bf16/i.test(opts.lmModel)) {
      log(job, 'warn',
        `${opts.lmModel} does not look like a BF16 base — quantized LM bases cannot be trained against`);
    }

    const args = buildTrainLmArgs({ opts, modelsDir: config.aceServer.models });

    // ── 2. engine handoff (L19) ──────────────────────────────────────────
    let stopped = false;
    let engineExited = true;
    // NOT gated on getAceProcess() being non-null: a crashed engine leaves a
    // respawn scheduled 3 s out, and stopAceServer() is what cancels it. Skipping
    // the call would let that timer spawn an engine into the middle of the job
    // and compete for VRAM — which for training is worse than for preprocess,
    // because ace-train sized its sequence length against the free VRAM it
    // measured while the card was empty (L8). With no live child the stop is a
    // cheap no-op.
    if (opts.stopEngine) {
      job.phase = 'engine-stop';
      emitProgress(job);
      log(job, 'info', 'Stopping the engine to free VRAM…');
      engineExited = await stopAceServer('Paused for LM training');
      stopped = true;
    }

    try {
      // Checked INSIDE the try so the `finally` still restores the engine.
      if (!engineExited) {
        throw new Error('The engine did not shut down — LM training needs its VRAM. Restart the app and try again.');
      }
      if (isCancelled(job)) return;

      // ── 3. spawn + relay ───────────────────────────────────────────────
      job.phase = 'loading-models';
      emitProgress(job);
      pushLog(`[Training] train-lm job ${job.id}: ace-train ${opts.lmSize} → ${opts.adapterDir}`);

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

      const state: RelayState = { fatalMessage: '', doneSeen: false };
      const rl = readline.createInterface({ input: child.stdout! });
      rl.on('line', (line) => {
        try {
          const ev = JSON.parse(line) as Record<string, unknown>;
          if (ev && typeof ev === 'object') relay(job, ev, state);
        } catch { /* non-JSON noise */ }
      });

      const timeoutMs = trainLmTimeoutMs(songCount, opts.epochs);
      let timedOut = false;
      const killer = setTimeout(() => {
        timedOut = true;
        console.error(`[Training] train-lm job ${job.id}: timed out — killing ace-train`);
        log(job, 'error', `Training exceeded its ${Math.round(timeoutMs / 60000)} min budget — stopping.`);
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
        // Checked BEFORE the exit-code branch: a killed child yields code null,
        // which would otherwise surface as "ace-train exited with code null".
        // Every finished epoch is already on disk (§3.5.4), so the partial
        // adapter survives this.
        throw new Error(
          `LM training timed out after ${Math.round(timeoutMs / 60000)} min and was stopped` +
          (stderrTail.length ? `: ${stderrTail.slice(-3).join(' | ')}` : ''));
      }
      if (code !== 0) {
        throw new Error(state.fatalMessage
          || `ace-train exited with code ${code === null ? 'null (killed)' : code}: ${stderrTail.slice(-5).join(' | ')}`);
      }
      // A structured `fatal` with a zero exit code should not happen, but if it
      // does the reason is the truth and the exit code is the bug.
      if (state.fatalMessage) throw new Error(state.fatalMessage);

      // ── 4. post-check ──────────────────────────────────────────────────
      if (opts.stages.includes('export')) {
        const model = path.join(opts.adapterDir, 'adapter_model.safetensors');
        const cfg = path.join(opts.adapterDir, 'adapter_config.json');
        if (!fs.existsSync(model) || !fs.existsSync(cfg)) {
          throw new Error('ace-train finished but wrote no adapter');
        }
      }
    } finally {
      // ── 5. ALWAYS restore the engine — success, failure, cancel, timeout ─
      if (stopped) {
        job.phase = 'engine-restart';
        emitProgress(job);
        log(job, 'info', 'Restarting the engine…');
        // On a cancel the SSE stream is already closed (cancelJob emits the
        // terminal status and endStream()s synchronously), so everything from
        // here on is written to a job nobody is listening to. Mirror it into the
        // app log or the up-to-90 s restart happens completely invisibly.
        pushLog(`[Training] train-lm job ${job.id}: restarting the engine…`);
        const back = await restartAceServer();
        if (!back) {
          log(job, 'warn',
            'Engine did not answer /health within 90 s — restart the app if generation fails');
          pushLog(`[Training] train-lm job ${job.id}: engine did not answer /health within 90 s`);
        } else {
          pushLog(`[Training] train-lm job ${job.id}: engine is back up`);
        }
      }
    }

    pushLog(`[Training] train-lm job ${job.id} finished — adapter at ${opts.adapterDir}`);
    finishJob(job, 'done');
  } catch (err: any) {
    if (isCancelled(job)) return;
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[Training] train-lm job ${job.id} FAILED — ${message}`);
    finishJob(job, 'failed', message);
  }
}
