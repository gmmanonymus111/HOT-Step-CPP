// training/auditionRunner.ts — runs one two-sided codes-audition job
//
// Lifecycle clone of trainLmRunner.ts WITH THE ENGINE STOP REMOVED. This is the
// one Training-Studio job that requires ace-server to be RUNNING: the only
// adapter-capable LM host in the tree is ace-server's /lm (ace-lm.cpp never
// reads req.lm_adapter), and /codes-decode lives in the same process.
//
// No new SSE event type. The shipped TrainingStreamEvent union already carries
// everything: {type:'job'} (phase + counters), 'progress', 'log', 'status'.
//
// Phase strings (FROZEN): audition-lm-base, audition-lm-adapter,
// audition-decode, audition-writing. Additive (2026-07-29): audition-render —
// the opt-in DiT pass between decode and writing.
//
// Spec: docs/plans/2026-07-28-codes-preview.md §5.4, C6, C15

import { randomUUID } from 'crypto';
import { config } from '../../config.js';
import { pushLog } from '../../routes/logs.js';
import { aceClient } from '../aceClient.js';
import { getDataset } from './datasetsRepo.js';
import { writePreview, type PreviewAudio } from './auditionStore.js';
import { resolveAuditionInputs, runOneSide } from './auditionService.js';
import {
  emitJob, emitProgress, finishJob, isCancelled, pushEvent, type TrainingJob,
} from './labelingQueue.js';
import type { AuditionOptions, AuditionPreview, AuditionSideResult } from './types.js';

function log(job: TrainingJob, level: 'info' | 'warn' | 'error', message: string): void {
  pushEvent(job, { type: 'log', level, message, ts: Date.now() });
}

export async function runAuditionJob(job: TrainingJob): Promise<void> {
  if (isCancelled(job)) return;

  // ── C15 eviction, hoisted out of the straight-line body ──────────────────
  // ?keep_loaded=1 flips the engine to EVICT_NEVER the moment the FIRST side
  // posts, permanently, for the ace-server lifetime. The eviction that mitigates
  // that must therefore run on EVERY exit — cancel mid-side, cancel after the
  // last side, and the outer catch — not only on the happy path. Leaving a 4B
  // planner (~8 GB BF16) pinned is especially bad here because the very next
  // thing the Training Studio does is size a train-lm run against free VRAM.
  //
  // Idempotent: the happy path calls it before finishJob so its log still
  // reaches the open SSE stream; the finally is the backstop for every other
  // exit.
  let evictLm = false;
  const evictLmIfFlipped = async (): Promise<void> => {
    if (!evictLm) return;
    evictLm = false;
    const evicted = await aceClient.unloadLabel('LM').catch(() => false);
    log(job, 'info', evicted
      ? 'Evicted the LM from VRAM (the engine stays in keep-loaded mode until a restart)'
      : 'Could not evict the LM — it stays resident until the engine restarts');
  };

  try {
    const ds = getDataset(job.datasetId);
    if (!ds) { finishJob(job, 'failed', 'Dataset not found'); return; }

    const opts = (job.opts || {}) as AuditionOptions;
    const resolved = resolveAuditionInputs(ds, opts);

    if (resolved.sides.length === 0) {
      finishJob(job, 'failed', 'No sides to audition');
      return;
    }

    job.status = 'running';
    job.startedAt = Date.now();
    job.total = resolved.sides.length;
    job.done = 0;
    job.failed = 0;
    job.phase = 'audition-lm-base';
    emitJob(job);

    log(job, 'info',
      `Audition seed ${resolved.seed} — ${resolved.sides.length} side(s), ${resolved.durationSec}s` +
      (resolved.ditModel ? `, detok from ${resolved.ditModel}` : ''));

    const results: AuditionSideResult[] = [];
    const audio: PreviewAudio[] = [];
    const previewId = randomUUID();

    // Armed BEFORE the first post, because the first post is what flips the flag.
    evictLm = resolved.coResident && !config.aceServer.keepLoaded;

    for (const side of resolved.sides) {
      if (isCancelled(job)) { finishJob(job, 'cancelled'); return; }

      job.phase = side.slot === 'base' ? 'audition-lm-base' : 'audition-lm-adapter';
      emitProgress(job);

      // Never throws — a failed side lands in .error and the other side runs.
      const outcome = await runOneSide(resolved, side, {
        onPhase: (phase) => { job.phase = phase; emitProgress(job); },
        isCancelled: () => isCancelled(job),
      });

      results.push(outcome.result);
      if (outcome.audio) {
        audio.push(outcome.audio);
        outcome.result.audioUrl = `/api/training/previews/${previewId}/${side.slot}`;
      }
      if (outcome.renderAudio) {
        audio.push(outcome.renderAudio);
        outcome.result.renderUrl = `/api/training/previews/${previewId}/${side.slot}-render`;
      }

      if (outcome.result.ok) {
        log(job, 'info',
          `${side.label}: ${outcome.result.codesCount} codes (${outcome.result.codesSha1.slice(0, 8)}) ` +
          `— LM ${outcome.result.lmMs} ms, decode ${outcome.result.decodeMs} ms` +
          (outcome.result.renderMs ? `, DiT render ${outcome.result.renderMs} ms` : ''));
        if (outcome.result.renderError) {
          log(job, 'warn', `${side.label}: DiT render failed — ${outcome.result.renderError} (the codes sketch is still playable)`);
        }
      } else {
        job.failed++;
        log(job, 'error', `${side.label}: ${outcome.result.error}`);
      }

      job.done++;
      emitProgress(job);
    }

    // A cancel arriving mid-side surfaces as a failed side (runOneSide never
    // throws), so the loop's top-of-iteration check cannot catch it on the last
    // side. Without this a cancelled single-side audition still writes a preview
    // dir full of a half-finished run.
    if (isCancelled(job)) { finishJob(job, 'cancelled'); return; }

    // ── Write the preview ────────────────────────────────────────────────
    job.phase = 'audition-writing';
    emitProgress(job);

    const preview: AuditionPreview = {
      previewId,
      datasetId: ds.id,
      kind: resolved.kind,
      createdAt: new Date().toISOString(),
      seed: resolved.seed,
      caption: resolved.caption,
      lyrics: resolved.lyrics,
      durationSec: resolved.durationSec,
      lmModel: resolved.lmModel,
      ditModel: resolved.ditModel,
      vaeModel: resolved.vaeModel,
      sampleId: resolved.sampleId,
      variantKey: resolved.variantKey,
      sides: results,
      ...(resolved.renderDit
        ? { renderDitModel: resolved.renderDitModel, renderSteps: resolved.renderSteps }
        : {}),
    };

    // Written UNCONDITIONALLY, even when every side failed. The record is
    // designed to hold failures (ok:false / error / audioUrl:''), and the seed
    // is the whole point of C14: on a server-picked random seed, skipping the
    // write because no audio came back throws away the only copy of the seed the
    // moment the SSE stream closes, so the user cannot re-run the same A/B after
    // fixing whatever broke.
    if (!writePreview(preview, audio)) {
      log(job, 'warn', 'The preview could not be written to disk');
      for (const r of preview.sides) { r.audioUrl = ''; r.renderUrl = ''; }
    }

    // C14 receipt, surfaced in the log as well as the UI: two identical hashes
    // means the adapter had no effect, which is this feature's likeliest silent
    // failure.
    const okHashes = results.filter(r => r.ok).map(r => r.codesSha1);
    if (okHashes.length === 2 && okHashes[0] === okHashes[1]) {
      log(job, 'warn',
        'Both sides produced identical codes — the adapter had no effect. ' +
        'Check ace_engine.log for "[Server] LM adapter: …".');
    }

    // C15: un-flip what we flipped, best effort. FSQ-Detok and VAE-Dec stay
    // resident on purpose — they are small and they are what makes the next
    // audition instant. Called here (not only in the finally) so the log line
    // still reaches the SSE stream while it is open.
    await evictLmIfFlipped();

    const allFailed = results.length > 0 && results.every(r => !r.ok);
    pushLog(`[Training] audition job ${job.id} finished — preview ${previewId}` +
      (allFailed ? ' (every side failed)' : ''));
    finishJob(job, allFailed ? 'failed' : 'done',
      allFailed ? (results[0]?.error || 'Every audition side failed') : undefined);
  } catch (err: any) {
    if (isCancelled(job)) return;
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[Training] audition job ${job.id} FAILED — ${message}`);
    finishJob(job, 'failed', message);
  } finally {
    // Backstop for every non-happy exit: cancel at the top of a side loop
    // iteration, cancel after the last side, and the catch above. No-op when the
    // happy path already ran it.
    await evictLmIfFlipped().catch(() => { /* never fail a job on cleanup */ });
  }
}
