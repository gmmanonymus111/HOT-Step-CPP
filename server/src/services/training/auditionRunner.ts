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
import {
  renderSideThroughDit, resolveAuditionInputs, runOneSide, type SideOutcome,
} from './auditionService.js';
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
  // `flipped` arms BOTH cleanups: the targeted mid-job LM evict (before the DiT
  // renders) and the end-of-job policy restore. Restore is the real fix
  // (2026-07-29): the latch used to survive the job, leaving the engine in
  // EVICT_NEVER — every module the audition (and every later generation!)
  // touched stayed resident for the rest of the session, which is why a
  // rendered audition ate far more VRAM than a normal STRICT-policy generation
  // and never gave it back.
  let flipped = false;
  const evictLmIfFlipped = async (): Promise<void> => {
    if (!flipped) return;
    const evicted = await aceClient.unloadLabel('LM').catch(() => false);
    log(job, 'info', evicted
      ? 'Evicted the LM from VRAM'
      : 'Could not evict the LM — it stays resident until the policy restore');
  };
  const restorePolicyIfFlipped = async (): Promise<void> => {
    if (!flipped) return;
    flipped = false;
    const restored = await aceClient.restoreEvictPolicy().catch(() => false);
    log(job, restored ? 'info' : 'warn', restored
      ? 'Freed every audition model and restored the normal eviction policy'
      : 'Could not restore the eviction policy — models may stay resident until the engine restarts');
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
    const renderQueue: SideOutcome[] = [];
    const previewId = randomUUID();

    // Armed BEFORE the first post, because the first post is what flips the flag.
    flipped = resolved.coResident && !config.aceServer.keepLoaded;

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
      renderQueue.push(outcome);

      if (outcome.result.ok) {
        log(job, 'info',
          `${side.label}: ${outcome.result.codesCount} codes (${outcome.result.codesSha1.slice(0, 8)}) ` +
          `— LM ${outcome.result.lmMs} ms, decode ${outcome.result.decodeMs} ms`);
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

    // ── Opt-in DiT renders — AFTER the LM sides, AFTER the LM is evicted ──
    // keep_loaded pins the planner (a 4B is ~8 GB BF16) for the whole job;
    // rendering inside a side loaded xl-turbo + VAE on top of it and spilled
    // VRAM into shared memory (Rob, 2026-07-29). Evicting first gives the
    // render a normal generation's footprint, and both LM sides still shared
    // one resident load — the whole point of co-residency.
    if (resolved.renderDit) {
      await evictLmIfFlipped();
      job.phase = 'audition-render';
      emitProgress(job);
      if (resolved.renderDitAdapterPath) {
        // The pinning is worth a log line: with adapter renders on, the user's
        // DiT pick / xl-turbo default is overridden by the adapter's training
        // base (resolveAuditionInputs) — otherwise "why did it render on
        // xl-thirds?" has no answer in the stream.
        log(job, 'info',
          `DiT-adapter renders on ${resolved.renderDitModel || '(engine default)'} — ` +
          `adapter ${resolved.renderDitAdapterPath}`);
      }
      // Bare renders first for BOTH sides, then adapter renders for both: the
      // engine caches the runtime delta between consecutive requests, so
      // grouping by adapter loads it once instead of twice.
      for (const withAdapter of resolved.renderDitAdapterPath ? [false, true] : [false]) {
        for (const outcome of renderQueue) {
          if (!outcome.result.ok || !outcome.audioCodes) continue;
          if (isCancelled(job)) { finishJob(job, 'cancelled'); return; }
          const what = withAdapter ? 'DiT-adapter render' : 'DiT render';
          const t0 = Date.now();
          try {
            const rendered = await renderSideThroughDit(resolved, outcome.result, outcome.audioCodes, {
              isCancelled: () => isCancelled(job),
            }, withAdapter ? resolved.renderDitAdapterPath : '');
            const ms = Date.now() - t0;
            audio.push(rendered);
            if (withAdapter) {
              outcome.result.renderAdapterMs = ms;
              outcome.result.renderAdapterUrl =
                `/api/training/previews/${previewId}/${outcome.result.slot}-render-adapter`;
            } else {
              outcome.result.renderMs = ms;
              outcome.result.renderUrl = `/api/training/previews/${previewId}/${outcome.result.slot}-render`;
            }
            log(job, 'info', `${outcome.result.label}: ${what} ${ms} ms`);
          } catch (err: any) {
            if (isCancelled(job)) { finishJob(job, 'cancelled'); return; }
            const message = err instanceof Error ? err.message : String(err);
            if (withAdapter) outcome.result.renderAdapterError = message;
            else outcome.result.renderError = message;
            log(job, 'warn',
              `${outcome.result.label}: ${what} failed — ${message} (the codes sketch is still playable)`);
          }
        }
      }
    }

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
      // Mirrored-generation receipts — what "Send to Custom-Gen" replays.
      captionInput: resolved.captionUntagged,
      bpm: resolved.bpm,
      keyscale: resolved.keyscale,
      timesignature: resolved.timesignature,
      lmTemperature: resolved.temperature,
      lmTopP: resolved.topP,
      lmCfgScale: resolved.cfgScale,
      lmRepPenalty: resolved.repPenalty,
      ...(resolved.renderDit
        ? { renderDitModel: resolved.renderDitModel, renderSteps: resolved.renderSteps }
        : {}),
      ...(resolved.renderDitAdapterPath
        ? { renderDitAdapter: resolved.renderDitAdapterPath }
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
      for (const r of preview.sides) { r.audioUrl = ''; r.renderUrl = ''; r.renderAdapterUrl = ''; }
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

    // C15: un-flip what we flipped, best effort — full eviction pass + STRICT
    // restored. Called here (not only in the finally) so the log line still
    // reaches the SSE stream while it is open. The next audition pays a cold
    // load again; that is the correct price for giving the VRAM back.
    await restorePolicyIfFlipped();

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
    await restorePolicyIfFlipped().catch(() => { /* never fail a job on cleanup */ });
  }
}
