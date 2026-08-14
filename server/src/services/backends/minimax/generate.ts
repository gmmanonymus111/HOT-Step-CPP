// backends/minimax/generate.ts — the MiniMax-Music3 generation path
//
// Runs a dequeued generation job against the /mm3/* engine family. Called from
// routes/generate.ts's runGeneration() when the active backend is
// 'minimax-m3'; everything MM3-specific lives here so that diff stays four
// lines (plan §2: shared code must not grow backend branches).
//
// WHAT IS SHARED WITH THE ACE PATH (deliberately, not by accident):
//   • the serial job queue + retry wrapper (routes/generate.ts:enqueueGeneration)
//   • the in-memory job map, /status/:id, /cancel/:id and the SSE progress shape
//   • pollUntilDone's watchdog (injected — MM3 jobs ride the SAME /job endpoints)
//   • the audio dir + /audio/<uuid>.wav URL convention, and the songs INSERT
//   • cover art (a text→image service with no engine coupling)
//
// WHAT IS NOT REUSED (and why — see the post-processing note further down):
//   • the LM phase, LM cache, LM-echo rebuild — MM3's planner is internal to
//     the engine pipeline; there is no two-request seam to cache or rebuild.
//   • translateParams / AceRequest — MM3's wire contract is 8 fields.
//   • adapters, latents, source audio, task modes, streaming, no-adapter render.
//   • the post-processing chain (see AUDIT below).
//
// POST-PROCESSING AUDIT (v1 decision: SKIP the whole chain, store the raw WAV).
// MM3 renders 44.1 kHz stereo; the shared chain is 48 kHz-minded and
// ACE-model-coupled at several stages:
//   • StableStep / SA3 refine — hardcodes `outSr: 48000`
//     (services/generation/postProcessing.ts:386) and needs ACE's SA3 GGUFs.
//   • PP-VAE re-encode — round-trips audio through the ACE VAE (wrong model,
//     wrong rate).
//   • Spectral Lifter / SuperSep / mastering-with-reference — engine endpoints
//     built and tuned for the ACE 48 kHz output.
// A resample or per-stage rate audit is real work with audible consequences,
// so v1 does none of it: correctness over polish. The unprocessed WAV is what
// the model produced, which is also the honest baseline for judging the port.
// Auto-trim is skipped for the same reason it doesn't apply: MM3's duration is
// exact frames, there is no ACE-style duration buffer to trim back off.
// Whisper transcription and LRC are skipped for v1 (no engine-side alignment,
// and the stem-mode path is entangled with the PP chain above).

import fs from 'fs';
import path from 'path';
import { performance } from 'perf_hooks';
import { v4 as uuidv4 } from 'uuid';

import { config } from '../../../config.js';
import { getDb } from '../../../db/database.js';
import { aceClient } from '../../aceClient.js';
import { wavDurationSec } from '../../audioCrop.js';
import {
  startGenerationLog, logGeneration, logGenerationParams,
  finishGenerationLog, failGenerationLog,
} from '../../logger.js';
import {
  mm3Synth, mm3JobDetail, mm3TokenizeCheck, mm3Unload,
  type Mm3SynthRequest, type Mm3JobDetail,
} from './client.js';
import type { GenerationJob, StageTiming } from '../../../routes/generate.js';

/** Injected so this module never imports back into routes/generate.ts at
 *  runtime (type-only import above is erased). pollUntilDone owns the stall
 *  watchdog, the wall-clock timeout and cancel checks — MM3 jobs use the same
 *  engine /job endpoints, so it applies unchanged. */
export interface MinimaxGenerationDeps {
  pollUntilDone(
    aceJobId: string, job: GenerationJob, signal: AbortSignal, timeoutMinutes?: number,
  ): Promise<void>;
}

/** MM3's own duration ceiling (mirrors the capability manifest). */
const MM3_MAX_DURATION_SEC = 300;
const MM3_DEFAULT_DURATION_SEC = 60;
/** Progress ticker interval — /mm3/job never takes the engine's MM3 mutex. */
const DETAIL_POLL_MS = 1_500;

// ── Param mapping ────────────────────────────────────────────────────────────

export interface MinimaxParamMapping {
  req: Mm3SynthRequest;
  /** Non-fatal notes (clamps, ignored knobs) surfaced to the generation log. */
  notes: string[];
}

/**
 * UI params → /mm3/synth request.
 *
 * The caption field list is IDENTICAL to translateParams.ts:49 on purpose: the
 * UI has one "what to generate" text field and both backends must read the
 * same one, or switching backends would silently change which box matters.
 * Everything ACE-specific (adapters, solver/scheduler/guidance, LM knobs,
 * task modes, latents, bpm/keyscale/timesignature, negative prompt) is ignored
 * — MM3 has no wire slot for any of it.
 */
export function mapMinimaxParams(params: any): MinimaxParamMapping {
  const notes: string[] = [];

  const caption: string = params.prompt || params.songDescription || params.caption || params.style || '';

  // Instrumental: MM3's contract is blank lyrics → the engine substitutes its
  // own instrumental token. Do NOT send ACE's '[Instrumental]' sentinel; that
  // would be tokenized as literal lyric text.
  const lyrics: string = params.instrumental ? '' : (params.lyrics || '');

  let duration = Number(params.duration);
  if (!(duration > 0)) {
    duration = MM3_DEFAULT_DURATION_SEC;
    notes.push(`no duration supplied — defaulting to ${MM3_DEFAULT_DURATION_SEC}s`);
  }
  // The ACE duration buffer (autoTrimEnabled + durationBuffer) is deliberately
  // NOT added: nothing trims it back off on this path.
  if (duration > MM3_MAX_DURATION_SEC) {
    notes.push(`duration ${Math.round(duration)}s exceeds the MiniMax-Music3 limit — clamped to ${MM3_MAX_DURATION_SEC}s`);
    duration = MM3_MAX_DURATION_SEC;
  }

  // -1 tells the engine to draw one; it echoes the resolved value back so the
  // take stays reproducible.
  const seed: number = params.randomSeed
    ? -1
    : (typeof params.seed === 'number' && params.seed >= 0 ? params.seed : -1);

  const requestedBatch = Number(params.batchSize) || 1;
  if (requestedBatch > 1) {
    notes.push(`batchSize ${requestedBatch} requested — MiniMax-Music3 v1 generates 1 track per job (capability manifest: batch.max = 1)`);
  }

  return {
    req: {
      caption,
      lyrics,
      duration: Math.round(duration * 1000) / 1000,
      seed,
      // Surfaced as capability `extensions` (backends/minimax/index.ts) rather
      // than left checkpoint-fixed: the flow stage is the majority of wall time
      // on a full-length render and `steps` is a linear dial on it. Both are
      // omitted from the wire when unset so the engine keeps applying the
      // checkpoint defaults (30 / 1.7) — an out-of-range value is dropped here
      // rather than sent for the engine to reject mid-job.
      ...(Number.isFinite(Number(params.mm3Steps)) && Number(params.mm3Steps) >= 8 &&
          Number(params.mm3Steps) <= 60
            ? { steps: Math.round(Number(params.mm3Steps)) }
            : {}),
      ...(Number.isFinite(Number(params.mm3CfgFlow)) && Number(params.mm3CfgFlow) >= 1.0 &&
          Number(params.mm3CfgFlow) <= 5.0
            ? { cfg_flow: Number(params.mm3CfgFlow) }
            : {}),
      get_wav_bits: 16,
    },
    notes,
  };
}

// ── Progress ─────────────────────────────────────────────────────────────────

/** MM3 stage → user-facing text + progress %.
 *
 *  `elapsedSec` is appended ONLY for stages the engine cannot report progress
 *  inside (weights load, arbitration, stitch/encode). That keeps the shared
 *  stall watchdog fed through a legitimately long, silent phase — a 24 GB cold
 *  load can exceed its 120 s limit — while leaving the stepped stages (ar /
 *  cond / flow / vocode) genuinely stall-detectable. */
export function minimaxStageText(d: Mm3JobDetail, elapsedSec: number): { stage: string; progress: number } {
  const frac = (a: number, b: number) => (b > 0 ? Math.max(0, Math.min(1, a / b)) : 0);
  const silent = (text: string, progress: number) => ({ stage: `${text} (${elapsedSec}s)`, progress });

  switch (d.stage) {
    case 'queued':
      return silent('MiniMax-Music3: queued', 2);
    case 'arbitrating':
      return silent('MiniMax-Music3: making room in VRAM', 4);
    case 'warming':
      return silent('MiniMax-Music3: loading weights', 6);
    case 'warm':
      return silent('MiniMax-Music3: weights resident', 8);
    case 'ar':
      // The autoregressive planner — MM3's analogue of the ACE LM phase.
      return { stage: `MiniMax-Music3: planning (frame ${d.step}/${d.n_steps})`, progress: 10 + Math.round(frac(d.step, d.n_steps) * 25) };
    case 'cond':
      return { stage: `MiniMax-Music3: conditioning (window ${d.window + 1}/${d.n_windows})`, progress: 36 + Math.round(frac(d.window + 1, d.n_windows) * 4) };
    case 'flow':
      return { stage: `MiniMax-Music3: flow step ${d.step}/${d.n_steps}${d.n_windows > 1 ? ` (window ${d.window + 1}/${d.n_windows})` : ''}`, progress: 40 + Math.round(frac(d.step, d.n_steps) * 45) };
    case 'vocode':
      return { stage: `MiniMax-Music3: vocoding (window ${d.window + 1}/${d.n_windows})`, progress: 85 + Math.round(frac(d.window + 1, d.n_windows) * 6) };
    case 'stitch':
      return silent('MiniMax-Music3: stitching windows', 92);
    case 'encoding':
      return silent('MiniMax-Music3: encoding WAV', 93);
    case 'done':
      return { stage: 'MiniMax-Music3: fetching audio', progress: 94 };
    default:
      return silent(`MiniMax-Music3: ${d.stage}`, 5);
  }
}

// ── The run ──────────────────────────────────────────────────────────────────

export async function runMinimaxGeneration(job: GenerationJob, deps: MinimaxGenerationDeps): Promise<void> {
  const pipelineStart = performance.now();
  const timing: StageTiming[] = [];
  const timeoutMinutes: number | undefined = job.params.generationTimeoutMinutes;
  const log = (level: 'INFO' | 'DEBUG' | 'WARNING' | 'ERROR', msg: string) => logGeneration(job.id, level, msg);

  if (job.status === 'cancelled') return;

  const { req, notes } = mapMinimaxParams(job.params);

  const abortController = new AbortController();
  (job as any)._abort = abortController;

  startGenerationLog(job.id, 'mm3-text2music');
  logGenerationParams(job.id, req as unknown as Record<string, unknown>);
  for (const n of notes) log('WARNING', `[MM3] ${n}`);

  console.log(`[Generate] Job ${job.id} — backend=minimax-m3, duration=${req.duration}s, seed=${req.seed}, lyrics=${req.lyrics ? `${req.lyrics.length} chars` : '(instrumental)'}`);

  let detailTimer: NodeJS.Timeout | undefined;

  try {
    if (!req.caption.trim()) {
      throw new Error('MiniMax-Music3 needs a caption — the Style Description field is empty');
    }

    // ── Defensive arbitration is NOT needed here ──
    // The MM3 job evicts the ACE side itself, on the GPU worker thread, only
    // when the weights would not otherwise fit (mm3-job.h:183 mm3_arbitrate_vram).
    // Unloading ACE from Node would fight that logic and slow down the common
    // case where both fit. The reverse direction (ACE defensively unloading
    // MM3) IS done in routes/generate.ts, because the ACE path has no
    // equivalent engine-side arbitration.

    // ── Pre-flight: prompt token budget ──
    // Cold-capable, so this catches an over-long caption+lyrics before any GPU
    // work. POST /mm3/synth re-checks and would 400 anyway; doing it here buys
    // a clear, actionable job error instead of a raw HTTP failure.
    job.status = 'synth_running';
    job.stage = 'MiniMax-Music3: checking prompt length...';
    job.progress = 1;
    const tokStart = performance.now();
    try {
      const tok = await mm3TokenizeCheck(req.caption, req.lyrics || '');
      log('INFO', `[MM3] Prompt: ${tok.tokens}/${tok.limit} tokens${tok.instrumental ? ' (instrumental)' : ''}`);
      if (!tok.ok) {
        throw new Error(
          `Prompt is too long for MiniMax-Music3: ${tok.tokens} tokens vs a ${tok.limit}-token limit. `
          + 'Shorten the caption (the Arrangement section is usually the longest part) or the lyrics.',
        );
      }
    } catch (tokErr: any) {
      // Only a genuine over-limit answer is fatal; a failed probe (engine cold,
      // busy, tokenizer unavailable) must not block the run — /mm3/synth
      // enforces the same limit synchronously.
      if (tokErr?.message?.startsWith('Prompt is too long')) throw tokErr;
      log('WARNING', `[MM3] Token pre-flight unavailable (continuing — /mm3/synth re-checks): ${tokErr.message}`);
    }
    const tokMs = Math.round(performance.now() - tokStart);
    if (tokMs > 50) timing.push({ name: 'Token pre-flight', ms: tokMs });

    // ── Submit ──
    job.stage = 'MiniMax-Music3: submitting...';
    job.progress = 2;
    const submitStart = performance.now();
    const sub = await mm3Synth(req);
    job.aceJobId = sub.job_id;   // standard /job id — /cancel/:id reaches it unchanged

    // Persist the RESOLVED seed so a randomized take is reproducible.
    job.params.seed = sub.seed;
    job.params.randomSeed = false;
    log('INFO',
      `[MM3] Job ${sub.job_id} submitted — ${sub.prompt_tokens} prompt tokens, ${sub.max_frames} frames `
      + `(${sub.duration.toFixed(1)}s), seed ${sub.seed}, ${sub.steps} steps, cfg ${sub.cfg_flow}`
      + `${sub.instrumental ? ', instrumental' : ''}`);

    // ── Progress ticker ──
    // GET /job (polled by pollUntilDone) gives the ACE-phase mapping; this adds
    // MM3's own vocabulary and keeps the stall watchdog fed.
    const runStart = Date.now();
    detailTimer = setInterval(() => {
      void (async () => {
        const d = await mm3JobDetail(sub.job_id);
        if (!d || job.status === 'cancelled') return;
        const elapsed = Math.round((Date.now() - runStart) / 1000);
        const { stage, progress } = minimaxStageText(d, elapsed);
        job.stage = stage;
        job.progress = progress;
        // MM3's stage is finer-grained than GET /job's phase mapping; surface
        // it through the existing ace_phase_progress channel.
        if (d.n_steps > 0) job.acePhaseProgress = `step ${d.step}/${d.n_steps}`;
      })();
    }, DETAIL_POLL_MS);
    detailTimer.unref?.();

    await deps.pollUntilDone(sub.job_id, job, abortController.signal, timeoutMinutes);
    clearInterval(detailTimer);
    detailTimer = undefined;
    timing.push({ name: 'MM3 Generate', ms: Math.round(performance.now() - submitStart) });

    // ── Result ──
    job.stage = 'MiniMax-Music3: saving audio...';
    job.progress = 95;
    const saveStart = performance.now();
    const audioRes = await aceClient.getJobResult(sub.job_id);
    if (!audioRes.ok) {
      throw new Error(`Failed to fetch MiniMax-Music3 result (HTTP ${audioRes.status})`);
    }
    const audioBuffer = Buffer.from(await audioRes.arrayBuffer());
    if (audioBuffer.length === 0) throw new Error('MiniMax-Music3 returned an empty audio body');

    const contentType = audioRes.headers.get('content-type') || 'audio/wav';
    const ext = contentType.includes('wav') ? 'wav' : 'bin';
    const filename = `${uuidv4()}.${ext}`;
    const filepath = path.join(config.data.audioDir, filename);
    fs.writeFileSync(filepath, audioBuffer);
    const audioUrl = `/audio/${filename}`;

    // Final MM3 detail: shape + per-stage timings, straight from the engine.
    const finalDetail = await mm3JobDetail(sub.job_id);
    if (finalDetail?.result) {
      const r = finalDetail.result;
      log('INFO',
        `[MM3] Rendered ${r.frames} frames → ${r.duration_sec.toFixed(1)}s @ ${r.sample_rate} Hz, `
        + `rms ${r.rms.toFixed(4)}, peak ${r.peak.toFixed(3)}${r.eos ? ', EOS hit' : ''}${r.has_nan ? ' — WARNING: NaNs present' : ''}`);
      if (r.ms) {
        timing.push({ name: '  AR planner', ms: Math.round(r.ms.ar) });
        timing.push({ name: '  Condition encoder', ms: Math.round(r.ms.cond) });
        timing.push({ name: '  Flow (DiT)', ms: Math.round(r.ms.flow) });
        timing.push({ name: '  Vocoder', ms: Math.round(r.ms.voc) });
      }
      if (finalDetail.evicted_modules) {
        log('INFO', `[MM3] VRAM arbitration evicted ${finalDetail.evicted_modules} ACE module(s) (${(finalDetail.evicted_mb ?? 0).toFixed(0)} MB)`);
      }
    }

    // Duration from the actual WAV header (MM3 renders 44.1 kHz — the parser
    // reads the real rate, so no 48 kHz assumption leaks in here).
    const measured = wavDurationSec(filepath);
    const duration = measured > 0
      ? Math.round(measured)
      : Math.round(finalDetail?.result?.duration_sec || sub.duration || 0);

    log('INFO', `[MM3] Saved ${filename} (${(audioBuffer.length / 1024).toFixed(0)} KB, ${duration}s)`);
    timing.push({ name: 'Save', ms: Math.round(performance.now() - saveStart) });

    // ── Persist ──
    const captionLine = req.caption.split('\n').map(s => s.trim()).find(s => s.length > 0) || '';
    const title: string = job.params.title || captionLine.substring(0, 60) || 'Untitled';
    const style: string = job.params.caption || job.params.style || '';
    const trackParams = {
      ...job.params,
      backend: 'minimax-m3',
      seed: sub.seed,
      duration: sub.duration,
      // The exact wire request, so a reproduce/A-B flow has the real inputs
      // rather than an ACE-shaped approximation of them.
      mm3Request: req,
      mm3: {
        max_frames: sub.max_frames,
        steps: sub.steps,
        cfg_flow: sub.cfg_flow,
        prompt_tokens: sub.prompt_tokens,
        instrumental: sub.instrumental,
        sample_rate: finalDetail?.result?.sample_rate ?? 44100,
      },
    };

    const songId = uuidv4();
    getDb().prepare(`
      INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url,
                         duration, bpm, key_scale, time_signature, tags, dit_model,
                         generation_params, mastered_audio_url, latent_url, quality_scores,
                         noadapter_audio_url, backend)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      songId, job.userId, title, req.lyrics || '', style, req.caption,
      audioUrl, duration, 0, '', '',
      JSON.stringify([]), 'mm3', JSON.stringify(trackParams),
      '', '', '',
      '', 'minimax-m3',
    );

    // ── Cover art (backend-agnostic: a text→image service, no engine coupling) ──
    if (job.params.coverArtEnabled) {
      const coverStart = performance.now();
      try {
        const { generateCoverArt, getCoverArtReadiness } = await import('../../coverArt/coverArtService.js');
        const readiness = getCoverArtReadiness();
        if (readiness.installed) {
          job.stage = 'Generating cover art...';
          job.progress = 97;
          await generateCoverArt({
            songId,
            title,
            style: style || captionLine,
            lyrics: req.lyrics || '',
            subject: job.params.coverArtSubject || job.params.subject || '',
          });
          log('INFO', `[CoverArt] Generated cover for song ${songId}`);
          if (job.params.coverArtSubject) {
            getDb().prepare('UPDATE songs SET cover_art_subject = ? WHERE id = ?')
              .run(job.params.coverArtSubject, songId);
          }
        } else {
          log('DEBUG', `[CoverArt] Skipped — not installed (missing: ${readiness.missingFiles.join(', ')})`);
        }
      } catch (coverErr: any) {
        log('WARNING', `[CoverArt] Failed (non-fatal): ${coverErr.message}`);
      }
      const coverMs = Math.round(performance.now() - coverStart);
      if (coverMs > 50) timing.push({ name: 'Cover Art', ms: coverMs });
    }

    const totalMs = Math.round(performance.now() - pipelineStart);
    timing.push({ name: 'TOTAL', ms: totalMs });

    job.status = 'succeeded';
    job.progress = 100;
    job.stage = 'Complete!';
    job.result = {
      audioUrls: [audioUrl],
      songIds: [songId],
      duration,
      timing,
      totalMs,
    };

    log('INFO', `[Result] 1 audio file saved, 1 song created (backend=minimax-m3)`);
    console.log(`[Generate] Job ${job.id} (minimax-m3) completed in ${(totalMs / 1000).toFixed(1)}s`);
    finishGenerationLog(job.id, 'mm3-text2music');

  } catch (err: any) {
    if (detailTimer) clearInterval(detailTimer);
    // `as string`: POST /api/generate/cancel/:id mutates job.status from
    // outside this function, which TS's control-flow narrowing can't see.
    if (err.message === 'Cancelled' || (job.status as string) === 'cancelled') {
      job.status = 'cancelled';
      job.stage = 'Cancelled';
      failGenerationLog(job.id, 'Cancelled by user', 'mm3-text2music');
    } else {
      job.status = 'failed';
      job.error = err.message || 'Unknown error';
      job.stage = 'Failed';
      console.error(`[Generate] Job ${job.id} (minimax-m3) failed:`, err.message);
      failGenerationLog(job.id, err.message || 'Unknown error', 'mm3-text2music');
    }
  } finally {
    if (detailTimer) clearInterval(detailTimer);
  }
}

/** Best-effort MM3 eviction for the ACE path (plan §4.4). Cheap when cold,
 *  short-fused so a hung engine can never stall an ACE generation. */
export async function releaseMinimaxVramForAce(): Promise<void> {
  try {
    const r = await mm3Unload(3_000);
    if (r?.unloaded) {
      console.log(`[Generate] Freed MiniMax-Music3 VRAM before ACE work (${(r.freed_mb ?? 0).toFixed(0)} MB)`);
    }
  } catch { /* never blocks ACE */ }
}
