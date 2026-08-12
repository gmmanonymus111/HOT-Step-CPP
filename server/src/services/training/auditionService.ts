// training/auditionService.ts — the codes-audition brain
//
// Resolves every default in ONE place, builds the /lm request field by field,
// runs one A/B side, and decodes stored codes for the synchronous sample
// audition. The only module that knows the shape of an LM request.
//
// ── THE LM ECHO SIDEBAND TRAP (read before touching runOneSide) ─────────────
// The known failure mode is reusing the AceRequest that /lm ECHOED BACK as the
// basis for the next engine call: server-only fields do not survive the round
// trip. This module never does that. The /lm response is read for exactly six
// values (audio_codes, caption, lyrics, bpm, keyscale, timesignature) and then
// discarded; the /codes-decode request is built fresh from
// {audio_codes, synth_model, vae, output_format, peak_clip} and nothing else.
// DO NOT "optimise" this by forwarding the echoed request object.
//
// Spec: docs/plans/2026-07-28-codes-preview.md §5.3, C5, C12, C13, C14
//
// Read-only reuse: trainLmStatus.ts (newestVariantKey, variantDitModel),
// aceTrain.ts (tensorsRoot), labelStore.ts (readLabel). None of those files is
// edited by this plan.

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { randomUUID } from 'crypto';
import { aceClient, type AceRequest } from '../aceClient.js';
import { getModelSnapshot, pickLmFor, tensorsRoot } from './aceTrain.js';
import { lmSizeOfAdapterPath } from './adapterLayout.js';
import { readLabel } from './labelStore.js';
import { newestVariantKey, variantDitModel, variantExists } from './trainLmStatus.js';
import { writePreview, type PreviewAudio } from './auditionStore.js';
import type {
  AuditionKind, AuditionOptions, AuditionPreview, AuditionSideResult, AuditionSideSpec,
  LmSize, SampleAuditionResponse, TrainingDatasetRow,
} from './types.js';

const CODES_FILE = 'lm_codes.jsonl';

/** Wall-clock bound on one /lm run. A 4B planner measures ~18 s of LM phase
 *  plus up to ~20 s of cold load, so 10 min is ~15x headroom — this is a
 *  "the engine is wedged" tripwire, not a performance budget. Deliberately a
 *  local constant: config.ts is outside this plan's editable set. */
const LM_DEADLINE_MS = 10 * 60_000;

/** An error the route can map straight onto an HTTP status. */
export class AuditionError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = 'AuditionError';
  }
}

function clamp(value: unknown, fallback: number, min: number, max: number): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function str(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function num(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

export function sha1(text: string): string {
  return crypto.createHash('sha1').update(text, 'utf8').digest('hex');
}

/** codesCount / 5, rounded to 0.1 (§3.1). */
function codesDuration(codesCount: number): number {
  return Math.round((codesCount / 5) * 10) / 10;
}

/**
 * The dataset's trigger word, applied EXACTLY as the trainer applied it
 * (lm_apply_tag, engine/src/train/lm-common.h:353): prepend → "tag, caption",
 * append → "caption, tag", and 'replace' applies NO tag (verbatim Side-Step —
 * those captions trained untagged, so claiming a trigger would be false).
 *
 * Free-text and LM-written audition prompts never carried the trigger, so the
 * adapter ran without the token it was trained on and both sides emitted
 * near-identical plans (Rob, 2026-07-29). Applied to BOTH sides identically —
 * A/B discipline — and skipped when the caption already carries the tag (the
 * lm_codes.jsonl row fallback is pre-tagged; users may also type it).
 */
function applyTriggerTag(caption: string, tag: string, position: string): string {
  const t = String(tag ?? '').trim();
  if (!t || !caption) return caption;
  const pos = position || 'prepend';
  // 'replace' collapses the caption to the trigger, which is what the dataset
  // was preprocessed with — so it runs BEFORE the already-contains check. That
  // check exists to avoid duplicating a token in a caption that keeps its prose
  // (pre-tagged lm_codes.jsonl rows, users typing the trigger themselves); for
  // 'replace' the prose has to go regardless of whether the tag is in it.
  if (pos === 'replace') return t;
  if (caption.toLowerCase().includes(t.toLowerCase())) return caption;
  if (pos === 'prepend') return `${t}, ${caption}`;
  if (pos === 'append') return `${caption}, ${t}`;
  return caption;
}

// ── Codes-row lookup ──────────────────────────────────────────────────────

export interface CodesRow {
  file: string;
  caption: string;
  lyrics: string;
  bpm: number;
  keyscale: string;
  timesignature: string;
  duration: number;
  codes: number[];
  id: string;
}

/**
 * The lm_codes.jsonl row for one dataset sample.
 *
 * `sampleId` is the studio's 16-hex TrainingSample id; the codes file stores the
 * 8-hex `datasetSampleId` (= the same sha1, first 8) — the identical `_id` /
 * `-([0-9a-f]{8}).safetensors` pair trainLmStatus.countCodes matches on. Falls
 * back to the id embedded in `row.file` when `_id` is missing (legacy rows).
 */
export function findCodesRow(slug: string, variantKey: string, sampleId: string): CodesRow | null {
  const id8 = String(sampleId ?? '').toLowerCase().slice(0, 8);
  if (!id8 || !variantKey) return null;
  const codesPath = path.join(tensorsRoot(slug), variantKey, CODES_FILE);
  let raw = '';
  try {
    if (!fs.existsSync(codesPath)) return null;
    raw = fs.readFileSync(codesPath, 'utf-8');
  } catch {
    return null;
  }
  for (const line of raw.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let row: Record<string, unknown>;
    try {
      row = JSON.parse(trimmed) as Record<string, unknown>;
    } catch {
      continue;   // a torn line is not a row
    }
    if (!row || typeof row !== 'object') continue;

    const rowId = str(row._id).toLowerCase();
    let hit = rowId === id8;
    if (!hit) {
      const m = /-([0-9a-f]{8})\.safetensors$/i.exec(str(row.file));
      hit = !!m && m[1].toLowerCase() === id8;
    }
    if (!hit) continue;

    const codes = Array.isArray(row.codes)
      ? (row.codes as unknown[]).map(v => Math.trunc(num(v, -1))).filter(v => v >= 0)
      : [];
    return {
      file: str(row.file),
      caption: str(row.caption),
      lyrics: str(row.lyrics),
      bpm: Math.trunc(num(row.bpm)),
      keyscale: str(row.keyscale),
      timesignature: str(row.timesignature),
      duration: Math.trunc(num(row.duration)),
      codes,
      id: rowId || id8,
    };
  }
  return null;
}

// ── Resolution ────────────────────────────────────────────────────────────

export interface ResolvedAudition {
  datasetId: string;
  slug: string;
  kind: AuditionKind;
  sides: AuditionSideSpec[];
  caption: string;
  lyrics: string;
  seed: number;
  durationSec: number;
  /** Dataset-sample metadata pinning (0/'' = let the LM predict). Filled from
   *  the sample's lm_codes.jsonl row so both sides plan at the ground-truth
   *  bpm/key/timesig — the engine force_fields any non-empty value into the
   *  CoT FSM (pipeline-lm.cpp:1592-1606). */
  bpm: number;
  keyscale: string;
  timesignature: string;
  lmModel: string;
  ditModel: string;
  vaeModel: string;
  variantKey: string;
  sampleId: string;
  temperature: number;
  topP: number;
  cfgScale: number;
  repPenalty: number;
  format: 'wav16' | 'mp3';
  coResident: boolean;
  renderDit: boolean;
  renderSteps: number;
  renderDitModel: string;   // '' = let the engine resolve its default DiT
}

/**
 * The DiT the opt-in render runs on when the user names none: the newest
 * installed xl-turbo (fast, 8-step-friendly), else any turbo, else the detok
 * DiT the codes were decoded against. Never a hard failure — a render on a
 * slower base is still a render.
 */
function pickRenderDit(detokDit: string): string {
  const dits = getModelSnapshot().dit;
  return dits.find(m => /xl[-_]?turbo/i.test(m))
    ?? dits.find(m => /turbo/i.test(m))
    ?? detokDit;
}

/**
 * ONE place decides every default (§5.3).
 *
 * `lmModel`: when empty, derived from the adapter's size suffix via pickLmFor —
 * letting the engine's resolve_name pick the sticky/first LM put a 4B adapter on
 * the 0.6B base ("36 layers but model has 28 — mismatch, refusing", found live
 * by Rob). An explicit lmModel that contradicts the adapter's size is a 400.
 */
export function resolveAuditionInputs(ds: TrainingDatasetRow, opts: AuditionOptions): ResolvedAudition {
  const variantKey = (typeof opts.variantKey === 'string' && variantExists(ds.slug, opts.variantKey))
    ? opts.variantKey
    : newestVariantKey(ds.slug);

  const sampleId = str(opts.sampleId);
  const row = sampleId ? findCodesRow(ds.slug, variantKey, sampleId) : null;

  // The literal strings the trainer conditioned on (C12a) fill the gaps, but
  // never override what the user actually typed. The fallback can only fire when
  // opts.sampleId is set — the free-text and "let the LM write it" sources never
  // send one, so an intentionally empty `lyrics` is never silently repopulated.
  const caption = str(opts.caption).trim() || (row ? row.caption : '');
  const lyrics = str(opts.lyrics) || (row ? row.lyrics : '');

  // The route lets caption be empty when a sampleId is present, precisely so the
  // fallback above can supply the TAGGED trainer caption. If the row is missing
  // or has no caption, there is nothing to fall back to and an empty prompt would
  // run two meaningless LM passes — fail loudly instead.
  if (!caption) {
    throw new AuditionError(409, sampleId
      ? `Sample ${sampleId} has no caption in ${CODES_FILE} for variant ${variantKey} — ` +
        'run Preprocess + Extract for this variant, or type a caption instead'
      : 'caption is required');
  }

  // Free-text / LM-written / hand-edited prompts get the dataset's trigger word
  // exactly as the trainer applied it; a pre-tagged row caption passes through
  // unchanged (the contains-check inside). See applyTriggerTag.
  const taggedCaption = applyTriggerTag(caption, ds.customTag, ds.tagPosition);

  // Never -1: the audition must be re-runnable from what we record (C14).
  const seed = Number.isFinite(Number(opts.seed)) && Number(opts.seed) >= 0
    ? Math.trunc(Number(opts.seed))
    : Math.floor(Math.random() * 2 ** 31);

  const ditModel = str(opts.ditModel) || variantDitModel(ds.slug, variantKey) || '';

  const sides: AuditionSideSpec[] = (Array.isArray(opts.sides) ? opts.sides : []).map(s => ({
    slot: s.slot === 'adapter' ? 'adapter' : 'base',
    label: str(s.label).slice(0, 64) || (s.slot === 'adapter' ? 'Adapter' : 'Base LM'),
    lmAdapter: str(s.lmAdapter),
    lmAdapterScale: clamp(s.lmAdapterScale, 1.0, 0, 2),
  }));

  // Base-LM pinning. An empty lmModel used to fall through to the engine's
  // resolve_name (sticky loaded LM, else registry[0] = the 0.6B), which made a
  // 4B adapter meet a 28-layer base and die with "36 layers but model has 28 —
  // mismatch, refusing" AFTER a full load cycle. Derive the base from the
  // adapter's own layout instead — its lm-<size> parent folder in the per-base
  // layout, the legacy -<size> name suffix otherwise — and reject an explicit
  // mismatch up front. Both sides of an A/B must run the SAME base for the
  // comparison to mean anything, so this is seed-discipline, not ergonomics.
  // Milestone dirs sit one level deeper (<adapter>/milestones/loss_x), so walk
  // up until a size is found or the adapters root is reached.
  let lmModel = str(opts.lmModel);
  const adapterSide = sides.find(s => s.lmAdapter);
  let adapterSize: string = '';
  if (adapterSide) {
    let probe = adapterSide.lmAdapter;
    for (let hops = 0; hops < 4 && probe && !adapterSize; hops++) {
      adapterSize = lmSizeOfAdapterPath(probe);
      probe = path.dirname(probe);
    }
  }
  if (adapterSize) {
    if (!lmModel) {
      lmModel = pickLmFor(adapterSize as LmSize, getModelSnapshot().lm);
      if (!lmModel) {
        throw new AuditionError(409,
          `Adapter ${adapterSide!.lmAdapter} needs a ${adapterSize} LM base, but none is installed`);
      }
    } else if (!new RegExp(`-${adapterSize.replace('.', '\\.')}(-|$)`, 'i').test(lmModel)) {
      throw new AuditionError(400,
        `Adapter ${adapterSide!.lmAdapter} is ${adapterSize} but lmModel "${lmModel}" is not — ` +
        'pick a matching base or clear the base-model field');
    }
  }

  return {
    datasetId: ds.id,
    slug: ds.slug,
    kind: opts.kind === 'milestone' ? 'milestone' : 'ab',
    sides,
    caption: taggedCaption,
    lyrics,
    seed,
    // Cap raised 120 → 300 (Rob, 2026-07-29): a 3-minute audition is a
    // legitimate ask; the LM cost scales linearly and the deadline has slack.
    // Default 30 → 180 (Rob, 2026-08-12): a 30 s plan is too short to judge.
    durationSec: Math.trunc(clamp(opts.durationSec, 180, 10, 300)),
    // Dataset Sample mode: pin the plan to the sample's ground-truth metadata.
    // Without this nothing anchors bpm/key/timesig, the FSM samples them
    // freely, and the plan drifts from the trainer's conditioning (a 2/4
    // timesig on a 4/4 dataset — Rob, 2026-08-12). Applied to BOTH sides
    // identically (A/B discipline); free-text prompts (no row) still let the
    // LM predict, which is the point of that mode. Duration is NOT taken from
    // the row — the user's audition length wins; the engine pins it already.
    bpm: row ? row.bpm : 0,
    keyscale: row ? row.keyscale : '',
    timesignature: row ? row.timesignature : '',
    lmModel,
    ditModel,
    vaeModel: str(opts.vaeModel),
    variantKey,
    sampleId,
    temperature: clamp(opts.temperature, 0.85, 0.1, 2),
    topP: clamp(opts.topP, 0.9, 0.05, 1),
    cfgScale: clamp(opts.cfgScale, 2.0, 0, 10),
    // Default 1.0 → 1.1 (Rob, 2026-08-12): adapters sharpen the code
    // distribution; a mild presence penalty is the better audition baseline.
    repPenalty: clamp(opts.repPenalty, 1.1, 1, 1.5),
    format: opts.format === 'mp3' ? 'mp3' : 'wav16',
    coResident: opts.coResident !== false,
    renderDit: opts.renderDit === true,
    renderSteps: Math.trunc(clamp(opts.renderSteps, 8, 2, 60)),
    renderDitModel: opts.renderDit === true
      ? (str(opts.renderDitModel) || pickRenderDit(ditModel))
      : '',
  };
}

// ── The LM request ────────────────────────────────────────────────────────

/**
 * The ONLY place an LM request is constructed. Every field written explicitly;
 * nothing inherited from a previous request object (see the module header).
 *
 * `lm_seed` is set explicitly as well as `seed` because an explicit lm_seed
 * always wins over the seed fallback in the engine — which is what makes both
 * sides of an A/B literally the same LM run apart from the adapter (C5/C14).
 */
export function buildLmRequest(
  resolved: ResolvedAudition,
  side: AuditionSideSpec,
): AceRequest & { lm_mode: string } {
  // `lm_mode` is a real engine field (request.cpp:132, default LM_MODE_NAME_GENERATE)
  // that AceRequest does not declare — submitLm only injects it for the
  // 'inspire'/'format' modes. Written explicitly here so the audition never
  // depends on the engine's default staying 'generate'.
  return {
    caption: resolved.caption,
    lyrics: resolved.lyrics,
    duration: resolved.durationSec,
    // Sample-mode metadata pins (empty = absent = LM predicts). The engine
    // force_fields these into the CoT, so the plan keeps the dataset's
    // bpm/key/timesig while the CoT caption rewrite still shows the
    // planner's own voice.
    ...(resolved.bpm > 0 ? { bpm: resolved.bpm } : {}),
    ...(resolved.keyscale ? { keyscale: resolved.keyscale } : {}),
    ...(resolved.timesignature ? { timesignature: resolved.timesignature } : {}),
    seed: resolved.seed,
    lm_seed: resolved.seed,
    lm_mode: 'generate',
    lm_batch_size: 1,
    lm_temperature: resolved.temperature,
    lm_top_p: resolved.topP,
    lm_cfg_scale: resolved.cfgScale,
    lm_rep_penalty: resolved.repPenalty,
    lm_model: resolved.lmModel,
    lm_adapter: side.lmAdapter,
    lm_adapter_scale: side.lmAdapterScale,
  };
}

export interface SideHooks {
  /** Set the job phase — 'audition-decode' between the LM and the decode.
   *  A callback rather than the TrainingJob itself so this module never imports
   *  labelingQueue (auditionRunner already does, and a cycle would be a lazy
   *  import for no gain). */
  onPhase?: (phase: string) => void;
  isCancelled?: () => boolean;
}

export interface SideOutcome {
  result: AuditionSideResult;
  audio: PreviewAudio | null;
  /** The raw codes CSV, kept so the runner can render this side through the
   *  DiT AFTER the pinned LM has been evicted. Rendering inside the side —
   *  with keep_loaded holding the 4B planner resident — loaded the DiT on top
   *  of it and spilled VRAM into shared memory (Rob, 2026-07-29). '' on
   *  failure. */
  audioCodes: string;
}

/** Poll one engine job to completion under a deadline, cancelling the engine
 *  job on our own cancel/timeout (same discipline as the /lm loop — an
 *  abandoned engine job pins VRAM and wedges the global queue chain). */
async function awaitEngineJob(
  jobId: string,
  what: string,
  hooks: SideHooks,
): Promise<void> {
  const deadline = Date.now() + LM_DEADLINE_MS;
  for (;;) {
    if (hooks.isCancelled?.()) {
      await aceClient.cancelJob(jobId).catch(() => { /* engine may already be gone */ });
      throw new Error('Cancelled');
    }
    if (Date.now() > deadline) {
      await aceClient.cancelJob(jobId).catch(() => { /* best effort */ });
      throw new Error(`${what} timed out after ${Math.round(LM_DEADLINE_MS / 60_000)} min`);
    }
    const status = await aceClient.pollJob(jobId);
    if (status.status === 'done') return;
    if (status.status === 'failed') throw new Error(`${what} failed — see ace_engine.log`);
    if (status.status === 'cancelled') throw new Error(`${what} cancelled`);
    await new Promise(r => setTimeout(r, 200));
  }
}

/**
 * The opt-in DiT render of one side's codes (§Rob 2026-07-29): a fresh /synth
 * request carrying the side's audio_codes and the ECHOED plan fields, at a
 * fixed step count on a DiT shared by both sides, with NO sound adapter — the
 * planner adapter must remain the A/B's only variable. Request built FRESH,
 * exactly like the /codes-decode one (module-header sideband rule).
 *
 * Called by the RUNNER after every LM side has finished and the pinned LM has
 * been evicted — never inside a side, where keep_loaded's EVICT_NEVER would
 * stack the DiT on top of a resident 4B planner and overflow into shared
 * memory. Exported for exactly that caller.
 */
export async function renderSideThroughDit(
  resolved: ResolvedAudition,
  result: AuditionSideResult,
  audioCodes: string,
  hooks: SideHooks,
): Promise<PreviewAudio> {
  const req: AceRequest = {
    caption: result.caption || resolved.caption,
    lyrics: result.lyrics || resolved.lyrics,
    duration: result.durationSec || resolved.durationSec,
    ...(result.bpm > 0 ? { bpm: result.bpm } : {}),
    ...(result.keyscale ? { keyscale: result.keyscale } : {}),
    ...(result.timesignature ? { timesignature: result.timesignature } : {}),
    seed: resolved.seed,
    audio_codes: audioCodes,
    inference_steps: resolved.renderSteps,
    task_type: 'text2music',
    ...(resolved.renderDitModel ? { synth_model: resolved.renderDitModel } : {}),
    ...(resolved.vaeModel ? { vae_model: resolved.vaeModel } : {}),
    // NO adapter / adapters / lm_* fields — deliberately.
  };
  const jobId = await aceClient.submitSynth(req, resolved.format);
  await awaitEngineJob(jobId, 'DiT render', hooks);
  const res = await aceClient.getJobResult(jobId);
  if (!res.ok) throw new Error(`DiT render result fetch failed (${res.status})`);
  const buf = Buffer.from(await res.arrayBuffer());
  if (!buf.length) throw new Error('DiT render returned no audio');
  return { slot: result.slot, buf, ext: resolved.format === 'mp3' ? 'mp3' : 'wav', render: true };
}

/**
 * One A/B side: /lm → /codes-decode.
 *
 * NEVER throws. A failed side lands in `.error` with `ok:false` and the other
 * side still runs — half an A/B is still evidence.
 */
export async function runOneSide(
  resolved: ResolvedAudition,
  side: AuditionSideSpec,
  hooks: SideHooks = {},
): Promise<SideOutcome> {
  const result: AuditionSideResult = {
    slot: side.slot,
    label: side.label,
    lmAdapter: side.lmAdapter,
    lmAdapterScale: side.lmAdapterScale,
    ok: false,
    error: '',
    audioUrl: '',
    codesCount: 0,
    codesSha1: '',
    durationSec: 0,
    caption: '',
    lyrics: '',
    bpm: 0,
    keyscale: '',
    timesignature: '',
    lmMs: 0,
    decodeMs: 0,
  };

  try {
    // ── 1. /lm ───────────────────────────────────────────────────────────
    const t0 = Date.now();
    const lmJobId = await aceClient.submitLm(
      buildLmRequest(resolved, side), undefined, resolved.coResident);

    // A deadline is not optional here. labelingQueue runs every training job
    // kind — label, enhance, build, preprocess, train-lm, train-dit — on ONE
    // global promise chain (`queueTail = queueTail.then(...)`), so an unbounded
    // wait in this loop does not just hang the audition: it wedges that chain
    // for the rest of the server's lifetime, leaving every subsequent job for
    // EVERY dataset stuck at 'queued' with no error and no log line. A wedged
    // engine LM job (OOM-stall, driver hang, or a worker that died without ever
    // setting status 2) is exactly how that happens. Bound + cancel, mirroring
    // understandClient's deadline/abort handling. (awaitEngineJob also cancels
    // the ENGINE job on our cancel/timeout: abandoning it leaves the GPU
    // working on a result nobody will fetch, and with ?keep_loaded=1 the 4B
    // stays pinned — the next train-lm run then kills ace-server mid-flight.)
    await awaitEngineJob(lmJobId, 'LM job', hooks);
    const lmRes = await aceClient.getJobResult(lmJobId);
    if (!lmRes.ok) throw new Error(`LM result fetch failed (${lmRes.status})`);
    const echoed = await lmRes.json() as AceRequest[];
    result.lmMs = Date.now() - t0;

    // Exactly six values are read out of the echo. It is then DISCARDED.
    const plan = Array.isArray(echoed) ? echoed[0] : undefined;
    const audioCodes = str(plan?.audio_codes);
    result.caption = str(plan?.caption);
    result.lyrics = str(plan?.lyrics);
    result.bpm = Math.trunc(num(plan?.bpm));
    result.keyscale = str(plan?.keyscale);
    result.timesignature = str(plan?.timesignature);

    // ── 2. no codes → a soft failure, not a thrown job ───────────────────
    if (!audioCodes.trim()) {
      result.error = 'LM returned no audio codes';
      return { result, audio: null, audioCodes: '' };
    }

    result.codesSha1 = sha1(audioCodes);
    result.codesCount = audioCodes.split(',').filter(t => t.trim().length > 0).length;
    result.durationSec = codesDuration(result.codesCount);

    // ── 3. /codes-decode — request built FRESH, never from `plan` ─────────
    hooks.onPhase?.('audition-decode');
    const t1 = Date.now();
    const buf = await aceClient.codesDecode({
      audio_codes: audioCodes,
      synth_model: resolved.ditModel,
      vae: resolved.vaeModel,
      output_format: resolved.format,
    }, undefined, hooks.isCancelled);
    result.decodeMs = Date.now() - t1;
    result.ok = true;

    // The DiT render deliberately does NOT happen here — see SideOutcome.
    return {
      result,
      audio: { slot: side.slot, buf, ext: resolved.format === 'mp3' ? 'mp3' : 'wav' },
      audioCodes,
    };
  } catch (err: any) {
    result.ok = false;
    result.error = err instanceof Error ? err.message : String(err);
    return { result, audio: null, audioCodes: '' };
  }
}

// ── The synchronous sample audition ───────────────────────────────────────

/**
 * Stored codes → audio, no /lm at all — which is exactly why it is fast enough
 * to be synchronous (C7).
 *
 * Precedence (FROZEN, C12):
 *   1. lm_codes.jsonl's `codes` array — the canonical source, literally the
 *      token stream the trainer conditioned on.
 *   2. labelStore's SampleLabelRecord.audioCodes — the legacy /understand payload.
 *   3. Neither → 409. Codes are NEVER silently synthesised.
 *
 * The `source` field says which was used: (1) and (2) can disagree if the DiT
 * changed since labelling, and the user must be able to see which they heard.
 */
export async function decodeStoredCodes(
  ds: TrainingDatasetRow,
  sampleId: string,
  format: 'wav16' | 'mp3',
): Promise<SampleAuditionResponse> {
  const variantKey = newestVariantKey(ds.slug);
  const row = findCodesRow(ds.slug, variantKey, sampleId);

  let codesCsv = '';
  let source: 'lm_codes' | 'label' = 'lm_codes';
  let caption = '';
  let lyrics = '';
  let bpm = 0;
  let keyscale = '';
  let timesignature = '';

  if (row && row.codes.length > 0) {
    codesCsv = row.codes.join(',');
    source = 'lm_codes';
    caption = row.caption;
    lyrics = row.lyrics;
    bpm = row.bpm;
    keyscale = row.keyscale;
    timesignature = row.timesignature;
  } else {
    const label = readLabel(ds.slug, sampleId);
    const stored = str(label?.audioCodes).trim();
    if (stored) {
      codesCsv = stored;
      source = 'label';
      caption = str(label?.understand?.caption);
      lyrics = str(label?.understand?.lyrics);
      bpm = Math.trunc(num(label?.understand?.bpm));
      keyscale = str(label?.understand?.keyscale);
      timesignature = str(label?.understand?.timesignature);
    }
  }

  if (!codesCsv) {
    throw new AuditionError(409,
      'This sample has no stored codes — run Preprocess + Extract, or label it with Understand');
  }

  const codesCount = codesCsv.split(',').filter(t => t.trim().length > 0).length;
  const ditModel = variantDitModel(ds.slug, variantKey) || '';

  const t0 = Date.now();
  const buf = await aceClient.codesDecode({
    audio_codes: codesCsv,
    synth_model: ditModel,
    vae: '',
    output_format: format,
  });
  const decodeMs = Date.now() - t0;

  const previewId = randomUUID();
  const ext: 'wav' | 'mp3' = format === 'mp3' ? 'mp3' : 'wav';
  const durationSec = codesDuration(codesCount);

  const sideResult: AuditionSideResult = {
    slot: 'base',
    label: source === 'lm_codes' ? 'Stored codes (lm_codes.jsonl)' : 'Stored codes (Understand)',
    lmAdapter: '',
    lmAdapterScale: 1,
    ok: true,
    error: '',
    audioUrl: `/api/training/previews/${previewId}/base`,
    codesCount,
    codesSha1: sha1(codesCsv),
    durationSec,
    caption,
    lyrics,
    bpm,
    keyscale,
    timesignature,
    lmMs: 0,
    decodeMs,
  };

  const preview: AuditionPreview = {
    previewId,
    datasetId: ds.id,
    kind: 'sample',
    createdAt: new Date().toISOString(),
    seed: 0,
    caption,
    lyrics,
    durationSec,
    lmModel: '',
    ditModel,
    vaeModel: '',
    sampleId,
    variantKey: source === 'lm_codes' ? variantKey : '',
    sides: [sideResult],
  };

  const wrote = writePreview(preview, [{ slot: 'base', buf, ext }]);
  if (!wrote) {
    // The audio decoded but could not be stored — say so rather than handing
    // back a URL that 404s.
    preview.sides[0].audioUrl = '';
    preview.sides[0].error = 'Decoded, but the preview could not be written to disk';
  }

  return { preview, source };
}
