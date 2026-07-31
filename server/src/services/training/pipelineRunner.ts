// training/pipelineRunner.ts — batch import orchestrator
//
// Runs N audio folders through create → label → build → preprocess → train-dit
// → train-lm with no user intervention, using the stored stage defaults.
//
// Two rules shape the whole file:
//   * stages are started by an INTERNAL HTTP POST to this server's own
//     /api/training routes — the route layer stays the single source of
//     validation and of the stage default literals (§3.2). Progress is then
//     read straight from the job map (no HTTP).
//   * the orchestrator never touches the engine (§3.6). Every runner already
//     owns its own ace-server stop/restart.
//
// One pipeline at a time; items run sequentially, which costs nothing — the job
// queue is one global serial chain anyway. A failed folder stops that folder
// only; the batch carries on.
//
// Spec: docs/plans/2026-07-28-training-batch-pipeline.md §3

import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';
import { config } from '../../config.js';
import { createDatasetFromFolder } from './datasetCreate.js';
import { detailFor } from './datasetDetail.js';
import * as repo from './datasetsRepo.js';
import * as queue from './labelingQueue.js';
import { trainingBaseDir } from './paths.js';
import { getTrainingDefaults } from './trainingDefaults.js';
import type {
  PipelineFolderSpec, PipelineItem, PipelineStage, PipelineStageResult, PipelineSummary,
  TrainingDefaults,
} from './types.js';

/** Canonical stage order. Any subset a caller asks for is filtered through it. */
// 'lyric-studio' runs LAST, after both trainers, because that is what makes it
// useful: the export links the run's freshly trained adapters onto the album
// preset (linkAdapters defaults true), so running it earlier would publish an
// artist/album page pointing at nothing. Added 2026-07-31 — a bulk run used to
// finish with trained adapters that no album preset referenced.
export const PIPELINE_STAGES: readonly PipelineStage[] =
  ['label', 'build', 'preprocess', 'train-dit', 'train-lm', 'lyric-studio'];

const STAGE_PATH: Record<PipelineStage, string> = {
  'label': 'label',
  'build': 'build',
  'preprocess': 'preprocess',
  'train-dit': 'train-dit',
  'train-lm': 'train-lm',
  'lyric-studio': 'lyric-studio',
};

/** Stages that complete SYNCHRONOUSLY — they return their result, not a jobId,
 *  so there is no queue job to poll. */
const SYNC_STAGES: ReadonlySet<PipelineStage> = new Set<PipelineStage>(['lyric-studio']);

/** Which stored-defaults section feeds each stage's POST body. Build takes no
 *  options beyond outputPath, which must stay the route's own default. */
const STAGE_DEFAULTS_KEY: Record<PipelineStage, keyof TrainingDefaults | ''> = {
  'label': 'label',
  'build': '',
  'preprocess': 'preprocess',
  'train-dit': 'trainDit',
  'train-lm': 'trainLm',
  // No stored section: artist/album are derived from the dataset's own tags and
  // linkAdapters already defaults true on the route.
  'lyric-studio': '',
};

const POLL_MS = 1500;
/** How long to wait for a job the USER started on the same dataset (§3.5). */
const IDLE_WAIT_MS = 10 * 60 * 1000;
const MAX_LISTED = 20;

interface PipelineState extends PipelineSummary {
  /** Set by cancelPipeline; every loop checks it between awaits. */
  cancelRequested: boolean;
  /** Per-item creation inputs — not part of the wire contract. */
  specs: PipelineFolderSpec[];
}

const pipelines = new Map<string, PipelineState>();

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function toSummary(state: PipelineState): PipelineSummary {
  const { cancelRequested, specs, ...summary } = state;
  return summary;
}

// ── Persistence (§2.3) ───────────────────────────────────────────────────

function pipelinesDir(): string {
  return path.join(trainingBaseDir, 'pipelines');
}

/** Snapshot on every item/stage transition — same spirit as jobs' _meta.json. */
function persist(state: PipelineState): void {
  try {
    fs.mkdirSync(pipelinesDir(), { recursive: true });
    fs.writeFileSync(
      path.join(pipelinesDir(), `${state.id}.json`),
      JSON.stringify(toSummary(state), null, 2),
      'utf-8',
    );
  } catch (err: any) {
    console.warn(`[Training] Could not persist pipeline ${state.id}: ${err.message}`);
  }
}

function readSnapshots(): PipelineSummary[] {
  const out: PipelineSummary[] = [];
  try {
    const base = pipelinesDir();
    if (!fs.existsSync(base)) return out;
    for (const entry of fs.readdirSync(base, { withFileTypes: true })) {
      if (!entry.isFile() || !entry.name.endsWith('.json')) continue;
      try {
        out.push(JSON.parse(fs.readFileSync(path.join(base, entry.name), 'utf-8')) as PipelineSummary);
      } catch { /* skip corrupted snapshot */ }
    }
  } catch { /* pipelines dir unreadable — memory list is still valid */ }
  return out;
}

/**
 * v1 does not auto-resume (§7): a server restart loses the loop, so rewrite any
 * `running` snapshot as failed rather than leaving a pipeline that looks live
 * forever. Mirrors labelingQueue's recoverStaleJobs.
 */
function recoverStalePipelines(): void {
  try {
    const base = pipelinesDir();
    if (!fs.existsSync(base)) return;
    for (const entry of fs.readdirSync(base, { withFileTypes: true })) {
      if (!entry.isFile() || !entry.name.endsWith('.json')) continue;
      const file = path.join(base, entry.name);
      try {
        const snap = JSON.parse(fs.readFileSync(file, 'utf-8')) as PipelineSummary;
        if (snap.status !== 'running') continue;
        snap.status = 'failed';
        snap.finishedAt = Date.now();
        for (const item of snap.items ?? []) {
          for (const stage of item.stages ?? []) {
            if (stage.status === 'running') {
              stage.status = 'failed';
              stage.error = 'Server restarted';
              stage.finishedAt = Date.now();
            } else if (stage.status === 'pending' || stage.status === 'creating') {
              stage.status = 'cancelled';
            }
          }
          if (item.status === 'pending' || item.status === 'creating' || item.status === 'running') {
            item.status = 'failed';
            item.error = 'Server restarted';
            item.currentStage = null;
          }
        }
        fs.writeFileSync(file, JSON.stringify(snap, null, 2), 'utf-8');
      } catch { /* skip corrupted snapshot */ }
    }
  } catch { /* pipelines dir unreadable */ }
}
recoverStalePipelines();

// ── Public API ───────────────────────────────────────────────────────────

export function hasActivePipeline(): boolean {
  for (const state of pipelines.values()) {
    if (state.status === 'running') return true;
  }
  return false;
}

export function getPipeline(id: string): PipelineSummary | null {
  const state = pipelines.get(id);
  if (state) return toSummary(state);
  return readSnapshots().find(p => p.id === id) ?? null;
}

/** Active first, then the most recent snapshots from disk, max 20. */
export function listPipelines(): PipelineSummary[] {
  const live = [...pipelines.values()].map(toSummary);
  const seen = new Set(live.map(p => p.id));
  const disk = readSnapshots().filter(p => !seen.has(p.id));

  const byNewest = (a: PipelineSummary, b: PipelineSummary): number => b.createdAt - a.createdAt;
  const running = live.filter(p => p.status === 'running').sort(byNewest);
  const rest = [...live.filter(p => p.status !== 'running'), ...disk].sort(byNewest);
  return [...running, ...rest].slice(0, MAX_LISTED);
}

export function startPipeline(folders: PipelineFolderSpec[], stages: PipelineStage[]): PipelineSummary {
  const state: PipelineState = {
    id: randomUUID(),
    status: 'running',
    stages: [...stages],
    items: folders.map(spec => ({
      sourceDir: spec.sourceDir,
      name: (spec.name || '').trim() || path.basename(path.resolve(spec.sourceDir)) || spec.sourceDir,
      datasetId: '',
      reusedExisting: false,
      status: 'pending',
      currentStage: null,
      stages: stages.map(stage => ({
        stage, jobId: '', status: 'pending', error: null, startedAt: null, finishedAt: null,
      })),
      error: null,
    })),
    createdAt: Date.now(),
    finishedAt: null,
    cancelRequested: false,
    specs: folders.map(spec => ({ ...spec })),
  };

  pipelines.set(state.id, state);
  persist(state);
  console.log(
    `[Training] Pipeline ${state.id} started — ${state.items.length} folder(s), stages: ${stages.join(' → ') || 'none'}`);
  // Fire and forget, one tick later: the first folder's create is a synchronous
  // folder scan, and running it inside the handler would hold up the 202.
  setImmediate(() => { void runPipeline(state); });
  return toSummary(state);
}

/** Cancels the running job, then marks everything still pending as cancelled. */
export function cancelPipeline(id: string): boolean {
  const state = pipelines.get(id);
  // A pipeline from a previous process run exists but has nothing to cancel —
  // still a 200, since the id is real.
  if (!state) return readSnapshots().some(p => p.id === id);
  if (state.status !== 'running') return true;

  state.cancelRequested = true;
  for (const item of state.items) {
    for (const stage of item.stages) {
      if (stage.status === 'running') {
        // The runner's poll loop maps the cancelled job onto this stage; killing
        // the job here is what actually stops the work.
        if (stage.jobId) queue.cancelJob(stage.jobId);
      } else if (stage.status === 'pending') {
        stage.status = 'cancelled';
      }
    }
    if (item.status === 'pending') item.status = 'cancelled';
  }
  persist(state);
  console.log(`[Training] Pipeline ${id} cancel requested`);
  return true;
}

// ── Orchestration (§3.1) ─────────────────────────────────────────────────

async function runPipeline(state: PipelineState): Promise<void> {
  try {
    for (let i = 0; i < state.items.length; i++) {
      const item = state.items[i];
      if (state.cancelRequested) {
        cancelItem(item);
        continue;
      }
      await runItem(state, item, state.specs[i]);
    }
  } catch (err: any) {
    // A throw here would leave the pipeline 'running' forever — the finaliser
    // below has to run whatever happened.
    console.error(`[Training] Pipeline ${state.id} crashed: ${err?.message || err}`);
  }

  state.status = state.cancelRequested
    ? 'cancelled'
    : (state.items.some(i => i.status === 'failed') ? 'failed' : 'done');
  state.finishedAt = Date.now();
  persist(state);
  console.log(`[Training] Pipeline ${state.id} ${state.status}`);
}

function cancelItem(item: PipelineItem): void {
  for (const stage of item.stages) {
    if (stage.status === 'pending' || stage.status === 'running') stage.status = 'cancelled';
  }
  if (item.status === 'pending' || item.status === 'creating' || item.status === 'running') {
    item.status = 'cancelled';
  }
  item.currentStage = null;
}

function failItem(state: PipelineState, item: PipelineItem, error: string): void {
  item.status = 'failed';
  item.error = error;
  item.currentStage = null;
  for (const stage of item.stages) {
    if (stage.status === 'pending') stage.status = 'cancelled';
  }
  persist(state);
  console.warn(`[Training] Pipeline ${state.id}: ${item.name} failed — ${error}`);
}

async function runItem(state: PipelineState, item: PipelineItem, spec: PipelineFolderSpec): Promise<void> {
  // ── 1. dataset create/reuse (§3.3) — direct service calls, not HTTP ────
  const sourceDir = path.resolve(item.sourceDir);
  if (!fs.existsSync(sourceDir) || !fs.statSync(sourceDir).isDirectory()) {
    failItem(state, item, `Folder does not exist: ${item.sourceDir}`);
    return;
  }

  const existing = repo.getDatasetBySourceDir(sourceDir);
  if (existing) {
    item.datasetId = existing.id;
    item.reusedExisting = true;
    item.name = existing.name;
  } else {
    item.status = 'creating';
    persist(state);
    try {
      const detail = await createDatasetFromFolder({
        name: item.name,
        sourceDir,
        ...(spec.customTag ? { customTag: spec.customTag } : {}),
      });
      item.datasetId = detail.id;
    } catch (err: any) {
      failItem(state, item, err?.message || String(err));
      return;
    }
  }

  // ── 2. stages ─────────────────────────────────────────────────────────
  item.status = 'running';
  persist(state);

  for (const result of item.stages) {
    if (state.cancelRequested) {
      cancelItem(item);
      persist(state);
      return;
    }
    await runStage(state, item, result);

    if (result.status !== 'done') {
      if (result.status === 'cancelled') {
        cancelItem(item);
        persist(state);
      } else {
        failItem(state, item, `${result.stage}: ${result.error || 'failed'}`);
      }
      return;
    }

    // ── Quality gate (§3.4): build and train would happily run on empty
    // captions, and the result is silent garbage nobody notices until it is
    // trained. Only a fresh scan can answer this.
    if (result.stage === 'label') {
      const gate = await labelGateError(item.datasetId);
      if (gate) {
        failItem(state, item, gate);
        return;
      }
    }
  }

  item.status = 'done';
  item.currentStage = null;
  persist(state);
}

/** `null` when the dataset has at least one caption after labeling. */
async function labelGateError(datasetId: string): Promise<string | null> {
  const ds = repo.getDataset(datasetId);
  if (!ds) return 'Dataset disappeared';
  try {
    const detail = await detailFor(ds);
    if (detail.labeledCount > 0) return null;
  } catch (err: any) {
    return `Could not re-scan after labeling: ${err?.message || err}`;
  }
  return 'Labeling produced no labeled samples — check labeling capabilities';
}

// ── One stage (§3.2) ─────────────────────────────────────────────────────

async function runStage(
  state: PipelineState,
  item: PipelineItem,
  result: PipelineStageResult,
): Promise<void> {
  result.status = 'running';
  result.startedAt = Date.now();
  item.currentStage = result.stage;
  persist(state);

  // §3.5 — a job the USER started on this dataset would make the route answer
  // 409. Wait it out instead of consuming that as a stage failure.
  await waitForDatasetIdle(state, item.datasetId);
  if (state.cancelRequested) {
    finishStage(state, result, 'cancelled', null);
    return;
  }

  const section = STAGE_DEFAULTS_KEY[result.stage];
  const body: Record<string, unknown> = section ? { ...getTrainingDefaults()[section] } : {};
  // A dataset that arrives already labelled is a DONE label stage, not a failed
  // one. Without this the route answers 400 "Nothing to label" and the item is
  // abandoned before preprocess/train ever run (2026-07-31).
  if (result.stage === 'label') body.allowEmpty = true;
  const url = `http://127.0.0.1:${config.server.port}`
    + `/api/training/datasets/${encodeURIComponent(item.datasetId)}/${STAGE_PATH[result.stage]}`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err: any) {
    finishStage(state, result, 'failed', err?.message || String(err));
    return;
  }

  if (!response.ok) {
    // The route's own messages are already user-legible — pass them through.
    finishStage(state, result, 'failed', await errorTextOf(response));
    return;
  }

  let jobId = '';
  let skipped = '';
  try {
    const payload = await response.json() as { jobId?: unknown; skipped?: unknown };
    if (typeof payload?.jobId === 'string') jobId = payload.jobId;
    if (typeof payload?.skipped === 'string') skipped = payload.skipped;
  } catch { /* handled by the empty check below */ }
  // A 200 with no jobId and an explicit `skipped` is a stage that had nothing to
  // do — complete, not failed, and the item carries on to the next stage.
  if (!jobId && skipped) {
    finishStage(state, result, 'done', null);
    return;
  }
  // Synchronous stages have already finished by the time they answer 200.
  if (SYNC_STAGES.has(result.stage)) {
    finishStage(state, result, 'done', null);
    return;
  }
  if (!jobId) {
    finishStage(state, result, 'failed', 'Stage returned no jobId');
    return;
  }
  result.jobId = jobId;
  persist(state);

  await pollStageJob(state, result, jobId);
}

async function waitForDatasetIdle(state: PipelineState, datasetId: string): Promise<void> {
  const deadline = Date.now() + IDLE_WAIT_MS;
  while (queue.activeJobForDataset(datasetId) && !state.cancelRequested && Date.now() < deadline) {
    await sleep(POLL_MS);
  }
}

async function pollStageJob(state: PipelineState, result: PipelineStageResult, jobId: string): Promise<void> {
  for (;;) {
    if (state.cancelRequested) queue.cancelJob(jobId);

    const job = queue.getJob(jobId);
    // cancelJob DELETES the job from the map (ours above, or a user hitting
    // DELETE /jobs/:id) — a job that was there a moment ago and is now gone was
    // cancelled, not lost.
    if (!job) {
      finishStage(state, result, 'cancelled', null);
      return;
    }
    if (job.status === 'done' || job.status === 'failed' || job.status === 'cancelled') {
      finishStage(state, result, job.status, job.error);
      return;
    }
    await sleep(POLL_MS);
  }
}

function finishStage(
  state: PipelineState,
  result: PipelineStageResult,
  status: 'done' | 'failed' | 'cancelled',
  error: string | null,
): void {
  result.status = status;
  result.error = status === 'done' ? null : error;
  result.finishedAt = Date.now();
  persist(state);
}

async function errorTextOf(response: Response): Promise<string> {
  try {
    const payload = await response.json() as { error?: unknown };
    if (typeof payload?.error === 'string' && payload.error) return payload.error;
  } catch { /* not a JSON body */ }
  return `Stage request failed (HTTP ${response.status})`;
}
