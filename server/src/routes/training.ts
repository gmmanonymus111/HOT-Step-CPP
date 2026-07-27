// training.ts — Dataset Studio HTTP surface (Training Studio phase 1)
//
// Import audio → local label (/understand + Essentia) → optional cloud enhance
// (Genius lyrics, LLM captions) → review/edit grid → build dataset.json.
//
// Disk is the source of truth (D3): sidecar .txt files next to the audio plus
// dataset.json in the source folder. SQLite holds ONE row per dataset for
// listing and status — never per-sample rows. Every GET re-scans.
//
// There is no global error middleware, so every handler is wrapped in
// try/catch and must respond, or the request hangs. Static routes are declared
// before any /:param routes.
//
// Mounts at: /api/training
// Routes:
//   GET    /capabilities                                — engine/Essentia/Genius/LLM probes
//   GET    /scan-preview                                — pre-create folder summary
//   GET    /datasets                                    — list (newest first)
//   POST   /datasets                                    — create
//   GET    /jobs                                        — active + finished jobs
//   GET    /jobs/:jobId                                 — poll a job
//   DELETE /jobs/:jobId                                 — cancel/forget a job
//   GET    /jobs/:jobId/stream                          — SSE: live job events
//   GET    /datasets/:id                                — detail (re-scans disk)
//   PATCH  /datasets/:id                                — settings
//   POST   /datasets/:id/rescan                         — pick up added/removed files
//   DELETE /datasets/:id                                — forget (never touches sourceDir)
//   POST   /datasets/:id/samples/bulk                   — bulk field set
//   PATCH  /datasets/:id/samples/:sampleId              — edit one sample
//   GET    /datasets/:id/samples/:sampleId/audio        — stream audio (Range aware)
//   POST   /datasets/:id/label                          — start a labeling job
//   POST   /datasets/:id/enhance/genius                 — start a Genius job
//   POST   /datasets/:id/enhance/caption                — start an LLM caption job
//   POST   /datasets/:id/build                          — start a build job
//   GET    /datasets/:id/dataset-json                   — read back the built file
//   POST   /datasets/:id/preprocess                     — start a tensor-cache job
//   GET    /datasets/:id/preprocess                     — tensor-cache status
//   DELETE /datasets/:id/preprocess/:variantKey         — delete one cache variant

import { Router, Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';
import { config } from '../config.js';
import { engineReady } from '../engineState.js';
import { aceClient } from '../services/aceClient.js';
import { listProviders, getProvider } from '../services/lireek/llm/registry.js';
import * as repo from '../services/training/datasetsRepo.js';
import {
  buildSamples, loadSidecarMetadata, sampleFromParts,
  scanPreview as scanPreviewFolder, ScanLimitError,
} from '../services/training/datasetScan.js';
import {
  isInside, labelsDir, slugify, trainingBaseDir, uniqueSlug,
} from '../services/training/paths.js';
import { deleteLabel, deleteLabels, patchLabel, readLabel } from '../services/training/labelStore.js';
import { readDatasetJsonMetadata } from '../services/training/datasetBuilder.js';
import { writeSidecar } from '../services/training/sidecarIO.js';
import { essentiaAvailable } from '../services/training/essentiaClient.js';
import { engineQueueDepth, engineUnderstandReady, pickBestLm } from '../services/training/understandClient.js';
import * as queue from '../services/training/labelingQueue.js';
import { isEngineSuspended } from '../services/aceEngineProcess.js';
import {
  aceTrainExe, getModelSnapshot, pickBf16, refreshModelSnapshot,
  tensorsDir, tensorsRoot, variantKeyFor, type ResolvedPreprocessOptions,
} from '../services/training/aceTrain.js';
import { countPreprocessedVariants, readPreprocessStatus } from '../services/training/preprocessStatus.js';
import type {
  BulkSetInput, CaptionOptions, FieldSource, GeniusOptions, LabelOptions, PatchSampleInput,
  PreprocessCompat, PreprocessDtype, PreprocessNormalize, PreprocessOptions,
  TrainingCapabilities, TrainingDatasetDetail, TrainingDatasetRow, TrainingSample,
} from '../services/training/types.js';

const router = Router();

const MIME_BY_EXT: Record<string, string> = {
  '.wav': 'audio/wav',
  '.mp3': 'audio/mpeg',
  '.flac': 'audio/flac',
  '.ogg': 'audio/ogg',
  '.opus': 'audio/ogg',
  '.m4a': 'audio/mp4',
  '.aac': 'audio/aac',
};

// ── Helpers ──────────────────────────────────────────────────────────────

/**
 * The dataset status the fresh scan implies (§2.0). Both the response body and
 * the DB cache are derived from this, so a GET can never report the pre-sync
 * value (e.g. 'labeling' for one call after a job ended).
 */
function computeStatus(ds: TrainingDatasetRow, samples: TrainingSample[]): TrainingDatasetRow['status'] {
  if (ds.status === 'built') return 'built';
  if (queue.activeJobForDataset(ds.id)) return 'labeling';
  return samples.some(s => !!s.caption.trim()) ? 'labeled' : 'draft';
}

/** Build the full detail payload — always a fresh scan (D3). */
async function detailFor(ds: TrainingDatasetRow): Promise<TrainingDatasetDetail> {
  const warnings: string[] = [];
  const samples = await buildSamples(ds, { warnings });

  const included = samples.filter(s => !s.excluded);
  if (included.length > 0 && included.length < 10) {
    warnings.push(`Only ${included.length} samples — 10+ recommended`);
  }
  const missing = included.filter(s => s.fileMissing).length;
  if (missing > 0) warnings.push(`${missing} file(s) are missing from disk`);

  const active = queue.activeJobForDataset(ds.id);

  let preprocessedVariants = 0;
  try { preprocessedVariants = countPreprocessedVariants(ds.slug); } catch { /* stays 0 */ }

  return {
    ...ds,
    // The DB counters are a cache; the fresh scan is what the client sees.
    sampleCount: samples.length,
    labeledCount: samples.filter(s => !!s.caption.trim()).length,
    excludedCount: samples.filter(s => s.excluded).length,
    status: computeStatus(ds, samples),
    samples,
    warnings,
    activeJobId: active ? active.id : null,
    preprocessedVariants,
  };
}

/** Refresh the cached counters after a scan. */
function syncCounters(ds: TrainingDatasetRow, samples: TrainingSample[]): void {
  const labeled = samples.filter(s => !!s.caption.trim()).length;
  const excluded = samples.filter(s => s.excluded).length;
  const status = computeStatus(ds, samples);
  try {
    repo.updateCounters(ds.id, {
      sampleCount: samples.length,
      labeledCount: labeled,
      excludedCount: excluded,
      status,
    });
  } catch (err: any) {
    console.warn(`[Training] Counter sync failed: ${err.message}`);
  }
}

/** Re-read one sample from disk after a write — avoids a full rescan per edit. */
function reloadSample(ds: TrainingDatasetRow, sample: TrainingSample): TrainingSample {
  let sizeBytes = sample.sizeBytes;
  let mtimeMs = 0;
  try {
    const st = fs.statSync(sample.audioPath);
    sizeBytes = st.size;
    mtimeMs = st.mtimeMs;
  } catch { /* file vanished — keep the sizes we already had */ }
  const label = readLabel(ds.slug, sample.sampleId) ?? undefined;
  const meta = loadSidecarMetadata(sample.audioPath);
  return sampleFromParts(
    ds.id,
    { relPath: sample.relPath, absPath: sample.audioPath, sizeBytes, mtimeMs },
    meta, label, label?.durationCache?.seconds ?? sample.duration, sample.fileMissing,
  );
}

/** Sidecar keys for the editable sample fields (§6.4 write whitelist). */
const SIDECAR_KEY_OF: Record<string, string> = {
  caption: 'caption',
  genre: 'genre',
  bpm: 'bpm',
  key: 'key',
  signature: 'signature',
  language: 'language',
  isInstrumental: 'is_instrumental',
  lyrics: 'lyrics',
  customTag: 'custom_tag',
  repeat: 'repeat',
  promptOverride: 'prompt_override',
};

/** Sample fields whose provenance the contract tracks (§2.0 `sources`). */
const SOURCE_TRACKED_FIELDS: ReadonlySet<string> = new Set([
  'caption', 'lyrics', 'genre', 'bpm', 'key', 'signature', 'language',
]);

function metaString(meta: Record<string, unknown>, key: string): string {
  const v = meta[key];
  return typeof v === 'string' ? v : '';
}

function isWritableDir(dir: string): boolean {
  const probe = path.join(dir, '.hotstep-write-test');
  try {
    fs.writeFileSync(probe, 'x');
    fs.unlinkSync(probe);
    return true;
  } catch {
    return false;
  }
}

// ── Capabilities (§2.1) ──────────────────────────────────────────────────

router.get('/capabilities', async (_req: Request, res: Response) => {
  // Every probe degrades independently — this endpoint never throws upward.
  const caps: TrainingCapabilities = {
    engine: { up: false, ready: engineReady, understandSupported: false, missingModels: [], queueDepth: 0, lmModels: [], defaultLmModel: '' },
    essentia: { available: false, binPath: config.essentia.bin },
    genius: { configured: false },
    llm: { configured: false, defaultProvider: config.lireek.defaultProvider, providers: [] },
    preprocess: {
      available: false, binPath: '', ditModels: [], vaeModels: [], textEncoders: [],
      defaultDit: '', defaultVae: '', defaultTextEnc: '', modelsCachedAt: 0,
      engineSuspended: false,
    },
  };

  try { caps.engine.up = await aceClient.isReachable(); } catch { /* stays false */ }
  if (caps.engine.up) {
    try {
      const ready = await engineUnderstandReady();
      caps.engine.understandSupported = ready.ok;
      caps.engine.missingModels = ready.missing;
      caps.engine.lmModels = ready.lmModels;
      caps.engine.defaultLmModel = pickBestLm(ready.lmModels);
    } catch { caps.engine.missingModels = ['lm', 'dit', 'vae']; }
    try { caps.engine.queueDepth = await engineQueueDepth(); } catch { /* stays 0 */ }
  }

  try { caps.essentia.available = essentiaAvailable(); } catch { /* stays false */ }
  try { caps.genius.configured = !!config.lireek.geniusAccessToken; } catch { /* stays false */ }

  try {
    const providers = await listProviders();
    caps.llm.providers = providers.map(p => ({
      id: p.id,
      name: p.name,
      available: p.available,
      models: p.models,
      defaultModel: p.default_model,
    }));
    caps.llm.configured = caps.llm.providers.some(p => p.available);
  } catch (err: any) {
    console.warn(`[Training] Provider probe failed: ${err.message}`);
  }

  // ── Preprocess (phase 2). Every probe degrades independently; the model
  // lists come from a CACHED /props snapshot so the picker survives the engine
  // being stopped by a running preprocess job (P28).
  try {
    const exe = aceTrainExe();
    caps.preprocess.available = !!exe;
    caps.preprocess.binPath = exe ?? '';
  } catch { /* stays unavailable */ }
  try { caps.preprocess.engineSuspended = isEngineSuspended(); } catch { /* stays false */ }
  try {
    const snap = caps.engine.up ? await refreshModelSnapshot() : getModelSnapshot();
    caps.preprocess.ditModels = snap.dit;
    caps.preprocess.vaeModels = snap.vae;
    caps.preprocess.textEncoders = snap.textEnc;
    caps.preprocess.defaultDit = pickBf16(snap.dit);
    caps.preprocess.defaultVae = pickBf16(snap.vae) || snap.vae[0] || '';
    caps.preprocess.defaultTextEnc = snap.textEnc[0] || '';
    caps.preprocess.modelsCachedAt = snap.cachedAt;
  } catch { /* stays empty */ }

  res.json(caps);
});

// ── Scan preview (§2.2) ──────────────────────────────────────────────────

router.get('/scan-preview', (req: Request, res: Response) => {
  try {
    const raw = (req.query.path as string) || '';
    if (!raw.trim()) {
      res.status(400).json({ error: 'path is required' });
      return;
    }
    let root: string;
    try { root = path.resolve(raw); } catch {
      res.status(400).json({ error: 'Invalid path' });
      return;
    }
    if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) {
      res.status(404).json({ error: 'Directory not found' });
      return;
    }
    const recursive = req.query.recursive !== '0' && req.query.recursive !== 'false';
    res.json(scanPreviewFolder(root, recursive));
  } catch (err: any) {
    if (err instanceof ScanLimitError) {
      res.status(400).json({ error: err.message });
      return;
    }
    console.error(`[Training] scan-preview failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Dataset list / create (§2.3) ─────────────────────────────────────────

router.get('/datasets', (_req: Request, res: Response) => {
  try {
    res.json({ datasets: repo.listDatasets() });
  } catch (err: any) {
    console.error(`[Training] List datasets failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.post('/datasets', async (req: Request, res: Response) => {
  try {
    const body = req.body || {};
    const name = typeof body.name === 'string' ? body.name.trim() : '';
    const rawDir = typeof body.sourceDir === 'string' ? body.sourceDir.trim() : '';

    if (!name || name.length > 100) {
      res.status(400).json({ error: 'name is required' });
      return;
    }
    if (!rawDir) {
      res.status(400).json({ error: 'sourceDir is required' });
      return;
    }

    const sourceDir = path.resolve(rawDir);
    if (!fs.existsSync(sourceDir) || !fs.statSync(sourceDir).isDirectory()) {
      res.status(400).json({ error: 'sourceDir is not a directory' });
      return;
    }
    if (!isWritableDir(sourceDir)) {
      res.status(400).json({ error: 'sourceDir is not writable — sidecars cannot be saved there' });
      return;
    }
    if (repo.getDatasetBySourceDir(sourceDir)) {
      res.status(409).json({ error: 'A dataset already exists for this folder' });
      return;
    }

    const recursive = body.recursive !== false;
    const preview = scanPreviewFolder(sourceDir, recursive);
    if (preview.audioFiles === 0) {
      res.status(400).json({ error: 'No audio files found in sourceDir' });
      return;
    }

    const slug = uniqueSlug(slugify(name), repo.listSlugs());

    // §8.5 — a folder that already has a dataset.json owns these settings.
    // Seeding the row from it is what makes "existing labels will be imported
    // and kept" true: the builder rewrites the same values instead of blanking
    // custom_tag / genre_ratio / default_artist / default_album on the first build.
    const priorMeta = readDatasetJsonMetadata(sourceDir);
    const priorTagPosition = metaString(priorMeta, 'tag_position');
    const priorGenreRatio = Number(priorMeta.genre_ratio);

    const now = new Date().toISOString();
    const row: TrainingDatasetRow = {
      id: randomUUID(),
      slug,
      name,
      sourceDir,
      recursive,
      customTag: typeof body.customTag === 'string'
        ? body.customTag.trim()
        : (metaString(priorMeta, 'custom_tag') || slug),
      tagPosition: ['prepend', 'append', 'replace'].includes(body.tagPosition)
        ? body.tagPosition
        : (['prepend', 'append', 'replace'].includes(priorTagPosition)
          ? priorTagPosition as TrainingDatasetRow['tagPosition']
          : 'prepend'),
      genreRatio: Number.isFinite(Number(body.genreRatio))
        ? Math.min(100, Math.max(0, Math.trunc(Number(body.genreRatio))))
        : (Number.isFinite(priorGenreRatio) ? Math.min(100, Math.max(0, Math.trunc(priorGenreRatio))) : 30),
      defaultArtist: typeof body.defaultArtist === 'string' ? body.defaultArtist : metaString(priorMeta, 'default_artist'),
      defaultAlbum: typeof body.defaultAlbum === 'string' ? body.defaultAlbum : metaString(priorMeta, 'default_album'),
      defaultGenre: typeof body.defaultGenre === 'string' ? body.defaultGenre : metaString(priorMeta, 'default_genre'),
      defaultLanguage: typeof body.defaultLanguage === 'string' && body.defaultLanguage.trim()
        ? body.defaultLanguage.trim().toLowerCase()
        : (metaString(priorMeta, 'default_language') || 'english'),
      sampleCount: preview.audioFiles,
      labeledCount: preview.withCaption,
      excludedCount: 0,
      status: 'draft',
      builtAt: '',
      datasetJsonPath: '',
      createdAt: now,
      updatedAt: now,
    };

    repo.insertDataset(row);
    fs.mkdirSync(labelsDir(slug), { recursive: true });

    const detail = await detailFor(row);
    syncCounters(row, detail.samples);
    console.log(`[Training] Created dataset "${name}" (${slug}) — ${detail.samples.length} files in ${sourceDir}`);
    res.status(201).json({ dataset: detail });
  } catch (err: any) {
    if (err instanceof ScanLimitError) {
      res.status(400).json({ error: err.message });
      return;
    }
    console.error(`[Training] Create dataset failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Job control (§2.8, §2.9) — declared before /datasets/:id ──────────────

router.get('/jobs', (req: Request, res: Response) => {
  try {
    const datasetId = typeof req.query.datasetId === 'string' ? req.query.datasetId : undefined;
    res.json({ jobs: queue.listJobs(datasetId) });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.get('/jobs/:jobId', (req: Request, res: Response) => {
  try {
    const job = queue.getJob(req.params.jobId as string);
    if (job) {
      res.json(queue.toSummary(job));
      return;
    }
    const finished = queue.listJobs().find(j => j.id === req.params.jobId);
    if (!finished) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }
    res.json(finished);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.delete('/jobs/:jobId', (req: Request, res: Response) => {
  try {
    const jobId = req.params.jobId as string;
    if (!queue.cancelJob(jobId)) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }
    res.json({ ok: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.get('/jobs/:jobId/stream', (req: Request, res: Response) => {
  try {
    const job = queue.getJob(req.params.jobId as string);
    if (!job) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }
    queue.attachStream(job, res);
  } catch (err: any) {
    if (!res.headersSent) res.status(500).json({ error: err.message });
    else { try { res.end(); } catch { /* already closed */ } }
  }
});

// ── Dataset detail / patch / rescan / delete (§2.3) ───────────────────────

router.get('/datasets/:id', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const detail = await detailFor(ds);
    syncCounters(ds, detail.samples);
    res.json(detail);
  } catch (err: any) {
    if (err instanceof ScanLimitError) {
      res.status(400).json({ error: err.message });
      return;
    }
    console.error(`[Training] Dataset detail failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.patch('/datasets/:id', (req: Request, res: Response) => {
  try {
    const id = req.params.id as string;
    const ds = repo.getDataset(id);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const body = req.body || {};
    const patch: Partial<TrainingDatasetRow> = {};

    if (typeof body.name === 'string') {
      const name = body.name.trim();
      if (!name || name.length > 100) {
        res.status(400).json({ error: 'name is required' });
        return;
      }
      patch.name = name;
    }
    if (typeof body.customTag === 'string') patch.customTag = body.customTag.trim();
    if (typeof body.tagPosition === 'string') {
      if (!['prepend', 'append', 'replace'].includes(body.tagPosition)) {
        res.status(400).json({ error: 'Invalid tagPosition' });
        return;
      }
      patch.tagPosition = body.tagPosition;
    }
    if (body.genreRatio !== undefined) {
      const n = Number(body.genreRatio);
      if (!Number.isFinite(n)) {
        res.status(400).json({ error: 'genreRatio must be a number' });
        return;
      }
      patch.genreRatio = Math.min(100, Math.max(0, Math.trunc(n)));
    }
    if (typeof body.defaultArtist === 'string') patch.defaultArtist = body.defaultArtist;
    if (typeof body.defaultAlbum === 'string') patch.defaultAlbum = body.defaultAlbum;
    if (typeof body.defaultGenre === 'string') patch.defaultGenre = body.defaultGenre;
    if (typeof body.defaultLanguage === 'string') patch.defaultLanguage = body.defaultLanguage.trim().toLowerCase();
    if (typeof body.recursive === 'boolean') patch.recursive = body.recursive;

    // slug and sourceDir are immutable after creation.
    repo.updateDataset(id, patch);
    const updated = repo.getDataset(id);
    res.json({ dataset: updated });
  } catch (err: any) {
    console.error(`[Training] Patch dataset failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.post('/datasets/:id/rescan', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is running for this dataset' });
      return;
    }

    let detail = await detailFor(ds);

    // §7.4 — prune orphaned label records only after TWO consecutive misses;
    // a disconnected network drive must not wipe the user's edits.
    const now = Date.now();
    let pruned = false;
    for (const sample of detail.samples) {
      const rec = readLabel(ds.slug, sample.sampleId);
      if (!rec) continue;
      if (!sample.fileMissing) {
        if (rec.missingSince) patchLabel(ds.slug, sample.sampleId, { missingSince: null });
        continue;
      }
      if (!rec.missingSince) {
        patchLabel(ds.slug, sample.sampleId, { missingSince: now });
        continue;
      }
      // Second consecutive miss — the ghost row goes, and with it the Build block.
      deleteLabel(ds.slug, sample.sampleId);
      pruned = true;
    }
    if (pruned) detail = await detailFor(ds);

    syncCounters(ds, detail.samples);
    res.json(detail);
  } catch (err: any) {
    if (err instanceof ScanLimitError) {
      res.status(400).json({ error: err.message });
      return;
    }
    console.error(`[Training] Rescan failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.delete('/datasets/:id', (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is running for this dataset' });
      return;
    }
    // D20: the DB row and data/training/datasets/<slug>/ only. The user's audio,
    // sidecars and dataset.json are their files and are never touched.
    repo.deleteDataset(ds.id);
    deleteLabels(ds.slug);
    console.log(`[Training] Deleted dataset ${ds.slug} (source folder untouched: ${ds.sourceDir})`);
    res.json({ ok: true });
  } catch (err: any) {
    console.error(`[Training] Delete dataset failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Sample edits (§2.4) ──────────────────────────────────────────────────

/** Apply a patch to one sample: sidecar fields to disk, `excluded` to the label. */
async function applySamplePatch(
  ds: TrainingDatasetRow,
  sample: TrainingSample,
  patch: PatchSampleInput,
): Promise<TrainingSample> {
  const meta = loadSidecarMetadata(sample.audioPath);
  const sources: Record<string, FieldSource> = {};
  let touchedSidecar = false;

  for (const [field, value] of Object.entries(patch)) {
    if (value === undefined) continue;
    if (field === 'excluded') continue;
    const key = SIDECAR_KEY_OF[field];
    if (!key) continue;

    if (value === null) meta[key] = '';
    else if (typeof value === 'boolean') meta[key] = value ? 'true' : 'false';
    else meta[key] = String(value);

    touchedSidecar = true;
    // §2.0 types `sources` over seven field names only — editing
    // isInstrumental/customTag/promptOverride must not inject keys outside it.
    if (SOURCE_TRACKED_FIELDS.has(key)) sources[key] = 'user';
  }

  if (touchedSidecar) await writeSidecar(sample.sidecarPath, meta);

  patchLabel(ds.slug, sample.sampleId, {
    relPath: sample.relPath,
    sources,
    ...(typeof patch.excluded === 'boolean' ? { excluded: patch.excluded } : {}),
  });

  return reloadSample(ds, sample);
}

router.post('/datasets/:id/samples/bulk', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const body = req.body || {};
    const sampleIds: string[] = Array.isArray(body.sampleIds) ? body.sampleIds.filter((s: unknown) => typeof s === 'string') : [];
    const set = (body.set || {}) as BulkSetInput;
    if (sampleIds.length === 0) {
      res.status(400).json({ error: 'sampleIds is required' });
      return;
    }
    if (Object.keys(set).length === 0) {
      res.status(400).json({ error: 'set is empty' });
      return;
    }

    const samples = await buildSamples(ds);
    const byId = new Map(samples.map(s => [s.sampleId, s]));
    const failed: Array<{ sampleId: string; error: string }> = [];
    let updated = 0;

    for (const sampleId of sampleIds) {
      const sample = byId.get(sampleId);
      if (!sample) {
        failed.push({ sampleId, error: 'Sample not found' });
        continue;
      }
      if (sample.fileMissing && Object.keys(set).some(k => k !== 'excluded')) {
        failed.push({ sampleId, error: 'Audio file is missing from disk' });
        continue;
      }
      try {
        await applySamplePatch(ds, sample, set as PatchSampleInput);
        updated++;
      } catch (err: any) {
        failed.push({ sampleId, error: err.message });
      }
    }

    res.json({ updated, failed });
  } catch (err: any) {
    console.error(`[Training] Bulk edit failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.patch('/datasets/:id/samples/:sampleId', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const body = (req.body || {}) as PatchSampleInput;
    const editable = Object.keys(body).filter(
      k => (k in SIDECAR_KEY_OF || k === 'excluded') && (body as Record<string, unknown>)[k] !== undefined,
    );
    if (editable.length === 0) {
      res.status(400).json({ error: 'No editable fields in body' });
      return;
    }
    if (body.repeat !== undefined && (!Number.isFinite(Number(body.repeat)) || Number(body.repeat) < 1)) {
      res.status(400).json({ error: 'repeat must be >= 1' });
      return;
    }

    const samples = await buildSamples(ds);
    const sample = samples.find(s => s.sampleId === req.params.sampleId);
    if (!sample) {
      res.status(404).json({ error: 'Sample not found' });
      return;
    }
    if (sample.fileMissing) {
      res.status(409).json({ error: 'Audio file is missing from disk' });
      return;
    }

    const updated = await applySamplePatch(ds, sample, body);
    res.json({ sample: updated });
  } catch (err: any) {
    console.error(`[Training] Sample patch failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/datasets/:id/samples/:sampleId/audio', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Sample not found' });
      return;
    }
    const samples = await buildSamples(ds);
    const sample = samples.find(s => s.sampleId === req.params.sampleId);
    // The path always comes from our own scan, never from the client (§7.8).
    if (!sample || !isInside(ds.sourceDir, sample.audioPath)) {
      res.status(404).json({ error: 'Sample not found' });
      return;
    }
    if (!fs.existsSync(sample.audioPath)) {
      res.status(404).json({ error: 'Audio file is missing from disk' });
      return;
    }

    const stat = fs.statSync(sample.audioPath);
    const mime = MIME_BY_EXT[path.extname(sample.audioPath).toLowerCase()] || 'application/octet-stream';
    res.setHeader('Content-Type', mime);
    res.setHeader('Accept-Ranges', 'bytes');
    res.setHeader('Cache-Control', 'no-store');

    const range = req.headers.range;
    if (range) {
      const m = /bytes=(\d*)-(\d*)/.exec(range);
      if (m) {
        const start = m[1] ? parseInt(m[1], 10) : 0;
        const end = m[2] ? parseInt(m[2], 10) : stat.size - 1;
        if (Number.isFinite(start) && start < stat.size && end >= start) {
          const last = Math.min(end, stat.size - 1);
          res.status(206);
          res.setHeader('Content-Range', `bytes ${start}-${last}/${stat.size}`);
          res.setHeader('Content-Length', last - start + 1);
          fs.createReadStream(sample.audioPath, { start, end: last }).pipe(res);
          return;
        }
      }
    }

    res.setHeader('Content-Length', stat.size);
    fs.createReadStream(sample.audioPath).pipe(res);
  } catch (err: any) {
    console.error(`[Training] Audio stream failed: ${err.message}`);
    if (!res.headersSent) res.status(500).json({ error: err.message });
  }
});

// ── Labeling job (§2.5) ──────────────────────────────────────────────────

/** Resolve the sample ids a job should work on. */
function pickTargets(
  samples: TrainingSample[],
  sampleIds: unknown,
  scope: 'all' | 'unlabeled',
  extraFilter?: (s: TrainingSample) => boolean,
): string[] {
  if (Array.isArray(sampleIds) && sampleIds.length > 0) {
    const wanted = new Set(sampleIds.filter((s): s is string => typeof s === 'string'));
    return samples.filter(s => wanted.has(s.sampleId) && !s.fileMissing).map(s => s.sampleId);
  }
  return samples
    .filter(s => !s.excluded && !s.fileMissing)
    .filter(s => (scope === 'all' ? true : !s.caption.trim()))
    .filter(s => (extraFilter ? extraFilter(s) : true))
    .map(s => s.sampleId);
}

router.post('/datasets/:id/label', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is already running for this dataset' });
      return;
    }

    const body = (req.body || {}) as LabelOptions;
    const useEssentia = body.useEssentia !== false;
    // 2026-07-27 pivot: understand is LEGACY and opt-in; the default flow is
    // Essentia + Genius + LLM caption, all engine-free.
    const useUnderstand = body.useUnderstand === true;
    const useGenius = body.useGenius === true;
    const useCaption = body.useCaption === true;
    if (!useEssentia && !useUnderstand && !useGenius && !useCaption) {
      res.status(400).json({ error: 'No labeling steps enabled' });
      return;
    }

    if (useGenius && !config.lireek.geniusAccessToken) {
      res.status(503).json({ error: 'GENIUS_ACCESS_TOKEN is not set' });
      return;
    }
    if (useCaption) {
      const providerName = body.caption?.provider || config.lireek.defaultProvider;
      let provider;
      try {
        provider = getProvider(providerName);
      } catch (err: any) {
        res.status(400).json({ error: err.message });
        return;
      }
      if (!provider.isAvailable()) {
        res.status(503).json({
          error: `Provider ${providerName} is not available. Check API keys in Settings → AI Services.`,
        });
        return;
      }
    }

    if (useUnderstand) {
      let ready: { ok: boolean; missing: string[]; lmModels: string[] };
      try {
        ready = await engineUnderstandReady();
      } catch {
        res.status(503).json({ error: 'Engine is not running' });
        return;
      }
      if (!ready.ok) {
        res.status(503).json({ error: `Engine is missing models: ${ready.missing.join(', ')}` });
        return;
      }
      // Without an explicit model the engine falls back to whatever is loaded,
      // else the alphabetically-first registry entry — the 0.6B. Always pin the
      // best LM unless the caller chose one.
      if (!body.understand?.lmModel) {
        const best = pickBestLm(ready.lmModels);
        if (best) body.understand = { ...body.understand, lmModel: best };
      }
    }

    const samples = await buildSamples(ds);
    const scope = body.scope === 'all' ? 'all' : 'unlabeled';
    const targets = pickTargets(samples, body.sampleIds, scope);
    if (targets.length === 0) {
      res.status(400).json({ error: 'Nothing to label' });
      return;
    }

    const job = queue.startLabelJob(ds.id, targets, {
      ...body,
      useEssentia,
      useUnderstand,
      useGenius,
      useCaption,
      mergePolicy: body.mergePolicy || 'fill_missing',
    });
    repo.updateDataset(ds.id, { status: 'labeling' });
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] Label start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Enhance jobs (§2.6) ──────────────────────────────────────────────────

router.post('/datasets/:id/enhance/genius', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is already running for this dataset' });
      return;
    }
    if (!config.lireek.geniusAccessToken) {
      res.status(503).json({ error: 'GENIUS_ACCESS_TOKEN is not set' });
      return;
    }

    const body = (req.body || {}) as GeniusOptions;
    const samples = await buildSamples(ds);
    const targets = pickTargets(samples, body.sampleIds, 'all', s => !s.isInstrumental);
    if (targets.length === 0) {
      res.status(400).json({ error: 'Nothing to enhance' });
      return;
    }

    const job = queue.startGeniusJob(ds.id, targets, {
      ...body,
      mergePolicy: body.mergePolicy || 'overwrite_lyrics',
      sanitizeHeaders: body.sanitizeHeaders !== false,
    });
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] Genius start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.post('/datasets/:id/enhance/caption', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is already running for this dataset' });
      return;
    }

    const body = (req.body || {}) as CaptionOptions;
    const providerName = body.provider || config.lireek.defaultProvider;
    let provider;
    try {
      provider = getProvider(providerName);
    } catch (err: any) {
      res.status(400).json({ error: err.message });
      return;
    }
    if (!provider.isAvailable()) {
      res.status(503).json({
        error: `Provider ${providerName} is not available. Check API keys in Settings → AI Services.`,
      });
      return;
    }

    const samples = await buildSamples(ds);
    const targets = pickTargets(samples, body.sampleIds, 'all');
    if (targets.length === 0) {
      res.status(400).json({ error: 'Nothing to enhance' });
      return;
    }

    const job = queue.startCaptionJob(ds.id, targets, {
      ...body,
      provider: providerName,
      mergePolicy: body.mergePolicy || 'overwrite_caption',
      includeLyricsExcerpt: body.includeLyricsExcerpt !== false,
      temperature: typeof body.temperature === 'number' ? body.temperature : 0.45,
    });
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] Caption start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Build (§2.7) ─────────────────────────────────────────────────────────

router.post('/datasets/:id/build', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is already running for this dataset' });
      return;
    }

    const raw = typeof req.body?.outputPath === 'string' ? req.body.outputPath.trim() : '';
    let outputPath = path.join(ds.sourceDir, 'dataset.json');
    if (raw) {
      const resolved = path.resolve(raw);
      if (!isInside(ds.sourceDir, resolved) && !isInside(trainingBaseDir, resolved)) {
        res.status(400).json({ error: 'outputPath must be inside the dataset folder' });
        return;
      }
      outputPath = resolved;
    }

    const samples = await buildSamples(ds);
    if (samples.filter(s => !s.excluded).length === 0) {
      res.status(400).json({ error: 'Dataset has no includable samples' });
      return;
    }

    const job = queue.startBuildJob(ds.id, { outputPath });
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] Build start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/datasets/:id/dataset-json', (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const target = ds.datasetJsonPath || path.join(ds.sourceDir, 'dataset.json');
    if (!ds.datasetJsonPath || !fs.existsSync(target)) {
      res.status(404).json({ error: 'dataset.json has not been built yet' });
      return;
    }
    const dataset = JSON.parse(fs.readFileSync(target, 'utf-8')) as unknown;
    res.json({ path: target, builtAt: ds.builtAt, dataset });
  } catch (err: any) {
    console.error(`[Training] dataset-json read failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Preprocess (§2.8, phase 2) ───────────────────────────────────────────

/** Clamp a numeric option to its default when the client omitted it. */
function numOpt(value: unknown, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

router.post('/datasets/:id/preprocess', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is already running for this dataset' });
      return;
    }
    // The Build step is what produces dataset.json — and with it the stable
    // sample ids the tensor cache filenames are keyed on.
    if (!ds.builtAt || !ds.datasetJsonPath || !fs.existsSync(ds.datasetJsonPath)) {
      res.status(400).json({ error: 'Dataset must be built first — run Build before Preprocess' });
      return;
    }
    if (!aceTrainExe()) {
      res.status(503).json({ error: 'ace-train was not found next to ace-server — rebuild the engine' });
      return;
    }

    const body = (req.body || {}) as PreprocessOptions;

    const samples = await buildSamples(ds);
    const wanted = Array.isArray(body.sampleIds)
      ? new Set(body.sampleIds.filter((s): s is string => typeof s === 'string'))
      : null;
    const targets = samples
      .filter(s => !s.excluded && !s.fileMissing)
      .filter(s => (wanted ? wanted.has(s.sampleId) : true))
      .map(s => s.sampleId);
    if (targets.length === 0) {
      res.status(400).json({ error: 'Dataset has no includable samples' });
      return;
    }

    // Models come from the CACHED snapshot — the engine may be stopped by a job
    // on another dataset. An empty snapshot means /props was never read, in
    // which case an explicit name is passed straight through to ace-train.
    // A never-probed snapshot (fresh boot, or a /capabilities read that raced the
    // engine coming up) would otherwise 400 with 'No DiT base model available'
    // even though the engine is reachable — probe once instead of rejecting.
    let snap = getModelSnapshot();
    if (!snap.cachedAt && !isEngineSuspended()) snap = await refreshModelSnapshot();
    const dit = (typeof body.ditModel === 'string' ? body.ditModel.trim() : '') || pickBf16(snap.dit);
    if (!dit) {
      res.status(400).json({ error: 'No DiT base model available' });
      return;
    }
    if (body.ditModel && snap.dit.length > 0 && !snap.dit.includes(dit)) {
      res.status(400).json({ error: `Unknown DiT model: ${dit}` });
      return;
    }
    const vae = (typeof body.vaeModel === 'string' ? body.vaeModel.trim() : '')
      || pickBf16(snap.vae) || snap.vae[0] || '';
    const textEnc = (typeof body.textEncoder === 'string' ? body.textEncoder.trim() : '')
      || snap.textEnc[0] || '';

    const maxDuration = numOpt(body.maxDuration, 240);
    const vaeChunk = numOpt(body.vaeChunk, 384);
    const vaeOverlap = numOpt(body.vaeOverlap, 48);
    const maxCaptionTokens = numOpt(body.maxCaptionTokens, 256);
    const maxLyricTokens = numOpt(body.maxLyricTokens, 512);
    const targetDb = numOpt(body.targetDb, -1.0);

    if (maxDuration < 0) {
      res.status(400).json({ error: 'maxDuration must be >= 0' });
      return;
    }
    if (vaeChunk < 64) {
      res.status(400).json({ error: 'vaeChunk must be >= 64' });
      return;
    }
    if (vaeOverlap < 0 || vaeOverlap >= vaeChunk) {
      res.status(400).json({ error: 'vaeOverlap must be >= 0 and less than vaeChunk' });
      return;
    }
    if (maxCaptionTokens < 16 || maxCaptionTokens > 4096) {
      res.status(400).json({ error: 'maxCaptionTokens must be between 16 and 4096' });
      return;
    }
    if (maxLyricTokens < 16 || maxLyricTokens > 4096) {
      res.status(400).json({ error: 'maxLyricTokens must be between 16 and 4096' });
      return;
    }
    if (targetDb < -60 || targetDb > 0) {
      res.status(400).json({ error: 'targetDb must be between -60 and 0' });
      return;
    }

    const variantKey = variantKeyFor(dit);
    let outputDir = tensorsDir(ds.slug, dit);
    if (typeof body.outputDir === 'string' && body.outputDir.trim()) {
      // Containment, same rule the sibling DELETE handler applies (§7.8). This
      // path is mkdir'd, ace-train creates <out>/.tmp/ in it and deletes orphan
      // *.__writing__ files there, so an unchecked absolute path from the
      // request body is a write primitive. Staying under the dataset's tensors
      // root also keeps the cache visible to GET/DELETE .../preprocess — a
      // cache written anywhere else is unmanaged disk the UI can never see.
      const root = tensorsRoot(ds.slug);
      const resolved = path.resolve(body.outputDir.trim());
      if (!isInside(root, resolved) || path.resolve(root) === resolved) {
        res.status(400).json({ error: `outputDir must be a subdirectory of ${root}` });
        return;
      }
      outputDir = resolved;
    }

    const opts: ResolvedPreprocessOptions = {
      ditModel: dit,
      vaeModel: vae,
      textEncoder: textEnc,
      maxDuration: Math.trunc(maxDuration),
      normalize: (body.normalize === 'none' ? 'none' : 'peak') as PreprocessNormalize,
      targetDb,
      dtype: (body.dtype === 'bf16' ? 'bf16' : 'f32') as PreprocessDtype,
      compat: (body.compat === 'sidestep' ? 'sidestep' : 'hotstep') as PreprocessCompat,
      maxCaptionTokens: Math.trunc(maxCaptionTokens),
      maxLyricTokens: Math.trunc(maxLyricTokens),
      vaeChunk: Math.trunc(vaeChunk),
      vaeOverlap: Math.trunc(vaeOverlap),
      overwrite: body.overwrite === true,
      stopEngine: body.stopEngine !== false,
      outputDir,
      variantKey,
    };

    const job = queue.startPreprocessJob(ds.id, targets, opts);
    console.log(`[Training] Preprocess job ${job.id} queued — ${targets.length} songs, variant ${variantKey}`);
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] Preprocess start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/datasets/:id/preprocess', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    res.json(readPreprocessStatus(ds, await buildSamples(ds)));
  } catch (err: any) {
    console.error(`[Training] Preprocess status failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.delete('/datasets/:id/preprocess/:variantKey', (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is running for this dataset' });
      return;
    }
    const root = tensorsRoot(ds.slug);
    const dir = path.join(root, path.basename(String(req.params.variantKey ?? '')));
    // §7.8 — the client-supplied key never escapes the dataset's tensors root.
    if (!isInside(root, dir) || root === path.resolve(dir) || !fs.existsSync(dir)) {
      res.status(404).json({ error: 'Variant not found' });
      return;
    }
    fs.rmSync(dir, { recursive: true, force: true });
    console.log(`[Training] Deleted tensor cache ${dir}`);
    res.json({ ok: true });
  } catch (err: any) {
    console.error(`[Training] Preprocess delete failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

export default router;
