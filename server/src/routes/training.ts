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
//   POST   /datasets/:id/train-lm                       — start an LM LoRA training job
//   GET    /datasets/:id/train-lm                       — LM adapter / codes status
//   POST   /datasets/:id/train-dit                      — start a DiT LoRA training job
//   GET    /datasets/:id/train-dit                      — DiT adapter / variant status

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
  aceTrainExe, getModelSnapshot, pickBf16, pickDitBaseFor, pickLmFor, refreshModelSnapshot,
  tensorsDir, tensorsRoot, variantKeyFor,
  type ResolvedPreprocessOptions, type ResolvedTrainDitOptions, type ResolvedTrainLmOptions,
} from '../services/training/aceTrain.js';
import { countPreprocessedVariants, readPreprocessStatus } from '../services/training/preprocessStatus.js';
import {
  adapterDirFor, adapterLmRoot, newestVariantKey, readTrainLmStatus,
  safeAdapterName, variantDitModel, variantExists,
} from '../services/training/trainLmStatus.js';
import {
  adapterDitDirFor, adapterDitRoot, readTrainDitStatus,
} from '../services/training/trainDitStatus.js';
import type {
  BulkSetInput, CaptionOptions, FieldSource, GeniusOptions, LabelOptions, LmSize,
  PatchSampleInput, PreprocessCompat, PreprocessDtype, PreprocessNormalize, PreprocessOptions,
  TrainingCapabilities, TrainingDatasetDetail, TrainingDatasetRow, TrainingSample,
  TrainDitOptions, TrainDitStage, TrainLmOptions, TrainLmStage,
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
    trainLm: {
      available: false, lmModels: [], sizes: ['0.6B', '1.7B', '4B'],
      defaultLmBySize: { '0.6B': '', '1.7B': '', '4B': '' }, adaptersRoot: '',
    },
    trainDit: {
      available: false, adapterTypes: ['lora'], adaptersRoot: '', minVramMb: 16384,
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

  // ── LM trainer (phase 3). Same rules: every probe individually caught, the
  // model list comes from the cached snapshot so the picker survives the engine
  // being stopped by a running training job.
  try { caps.trainLm.available = !!aceTrainExe(); } catch { /* stays false */ }
  try { caps.trainLm.adaptersRoot = adapterLmRoot(); } catch { /* stays '' */ }
  try {
    const snap = getModelSnapshot();
    caps.trainLm.lmModels = snap.lm;
    caps.trainLm.defaultLmBySize = {
      '0.6B': pickLmFor('0.6B', snap.lm),
      '1.7B': pickLmFor('1.7B', snap.lm),
      '4B': pickLmFor('4B', snap.lm),
    };
  } catch { /* stays empty */ }

  // ── DiT trainer (phase 4). No model list: the base is forced to the one the
  // chosen preprocess variant was made against (§4.2 base-match guard), so
  // there is nothing for the UI to pick. `minVramMb` is ADVISORY — the real
  // gate is ace-train's own footprint solve, which can only run once the base
  // is loaded and the engine is already down (§4.5).
  try { caps.trainDit.available = !!aceTrainExe(); } catch { /* stays false */ }
  try { caps.trainDit.adaptersRoot = adapterDitRoot(); } catch { /* stays '' */ }

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

// ── LM LoRA training (§2.8, phase 3) ─────────────────────────────────────

const TRAIN_LM_STAGES: readonly TrainLmStage[] = ['extract', 'train', 'export'];
const ADAPTER_NAME_RE = /^[A-Za-z0-9._-]{1,64}$/;

/** `null` when in range, else the §2.8-shaped 400 message. */
function outOfRange(name: string, n: number, min: number, max: number): string | null {
  if (!Number.isFinite(n) || n < min || n > max) return `${name} must be between ${min} and ${max}`;
  return null;
}

router.post('/datasets/:id/train-lm', async (req: Request, res: Response) => {
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
    if (!aceTrainExe()) {
      res.status(503).json({ error: 'ace-train was not found next to ace-server — rebuild the engine' });
      return;
    }
    if (!ds.builtAt || !ds.datasetJsonPath || !fs.existsSync(ds.datasetJsonPath)) {
      res.status(400).json({ error: 'Dataset must be built first — run Build before Training' });
      return;
    }

    const body = (req.body || {}) as TrainLmOptions & { ditModel?: unknown };

    // ── variant ──────────────────────────────────────────────────────────
    const requestedVariant = typeof body.variantKey === 'string' ? body.variantKey.trim() : '';
    // isSafeVariantKey (inside variantExists) rejects any key that is not a
    // single directory name: without it `../../otherslug/…` escapes the tensors
    // root and ace-train would read from — and write lm_codes.jsonl into — an
    // arbitrary directory. Same §7.8 rule the two preprocess routes apply.
    if (requestedVariant && !variantExists(ds.slug, requestedVariant)) {
      res.status(400).json({ error: `Unknown preprocess variant: ${requestedVariant}` });
      return;
    }
    const variantKey = requestedVariant || newestVariantKey(ds.slug);
    if (!variantKey) {
      res.status(400).json({ error: 'Dataset has no preprocessed tensors — run Preprocess first' });
      return;
    }

    // ── base size ────────────────────────────────────────────────────────
    // All three sizes are accepted: 4B trains through the engine's low-VRAM
    // path (per-layer checkpointing + chunked CE). An unaffordable 4B request
    // is now refused by ace-train's own VRAM solve with real numbers, not by a
    // blanket 400 here.
    const lmSize: LmSize =
      body.lmSize === '1.7B' ? '1.7B' : body.lmSize === '4B' ? '4B' : '0.6B';

    // ── models ───────────────────────────────────────────────────────────
    // Same never-probed race fix as preprocess: a fresh boot would otherwise
    // 400 with "No LM base model available" while the engine is reachable.
    let snap = getModelSnapshot();
    if (!snap.cachedAt && !isEngineSuspended()) snap = await refreshModelSnapshot();
    const requestedLm = typeof body.lmModel === 'string' ? body.lmModel.trim() : '';
    const lmModel = requestedLm || pickLmFor(lmSize, snap.lm);
    if (!lmModel) {
      res.status(400).json({ error: `No LM base model available for ${lmSize}` });
      return;
    }
    if (requestedLm && snap.lm.length > 0 && !snap.lm.includes(lmModel)) {
      res.status(400).json({ error: `Unknown LM model: ${lmModel}` });
      return;
    }

    // The FSQ tokenizer lives inside the DiT, and it must be the one the
    // latents were made against — the variant's own record wins by default.
    const ditOverride = typeof body.ditModel === 'string' ? body.ditModel.trim() : '';
    const ditModel = ditOverride || variantDitModel(ds.slug, variantKey);

    // ── adapter name / dir ───────────────────────────────────────────────
    const adapterName = (typeof body.adapterName === 'string' ? body.adapterName.trim() : '') || ds.slug;
    // The regex alone accepts '.', '..', '.hidden' — all of which safeAdapterName()
    // silently REWRITES ('..' and '.' both collapse to 'adapter', '.hidden' to
    // 'hidden'). The stored/logged/returned adapterName would then name a
    // directory that does not exist, and two distinct requests would write the
    // same dir. Reject anything the sanitiser would have to change.
    if (!ADAPTER_NAME_RE.test(adapterName) || safeAdapterName(adapterName) !== adapterName) {
      res.status(400).json({ error: 'adapterName must match [A-Za-z0-9._-]{1,64}' });
      return;
    }
    const adaptersRoot = adapterLmRoot();
    const adapterDir = adapterDirFor(adapterName, lmSize);
    // Containment, same rule the preprocess outputDir uses (§7.8): this path is
    // mkdir'd and written into by a spawned process.
    if (!isInside(adaptersRoot, adapterDir) || path.resolve(adaptersRoot) === path.resolve(adapterDir)) {
      res.status(400).json({ error: 'adapterName must match [A-Za-z0-9._-]{1,64}' });
      return;
    }

    // ── numeric clamps (§4.5 step 8) ─────────────────────────────────────
    const epochs = numOpt(body.epochs, 75);
    const targetLoss = numOpt(body.targetLoss, 4.0);
    const rank = numOpt(body.rank, 16);
    const alpha = numOpt(body.alpha, 32);
    const learningRate = numOpt(body.learningRate, 0.0001);
    const gradAccum = numOpt(body.gradAccum, 2);
    const gradClip = numOpt(body.gradClip, 1.0);
    const warmupRatio = numOpt(body.warmupRatio, 0.05);
    const weightDecay = numOpt(body.weightDecay, 0.01);
    const maxLen = numOpt(body.maxLen, 0);
    const seed = numOpt(body.seed, 42);
    const milestoneStep = numOpt(body.milestoneStep, 1);
    const milestoneKeep = numOpt(body.milestoneKeep, 6);

    const rangeFailure =
      outOfRange('epochs', epochs, 1, 200)
      ?? outOfRange('targetLoss', targetLoss, 0, 20)
      ?? outOfRange('rank', rank, 1, 256)
      ?? outOfRange('alpha', alpha, 1, 1024)
      ?? outOfRange('gradAccum', gradAccum, 1, 64)
      ?? outOfRange('gradClip', gradClip, 0, 100)
      ?? outOfRange('warmupRatio', warmupRatio, 0, 0.5)
      ?? outOfRange('weightDecay', weightDecay, 0, 1)
      ?? outOfRange('seed', seed, 0, 2 ** 31 - 1)
      ?? outOfRange('milestoneStep', milestoneStep, 0, 5)
      ?? outOfRange('milestoneKeep', milestoneKeep, 0, 64);
    if (rangeFailure) {
      res.status(400).json({ error: rangeFailure });
      return;
    }
    if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) {
      res.status(400).json({ error: 'learningRate must be greater than 0 and at most 1' });
      return;
    }
    // 0 means "auto-fit from free VRAM" (L8); any explicit value must be usable.
    if (maxLen !== 0 && (maxLen < 512 || maxLen > 16384)) {
      res.status(400).json({ error: 'maxLen must be 0 (auto) or between 512 and 16384' });
      return;
    }

    // ── low-VRAM knobs (4B §2.5) ─────────────────────────────────────────
    // 'auto' is the engine's own default, so an omitted field emits no flag at
    // all (see buildTrainLmArgs) and the argv stays byte-identical to today.
    const lowVram = body.lowVram === undefined ? 'auto' : body.lowVram;
    if (lowVram !== 'auto' && lowVram !== 'on' && lowVram !== 'off') {
      res.status(400).json({ error: 'lowVram must be auto, on or off' });
      return;
    }
    const attnHeadBlock = numOpt(body.attnHeadBlock, 0);
    const chunk = numOpt(body.chunk, 0);
    if (!Number.isFinite(attnHeadBlock) || attnHeadBlock < 0 || attnHeadBlock > 128) {
      res.status(400).json({ error: 'attnHeadBlock must be between 0 and 128' });
      return;
    }
    // 0 = "engine default (128)"; any explicit value must be a usable chunk.
    if (chunk !== 0 && (chunk < 16 || chunk > 1024)) {
      res.status(400).json({ error: 'chunk must be between 16 and 1024' });
      return;
    }

    // ── speed levers (2026-07-28 plan §2.5) ──────────────────────────────
    // Both defaults ARE the CLI defaults, so an omitted field emits no flag at
    // all (buildTrainLmArgs) and the argv stays byte-identical to today. The
    // engine owns the semantic rules (bf16 needs a BF16 base; batch>1 implies
    // low-VRAM) — this is a value whitelist only, so a stale UI can never make
    // the runner stop ace-server for an argument ace-train would reject.
    const weights = body.weights === undefined ? 'f32-window' : body.weights;
    if (weights !== 'f32-window' && weights !== 'bf16') {
      res.status(400).json({ error: 'weights must be f32-window or bf16' });
      return;
    }
    const rawBatch = body.batch === undefined ? 1 : body.batch;
    let batch: number | 'auto';
    if (rawBatch === 'auto') {
      batch = 'auto';
    } else {
      const n = Number(rawBatch);
      if (!Number.isFinite(n) || Math.trunc(n) !== n || n < 1 || n > 8) {
        res.status(400).json({ error: 'batch must be 1-8 or auto' });
        return;
      }
      batch = n;
    }
    // Lever B (micro-batching) was never written — its §6.1 build gate measures
    // 9.3% amortisable host overhead at 4B against a 10% bar — so ace-train
    // refuses anything but 1 with exit 2. That refusal has to happen HERE and
    // not there: trainLmRunner stops ace-server before it spawns ace-train, so
    // letting a batch>1 request through would shut the user's engine down, fail
    // instantly, and restart it — an engine bounce as the reward for touching a
    // dropdown. Reject before the job is queued.
    if (batch !== 1) {
      res.status(400).json({
        error: 'batch must be 1 — micro-batching is not built in this engine. The host overhead it would '
             + 'amortise measures 9.3% at 4B, under the 10% bar its build gate required. Use gradient '
             + 'accumulation to change the effective batch size.',
      });
      return;
    }

    // ── stages ───────────────────────────────────────────────────────────
    const requestedStages = Array.isArray(body.stages) ? body.stages : [];
    const stages = TRAIN_LM_STAGES.filter(s => requestedStages.includes(s));
    const resolvedStages: TrainLmStage[] = stages.length > 0 ? [...stages] : [...TRAIN_LM_STAGES];

    // Belt and braces: the key is already segment-validated above, but this is
    // the path a spawned process reads and writes, so assert containment too.
    const tensorsPath = path.join(tensorsRoot(ds.slug), variantKey);
    if (!isInside(tensorsRoot(ds.slug), tensorsPath)) {
      res.status(400).json({ error: `Unknown preprocess variant: ${variantKey}` });
      return;
    }
    const opts: ResolvedTrainLmOptions = {
      lmSize,
      lmModel,
      ditModel,
      variantKey,
      tensorsDir: tensorsPath,
      codesPath: path.join(tensorsPath, 'lm_codes.jsonl'),
      adapterName,
      adapterDir,
      targetLoss,
      epochs: Math.trunc(epochs),
      rank: Math.trunc(rank),
      alpha: Math.trunc(alpha),
      learningRate,
      gradAccum: Math.trunc(gradAccum),
      gradClip,
      warmupRatio,
      weightDecay,
      maxLen: Math.trunc(maxLen),
      seed: Math.trunc(seed),
      lossOnCot: body.lossOnCot !== false,
      order: body.order === 'fixed' ? 'fixed' : 'shuffle',
      milestoneStep,
      milestoneKeep: Math.trunc(milestoneKeep),
      stages: resolvedStages,
      overwrite: body.overwrite === true,
      stopEngine: body.stopEngine !== false,
      lowVram,
      attnHeadBlock: Math.trunc(attnHeadBlock),
      chunk: Math.trunc(chunk),
      weights,
      batch,
    };

    const job = queue.startTrainLmJob(ds.id, opts);
    console.log(
      `[Training] train-lm job ${job.id} queued — ${lmSize} ${lmModel}, variant ${variantKey} → ${adapterDir}`);
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] train-lm start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/datasets/:id/train-lm', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const lmSizeQuery: LmSize | undefined =
      req.query.lmSize === '1.7B' ? '1.7B'
        : req.query.lmSize === '4B' ? '4B'
          : req.query.lmSize === '0.6B' ? '0.6B'
            : undefined;
    // `[]`, not `await buildSamples(ds)`: readTrainLmStatus names the parameter
    // `_samples` and never reads it (its counters are measured against the tensor
    // cache, not the dataset). buildSamples walks the whole source tree, stats
    // every audio file and reads every sidecar — a full recursive scan per poll,
    // thrown away, on the same process relaying the training JSONL.
    res.json(readTrainLmStatus(ds, [], {
      variantKey: typeof req.query.variantKey === 'string' ? req.query.variantKey : undefined,
      adapterName: typeof req.query.adapterName === 'string' ? req.query.adapterName : undefined,
      lmSize: lmSizeQuery,
    }));
  } catch (err: any) {
    console.error(`[Training] train-lm status failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── DiT LoRA training (§2.7, phase 4) ────────────────────────────────────

const TRAIN_DIT_STAGES: readonly TrainDitStage[] = ['train', 'export'];

router.post('/datasets/:id/train-dit', async (req: Request, res: Response) => {
  try {
    // ── 1. dataset / job / binary ────────────────────────────────────────
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is already running for this dataset' });
      return;
    }
    if (!aceTrainExe()) {
      res.status(503).json({ error: 'ace-train was not found next to ace-server — rebuild the engine' });
      return;
    }
    // ── 2. built ─────────────────────────────────────────────────────────
    if (!ds.builtAt || !ds.datasetJsonPath || !fs.existsSync(ds.datasetJsonPath)) {
      res.status(400).json({ error: 'Dataset must be built first — run Build before Training' });
      return;
    }

    const body = (req.body || {}) as TrainDitOptions;

    // ── 3. variant ───────────────────────────────────────────────────────
    // isSafeVariantKey (inside variantExists) rejects any key that is not a
    // single directory name: without it `../../otherslug/…` escapes the tensors
    // root and ace-train would read an arbitrary directory. Same §7.8 rule the
    // preprocess and train-lm routes apply.
    const requestedVariant = typeof body.variantKey === 'string' ? body.variantKey.trim() : '';
    if (requestedVariant && !variantExists(ds.slug, requestedVariant)) {
      res.status(400).json({ error: `Unknown preprocess variant: ${requestedVariant}` });
      return;
    }
    const variantKey = requestedVariant || newestVariantKey(ds.slug);
    if (!variantKey) {
      res.status(400).json({ error: 'Dataset has no preprocessed tensors — run Preprocess first' });
      return;
    }
    // Belt and braces: the key is already segment-validated above, but this is
    // the path a spawned process reads, so assert containment too.
    const tensorsPath = path.join(tensorsRoot(ds.slug), variantKey);
    if (!isInside(tensorsRoot(ds.slug), tensorsPath)) {
      res.status(400).json({ error: `Unknown preprocess variant: ${variantKey}` });
      return;
    }

    // ── 4. adapter type ──────────────────────────────────────────────────
    // Refused HERE rather than by the engine (which answers `fatal
    // reason=unsupported-adapter`, exit 1): the engine's refusal arrives only
    // after the runner has already stopped the app's engine, so a typo would
    // cost a full stop/restart cycle.
    if (body.adapterType !== undefined && body.adapterType !== 'lora') {
      res.status(400).json({ error: 'LoKR training is not available yet — use LoRA' });
      return;
    }

    // ── 5. base model ────────────────────────────────────────────────────
    // NEVER from user input: the cached encoder states and context latents are
    // this exact model's outputs, so training against another base is silently
    // wrong (§4.2 base-match guard).
    const ditModel = variantDitModel(ds.slug, variantKey);
    const ditPath = pickDitBaseFor(variantKey, tensorsPath);
    if (!ditPath) {
      res.status(400).json({
        error: 'This preprocess variant was made against a base that is no longer installed',
      });
      return;
    }

    // ── 6. adapter name / dir ────────────────────────────────────────────
    const adapterName = (typeof body.adapterName === 'string' ? body.adapterName.trim() : '') || ds.slug;
    // The regex alone accepts '.', '..', '.hidden' — all of which
    // safeAdapterName() silently REWRITES, so the returned/logged adapterName
    // would name a directory that does not exist and two distinct requests would
    // write the same dir. Reject anything the sanitiser would have to change.
    if (!ADAPTER_NAME_RE.test(adapterName) || safeAdapterName(adapterName) !== adapterName) {
      res.status(400).json({ error: 'adapterName must match [A-Za-z0-9._-]{1,64}' });
      return;
    }
    const adaptersRoot = adapterDitRoot();
    const adapterDir = adapterDitDirFor(adapterName);
    // Containment, same rule the preprocess outputDir uses (§7.8): this path is
    // mkdir'd and written into by a spawned process. The root itself is refused
    // — a run writing adapter_model.safetensors into the adapters root would
    // put a nameless adapter in every user's dropdown.
    // ...and so is the LM adapter root: adapterDitDirFor('lm') resolves to exactly
    // adapterLmRoot(), so a dataset slugged 'lm' would drop a DiT PEFT dir into
    // <adapters>/lm, where GET /adapters/lm lists its bare .safetensors as a
    // planner-LM adapter and POST /scan lists the directory itself as a top-level one.
    if (
      !isInside(adaptersRoot, adapterDir) ||
      path.resolve(adaptersRoot) === path.resolve(adapterDir) ||
      path.resolve(adapterLmRoot()) === path.resolve(adapterDir)
    ) {
      res.status(400).json({ error: 'adapterName must match [A-Za-z0-9._-]{1,64}' });
      return;
    }

    // ── 7. numeric clamps (§4.5 step 7) ──────────────────────────────────
    const epochs = numOpt(body.epochs, 100);
    const targetLoss = numOpt(body.targetLoss, 0.4);
    const rank = numOpt(body.rank, 16);
    const alpha = numOpt(body.alpha, 32);
    const layers = numOpt(body.layers, 0);
    const crop = numOpt(body.crop, 0);
    const cropMin = numOpt(body.cropMin, 375);
    const cropMax = numOpt(body.cropMax, 1250);
    const learningRate = numOpt(body.learningRate, 0.0005);
    const gradAccum = numOpt(body.gradAccum, 4);
    const gradClip = numOpt(body.gradClip, 1.0);
    const warmupRatio = numOpt(body.warmupRatio, 0.05);
    const weightDecay = numOpt(body.weightDecay, 0.01);
    const snrGamma = numOpt(body.snrGamma, 5);
    const tBias = numOpt(body.tBias, 0.5);
    const timestepMu = numOpt(body.timestepMu, -0.4);
    const timestepSigma = numOpt(body.timestepSigma, 1.0);
    const tMin = numOpt(body.tMin, 0);
    const tMax = numOpt(body.tMax, 1);
    const cfgRatio = numOpt(body.cfgRatio, 0.15);
    const genreRatio = numOpt(body.genreRatio, 0);
    const seed = numOpt(body.seed, 42);
    const milestoneStep = numOpt(body.milestoneStep, 0.1);
    const milestoneKeep = numOpt(body.milestoneKeep, 6);
    const vramReserveMb = numOpt(body.vramReserveMb, 2048);

    const rangeFailure =
      outOfRange('epochs', epochs, 1, 2000)
      ?? outOfRange('targetLoss', targetLoss, 0, 20)
      ?? outOfRange('rank', rank, 1, 256)
      ?? outOfRange('alpha', alpha, 1, 1024)
      ?? outOfRange('layers', layers, 0, 64)
      ?? outOfRange('cropMin', cropMin, 128, 8192)
      ?? outOfRange('cropMax', cropMax, 128, 8192)
      ?? outOfRange('gradAccum', gradAccum, 1, 64)
      ?? outOfRange('gradClip', gradClip, 0, 100)
      ?? outOfRange('warmupRatio', warmupRatio, 0, 0.5)
      ?? outOfRange('weightDecay', weightDecay, 0, 1)
      ?? outOfRange('snrGamma', snrGamma, 1, 100)
      ?? outOfRange('tBias', tBias, 0, 4)
      ?? outOfRange('timestepMu', timestepMu, -4, 4)
      ?? outOfRange('tMin', tMin, 0, 1)
      ?? outOfRange('tMax', tMax, 0, 1)
      ?? outOfRange('cfgRatio', cfgRatio, 0, 1)
      ?? outOfRange('genreRatio', genreRatio, 0, 100)
      ?? outOfRange('seed', seed, 0, 2 ** 31 - 1)
      ?? outOfRange('milestoneStep', milestoneStep, 0, 5)
      ?? outOfRange('milestoneKeep', milestoneKeep, 0, 64)
      ?? outOfRange('vramReserveMb', vramReserveMb, 0, 16384);
    if (rangeFailure) {
      res.status(400).json({ error: rangeFailure });
      return;
    }
    // 0 means "auto-fit from free VRAM" (D10); any explicit value must be usable.
    if (crop !== 0 && (crop < 128 || crop > 8192)) {
      res.status(400).json({ error: 'crop must be 0 or between 128 and 8192 frames' });
      return;
    }
    if (cropMax < cropMin) {
      res.status(400).json({ error: 'cropMax must be greater than or equal to cropMin' });
      return;
    }
    if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) {
      res.status(400).json({ error: 'learningRate must be greater than 0 and at most 1' });
      return;
    }
    // Exclusive lower bound: sigma 0 makes the logit-normal timestep sampler
    // degenerate to a single t, which trains one point of the schedule (D12).
    if (!Number.isFinite(timestepSigma) || timestepSigma <= 0 || timestepSigma > 4) {
      res.status(400).json({ error: 'timestepSigma must be greater than 0 and at most 4' });
      return;
    }
    // An empty interval makes dit_sample_t's rejection loop exhaust its 64 tries
    // on every micro-step and clamp — silently training one timestep.
    if (tMin >= tMax) {
      res.status(400).json({ error: 'tMin must be less than tMax' });
      return;
    }

    // ── 8. stages ────────────────────────────────────────────────────────
    const requestedStages = Array.isArray(body.stages) ? body.stages : [];
    const stages = TRAIN_DIT_STAGES.filter(s => requestedStages.includes(s));
    const resolvedStages: TrainDitStage[] = stages.length > 0 ? [...stages] : [...TRAIN_DIT_STAGES];

    // ── 9. queue ─────────────────────────────────────────────────────────
    // No VRAM gating here (§4.5): only ace-train knows the mirror size, and only
    // after the base is loaded with the engine already stopped.
    // capabilities.trainDit.minVramMb is advisory for the UI banner alone.
    const opts: ResolvedTrainDitOptions = {
      variantKey,
      tensorsDir: tensorsPath,
      ditModel,
      ditPath,
      adapterName,
      adapterDir,
      adapterType: 'lora',
      rank: Math.trunc(rank),
      alpha: Math.trunc(alpha),
      targetMlp: body.targetMlp === true,
      layers: Math.trunc(layers),
      crop: Math.trunc(crop),
      cropMin: Math.trunc(cropMin),
      cropMax: Math.trunc(cropMax),
      targetLoss,
      epochs: Math.trunc(epochs),
      learningRate,
      gradAccum: Math.trunc(gradAccum),
      gradClip,
      warmupRatio,
      weightDecay,
      lossWeighting: body.lossWeighting === 'none' ? 'none' : 'flow_snr',
      snrGamma,
      tBias,
      channelBalance: body.channelBalance !== false,
      timestepMu,
      timestepSigma,
      tMin,
      tMax,
      cfgRatio,
      genreRatio: Math.trunc(genreRatio),
      seed: Math.trunc(seed),
      order: body.order === 'fixed' ? 'fixed' : 'shuffle',
      milestoneStep,
      milestoneKeep: Math.trunc(milestoneKeep),
      vramReserveMb: Math.trunc(vramReserveMb),
      stages: resolvedStages,
      overwrite: body.overwrite === true,
      stopEngine: body.stopEngine !== false,
    };

    const job = queue.startTrainDitJob(ds.id, opts);
    console.log(
      `[Training] train-dit job ${job.id} queued — lora r${opts.rank}, variant ${variantKey} → ${adapterDir}`);
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] train-dit start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/datasets/:id/train-dit', async (req: Request, res: Response) => {
  try {
    const ds = repo.getDataset(req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    res.json(readTrainDitStatus(ds, {
      variantKey: typeof req.query.variantKey === 'string' ? req.query.variantKey : undefined,
      adapterName: typeof req.query.adapterName === 'string' ? req.query.adapterName : undefined,
    }));
  } catch (err: any) {
    console.error(`[Training] train-dit status failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

export default router;
