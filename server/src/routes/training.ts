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
//   POST   /pipeline                                    — start a batch import + stage chain
//   GET    /pipeline                                    — active + recent pipelines
//   GET    /pipeline/:id                                — one pipeline
//   DELETE /pipeline/:id                                — cancel a pipeline
//   GET    /defaults                                    — stored per-stage defaults
//   PUT    /defaults                                    — set per-stage defaults
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
//   GET    /datasets/:id/lyric-studio                   — export preview (detected artist/album, adapters)
//   POST   /datasets/:id/lyric-studio                   — commit export into Lyric Studio
//   POST   /datasets/:id/preprocess                     — start a tensor-cache job
//   GET    /datasets/:id/preprocess                     — tensor-cache status
//   DELETE /datasets/:id/preprocess/:variantKey         — delete one cache variant
//   POST   /datasets/:id/train-lm                       — start an LM LoRA training job
//   GET    /datasets/:id/train-lm                       — LM adapter / codes status
//   POST   /datasets/:id/train-dit                      — start a DiT LoRA training job
//   GET    /datasets/:id/train-dit                      — DiT adapter / variant status
//   GET    /previews/:previewId/:slot                   — stream a codes preview (Range aware)
//   POST   /datasets/:id/audition                       — start an A/B codes-audition job
//   GET    /datasets/:id/audition                       — recent previews for a dataset
//   POST   /datasets/:id/samples/:sampleId/audition     — SYNC decode of a sample's stored codes

import { Router, Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import { config } from '../config.js';
import { engineReady } from '../engineState.js';
import { aceClient } from '../services/aceClient.js';
import { listProviders, getProvider } from '../services/lireek/llm/registry.js';
import * as repo from '../services/training/datasetsRepo.js';
import {
  buildSamples, loadSidecarMetadata, sampleFromParts,
  scanPreview as scanPreviewFolder, ScanLimitError,
} from '../services/training/datasetScan.js';
import { isInside, trainingBaseDir } from '../services/training/paths.js';
import { deleteLabel, deleteLabels, patchLabel, readLabel } from '../services/training/labelStore.js';
import { createDatasetFromFolder, DatasetCreateError } from '../services/training/datasetCreate.js';
import { detailFor, syncCounters } from '../services/training/datasetDetail.js';
import {
  cancelPipeline, getPipeline, hasActivePipeline, listPipelines, PIPELINE_STAGES, startPipeline,
} from '../services/training/pipelineRunner.js';
import { getTrainingDefaults, setTrainingDefaults } from '../services/training/trainingDefaults.js';
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
import { readPreprocessStatus } from '../services/training/preprocessStatus.js';
import {
  adapterLmRoot, lmRunDirFor, newestVariantKey, readTrainLmStatus,
  safeAdapterName, variantDitModel, variantExists,
} from '../services/training/trainLmStatus.js';
import {
  adapterDitRoot, ditRunDirFor, readTrainDitStatus,
} from '../services/training/trainDitStatus.js';
import { hasWeights, lmAdapterRoots } from '../services/training/adapterLayout.js';
import { getUserId } from './auth.js';
import {
  deleteDatasetPreviews, isPreviewFileKey, isPreviewId, listPreviews, previewsRoot,
  prunePreviews, resolvePreviewFile,
} from '../services/training/auditionStore.js';
import { AuditionError, decodeStoredCodes } from '../services/training/auditionService.js';
import {
  commitLyricStudioExport, LyricStudioExportError, previewLyricStudioExport,
} from '../services/training/lyricStudioExport.js';
import type {
  AuditionListResponse, AuditionOptions, AuditionSideSpec,
  BulkSetInput, CaptionOptions, CreateDatasetInput, FieldSource, GeniusOptions, LabelOptions, LmSize,
  LyricStudioExportInput,
  PatchSampleInput, PipelineFolderSpec, PipelineStage,
  PreprocessCompat, PreprocessDtype, PreprocessNormalize, PreprocessOptions,
  TrainingCapabilities, TrainingDatasetRow, TrainingDefaults, TrainingSample,
  DitAdapterType, TrainDitOptions, TrainDitStage, TrainLmOptions, TrainLmStage,
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

/** Re-read one sample from disk after a write — avoids a full rescan per edit. */
function reloadSample(ds: TrainingDatasetRow, sample: TrainingSample): TrainingSample {
  let sizeBytes = sample.sizeBytes;
  let mtimeMs = 0;
  try {
    const st = fs.statSync(sample.audioPath);
    sizeBytes = st.size;
    mtimeMs = st.mtimeMs;
  } catch { /* file vanished — keep the sizes we already had */ }
  const label = readLabel(ds.userId, ds.slug, sample.sampleId) ?? undefined;
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

// ── Capabilities (§2.1) ──────────────────────────────────────────────────

router.get('/capabilities', async (req: Request, res: Response) => {
  const userId = req.user!.userId;
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
      available: false, adapterTypes: ['lora', 'lokr'], adaptersRoot: '', minVramMb: 16384,
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
  try { caps.trainLm.adaptersRoot = adapterLmRoot(userId); } catch { /* stays '' */ }
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
  try { caps.trainDit.adaptersRoot = adapterDitRoot(userId); } catch { /* stays '' */ }

  res.json(caps);
});

// ── Scan preview (§2.2) ──────────────────────────────────────────────────

router.get('/scan-preview', (req: Request, res: Response) => {
  try {
    const userId = req.user!.userId;
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

router.get('/datasets', (req: Request, res: Response) => {
  try {
    const userId = req.user!.userId;
    res.json({ datasets: repo.listDatasets(userId) });
  } catch (err: any) {
    console.error(`[Training] List datasets failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.post('/datasets', async (req: Request, res: Response) => {
  try {
    const userId = req.user!.userId;
    const detail = await createDatasetFromFolder((req.body || {}) as CreateDatasetInput, userId);
    res.status(201).json({ dataset: detail });
  } catch (err: any) {
    if (err instanceof DatasetCreateError) {
      res.status(err.status).json({ error: err.message });
      return;
    }
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
    const userId = req.user!.userId;
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
    const userId = req.user!.userId;
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

// ── Codes-audition preview streaming (codes-preview §3.2) ────────────────
//
// Declared here, in the static-first-segment block, per [DS] §2 — `/previews`
// cannot collide with `/datasets/:id`, but the house rule is the house rule.
//
// The served path is BUILT SERVER-SIDE from two validated tokens (a uuid v4 and
// `base|adapter`) and re-checked with isInside(previewsRoot(), file). No
// client-supplied path ever reaches fs.
//
// Range handling is a verbatim clone of the sample-audio route below with
// sample.audioPath swapped for the validated preview path. Deliberately NOT
// refactored into a shared helper: the sample route is outside this plan's
// editable set and a shared helper would drag it in.
router.get('/previews/:previewId/:slot', (req: Request, res: Response) => {
  try {
    const userId = req.user!.userId;
    const previewId = req.params.previewId;
    const slot = req.params.slot;   // 'base' | 'adapter' | '<slot>-render'
    if (!isPreviewId(previewId) || !isPreviewFileKey(slot)) {
      res.status(404).json({ error: 'Preview not found' });
      return;
    }
    const file = resolvePreviewFile(previewId, slot);
    if (!file || !isInside(previewsRoot(), file) || !fs.existsSync(file)) {
      res.status(404).json({ error: 'Preview not found' });
      return;
    }

    const stat = fs.statSync(file);
    const mime = MIME_BY_EXT[path.extname(file).toLowerCase()] || 'application/octet-stream';
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
          fs.createReadStream(file, { start, end: last }).pipe(res);
          return;
        }
      }
    }

    res.setHeader('Content-Length', stat.size);
    fs.createReadStream(file).pipe(res);
  } catch (err: any) {
    console.error(`[Training] Preview stream failed: ${err.message}`);
    if (!res.headersSent) res.status(500).json({ error: err.message });
  }
});

// ── Batch pipeline (batch-pipeline §2.2) — declared before /datasets/:id ──

router.post('/pipeline', (req: Request, res: Response) => {
  try {
    const userId = req.user!.userId;
    const body = req.body || {};
    const rawFolders = Array.isArray(body.folders) ? body.folders : [];
    if (rawFolders.length === 0) {
      res.status(400).json({ error: 'folders is required' });
      return;
    }

    const folders: PipelineFolderSpec[] = [];
    for (const raw of rawFolders) {
      const dir = typeof raw?.sourceDir === 'string' ? raw.sourceDir.trim() : '';
      if (!dir) {
        res.status(400).json({ error: 'folders is required' });
        return;
      }
      // Checked up front for EVERY folder: a typo in the last row of a 20-album
      // batch should not surface an hour into the run.
      const sourceDir = path.resolve(dir);
      if (!fs.existsSync(sourceDir) || !fs.statSync(sourceDir).isDirectory()) {
        res.status(400).json({ error: `Folder does not exist: ${dir}` });
        return;
      }
      folders.push({
        sourceDir,
        name: typeof raw?.name === 'string' ? raw.name.trim() : undefined,
        customTag: typeof raw?.customTag === 'string' ? raw.customTag.trim() : undefined,
      });
    }

    let stages: PipelineStage[] = [...PIPELINE_STAGES];
    if (Array.isArray(body.stages)) {
      for (const stage of body.stages) {
        if (!PIPELINE_STAGES.includes(stage)) {
          res.status(400).json({ error: `Unknown stage: ${stage}` });
          return;
        }
      }
      // Canonical order, whatever order the client listed them in.
      stages = PIPELINE_STAGES.filter(s => body.stages.includes(s));
    }

    if (hasActivePipeline()) {
      res.status(409).json({ error: 'A pipeline is already running' });
      return;
    }

    res.status(202).json({ pipeline: startPipeline(folders, stages, userId) });
  } catch (err: any) {
    console.error(`[Training] Pipeline start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/pipeline', (_req: Request, res: Response) => {
  try {
    res.json({ pipelines: listPipelines() });
  } catch (err: any) {
    console.error(`[Training] Pipeline list failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/pipeline/:id', (req: Request, res: Response) => {
  try {
    const userId = req.user!.userId;
    const pipeline = getPipeline(req.params.id as string);
    if (!pipeline) {
      res.status(404).json({ error: 'Pipeline not found' });
      return;
    }
    res.json(pipeline);
  } catch (err: any) {
    console.error(`[Training] Pipeline read failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.delete('/pipeline/:id', (req: Request, res: Response) => {
  try {
    const userId = req.user!.userId;
    if (!cancelPipeline(req.params.id as string)) {
      res.status(404).json({ error: 'Pipeline not found' });
      return;
    }
    res.json({ ok: true });
  } catch (err: any) {
    console.error(`[Training] Pipeline cancel failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Stage defaults (batch-pipeline §2.2) ─────────────────────────────────
//
// No deep validation on PUT: a section is an opaque bag of the stage route's
// own option fields, and that route re-validates every one of them when the
// pipeline POSTs it — which is the honest gate.

const DEFAULTS_SECTIONS: ReadonlyArray<keyof TrainingDefaults> =
  ['label', 'preprocess', 'trainLm', 'trainDit'];

router.get('/defaults', (_req: Request, res: Response) => {
  try {
    res.json(getTrainingDefaults());
  } catch (err: any) {
    console.error(`[Training] Defaults read failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.put('/defaults', (req: Request, res: Response) => {
  try {
  const userId = req.user!.userId;
    const body = req.body || {};
    const patch: Partial<TrainingDefaults> = {};
    for (const section of DEFAULTS_SECTIONS) {
      const value = body[section];
      if (value === undefined) continue;
      if (!value || typeof value !== 'object' || Array.isArray(value)) {
        res.status(400).json({ error: `${section} must be an object` });
        return;
      }
      patch[section] = value as Record<string, unknown>;
    }
    res.json(setTrainingDefaults(patch));
  } catch (err: any) {
    console.error(`[Training] Defaults write failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Dataset detail / patch / rescan / delete (§2.3) ───────────────────────

router.get('/datasets/:id', async (req: Request, res: Response) => {
  try {
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
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
  const userId = req.user!.userId;
    const id = req.params.id as string;
    const ds = repo.getDataset(userId, id);
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
    repo.updateDataset(userId, id, patch);
    const updated = repo.getDataset(userId, id);
    res.json({ dataset: updated });
  } catch (err: any) {
    console.error(`[Training] Patch dataset failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.post('/datasets/:id/rescan', async (req: Request, res: Response) => {
  try {
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
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
      const rec = readLabel(ds.userId, ds.slug, sample.sampleId);
      if (!rec) continue;
      if (!sample.fileMissing) {
        if (rec.missingSince) patchLabel(userId, ds.slug, sample.sampleId, { missingSince: null });
        continue;
      }
      if (!rec.missingSince) {
        patchLabel(userId, ds.slug, sample.sampleId, { missingSince: now });
        continue;
      }
      // Second consecutive miss — the ghost row goes, and with it the Build block.
      deleteLabel(userId, ds.slug, sample.sampleId);
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
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
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
    repo.deleteDataset(userId, ds.id);
    deleteLabels(userId, ds.slug);
    deleteDatasetPreviews(ds.id);
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
  userId: string,
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

  patchLabel(userId, ds.slug, sample.sampleId, {
    relPath: sample.relPath,
    sources,
    ...(typeof patch.excluded === 'boolean' ? { excluded: patch.excluded } : {}),
  });

  return reloadSample(ds, sample);
}

router.post('/datasets/:id/samples/bulk', async (req: Request, res: Response) => {
  try {
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
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
        await applySamplePatch(userId, ds, sample, set as PatchSampleInput);
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
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
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

    const updated = await applySamplePatch(userId, ds, sample, body);
    res.json({ sample: updated });
  } catch (err: any) {
    console.error(`[Training] Sample patch failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/datasets/:id/samples/:sampleId/audio', async (req: Request, res: Response) => {
  try {
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
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
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
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
    repo.updateDataset(userId, ds.id, { status: 'labeling' });
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] Label start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Enhance jobs (§2.6) ──────────────────────────────────────────────────

router.post('/datasets/:id/enhance/genius', async (req: Request, res: Response) => {
  try {
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
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
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
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
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
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
    const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
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

// ── Lyric Studio export ──────────────────────────────────────────────────

router.get('/datasets/:id/lyric-studio', async (req: Request, res: Response) => {
  try {
    const userId = getUserId(req);
    if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }
    const ds = repo.getDataset(userId, req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const samples = await buildSamples(ds, { warnings: [] });
    res.json(previewLyricStudioExport(userId, ds, samples));
  } catch (err: any) {
    if (err instanceof ScanLimitError) {
      res.status(400).json({ error: err.message });
      return;
    }
    console.error(`[Training] Lyric Studio preview failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.post('/datasets/:id/lyric-studio', async (req: Request, res: Response) => {
  try {
    const userId = getUserId(req);
    if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }
    const ds = repo.getDataset(userId, req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    const body = (req.body ?? {}) as LyricStudioExportInput;
    const input: LyricStudioExportInput = {
      artist: typeof body.artist === 'string' ? body.artist.slice(0, 200) : undefined,
      album: typeof body.album === 'string' ? body.album.slice(0, 200) : undefined,
      linkAdapters: body.linkAdapters !== false,
    };
    const samples = await buildSamples(ds, { warnings: [] });
    const result = await commitLyricStudioExport(userId, ds, samples, input);
    res.json(result);
  } catch (err: any) {
    if (err instanceof LyricStudioExportError || err instanceof ScanLimitError) {
      res.status(400).json({ error: err.message });
      return;
    }
    console.error(`[Training] Lyric Studio export failed: ${err.message}`);
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
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
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
    let outputDir = tensorsDir(userId, ds.slug, dit);
    if (typeof body.outputDir === 'string' && body.outputDir.trim()) {
      // Containment, same rule the sibling DELETE handler applies (§7.8). This
      // path is mkdir'd, ace-train creates <out>/.tmp/ in it and deletes orphan
      // *.__writing__ files there, so an unchecked absolute path from the
      // request body is a write primitive. Staying under the dataset's tensors
      // root also keeps the cache visible to GET/DELETE .../preprocess — a
      // cache written anywhere else is unmanaged disk the UI can never see.
      const root = tensorsRoot(userId, ds.slug);
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
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
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
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is running for this dataset' });
      return;
    }
    const root = tensorsRoot(userId, ds.slug);
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
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
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
    if (requestedVariant && !variantExists(userId, ds.slug, requestedVariant)) {
      res.status(400).json({ error: `Unknown preprocess variant: ${requestedVariant}` });
      return;
    }
    const variantKey = requestedVariant || newestVariantKey(userId, ds.slug);
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
    const ditModel = ditOverride || variantDitModel(userId, ds.slug, variantKey);

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
    // Per-base + per-run layout: every run writes a FRESH
    // <adapters>/lm-<sizeSlug>/<name>/<stamp> dir, so retraining an artist
    // never overwrites an earlier adapter. Contain against the adapters root,
    // refusing the root itself and any size root — this path is mkdir'd and
    // written into by a spawned process (§7.8).
    const adaptersRoot = config.aceServer.adapters;
    const adapterDir = lmRunDirFor(userId, adapterName, lmSize);
    const isARoot = [adaptersRoot, adapterLmRoot(userId), ...lmAdapterRoots(userId).map(r => r.dir)]
      .some(r => path.resolve(r) === path.resolve(adapterDir));
    if (!isInside(adaptersRoot, adapterDir) || isARoot) {
      res.status(400).json({ error: 'adapterName must match [A-Za-z0-9._-]{1,64}' });
      return;
    }

    // ── numeric clamps (§4.5 step 8) ─────────────────────────────────────
    const epochs = numOpt(body.epochs, 75);
    const targetLoss = numOpt(body.targetLoss, 4.0);
    const rank = numOpt(body.rank, 16);
    // LoKr (2026-07-30). Default 'lora' — the shipped path — so an omitted
    // field can never move an existing caller onto a different parameterization.
    const lmIsLokr = body.adapterType === 'lokr';
    const lmMuonLrScale = numOpt(body.muonLrScale, 20.0);
    const lmMuonNsSteps = numOpt(body.muonNsSteps, 5);
    if (body.optimizer !== undefined && body.optimizer !== 'adamw' && body.optimizer !== 'muon') {
      res.status(400).json({ error: 'optimizer must be adamw or muon' });
      return;
    }
    if (lmMuonLrScale < 0.001 || lmMuonLrScale > 1000) {
      res.status(400).json({ error: 'muonLrScale must be between 0.001 and 1000' });
      return;
    }
    if (lmMuonNsSteps < 1 || lmMuonNsSteps > 20) {
      res.status(400).json({ error: 'muonNsSteps must be between 1 and 20' });
      return;
    }
    const lmLokrDim = numOpt(body.lokrDim, 512);
    const lmLokrAlpha = numOpt(body.lokrAlpha, 512);
    const lmLokrFactor = numOpt(body.lokrFactor, 6);
    if (body.adapterType !== undefined && body.adapterType !== 'lora' && body.adapterType !== 'lokr') {
      res.status(400).json({ error: 'adapterType must be lora or lokr' });
      return;
    }
    if (lmIsLokr && (lmLokrDim < 4 || lmLokrDim > 4096)) {
      res.status(400).json({ error: 'lokrDim must be between 4 and 4096' });
      return;
    }
    if (lmIsLokr && (lmLokrFactor !== -1 && (lmLokrFactor < 2 || lmLokrFactor > 64))) {
      res.status(400).json({ error: 'lokrFactor must be -1 or between 2 and 64' });
      return;
    }
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
    // The 'weights' default flipped to 'bf16' (2026-07-29) and is no longer the
    // CLI default ('f32-window'), so an omitted field now DOES emit an explicit
    // --weights bf16 flag (buildTrainLmArgs). The engine owns the semantic
    // rules (bf16 needs a BF16 base; batch>1 implies low-VRAM) — this is a
    // value whitelist only, so a stale UI can never make the runner stop
    // ace-server for an argument ace-train would reject.
    const weights = body.weights === undefined ? 'bf16' : body.weights;
    if (weights !== 'f32-window' && weights !== 'bf16') {
      res.status(400).json({ error: 'weights must be f32-window or bf16' });
      return;
    }
    // MUL_MAT activation-gradient formulation.
    //
    // THE LM DEFAULT IS 'outprod', NOT 'mm' — deliberately different from
    // train-dit's. `weights` above already defaults to 'bf16', and lm-bf16.h's
    // Lever A reaches the same mul_mat backward by rewriting ggml's out_prod
    // nodes in place, asserting exactly 7 rewrites per segment. Under --bwd mm
    // ggml emits mul_mat directly, that surgery finds nothing, and the S18
    // tripwire GGML_ABORTs the run. Defaulting the LM to 'mm' would therefore
    // brick the DEFAULT LM training job. It would also buy nothing on the
    // f32-window path, where the transposed weight is the F32 window and the
    // GEMM stays TF32 while paying an extra cont. --bwd mm is a train-dit win.
    const bwd = body.bwd === undefined ? 'outprod' : body.bwd;
    if (bwd !== 'outprod' && bwd !== 'mm') {
      res.status(400).json({ error: 'bwd must be outprod or mm' });
      return;
    }
    // Refused HERE so the user gets a 400 instead of the runner stopping
    // ace-server for a run ace-train exits 2 on (same rule as the other levers).
    if (weights === 'bf16' && bwd === 'mm') {
      res.status(400).json({
        error: 'weights bf16 and bwd mm are two routes to the same mul_mat backward and cannot be '
             + 'combined — the bf16 lever rewrites ggml out_prod nodes in place and aborts when there '
             + 'are none. Use weights bf16 on its own, or pair bwd mm with weights f32-window.',
      });
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
    const tensorsPath = path.join(tensorsRoot(userId, ds.slug), variantKey);
    if (!isInside(tensorsRoot(userId, ds.slug), tensorsPath)) {
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
      // Only the exact string 'lokr' opts in; anything else is a LoRA.
      adapterType: lmIsLokr ? 'lokr' : 'lora',
      // Only the exact string 'muon' opts in; anything else is AdamW.
      optimizer: body.optimizer === 'muon' ? 'muon' : 'adamw',
      muonLrScale: lmMuonLrScale,
      muonNsSteps: Math.trunc(lmMuonNsSteps),
      rank: Math.trunc(rank),
      alpha: Math.trunc(alpha),
      lokrDim: Math.trunc(lmLokrDim),
      lokrAlpha: lmLokrAlpha,
      lokrFactor: Math.trunc(lmLokrFactor),
      lokrDecomposeBoth: body.lokrDecomposeBoth !== false,
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
      bwd,
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
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
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
    res.json(readTrainLmStatus(userId, ds, [], {
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
  const userId = req.user!.userId;
    // ── 1. dataset / job / binary ────────────────────────────────────────
    const ds = repo.getDataset(userId, req.params.id as string);
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
    if (requestedVariant && !variantExists(userId, ds.slug, requestedVariant)) {
      res.status(400).json({ error: `Unknown preprocess variant: ${requestedVariant}` });
      return;
    }
    const variantKey = requestedVariant || newestVariantKey(userId, ds.slug);
    if (!variantKey) {
      res.status(400).json({ error: 'Dataset has no preprocessed tensors — run Preprocess first' });
      return;
    }
    // Belt and braces: the key is already segment-validated above, but this is
    // the path a spawned process reads, so assert containment too.
    const tensorsPath = path.join(tensorsRoot(userId, ds.slug), variantKey);
    if (!isInside(tensorsRoot(userId, ds.slug), tensorsPath)) {
      res.status(400).json({ error: `Unknown preprocess variant: ${variantKey}` });
      return;
    }

    // ── 4. adapter type ──────────────────────────────────────────────────
    // Refused HERE rather than by the engine (which answers `fatal
    // reason=unsupported-adapter`, exit 1): the engine's refusal arrives only
    // after the runner has already stopped the app's engine, so a typo would
    // cost a full stop/restart cycle.
    if (body.adapterType !== undefined && body.adapterType !== 'lora' && body.adapterType !== 'lokr') {
      res.status(400).json({ error: 'adapterType must be "lora" or "lokr"' });
      return;
    }
    const adapterType: DitAdapterType = body.adapterType === 'lokr' ? 'lokr' : 'lora';
    const isLokr = adapterType === 'lokr';

    // ── 5. base model ────────────────────────────────────────────────────
    // NEVER from user input: the cached encoder states and context latents are
    // this exact model's outputs, so training against another base is silently
    // wrong (§4.2 base-match guard).
    const ditModel = variantDitModel(userId, ds.slug, variantKey);
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
    const adaptersRoot = adapterDitRoot(userId);
    // Per-base + per-run layout: <adapters>/dit-<shorthand>/<name>/<stamp>,
    // the shorthand coming from the variant's base resolved above
    // (adapterLayout.ts). A fresh stamped dir per run — retraining an artist
    // never overwrites an earlier adapter.
    const adapterDir = ditRunDirFor(userId, adapterName, ditModel);
    // Containment, same rule the preprocess outputDir uses (§7.8): this path is
    // mkdir'd and written into by a spawned process. The root itself is refused
    // — a run writing adapter_model.safetensors into the adapters root would
    // put a nameless adapter in every user's dropdown — and so are the LM
    // adapter roots (legacy flat `lm/` and the per-size `lm-*` dirs), where a
    // DiT PEFT dir would show up in the planner-adapter dropdown.
    const clashesRoot = [adaptersRoot, adapterLmRoot(userId), ...lmAdapterRoots(userId).map(r => r.dir)]
      .some(r => path.resolve(r) === path.resolve(adapterDir));
    if (!isInside(adaptersRoot, adapterDir) || clashesRoot) {
      res.status(400).json({ error: 'adapterName must match [A-Za-z0-9._-]{1,64}' });
      return;
    }

    // ── 7. numeric clamps (§4.5 step 7) ──────────────────────────────────
    // train-dit defaults. These are what the UI path actually gets: TrainDitForm
    // sends every field, but any client that omits one lands here.
    //
    // K2 (lokr-dit-training plan §0): when adapterType==='lokr', four of these
    // omitted-field fallbacks change to Rob's Uber-LoKR-4 preset values. The
    // 'lora' path's fallbacks are untouched — byte-identical to before LoKR.
    // 250 for LoKR (2026-07-30 retune): a 400-epoch cosine horizon left every
    // measured run stopping at ~50% of peak LR, so the schedule never decayed
    // into the target. 250 cut epochs-to-0.6 from 228 to 203. LoRA keeps 400.
    const epochs = numOpt(body.epochs, isLokr ? 250 : 400);
    const targetLoss = numOpt(body.targetLoss, isLokr ? 0.6 : 0.4);
    const rank = numOpt(body.rank, 128);
    const alpha = numOpt(body.alpha, 256);
    const lokrDim = numOpt(body.lokrDim, 512);
    const lokrAlpha = numOpt(body.lokrAlpha, 512);
    const lokrFactor = numOpt(body.lokrFactor, 6);
    const lokrDecomposeBoth = body.lokrDecomposeBoth !== false;
    const layers = numOpt(body.layers, 0);
    const crop = numOpt(body.crop, 0);
    const cropMin = numOpt(body.cropMin, 375);
    const cropMax = numOpt(body.cropMax, 1250);
    // LoKR 2e-3 @ GA 4 (2026-07-30 retune) replaces 1e-2 @ GA 20. Side-Step
    // reaches an effective batch of 20 as batch 5 x GA 4; we reached it by
    // accumulating 20, which is the same effective LR per sample under linear
    // scaling. Measured on gunship_unicorn: IDENTICAL epochs-to-target (227 vs
    // 228) with strictly better-behaved gradients — median grad-norm 0.062 vs
    // 0.031 (the sqrt(5) a 5x smaller batch predicts) and no warmup spike, where
    // GA 20 peaked at 13.5 on epoch 1. THE TWO MOVE TOGETHER: 2e-3 at GA 20, or
    // 1e-2 at GA 4, are both untested configurations. The LoRA path (5e-4, GA 4)
    // is unchanged, as is ace-train's own CLI default.
    const learningRate = numOpt(body.learningRate, isLokr ? 0.002 : 0.0005);
    const gradAccum = numOpt(body.gradAccum, 4);
    const gradClip = numOpt(body.gradClip, 1.0);
    const warmupRatio = numOpt(body.warmupRatio, 0.05);
    const weightDecay = numOpt(body.weightDecay, isLokr ? 0.001 : 0.01);
    const snrGamma = numOpt(body.snrGamma, 5);
    const tBias = numOpt(body.tBias, 0.5);
    const timestepMu = numOpt(body.timestepMu, -0.4);
    const timestepSigma = numOpt(body.timestepSigma, 1.0);
    const tMin = numOpt(body.tMin, 0);
    const tMax = numOpt(body.tMax, 1);
    const cfgRatio = numOpt(body.cfgRatio, 0.15);
    // Percent of micro-steps conditioned on the dataset's genre text instead of
    // the caption (D14). Default 30 — enough for the adapter to learn the genre
    // handle without the caption path going untrained.
    const genreRatio = numOpt(body.genreRatio, 30);
    const seed = numOpt(body.seed, 42);
    const milestoneStep = numOpt(body.milestoneStep, 0.1);
    const milestoneKeep = numOpt(body.milestoneKeep, 6);
    const vramReserveMb = numOpt(body.vramReserveMb, 2048);
    // Micro-batching / checkpointing (design §2.2). ckptSegments mirrors the
    // engine's --ckpt semantics directly: 0=off, 1=auto, 2-32=fixed segments.
    // batch defaults to 1 = OFF (2026-07-29): measured ~2.5x SLOWER at full depth
    // on a 32 GB card, ~2.4x faster on shallow/partial-depth runs. Same default
    // as the engine's own DitTrainArgs, so an omitted field and an absent flag
    // land on the same behaviour.
    const batch = numOpt(body.batch, 1);
    const ckptSegments = numOpt(body.ckptSegments, 1);
    // Optimizer (2026-07-30). Default 'adamw' — the shipped path — so an
    // omitted field can never move an existing caller onto Muon.
    const muonLrScale = numOpt(body.muonLrScale, 20.0);
    const muonMomentum = numOpt(body.muonMomentum, 0.95);
    const muonNsSteps = numOpt(body.muonNsSteps, 5);
    const muonMinDim = numOpt(body.muonMinDim, 16);

    const rangeFailure =
      outOfRange('epochs', epochs, 1, 2000)
      ?? outOfRange('targetLoss', targetLoss, 0, 20)
      ?? outOfRange('rank', rank, 1, 256)
      ?? outOfRange('alpha', alpha, 1, 1024)
      ?? outOfRange('lokrDim', lokrDim, 4, 4096)
      // §2.1: (0,8192] with 0 as the "-> dim" sentinel (K6) — 0 is valid input.
      ?? outOfRange('lokrAlpha', lokrAlpha, 0, 8192)
      ?? (lokrFactor !== -1 && (lokrFactor < 2 || lokrFactor > 64)
        ? 'lokrFactor must be -1 or between 2 and 64' : null)
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
      ?? outOfRange('vramReserveMb', vramReserveMb, 0, 16384)
      ?? outOfRange('batch', batch, 1, 16)
      ?? outOfRange('muonLrScale', muonLrScale, 0.001, 1000)
      ?? outOfRange('muonMomentum', muonMomentum, 0, 0.999)
      ?? outOfRange('muonNsSteps', muonNsSteps, 1, 20)
      ?? outOfRange('muonMinDim', muonMinDim, 1, 4096)
      // ckptSegments: 0=off, 1=auto, 2-32=fixed segment count (design §2.2).
      ?? (ckptSegments !== 0 && ckptSegments !== 1 && (ckptSegments < 2 || ckptSegments > 32)
        ? 'ckptSegments must be 0, 1, or between 2 and 32' : null);
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
    // Refused here rather than coerced: an unrecognised value must not
    // silently land on either side. 'bf16' halves the frozen-weight mirror
    // and is the default (2026-07-29); the engine falls back to 'f32' itself
    // on a non-CUDA backend, so an explicit 'f32' remains the opt-out.
    if (body.mirror !== undefined && body.mirror !== 'f32' && body.mirror !== 'bf16') {
      res.status(400).json({ error: 'mirror must be f32 or bf16' });
      return;
    }
    // Same rule for the MUL_MAT activation-gradient formulation: refused, not
    // coerced. Default is 'mm' (engine/patches/mm-backward.patch), not
    // ace-train's own 'outprod'.
    if (body.optimizer !== undefined && body.optimizer !== 'adamw' && body.optimizer !== 'muon') {
      res.status(400).json({ error: 'optimizer must be adamw or muon' });
      return;
    }
    if (body.bwd !== undefined && body.bwd !== 'outprod' && body.bwd !== 'mm') {
      res.status(400).json({ error: 'bwd must be outprod or mm' });
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
      adapterType,
      rank: Math.trunc(rank),
      alpha: Math.trunc(alpha),
      lokrDim: Math.trunc(lokrDim),
      lokrAlpha,
      lokrFactor: Math.trunc(lokrFactor),
      lokrDecomposeBoth,
      // Default ON: an attention-only DiT LoRA leaves the MLP projections —
      // where most of the timbre lives — frozen. Same !== false shape as
      // channelBalance/stopEngine, so an omitting client gets the default.
      targetMlp: body.targetMlp !== false,
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
      // K2: omitted-field fallback is 'none' for lokr, 'flow_snr' for lora —
      // an explicit value from the client always wins either way.
      lossWeighting: body.lossWeighting === 'none' ? 'none'
        : body.lossWeighting === 'flow_snr' ? 'flow_snr'
          : (isLokr ? 'none' : 'flow_snr'),
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
      // Frozen-weight mirror precision. Default is 'bf16' (2026-07-29); only
      // the exact string 'f32' opts back out. The engine itself falls back to
      // f32 with a warning on a non-CUDA backend (dit-train-run.h).
      mirror: body.mirror === 'f32' ? 'f32' : 'bf16',
      // MUL_MAT activation-gradient formulation. Default 'mm' (2026-07-29);
      // only the exact string 'outprod' opts back out to upstream ggml's
      // F32-only out_prod backward.
      bwd: body.bwd === 'outprod' ? 'outprod' : 'mm',
      // DEFAULT MUON (2026-07-30, after the ear test). Only the exact string
      // 'adamw' opts back out. Measured on gunship_unicorn: 161 epochs to ma5
      // 0.6 vs AdamW's 227, and with bucketing that is ~1.23x on wall-clock —
      // Rob's own run reached 0.6 in ~5 minutes and the adapter was judged
      // perfect by ear, which is what made this a default rather than a flag.
      optimizer: body.optimizer === 'adamw' ? 'adamw' : 'muon',
      muonLrScale,
      muonMomentum,
      muonNsSteps: Math.trunc(muonNsSteps),
      muonMinDim: Math.trunc(muonMinDim),
      batch: Math.trunc(batch),
      ckptSegments: Math.trunc(ckptSegments),
      stages: resolvedStages,
      overwrite: body.overwrite === true,
      stopEngine: body.stopEngine !== false,
    };

    const job = queue.startTrainDitJob(ds.id, opts);
    const adapterDesc = isLokr ? `lokr dim${opts.lokrDim}` : `lora r${opts.rank}`;
    console.log(
      `[Training] train-dit job ${job.id} queued — ${adapterDesc}, variant ${variantKey} → ${adapterDir}`);
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] train-dit start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/datasets/:id/train-dit', async (req: Request, res: Response) => {
  try {
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    res.json(readTrainDitStatus(userId, ds, {
      variantKey: typeof req.query.variantKey === 'string' ? req.query.variantKey : undefined,
      adapterName: typeof req.query.adapterName === 'string' ? req.query.adapterName : undefined,
    }));
  } catch (err: any) {
    console.error(`[Training] train-dit status failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

// ── Codes audition (codes-preview §3.2) ──────────────────────────────────

const AUDITION_SLOTS: readonly string[] = ['base', 'adapter'];

/**
 * C16 — stricter than the Create panel, which accepts any absolute path.
 *
 * An lmAdapter is either '' (base LM), a bare registry name, or a directory
 * INSIDE adapters/lm that actually holds an adapter_model.safetensors. The third
 * clause is what lets a milestone dir
 * (`<adapters>/lm/<name>-<size>/milestones/loss_<v>`) through while refusing an
 * arbitrary absolute path. A Training-Studio route has no reason to be as
 * permissive as the Create panel.
 */
function auditionAdapterError(value: string, userId: string): string | null {
  if (value === '') return null;
  // The regex alone accepts '.', '..', '...' and '-'. Both other adapter-name
  // validators in this file (train-lm and train-dit) pair it with the
  // safeAdapterName cross-check for exactly that reason; being the one looser
  // validator in the file is how a future change to the engine's
  // resolve_lm_adapter_path heuristic turns into a real traversal. Today it is
  // contained — the engine only takes its path-fallback branch when the value
  // holds a '/' or '\' — but C16 is this plan's stated security clause and it
  // should not be the weakest of the three.
  if (ADAPTER_NAME_RE.test(value)) {
    return safeAdapterName(value) === value
      ? null
      : 'lmAdapter must match [A-Za-z0-9._-]{1,64} and cannot start with a dot';
  }
  // Any planner-adapter root counts: the per-size lm-* dirs plus the legacy
  // flat lm/ (adapterLayout.ts). Milestone dirs inside an adapter dir pass too.
  const roots = lmAdapterRoots(userId).map(r => r.dir);
  const resolved = path.resolve(value);
  const insideSome = roots.some(r => isInside(r, resolved) && path.resolve(r) !== resolved);
  if (!insideSome) {
    return `lmAdapter must be a registry name or a directory inside ${roots.join(' | ')}`;
  }
  // A LoKr adapter dir has NO adapter_model.safetensors — its weights live in
  // lokr_weights.safetensors and there is deliberately no adapter_config.json
  // (alpha rides the per-module tensors + __metadata__.lokr_config). Checking
  // only the PEFT name rejected every LoKr adapter before the engine ever saw
  // it; lm-adapter.h reads both layouts.
  if (!hasWeights(resolved)) {
    return `lmAdapter has no adapter_model.safetensors or lokr_weights.safetensors: ${resolved}`;
  }
  return null;
}

router.post('/datasets/:id/audition', async (req: Request, res: Response) => {
  try {
    const userId = req.user!.userId;
    const body = (req.body || {}) as AuditionOptions;

    // ── 1. dataset + queue ───────────────────────────────────────────────
    const ds = repo.getDataset(userId, req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (queue.activeJobForDataset(ds.id)) {
      res.status(409).json({ error: 'A job is already running for this dataset' });
      return;
    }

    // ── 2. the engine must be UP — the opposite of every ace-train job ───
    if (isEngineSuspended()) {
      res.status(503).json({
        error: 'Engine is stopped for a training run — audition needs ace-server up',
      });
      return;
    }
    // Suspended and down are different things, and this feature's own list
    // response already reports them separately (engineReady, engineSuspended) so
    // the client can tell them apart. Without this check a crashed ace-server
    // whose 3 s respawn has not landed yields a 202 + jobId, and ~15 s later the
    // job dies carrying the bare string 'fetch failed' — an error the user
    // cannot act on, where an immediate 503 is actionable.
    if (!engineReady) {
      res.status(503).json({
        error: 'Engine is not running — audition needs ace-server up',
      });
      return;
    }

    // ── 3. caption ───────────────────────────────────────────────────────
    // An EMPTY caption is legal when a sampleId is supplied, and that is not a
    // loophole — it is the only way §5.3's documented fallback can ever fire.
    // C12(a) says a sample-sourced audition must use "the literal strings the
    // trainer conditioned on", which is the lm_codes.jsonl row's caption, i.e.
    // lm_apply_tag(sidecar caption, custom_tag, position) — the TAGGED string
    // carrying the trigger word. The dataset sidecar caption the client can see
    // is the UNTAGGED one; sending it would run the adapter's prompt without its
    // trigger, both sides would emit near-identical plans, and the UI would
    // report "the adapter had no effect" — the exact misattribution C14 exists
    // to prevent. With an unconditional 400 here, resolveAuditionInputs' row
    // fallback was dead code for this route.
    const caption = typeof body.caption === 'string' ? body.caption.trim() : '';
    const hasSampleId = typeof body.sampleId === 'string' && body.sampleId.trim() !== '';
    if (!caption && !hasSampleId) {
      res.status(400).json({ error: 'caption is required' });
      return;
    }
    if (caption.length > 4000) {
      res.status(400).json({ error: 'caption must be at most 4000 characters' });
      return;
    }

    // ── 4. sides ─────────────────────────────────────────────────────────
    const rawSides = Array.isArray(body.sides) ? body.sides : [];
    if (rawSides.length < 1 || rawSides.length > 2) {
      res.status(400).json({ error: 'sides must hold 1 or 2 entries' });
      return;
    }
    const seenSlots = new Set<string>();
    const sides: AuditionSideSpec[] = [];
    for (const raw of rawSides) {
      const slot = typeof raw?.slot === 'string' ? raw.slot : '';
      if (!AUDITION_SLOTS.includes(slot)) {
        res.status(400).json({ error: "every side needs a slot of 'base' or 'adapter'" });
        return;
      }
      if (seenSlots.has(slot)) {
        res.status(400).json({ error: 'sides must have distinct slots' });
        return;
      }
      seenSlots.add(slot);

      const label = typeof raw?.label === 'string' ? raw.label : '';
      if (label.length > 64) {
        res.status(400).json({ error: 'side label must be at most 64 characters' });
        return;
      }

      // ── 5. adapter path safety (C16) ───────────────────────────────────
      const lmAdapter = typeof raw?.lmAdapter === 'string' ? raw.lmAdapter.trim() : '';
      const adapterFailure = auditionAdapterError(lmAdapter, userId);
      if (adapterFailure) {
        res.status(400).json({ error: adapterFailure });
        return;
      }

      sides.push({
        slot: slot as AuditionSideSpec['slot'],
        label,
        lmAdapter,
        lmAdapterScale: numOpt(raw?.lmAdapterScale, 1.0),
      });
    }

    // ── 6. variantKey ────────────────────────────────────────────────────
    const requestedVariant = typeof body.variantKey === 'string' ? body.variantKey.trim() : '';
    if (requestedVariant && !variantExists(userId, ds.slug, requestedVariant)) {
      res.status(400).json({ error: `Unknown preprocess variant: ${requestedVariant}` });
      return;
    }

    // ── 7. numeric fields clamp silently in resolveAuditionInputs (§3.2.7).
    // They never 400: an out-of-range temperature is a slider mishap, not a
    // reason to refuse an audition.
    const opts: AuditionOptions = {
      ...body,
      caption,
      sides,
      variantKey: requestedVariant || undefined,
    };

    const job = queue.startAuditionJob(ds.id, opts);
    console.log(
      `[Training] audition job ${job.id} queued — ${sides.length} side(s), dataset ${ds.slug}`);
    res.status(202).json({ jobId: job.id });
  } catch (err: any) {
    console.error(`[Training] audition start failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

router.get('/datasets/:id/audition', (req: Request, res: Response) => {
  try {
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    // Lazy prune (C11) — there is deliberately no boot hook.
    prunePreviews();

    const limit = numOpt(req.query.limit, 20);
    const payload: AuditionListResponse = {
      previews: listPreviews(ds.id, limit),
      engineReady,
      engineSuspended: isEngineSuspended(),
    };
    res.json(payload);
  } catch (err: any) {
    console.error(`[Training] audition list failed: ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

/**
 * SYNCHRONOUS (C7) — stored codes straight through the detokenizer + VAE. There
 * is no /lm call, which is exactly why this is fast enough not to be a job.
 * If gate V4 ever measures a warm total over 10 s, promote it to the 'audition'
 * job kind (the runner already handles a single-sided decode-only job).
 */
router.post('/datasets/:id/samples/:sampleId/audition', async (req: Request, res: Response) => {
  try {
  const userId = req.user!.userId;
    const ds = repo.getDataset(userId, req.params.id as string);
    if (!ds) {
      res.status(404).json({ error: 'Dataset not found' });
      return;
    }
    if (isEngineSuspended()) {
      res.status(503).json({
        error: 'Engine is stopped for a training run — audition needs ace-server up',
      });
      return;
    }

    const samples = await buildSamples(ds);
    const sample = samples.find(s => s.sampleId === req.params.sampleId);
    if (!sample) {
      res.status(404).json({ error: 'Sample not found' });
      return;
    }

    const format = (req.body || {}).format === 'mp3' ? 'mp3' : 'wav16';
    res.json(await decodeStoredCodes(ds, sample.sampleId, format));
  } catch (err: any) {
    if (err instanceof AuditionError) {
      res.status(err.status).json({ error: err.message });
      return;
    }
    const message = err?.message || String(err);
    // aceClient.codesDecode throws these two by exact text.
    if (/timed out/i.test(message)) {
      res.status(504).json({ error: 'Decode timed out after 90s' });
      return;
    }
    if (/fetch failed|ECONNREFUSED|unreachable|aborted/i.test(message)) {
      res.status(503).json({ error: `Engine unreachable: ${message}` });
      return;
    }
    console.error(`[Training] sample audition failed: ${message}`);
    res.status(500).json({ error: message });
  }
});

export default router;
