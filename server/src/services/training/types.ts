// training/types.ts — Dataset Studio API contract (FROZEN)
//
// Single source of truth for every type crossing the /api/training boundary.
// ui/src/services/trainingApi.ts holds a verbatim copy of the same text — keep
// the two in sync by hand; they are deliberately not shared at build time.
//
// Spec: docs/plans/2026-07-27-dataset-studio-implementation.md §2

// ─── Core enums ────────────────────────────────────────────────────────────
export type TagPosition = 'prepend' | 'append' | 'replace';

export type MergePolicy =
  | 'fill_missing'        // default: only write fields currently empty
  | 'overwrite_caption'
  | 'overwrite_lyrics'
  | 'overwrite_all';

export type TrainingJobKind = 'label' | 'enhance-genius' | 'enhance-caption' | 'build';

export type TrainingJobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

export type SampleLabelStatus =
  | 'unlabeled'   // no sidecar caption
  | 'labeled'     // sidecar caption present
  | 'pending'     // queued in an active job
  | 'processing'  // currently being labeled
  | 'error';      // last label attempt failed (see `error`)

export type FieldSource = 'sidecar' | 'understand' | 'essentia' | 'genius' | 'llm' | 'tags' | 'filename' | 'user';

// ─── Dataset ──────────────────────────────────────────────────────────────
export interface TrainingDatasetSummary {
  id: string;                 // uuid v4
  slug: string;               // filesystem-safe, unique
  name: string;
  sourceDir: string;          // absolute path to the user's audio folder
  recursive: boolean;
  customTag: string;          // trigger word; '' = none
  tagPosition: TagPosition;
  genreRatio: number;         // 0-100
  defaultArtist: string;
  defaultAlbum: string;
  defaultGenre: string;
  sampleCount: number;
  labeledCount: number;
  excludedCount: number;
  status: 'draft' | 'labeling' | 'labeled' | 'built' | 'error';
  builtAt: string;            // ISO or ''
  datasetJsonPath: string;    // absolute path or ''
  createdAt: string;          // ISO
  updatedAt: string;          // ISO
}

export interface TrainingSample {
  sampleId: string;           // sha1(relPath).slice(0,16) — stable
  relPath: string;            // path relative to sourceDir, forward slashes
  audioPath: string;          // absolute, OS-native separators
  filename: string;           // basename with extension
  sidecarPath: string;        // absolute path of <stem>.txt
  sidecarExists: boolean;
  fileMissing: boolean;       // audio file no longer on disk

  // Sidecar-backed editable fields
  caption: string;
  genre: string;
  bpm: number | null;
  key: string;                // e.g. "C minor"  (sidecar key `key`)
  signature: string;          // e.g. "4/4"
  language: string;           // ISO code or ''
  isInstrumental: boolean;
  lyrics: string;
  customTag: string;          // per-sample override; '' inherits dataset tag
  repeat: number;             // >= 1
  promptOverride: string | null;

  // Studio-private (labels/<sampleId>.json)
  excluded: boolean;
  labelStatus: SampleLabelStatus;
  error: string | null;
  hasAudioCodes: boolean;
  sources: Partial<Record<'caption' | 'lyrics' | 'genre' | 'bpm' | 'key' | 'signature' | 'language', FieldSource>>;

  // Derived / read-only
  duration: number;           // seconds, 0 if unknown
  sizeBytes: number;
  tagArtist: string;          // from embedded audio tags
  tagTitle: string;
  tagAlbum: string;
  labeledAt: string;          // ISO or ''
}

export interface TrainingDatasetDetail extends TrainingDatasetSummary {
  samples: TrainingSample[];
  warnings: string[];         // e.g. "Only 6 samples — 10+ recommended"
  activeJobId: string | null;
}

// ─── Jobs ─────────────────────────────────────────────────────────────────
export interface TrainingJobSummary {
  id: string;
  datasetId: string;
  kind: TrainingJobKind;
  status: TrainingJobStatus;
  total: number;
  done: number;
  failed: number;
  currentSampleId: string | null;
  phase: string;              // 'essentia' | 'understand' | 'waiting-for-engine' | 'genius' | 'llm' | 'writing' | 'build'
  engineQueueDepth: number;   // # of ace-server jobs ahead of ours; 0 if unknown
  error: string | null;
  createdAt: number;          // epoch ms
  startedAt: number | null;
  finishedAt: number | null;
}

// ─── Capabilities ─────────────────────────────────────────────────────────
export interface TrainingCapabilities {
  engine: {
    up: boolean;
    ready: boolean;               // from /api/health engine.ready
    understandSupported: boolean; // lm && dit && vae registries all non-empty
    missingModels: string[];      // e.g. ['lm'] — which registries are empty
    queueDepth: number;           // ace-server GET /jobs length
    lmModels: string[];           // LM registry names, for the understand model picker
    defaultLmModel: string;       // server's pick (biggest model, fast quant) when the UI sends none
  };
  essentia: { available: boolean; binPath: string };
  genius: { configured: boolean };
  llm: {
    configured: boolean;
    defaultProvider: string;
    providers: Array<{ id: string; name: string; available: boolean; models: string[]; defaultModel: string }>;
  };
}

// ─── SSE stream ───────────────────────────────────────────────────────────
export type TrainingStreamEvent =
  | { type: 'job'; job: TrainingJobSummary }                       // first event on connect + on any status change
  | {
      type: 'progress'; done: number; total: number; failed: number;
      phase: string; currentSampleId: string | null; engineQueueDepth: number;
    }
  | {
      type: 'sample'; sampleId: string; status: SampleLabelStatus;
      sample?: TrainingSample; error?: string;                     // `sample` present on success
    }
  | { type: 'log'; level: 'info' | 'warn' | 'error'; message: string; ts: number }
  | { type: 'status'; status: TrainingJobStatus; error?: string };  // terminal; server closes after this

// ─── Request payloads ─────────────────────────────────────────────────────

export interface ScanPreview {
  root: string;
  audioFiles: number;
  withSidecar: number;
  withCaption: number;
  hasDatasetJson: boolean;
  extensions: Record<string, number>;
  sampleNames: string[];
}

export interface CreateDatasetInput {
  name: string;
  sourceDir: string;
  recursive?: boolean;
  customTag?: string;
  tagPosition?: TagPosition;
  genreRatio?: number;
  defaultArtist?: string;
  defaultAlbum?: string;
  defaultGenre?: string;
}

export type PatchDatasetInput = Partial<Pick<
  TrainingDatasetSummary,
  'name' | 'customTag' | 'tagPosition' | 'genreRatio' | 'defaultArtist' | 'defaultAlbum' | 'defaultGenre' | 'recursive'
>>;

export type PatchSampleInput = Partial<Pick<
  TrainingSample,
  'caption' | 'genre' | 'bpm' | 'key' | 'signature' | 'language' | 'isInstrumental'
  | 'lyrics' | 'customTag' | 'repeat' | 'promptOverride' | 'excluded'
>>;

export type BulkSetInput = Partial<Pick<
  TrainingSample,
  'excluded' | 'isInstrumental' | 'genre' | 'customTag' | 'language' | 'repeat'
>>;

export interface BulkResult {
  updated: number;
  failed: Array<{ sampleId: string; error: string }>;
}

/** Optional `/understand` overrides forwarded into the engine `request` part. */
export interface UnderstandOverrides {
  lmModel?: string;
  synthModel?: string;
  lmTemperature?: number;
  lmTopP?: number;
  lmTopK?: number;
  seed?: number;
}

export interface LabelOptions {
  sampleIds?: string[];
  scope?: 'all' | 'unlabeled';
  useEssentia?: boolean;
  useUnderstand?: boolean;
  mergePolicy?: MergePolicy;
  understand?: UnderstandOverrides;
}

export interface GeniusOptions {
  sampleIds?: string[];
  mergePolicy?: MergePolicy;
  artist?: string;
  album?: string;
  sanitizeHeaders?: boolean;
}

export interface CaptionOptions {
  sampleIds?: string[];
  provider?: string;
  model?: string;
  mergePolicy?: MergePolicy;
  includeLyricsExcerpt?: boolean;
  temperature?: number;
}

export interface BuildOptions {
  outputPath?: string;
}

/** camelCase mirror of a `training_datasets` row. Identical in shape to
 *  TrainingDatasetSummary — the summary IS the row (D3: no per-sample table). */
export type TrainingDatasetRow = TrainingDatasetSummary;
