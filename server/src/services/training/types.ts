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

export type TrainingJobKind =
  | 'label' | 'enhance-genius' | 'enhance-caption' | 'build' | 'preprocess' | 'train-lm';

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
  defaultLanguage: string;    // language the user KNOWS the corpus is in; overrides understand's guess
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
  /** Subdirs of data/training/tensors/<slug> holding a preprocess_meta.json. */
  preprocessedVariants: number;
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
  // 'essentia' | 'understand' | 'waiting-for-engine' | 'genius' | 'llm' | 'writing' | 'build'
  // kind==='preprocess' adds: 'engine-stop' | 'loading-models' | 'preprocess' | 'stats' | 'engine-restart'
  // kind==='train-lm'   adds: 'engine-stop' | 'loading-models' | 'extract' | 'train' | 'export' | 'engine-restart'
  phase: string;
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
  preprocess: {
    available: boolean;      // ace-train binary found on disk
    binPath: string;         // '' when not found
    ditModels: string[];     // from the cached /props snapshot
    vaeModels: string[];
    textEncoders: string[];
    defaultDit: string;      // first /bf16/i match, else ''
    defaultVae: string;
    defaultTextEnc: string;
    modelsCachedAt: number;  // epoch ms of the snapshot; 0 = never probed
    engineSuspended: boolean;// true while a preprocess job owns the GPU
  };
  trainLm: TrainLmCapabilities;
}

// ─── SSE stream ───────────────────────────────────────────────────────────

/** Structured training numbers (L21). Additive member of TrainingStreamEvent.
 *  Consumers MUST ignore unknown `metric` values rather than throwing. */
export interface TrainingMetricEvent {
  type: 'metric';
  metric: 'vram' | 'data' | 'step' | 'epoch' | 'milestone';
  ts: number;
  // epoch / step
  epoch?: number;
  epochs?: number;
  step?: number;
  totalSteps?: number;
  loss?: number;
  lr?: number;
  gradNorm?: number;
  clipScale?: number;
  etaMs?: number;
  ms?: number;
  best?: boolean;
  // vram
  freeMb?: number;
  totalMb?: number;
  estMb?: number;
  vramMb?: number;
  maxLen?: number;
  // data
  samples?: number;
  skippedLong?: number;
  stepsPerEpoch?: number;
  loraParams?: number;
  // milestone
  path?: string;
}

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
  | { type: 'status'; status: TrainingJobStatus; error?: string }   // terminal; server closes after this
  | TrainingMetricEvent;

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
  defaultLanguage?: string;   // default 'english'
}

export type PatchDatasetInput = Partial<Pick<
  TrainingDatasetSummary,
  'name' | 'customTag' | 'tagPosition' | 'genreRatio' | 'defaultArtist' | 'defaultAlbum' | 'defaultGenre' | 'defaultLanguage' | 'recursive'
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
  useEssentia?: boolean;   // default true  — local BPM/key
  useGenius?: boolean;     // default false — canonical lyrics (needs token)
  useCaption?: boolean;    // default false — LLM caption+genre (audio-grounded on gemini)
  useUnderstand?: boolean; // default false — LEGACY /understand path (2026-07-27 pivot)
  mergePolicy?: MergePolicy;
  understand?: UnderstandOverrides;
  caption?: { provider?: string; model?: string };
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

// ─── Preprocess (Training Studio phase 2) ─────────────────────────────────
export type PreprocessDtype   = 'f32' | 'bf16';
export type PreprocessCompat  = 'hotstep' | 'sidestep';
export type PreprocessNormalize = 'none' | 'peak';

export interface PreprocessOptions {
  /** DiT base model name from capabilities.preprocess.ditModels.
   *  Omitted → server uses capabilities.preprocess.defaultDit; if that is
   *  empty the request is rejected with 400. */
  ditModel?: string;
  vaeModel?: string;          // default capabilities.preprocess.defaultVae
  textEncoder?: string;       // default capabilities.preprocess.defaultTextEnc
  sampleIds?: string[];       // omit = every non-excluded, non-missing sample
  maxDuration?: number;       // default 240; 0 = no truncation
  normalize?: PreprocessNormalize;  // default 'peak'
  targetDb?: number;          // default -1.0
  dtype?: PreprocessDtype;    // default 'f32'
  compat?: PreprocessCompat;  // default 'hotstep'
  maxCaptionTokens?: number;  // default 256
  maxLyricTokens?: number;    // default 512
  vaeChunk?: number;          // default 384    (latent frames; ~15.4s tiles — VRAM-bounded)
  vaeOverlap?: number;        // default 64     (latent frames)
  overwrite?: boolean;        // default false
  stopEngine?: boolean;       // default TRUE — stop ace-server for the job
  outputDir?: string;         // default data/training/tensors/<slug>/<variantKey>
}

export interface PreprocessVariantStatus {
  variantKey: string;      // directory name, e.g. 'acestep-v15-base-BF16'
  modelVariant: string;    // exact DiT file name from preprocess_meta.json
  outputDir: string;       // absolute
  createdAt: string;       // ISO, '' if unknown
  compat: string;          // 'hotstep' | 'sidestep' | '' (unknown/legacy)
  dtype: string;           // 'F32' | 'BF16' | ''
  total: number;           // songs the run attempted
  processed: number;
  failed: number;
  cachedCount: number;     // .safetensors files actually present on disk
  staleCount: number;      // cached entries whose source audio size/mtime changed
  missingCount: number;    // current dataset samples with no cache entry
  bytes: number;           // total size of the variant dir
  hasChannelStats: boolean;
}

export interface PreprocessStatus {
  tensorsRoot: string;                    // data/training/tensors/<slug>
  variants: PreprocessVariantStatus[];    // newest createdAt first
}

// ─── LM LoRA training (Training Studio phase 3) ───────────────────────────
// Spec: docs/plans/2026-07-27-lm-trainer-implementation.md §2.7
export type LmSize = '0.6B' | '1.7B';
export type TrainLmStage = 'extract' | 'train' | 'export';

export interface TrainLmOptions {
  /** Base LM size. v1 supports 0.6B and 1.7B only; 4B is rejected with 400. */
  lmSize?: LmSize;                 // default '0.6B'
  /** Explicit LM GGUF name from capabilities.trainLm.lmModels. Omit to let the
   *  server pick the BF16 model matching lmSize. */
  lmModel?: string;
  /** Which preprocess variant to train from. Omit = the newest variant. */
  variantKey?: string;
  /** Adapter directory stem; final dir is `<adapterName>-<lmSize>`.
   *  Omit = the dataset slug. */
  adapterName?: string;
  targetLoss?: number;             // default 0.4;  0 disables auto-stop
  epochs?: number;                 // default 16 (hard cap)
  rank?: number;                   // default 16
  alpha?: number;                  // default 32
  learningRate?: number;           // default 0.0001
  gradAccum?: number;              // default 4
  gradClip?: number;               // default 1.0;  0 disables
  warmupRatio?: number;            // default 0.05
  weightDecay?: number;            // default 0.01
  maxLen?: number;                 // default 0 = auto-fit from free VRAM
  seed?: number;                   // default 42
  lossOnCot?: boolean;             // default true
  order?: 'shuffle' | 'fixed';     // default 'shuffle'
  milestoneStep?: number;          // default 0.1;  0 disables
  milestoneKeep?: number;          // default 6
  stages?: TrainLmStage[];         // default ['extract','train','export']
  overwrite?: boolean;             // default false — re-extract every song
  stopEngine?: boolean;            // default TRUE — stop ace-server for the job
}

export interface TrainLmEpoch {
  epoch: number;
  loss: number;
  lr: number;
  gradNorm: number;
  ms: number;
}

export interface TrainLmStatus {
  /** '' when the dataset has no preprocessed variant at all. */
  variantKey: string;
  tensorsDir: string;              // absolute, '' if none
  /** lm_codes.jsonl */
  codesPath: string;               // absolute, '' if none
  codesExists: boolean;
  codesCount: number;              // rows in lm_codes.jsonl
  codesStale: number;              // rows whose tensor file changed since extract
  codesMissing: number;            // cached tensors with no codes row
  adapterName: string;             // stem, without the -<size> suffix
  adapterDir: string;              // absolute; where the next run would write
  adapterExists: boolean;          // adapter_model.safetensors present
  adapterBytes: number;
  lmSize: string;                  // from lm_train_log.json, '' if unknown
  trainedAt: string;               // ISO, '' if unknown
  finalLoss: number;               // -1 if unknown
  bestLoss: number;                // -1 if unknown
  epochsRun: number;               // 0 if unknown
  targetLoss: number;              // -1 if unknown
  stoppedOnTarget: boolean;
  epochs: TrainLmEpoch[];          // [] if unknown
  milestones: Array<{ loss: number; epoch: number; path: string }>;
}

export interface TrainLmCapabilities {
  available: boolean;              // ace-train binary found
  lmModels: string[];              // from the cached /props snapshot (models.lm)
  /** Sizes this build can train. v1: ['0.6B','1.7B']. */
  sizes: LmSize[];
  /** size -> preferred BF16 model name; '' when none is installed. */
  defaultLmBySize: Record<string, string>;
  adaptersRoot: string;            // <adapters>/lm
}

/** camelCase mirror of a `training_datasets` row. Identical in shape to
 *  TrainingDatasetSummary — the summary IS the row (D3: no per-sample table). */
export type TrainingDatasetRow = TrainingDatasetSummary;
