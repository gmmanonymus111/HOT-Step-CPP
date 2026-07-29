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
  | 'label' | 'enhance-genius' | 'enhance-caption' | 'build'
  | 'preprocess' | 'train-lm' | 'train-dit'
  | 'audition';

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
  // kind==='train-dit'  adds: 'engine-stop' | 'loading-models' | 'train' | 'export' | 'engine-restart'
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
  trainDit: TrainDitCapabilities;
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
  // train-dit additions (§2.6). No new union member — a DiT job reuses every
  // metric name above and adds these five.
  crop?: number;      // latent frames in the active crop window
  layers?: number;    // trained decoder-layer count (top-K)
  t?: number;         // last micro-step timestep (telemetry)
  ma5?: number;       // 5-epoch moving average (the target-loss quantity)
  rawLoss?: number;   // unweighted MSE, for telemetry
  // train-lm low-VRAM additions (4B plan §2.2). All optional, additive.
  mode?: string;      // 'naive' | 'lowvram'
  baseMb?: number;    // resident base weights
  ckptMb?: number;    // per-layer checkpoint buffers
  segPeakMb?: number; // transient peak of one segment graph
  // train-lm speed levers (2026-07-28 plan §2.2/§2.4). All optional, additive.
  // §2.4's verbatim three:
  weights?: string; batch?: number; padPct?: number;
  // …plus the rest of §2.2's JSONL fields. These are NOT in §2.4's block, but
  // §2.2 mandates them on the `vram` and `data` events and the relay whitelists
  // every field by name, so without a home on this interface they could never
  // reach a consumer. Same class of gap as the 4B plan's mode/baseMb/ckptMb.
  batchSource?: string;  // vram: 'user' | 'auto'
  batches?: number;      // data: micro-batches per epoch
  padTokens?: number;    // data: padding tokens added by batching
  // NB `samples` (declared above under `data`) is reused by `step` as the
  // per-optimizer-step sample count (micro * B_cur); no new field needed.
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
export type LmSize = '0.6B' | '1.7B' | '4B';
export type TrainLmStage = 'extract' | 'train' | 'export';

export interface TrainLmOptions {
  /** Base LM size. 4B trains through the engine's low-VRAM path (per-layer
   *  gradient checkpointing + chunked CE); 0.6B/1.7B stay on the naive path. */
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
  /** Per-layer gradient checkpointing + chunked CE. 'auto' (default) turns it on
   *  for 4B, and for smaller bases only when the naive path would have to skip
   *  full-song samples. */
  lowVram?: 'auto' | 'on' | 'off';   // default 'auto'
  attnHeadBlock?: number;            // default 0 = engine picks
  chunk?: number;                    // default 0 = engine default (128)
  // ── LM speed levers (2026-07-28 plan §2.4). Verbatim contract text. ──────
  /** Projection GEMM dtype. 'f32-window' is the shipped per-segment F32 weight
   *  cast (still the CLI's own default, ace-train.cpp). 'bf16' runs the
   *  projections in BF16 and rewrites the backward activation-gradient nodes;
   *  ~1.65-1.78x on the projection mix, but it changes the trained weights
   *  (BF16 gradient rounding) and requires the CUDA backend + a BF16-native
   *  base + low-VRAM (the engine forces low-VRAM mode itself). Neither of the
   *  other two is a hard fatal any more (2026-07-29): a non-CUDA backend or a
   *  non-BF16-native base each warn and fall back to 'f32-window'
   *  (lm-train-run.h) — the same graceful-fallback treatment as the DiT
   *  mirror. The SERVER default is 'bf16' (training.ts train-lm handler). */
  weights?: 'f32-window' | 'bf16';   // default 'bf16'
  /** Micro-batch size 1..8, or 'auto' (largest of {1,2,4} that fits without
   *  dropping a song). >1 forces low-VRAM mode. */
  batch?: number | 'auto';           // default 1
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
  adapterName: string;             // artist name (per-base layout: unsuffixed)
  adapterDir: string;              // absolute; the newest trained run, else the artist dir
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
  /** Sizes this build can train: ['0.6B','1.7B','4B']. */
  sizes: LmSize[];
  /** size -> preferred BF16 model name; '' when none is installed. */
  defaultLmBySize: Record<string, string>;
  adaptersRoot: string;            // <adapters>/lm
}

// ─── DiT LoRA/LoKR training (Training Studio phase 4 + D22 LoKR) ──────────
// Spec: docs/plans/2026-07-28-dit-trainer-implementation.md §2.6
//       docs/plans/2026-07-28-lokr-dit-training.md §3
export type DitAdapterType = 'lora' | 'lokr';
export type TrainDitStage  = 'train' | 'export';

export interface TrainDitOptions {
  /** Which preprocess variant to train from. Omit = the newest variant. */
  variantKey?: string;
  /** Adapter directory name under <adapters>/. Omit = the dataset slug. */
  adapterName?: string;
  /** Engine CLI default stays 'lora' for back-compat; the UI form defaults
   *  to 'lokr' (K1). */
  adapterType?: DitAdapterType;    // default 'lora'
  rank?: number;                   // default 128 (adapterType==='lora' only)
  alpha?: number;                  // default 256 (adapterType==='lora' only)
  /** LyCORIS LoKR factors, per Rob's Uber-LoKR-4 preset (K2). Ignored unless
   *  adapterType==='lokr'. */
  lokrDim?: number;                // default 512, range [4,4096]
  lokrAlpha?: number;              // default 512, range (0,8192]; 0 -> dim
  lokrFactor?: number;             // default 6, -1 or [2,64]
  lokrDecomposeBoth?: boolean;     // default true (parity knob; inert at dim 512)
  targetMlp?: boolean;             // default true
  layers?: number;                 // default 0 = auto (top-K depth)
  crop?: number;                   // default 0 = auto-fit
  cropMin?: number;                // default 375
  cropMax?: number;                // default 1250
  targetLoss?: number;             // default 0.4 (lora) / 0.6 (lokr, K2); 0 disables auto-stop
  epochs?: number;                 // default 400 (hard cap)
  learningRate?: number;           // default 0.0005 (lora) / 0.01 (lokr, K2)
  gradAccum?: number;              // default 4
  gradClip?: number;               // default 1.0;  0 disables
  warmupRatio?: number;            // default 0.05
  weightDecay?: number;            // default 0.01 (lora) / 0.001 (lokr, K2)
  lossWeighting?: 'none' | 'flow_snr';   // default 'flow_snr' (lora) / 'none' (lokr, K2)
  snrGamma?: number;               // default 5.0
  tBias?: number;                  // default 0.5
  channelBalance?: boolean;        // default true
  timestepMu?: number;             // default -0.4
  timestepSigma?: number;          // default 1.0
  tMin?: number;                   // default 0
  tMax?: number;                   // default 1
  cfgRatio?: number;               // default 0.15
  genreRatio?: number;             // default 30 (percent)
  seed?: number;                   // default 42
  order?: 'shuffle' | 'fixed';     // default 'shuffle'
  milestoneStep?: number;          // default 0.1;  0 disables
  milestoneKeep?: number;          // default 6
  vramReserveMb?: number;          // default 2048
  /** Frozen-weight mirror precision. Default 'bf16' (2026-07-29) — keeps the
   *  trainable layers' matmul weights in the base's native BF16 instead of
   *  promoting them to F32, roughly halving the mirror's VRAM. Needs the
   *  patched out_prod in engine/patches/bf16-out-prod.patch and the CUDA
   *  backend; on CPU/Vulkan the engine warns and falls back to 'f32' itself
   *  (dit-train-run.h), so an explicit 'f32' request is the only way to opt
   *  out deliberately. */
  mirror?: 'f32' | 'bf16';         // default 'bf16'
  /** Crops per micro-batch, from that many DIFFERENT songs (design §2.2 /
   *  C5). Effective samples/optimizer-step is batch x gradAccum.
   *  Default 1 = OFF (2026-07-29, measured): batching is ~2.5x SLOWER at full
   *  depth on a 32 GB card and ~2.4x faster on shallow/partial-depth runs, so
   *  raising it is a deliberate choice for the latter. */
  batch?: number;                  // default 1, range [1,16]
  /** Gradient-checkpointing segments (design §2.2 / C3). 0 = off (today's
   *  monolithic graph); 1 = auto (engine picks segment count from free
   *  VRAM); 2-32 = that many segments, fixed. */
  ckptSegments?: number;           // default 1 (auto); 0 = off; 2-32 fixed
  stages?: TrainDitStage[];        // default ['train','export']
  overwrite?: boolean;             // default false
  stopEngine?: boolean;            // default TRUE
}

/** Structurally identical to TrainLmEpoch so LossSparkline is reused unedited. */
export interface TrainDitEpoch {
  epoch: number;
  loss: number;
  lr: number;
  gradNorm: number;
  ms: number;
}

export interface TrainDitStatus {
  variantKey: string;              // '' when the dataset has no preprocessed variant
  tensorsDir: string;              // absolute, '' if none
  ditModel: string;                // the variant's base, from preprocess_meta.json
  sampleCount: number;             // usable cached songs in the variant
  channelStats: boolean;           // channel_stats.json present
  adapterName: string;
  adapterDir: string;              // absolute; the newest trained run, else the artist dir
  adapterExists: boolean;          // adapter_model.safetensors OR lokr_weights.safetensors present
  adapterBytes: number;
  /** From dit_train_log.json's config.adapter_type; '' when the log is
   *  missing/unreadable (adapter trained elsewhere, or not trained yet). */
  adapterType: string;
  trainedAt: string;               // ISO, '' if unknown
  finalLoss: number;               // -1 if unknown
  bestLoss: number;                // -1 if unknown
  epochsRun: number;               // 0 if unknown
  targetLoss: number;              // -1 if unknown
  stoppedOnTarget: boolean;
  crop: number;                    // 0 if unknown
  layers: number;                  // 0 if unknown
  partialDepth: boolean;
  epochs: TrainDitEpoch[];         // [] if unknown
  milestones: Array<{ loss: number; epoch: number; path: string }>;
}

export interface TrainDitCapabilities {
  available: boolean;              // ace-train binary found
  adapterTypes: DitAdapterType[];  // ['lora','lokr']
  adaptersRoot: string;            // <adapters>
  /** Minimum total VRAM this build will accept, MB. v1: 16384 (D9). */
  minVramMb: number;
}

/** camelCase mirror of a `training_datasets` row. Identical in shape to
 *  TrainingDatasetSummary — the summary IS the row (D3: no per-sample table). */
export type TrainingDatasetRow = TrainingDatasetSummary;

// ─── Codes audition (pure-LM preview) ─────────────────────────────────────

/** Which half of an A/B a preview file belongs to. A single-sided audition
 *  (sample codes, or a milestone with no baseline) always uses 'base'. */
export type AuditionSlot = 'base' | 'adapter';

export type AuditionKind = 'ab' | 'sample' | 'milestone';

/** One LM run to perform. Both sides of an A/B are identical except lmAdapter. */
export interface AuditionSideSpec {
  slot: AuditionSlot;
  label: string;            // free text shown on the player, e.g. 'Base 4B' / 'abba-4B'
  lmAdapter: string;        // '' = base LM; else a registry name or a path inside adapters/lm
  lmAdapterScale: number;   // default 1.0, clamp 0..2
}

export interface AuditionSideResult {
  slot: AuditionSlot;
  label: string;
  lmAdapter: string;
  lmAdapterScale: number;
  ok: boolean;
  error: string;            // '' when ok
  audioUrl: string;         // '' when !ok — /api/training/previews/<id>/<slot>
  /** Opt-in DiT render of this side's codes (same DiT/seed/steps both sides,
   *  NO sound adapter — the LM adapter stays the only variable). Absent on
   *  previews recorded before the feature and when renderDit was off. */
  renderUrl?: string;       // /api/training/previews/<id>/<slot>-render
  renderMs?: number;        // wall clock of the /synth render
  renderError?: string;     // render failed; the codes sketch above is still valid
  codesCount: number;       // number of 5 Hz codes the LM emitted
  codesSha1: string;        // sha1 of the raw audio_codes string — the determinism receipt
  durationSec: number;      // codesCount / 5, rounded to 0.1
  caption: string;          // the plan the LM handed back (C13)
  lyrics: string;
  bpm: number;              // 0 = not reported
  keyscale: string;
  timesignature: string;
  lmMs: number;             // wall clock of the /lm job
  decodeMs: number;         // wall clock of the /codes-decode job
}

export interface AuditionPreview {
  previewId: string;        // uuid v4
  datasetId: string;
  kind: AuditionKind;
  createdAt: string;        // ISO
  seed: number;             // the lm_seed actually used — never -1
  caption: string;          // the prompt that went IN
  lyrics: string;           // '' = the LM was asked to write its own
  durationSec: number;      // requested duration
  lmModel: string;
  ditModel: string;         // the DiT whose FSQ detokenizer decoded the codes
  vaeModel: string;
  sampleId: string;         // '' unless kind === 'sample' or the prompt came from a sample
  variantKey: string;       // '' when the prompt was free text
  sides: AuditionSideResult[];
  /** Set when the sides carry DiT renders — reproducibility receipt. */
  renderDitModel?: string;
  renderSteps?: number;
}

export interface AuditionOptions {
  sides: AuditionSideSpec[];      // 1..2 entries, slots must be distinct
  caption: string;                // required, 1..4000 chars
  lyrics?: string;                // default ''
  seed?: number;                  // >= 0; omitted or < 0 → the server picks one and records it
  durationSec?: number;           // default 30, clamp 10..300
  lmModel?: string;               // default: the newest variant's LM, else registry[0]
  ditModel?: string;              // default: variantDitModel(slug, variantKey)
  vaeModel?: string;              // default: engine's resolve_name default
  variantKey?: string;            // default: newestVariantKey(slug)
  sampleId?: string;              // when set, caption/lyrics default from its lm_codes.jsonl row
  temperature?: number;           // default 0.85, clamp 0.1..2
  topP?: number;                  // default 0.9,  clamp 0.05..1
  cfgScale?: number;              // default 2.0,  clamp 0..10
  repPenalty?: number;            // default 1.0,  clamp 1..1.5
  format?: 'wav16' | 'mp3';       // default 'wav16' (C10)
  coResident?: boolean;           // default true (C15)
  kind?: 'ab' | 'milestone';      // default 'ab'
  /** Also render each side's codes through a DiT so the A/B is audible as
   *  MUSIC, not a 5 Hz sketch. Same DiT, same seed, same steps on both sides
   *  and never a sound adapter — the LM adapter stays the only variable. */
  renderDit?: boolean;            // default false
  renderSteps?: number;           // default 8, clamp 2..60
  renderDitModel?: string;        // default: newest installed xl-turbo, else the detok DiT
}

export interface AuditionListResponse {
  previews: AuditionPreview[];    // newest first, max 20
  engineReady: boolean;
  engineSuspended: boolean;       // true while a preprocess/train job owns the GPU
}

/** Sync response of the sample-codes audition. */
export interface SampleAuditionResponse {
  preview: AuditionPreview;       // kind 'sample', exactly one side, slot 'base'
  source: 'lm_codes' | 'label';   // which stored-codes source was used (C12 / §5.3)
}

// ─── Lyric Studio export ──────────────────────────────────────────────────
//
// A labeled dataset already holds everything a Lyric Studio artist/album entry
// needs (Genius lyrics, LLM caption+genre, Essentia bpm/key/signature), so a
// dataset can be exported straight into the lireek tables. Artist/album are
// DETECTED (embedded-tag majority vote → dataset defaults → folder name) and
// user-overridable; an existing artist+album set is updated in place, never
// duplicated. Trained adapters for the dataset are offered as the album preset.

export type LyricStudioFieldSource = 'tags' | 'default' | 'filename' | 'folder' | 'dataset-name';

export interface LyricStudioExportSong {
  sampleId: string;
  title: string;
  album: string;              // per-song album (own tag first, set album else)
  hasCaption: boolean;
}

export interface LyricStudioAdapterHit {
  path: string;               // absolute adapter run dir (folder-only presets)
  kind: 'dit' | 'lm';
  detail: string;             // dit-<base> shorthand / LM size — display only
  trainedAt: string;          // ISO or ''
}

export interface LyricStudioExportPreview {
  artist: string;             // detected, pre-fills the override field
  album: string;
  artistSource: LyricStudioFieldSource;
  albumSource: LyricStudioFieldSource;
  songs: LyricStudioExportSong[];   // includable songs, dataset order
  skippedInstrumental: number;
  skippedNoLyrics: number;
  skippedExcluded: number;
  existingArtistId: number | null;    // COLLATE NOCASE match on the DETECTED names;
  existingLyricsSetId: number | null; // the commit re-resolves after overrides
  existingSongCount: number;
  ditAdapter: LyricStudioAdapterHit | null;
  lmAdapter: LyricStudioAdapterHit | null;
  geniusConfigured: boolean;  // album/artist art fetch will be attempted
}

export interface LyricStudioExportInput {
  artist?: string;            // override; omit/'' = detected value
  album?: string;
  linkAdapters?: boolean;     // default true — write the album preset from trained adapters
}

export interface LyricStudioExportResult {
  artistId: number;
  lyricsSetId: number;
  updatedExisting: boolean;
  songCount: number;
  presetUpdated: boolean;
  imageUrl: string;           // album art fetched during export, '' if none
}

// ─── Batch pipeline (multi-folder import + auto-chained stages) ────────────
//
// Spec: docs/plans/2026-07-28-training-batch-pipeline.md §2.1

export type PipelineStage = 'label' | 'build' | 'preprocess' | 'train-dit' | 'train-lm';
export type PipelineStatus = 'running' | 'done' | 'failed' | 'cancelled';
export type PipelineItemStatus = 'pending' | 'creating' | 'running' | 'done' | 'failed' | 'cancelled';

export interface PipelineFolderSpec {
  sourceDir: string;
  /** Dataset name; omit = folder basename. */
  name?: string;
  /** Trigger word; omit = slugified name. */
  customTag?: string;
}

export interface StartPipelineInput {
  folders: PipelineFolderSpec[];
  /** Which stages to run, in canonical order. Omit = all five. */
  stages?: PipelineStage[];
}

export interface PipelineStageResult {
  stage: PipelineStage;
  jobId: string;          // '' if the stage never started
  status: PipelineItemStatus;
  error: string | null;
  startedAt: number | null;
  finishedAt: number | null;
}

export interface PipelineItem {
  sourceDir: string;
  name: string;
  datasetId: string;      // '' until created/resolved
  reusedExisting: boolean;
  status: PipelineItemStatus;
  currentStage: PipelineStage | null;
  stages: PipelineStageResult[];
  error: string | null;
}

export interface PipelineSummary {
  id: string;
  status: PipelineStatus;
  stages: PipelineStage[];
  items: PipelineItem[];
  createdAt: number;
  finishedAt: number | null;
}

export interface TrainingDefaults {
  label: Record<string, unknown>;
  preprocess: Record<string, unknown>;
  trainLm: Record<string, unknown>;
  trainDit: Record<string, unknown>;
}
