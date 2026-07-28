// trainingApi.ts — API client for the Training Studio (Dataset phase)
//
// Talks to the Node server's /api/training endpoints. The type block below is
// a verbatim copy of the frozen contract shared with
// server/src/services/training/types.ts — keep both sides in sync by hand.

const API_BASE = '/api/training';

// ── Core enums ────────────────────────────────────────────────────────────
export type TagPosition = 'prepend' | 'append' | 'replace';

export type MergePolicy =
  | 'fill_missing'        // default: only write fields currently empty
  | 'overwrite_caption'
  | 'overwrite_lyrics'
  | 'overwrite_all';

export type TrainingJobKind =
  | 'label' | 'enhance-genius' | 'enhance-caption' | 'build'
  | 'preprocess' | 'train-lm' | 'train-dit';

export type TrainingJobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

export type SampleLabelStatus =
  | 'unlabeled'   // no sidecar caption
  | 'labeled'     // sidecar caption present
  | 'pending'     // queued in an active job
  | 'processing'  // currently being labeled
  | 'error';      // last label attempt failed (see `error`)

export type FieldSource = 'sidecar' | 'understand' | 'essentia' | 'genius' | 'llm' | 'tags' | 'filename' | 'user';

// ── Dataset ──────────────────────────────────────────────────────────────
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
  defaultLanguage: string;   // language the user KNOWS the corpus is in; overrides understand's guess
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
  preprocessedVariants: number;   // subdirs of data/training/tensors/<slug> with a preprocess_meta.json
}

// ── Jobs ─────────────────────────────────────────────────────────────────
export interface TrainingJobSummary {
  id: string;
  datasetId: string;
  kind: TrainingJobKind;
  status: TrainingJobStatus;
  total: number;
  done: number;
  failed: number;
  currentSampleId: string | null;
  phase: string;              // free-form: 'essentia' | 'understand' | 'waiting-for-engine' | 'genius' | 'llm' | 'writing' | 'build'
                              // kind==='preprocess' adds: 'engine-stop' | 'loading-models'
                              //                          | 'preprocess' | 'stats' | 'engine-restart'
                              // kind==='train-lm'   adds: 'engine-stop' | 'loading-models' | 'extract'
                              //                          | 'train' | 'export' | 'engine-restart'
                              // kind==='train-dit'  adds: 'engine-stop' | 'loading-models'
                              //                          | 'train' | 'export' | 'engine-restart'
  engineQueueDepth: number;   // # of ace-server jobs ahead of ours; 0 if unknown
  error: string | null;
  createdAt: number;          // epoch ms
  startedAt: number | null;
  finishedAt: number | null;
}

// ── Preprocess ───────────────────────────────────────────────────────────
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

// ── LM training ──────────────────────────────────────────────────────────
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

// ── DiT training ─────────────────────────────────────────────────────────
export type DitAdapterType = 'lora';                 // 'lokr' reserved (D22)
export type TrainDitStage  = 'train' | 'export';

export interface TrainDitOptions {
  /** Which preprocess variant to train from. Omit = the newest variant. */
  variantKey?: string;
  /** Adapter directory name under <adapters>/. Omit = the dataset slug. */
  adapterName?: string;
  adapterType?: DitAdapterType;    // default 'lora'
  rank?: number;                   // default 16
  alpha?: number;                  // default 32
  targetMlp?: boolean;             // default false
  layers?: number;                 // default 0 = auto (top-K depth)
  crop?: number;                   // default 0 = auto-fit
  cropMin?: number;                // default 375
  cropMax?: number;                // default 1250
  targetLoss?: number;             // default 0.4;  0 disables auto-stop
  epochs?: number;                 // default 100 (hard cap)
  learningRate?: number;           // default 0.0005
  gradAccum?: number;              // default 4
  gradClip?: number;               // default 1.0;  0 disables
  warmupRatio?: number;            // default 0.05
  weightDecay?: number;            // default 0.01
  lossWeighting?: 'none' | 'flow_snr';   // default 'flow_snr'
  snrGamma?: number;               // default 5.0
  tBias?: number;                  // default 0.5
  channelBalance?: boolean;        // default true
  timestepMu?: number;             // default -0.4
  timestepSigma?: number;          // default 1.0
  tMin?: number;                   // default 0
  tMax?: number;                   // default 1
  cfgRatio?: number;               // default 0.15
  genreRatio?: number;             // default 0 (percent)
  seed?: number;                   // default 42
  order?: 'shuffle' | 'fixed';     // default 'shuffle'
  milestoneStep?: number;          // default 0.1;  0 disables
  milestoneKeep?: number;          // default 6
  vramReserveMb?: number;          // default 2048
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
  adapterDir: string;              // absolute; where the next run would write
  adapterExists: boolean;          // adapter_model.safetensors present
  adapterBytes: number;
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
  adapterTypes: DitAdapterType[];  // v1: ['lora']
  adaptersRoot: string;            // <adapters>
  /** Minimum total VRAM this build will accept, MB. v1: 16384 (D9). */
  minVramMb: number;
}

// ── Capabilities ─────────────────────────────────────────────────────────
export interface TrainingCapabilities {
  engine: {
    up: boolean;
    ready: boolean;               // from /api/health engine.ready
    understandSupported: boolean; // lm && dit && vae registries all non-empty
    missingModels: string[];      // e.g. ['lm'] — which registries are empty
    queueDepth: number;           // ace-server GET /jobs length
    lmModels: string[];           // LM registry names, for the understand model picker
    defaultLmModel: string;       // server's pick (biggest LM, fast quant) when none is sent
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

// ── SSE stream ───────────────────────────────────────────────────────────

/** Structured training numbers (L21). Additive member of TrainingStreamEvent —
 *  consumers MUST ignore unknown `metric` values rather than throwing. */
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
  // train-dit additions (§2.6). Optional fields on the EXISTING interface —
  // deliberately not a new union member, so every consumer keeps working.
  crop?: number;      // latent frames in the active crop window
  layers?: number;    // trained decoder-layer count (top-K)
  t?: number;         // last micro-step timestep (telemetry)
  ma5?: number;       // 5-epoch moving average (the target-loss quantity)
  rawLoss?: number;   // unweighted MSE, for telemetry
}

export type TrainingStreamEvent =
  | { type: 'job'; job: TrainingJobSummary }                       // first event on connect + on any status change
  | {
      type: 'progress'; done: number; total: number; failed: number;
      phase: string; currentSampleId: string | null; engineQueueDepth: number;
    }
  | {
      type: 'sample'; sampleId: string; status: SampleLabelStatus;
      sample?: TrainingSample; error?: string;                      // `sample` present on success
    }
  | { type: 'log'; level: 'info' | 'warn' | 'error'; message: string; ts: number }
  | TrainingMetricEvent
  | { type: 'status'; status: TrainingJobStatus; error?: string };  // terminal; server closes after this

// ── Request / response shapes ────────────────────────────────────────────

export interface ScanPreview {
  root: string;
  audioFiles: number;
  withSidecar: number;
  withCaption: number;
  hasDatasetJson: boolean;
  extensions: Record<string, number>;   // { ".flac": 12, ".mp3": 3 }
  sampleNames: string[];                // first 10 basenames
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
  defaultLanguage?: string;  // default 'english'
}

export type PatchDatasetInput = Partial<{
  name: string;
  customTag: string;
  tagPosition: TagPosition;
  genreRatio: number;
  defaultArtist: string;
  defaultAlbum: string;
  defaultGenre: string;
  defaultLanguage: string;
  recursive: boolean;
}>;

export type PatchSampleInput = Partial<{
  caption: string;
  genre: string;
  bpm: number | null;
  key: string;
  signature: string;
  language: string;
  isInstrumental: boolean;
  lyrics: string;
  customTag: string;
  repeat: number;
  promptOverride: string | null;
  excluded: boolean;
}>;

export type BulkSetInput = Partial<{
  excluded: boolean;
  isInstrumental: boolean;
  genre: string;
  customTag: string;
  language: string;
  repeat: number;
}>;

export interface BulkResult {
  updated: number;
  failed: Array<{ sampleId: string; error: string }>;
}

export interface LabelOptions {
  sampleIds?: string[];
  scope?: 'all' | 'unlabeled';
  useEssentia?: boolean;   // default true  — local BPM/key
  useGenius?: boolean;     // default false — canonical lyrics (needs token)
  useCaption?: boolean;    // default false — LLM caption+genre (audio-grounded on gemini)
  useUnderstand?: boolean; // default false — LEGACY /understand path
  mergePolicy?: MergePolicy;
  caption?: { provider?: string; model?: string };
  understand?: {
    lmModel?: string;
    synthModel?: string;
    lmTemperature?: number;
    lmTopP?: number;
    lmTopK?: number;
    seed?: number;
  };
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

// ── fetch helpers ────────────────────────────────────────────────────────

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `API error: ${res.status}`);
  }
  return res.json();
}

function jsonBody(body: unknown): RequestInit {
  return { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}) };
}

// ── Capabilities ─────────────────────────────────────────────────────────

export async function getCapabilities(): Promise<TrainingCapabilities> {
  return request<TrainingCapabilities>('/capabilities');
}

// ── Scan preview ─────────────────────────────────────────────────────────

export async function scanPreview(path: string, recursive = true): Promise<ScanPreview> {
  return request<ScanPreview>(`/scan-preview?path=${encodeURIComponent(path)}&recursive=${recursive ? 1 : 0}`);
}

// ── Dataset CRUD ─────────────────────────────────────────────────────────

export async function listDatasets(): Promise<TrainingDatasetSummary[]> {
  const data = await request<{ datasets: TrainingDatasetSummary[] }>('/datasets');
  return data.datasets;
}

export async function createDataset(input: CreateDatasetInput): Promise<TrainingDatasetDetail> {
  const data = await request<{ dataset: TrainingDatasetDetail }>('/datasets', { method: 'POST', ...jsonBody(input) });
  return data.dataset;
}

export async function getDataset(id: string): Promise<TrainingDatasetDetail> {
  return request<TrainingDatasetDetail>(`/datasets/${encodeURIComponent(id)}`);
}

export async function patchDataset(id: string, patch: PatchDatasetInput): Promise<TrainingDatasetSummary> {
  const data = await request<{ dataset: TrainingDatasetSummary }>(
    `/datasets/${encodeURIComponent(id)}`, { method: 'PATCH', ...jsonBody(patch) },
  );
  return data.dataset;
}

export async function rescanDataset(id: string): Promise<TrainingDatasetDetail> {
  return request<TrainingDatasetDetail>(`/datasets/${encodeURIComponent(id)}/rescan`, { method: 'POST' });
}

export async function deleteDataset(id: string): Promise<void> {
  await request<{ ok: boolean }>(`/datasets/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

// ── Sample edits ─────────────────────────────────────────────────────────

export async function patchSample(id: string, sampleId: string, patch: PatchSampleInput): Promise<TrainingSample> {
  const data = await request<{ sample: TrainingSample }>(
    `/datasets/${encodeURIComponent(id)}/samples/${encodeURIComponent(sampleId)}`,
    { method: 'PATCH', ...jsonBody(patch) },
  );
  return data.sample;
}

export async function bulkSetSamples(id: string, sampleIds: string[], set: BulkSetInput): Promise<BulkResult> {
  return request<BulkResult>(
    `/datasets/${encodeURIComponent(id)}/samples/bulk`,
    { method: 'POST', ...jsonBody({ sampleIds, set }) },
  );
}

/** Direct audio URL for the preview player — never routed through the playback store. */
export function sampleAudioUrl(id: string, sampleId: string): string {
  return `${API_BASE}/datasets/${encodeURIComponent(id)}/samples/${encodeURIComponent(sampleId)}/audio`;
}

// ── Jobs ─────────────────────────────────────────────────────────────────

export async function startLabel(id: string, opts: LabelOptions): Promise<{ jobId: string }> {
  return request<{ jobId: string }>(`/datasets/${encodeURIComponent(id)}/label`, { method: 'POST', ...jsonBody(opts) });
}

export async function startGenius(id: string, opts: GeniusOptions): Promise<{ jobId: string }> {
  return request<{ jobId: string }>(`/datasets/${encodeURIComponent(id)}/enhance/genius`, { method: 'POST', ...jsonBody(opts) });
}

export async function startCaption(id: string, opts: CaptionOptions): Promise<{ jobId: string }> {
  return request<{ jobId: string }>(`/datasets/${encodeURIComponent(id)}/enhance/caption`, { method: 'POST', ...jsonBody(opts) });
}

export async function startBuild(id: string, outputPath?: string): Promise<{ jobId: string }> {
  return request<{ jobId: string }>(
    `/datasets/${encodeURIComponent(id)}/build`,
    { method: 'POST', ...jsonBody(outputPath ? { outputPath } : {}) },
  );
}

export async function startPreprocess(id: string, opts: PreprocessOptions): Promise<{ jobId: string }> {
  return request<{ jobId: string }>(
    `/datasets/${encodeURIComponent(id)}/preprocess`,
    { method: 'POST', ...jsonBody(opts) },
  );
}

export async function getPreprocessStatus(id: string): Promise<PreprocessStatus> {
  return request<PreprocessStatus>(`/datasets/${encodeURIComponent(id)}/preprocess`);
}

export async function deletePreprocessVariant(id: string, variantKey: string): Promise<void> {
  await request<{ ok: boolean }>(
    `/datasets/${encodeURIComponent(id)}/preprocess/${encodeURIComponent(variantKey)}`,
    { method: 'DELETE' },
  );
}

export async function startTrainLm(id: string, opts: TrainLmOptions): Promise<{ jobId: string }> {
  return request<{ jobId: string }>(
    `/datasets/${encodeURIComponent(id)}/train-lm`,
    { method: 'POST', ...jsonBody(opts) },
  );
}

export async function getTrainLmStatus(
  id: string,
  q?: { variantKey?: string; adapterName?: string; lmSize?: LmSize },
): Promise<TrainLmStatus> {
  const params = new URLSearchParams();
  if (q?.variantKey) params.set('variantKey', q.variantKey);
  if (q?.adapterName) params.set('adapterName', q.adapterName);
  if (q?.lmSize) params.set('lmSize', q.lmSize);
  const qs = params.toString();
  return request<TrainLmStatus>(
    `/datasets/${encodeURIComponent(id)}/train-lm${qs ? `?${qs}` : ''}`,
  );
}

export async function startTrainDit(id: string, opts: TrainDitOptions): Promise<{ jobId: string }> {
  return request<{ jobId: string }>(
    `/datasets/${encodeURIComponent(id)}/train-dit`,
    { method: 'POST', ...jsonBody(opts) },
  );
}

export async function getTrainDitStatus(
  id: string,
  q?: { variantKey?: string; adapterName?: string },
): Promise<TrainDitStatus> {
  const params = new URLSearchParams();
  if (q?.variantKey) params.set('variantKey', q.variantKey);
  if (q?.adapterName) params.set('adapterName', q.adapterName);
  const qs = params.toString();
  return request<TrainDitStatus>(
    `/datasets/${encodeURIComponent(id)}/train-dit${qs ? `?${qs}` : ''}`,
  );
}

export async function getJob(jobId: string): Promise<TrainingJobSummary> {
  return request<TrainingJobSummary>(`/jobs/${encodeURIComponent(jobId)}`);
}

export async function listJobs(datasetId?: string): Promise<TrainingJobSummary[]> {
  const qs = datasetId ? `?datasetId=${encodeURIComponent(datasetId)}` : '';
  const data = await request<{ jobs: TrainingJobSummary[] }>(`/jobs${qs}`);
  return data.jobs;
}

export async function cancelJob(jobId: string): Promise<void> {
  await request<{ ok: boolean }>(`/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
}

export const jobStreamUrl = (jobId: string) => `${API_BASE}/jobs/${encodeURIComponent(jobId)}/stream`;

/** Built dataset.json, parsed. */
export async function getDatasetJson(id: string): Promise<{ path: string; builtAt: string; dataset: unknown }> {
  return request<{ path: string; builtAt: string; dataset: unknown }>(`/datasets/${encodeURIComponent(id)}/dataset-json`);
}
