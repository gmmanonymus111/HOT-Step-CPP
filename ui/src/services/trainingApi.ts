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

export type TrainingJobKind = 'label' | 'enhance-genius' | 'enhance-caption' | 'build';

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
  engineQueueDepth: number;   // # of ace-server jobs ahead of ours; 0 if unknown
  error: string | null;
  createdAt: number;          // epoch ms
  startedAt: number | null;
  finishedAt: number | null;
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
}

// ── SSE stream ───────────────────────────────────────────────────────────
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
