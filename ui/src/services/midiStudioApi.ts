// midiStudioApi.ts — API client for MIDI Studio (audio → MIDI transcription)
//
// Communicates with Node server /api/midi-studio endpoints, which run
// MuScriptor (Kyutai & Mirelo) in a server-managed Python venv.

const API_BASE = '/api/midi-studio';

// ── Types ────────────────────────────────────────────────────────────────

export type MuscriptorModel = 'small' | 'medium' | 'large';

export interface ModelState {
  downloaded: boolean;
  sizeBytes: number;
  downloading: boolean;
  receivedBytes: number;
  totalBytes: number;
  error?: string;
  gated?: boolean;
}

export interface MidiStudioStatus {
  /** ace-midi engine binary present next to ace-server */
  engineAvailable: boolean;
  hfTokenSet: boolean;
  models: Record<MuscriptorModel, ModelState>;
}

export interface MidiJobProgress {
  status: 'queued' | 'transcribing' | 'done' | 'failed' | 'cancelled';
  chunksDone?: number;
  chunksTotal?: number;
  noteCount?: number;
  error?: string;
  gated?: boolean;
}

export interface MidiJobSummary {
  id: string;
  status: 'queued' | 'transcribing' | 'done' | 'failed' | 'cancelled';
  sourceFileName: string;
  /** URL of the original audio (for synced playback in the player) */
  sourceAudioUrl?: string;
  songId?: string;
  model: MuscriptorModel;
  noteCount: number;
  durationSec: number;
  createdAt: string;
  error?: string;
  /** Failure looks like gated-model / HF auth trouble */
  gated?: boolean;
  /** chunk progress — present on in-flight jobs */
  chunksDone?: number;
  chunksTotal?: number;
}

/**
 * The MuScriptor weights are GATED on Hugging Face — each user must request
 * access on the model page (free, instant after accepting the conditions)
 * and provide a read token before the first download.
 */
export const HF_MODEL_URLS: Record<MuscriptorModel, string> = {
  small: 'https://huggingface.co/MuScriptor/muscriptor-small',
  medium: 'https://huggingface.co/MuScriptor/muscriptor-medium',
  large: 'https://huggingface.co/MuScriptor/muscriptor-large',
};

export const HF_TOKEN_SETTINGS_URL = 'https://huggingface.co/settings/tokens';

export interface MidiNote {
  pitch: number;
  velocity: number;
  channel: number;
  start: number;
  duration: number;
}

export interface MidiChannelInfo {
  channel: number;
  program: number;
  isDrums: boolean;
  noteCount: number;
}

export interface ParsedMidi {
  durationSec: number;
  noteCount: number;
  channels: MidiChannelInfo[];
  notes: MidiNote[];
}

export interface TranscribeParams {
  sourceAudioUrl: string;
  sourceFileName: string;
  songId?: string;
  model: MuscriptorModel;
}

// ── Helpers ──────────────────────────────────────────────────────────────

async function jsonOrThrow<T>(res: Response, what: string): Promise<T> {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    throw new Error(err.error || `${what} failed: ${res.status}`);
  }
  return res.json();
}

// ── API Functions ────────────────────────────────────────────────────────

/** Get MuScriptor install/environment status. */
export async function getMidiStatus(): Promise<MidiStudioStatus> {
  return jsonOrThrow(await fetch(`${API_BASE}/status`), 'Status');
}

/** Save (or clear, with '') the Hugging Face read token for gated weights. */
export async function saveHfToken(token: string): Promise<void> {
  await jsonOrThrow(await fetch(`${API_BASE}/hf-token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token }),
  }), 'Save token');
}

/** Begin downloading a model's gated weights. Poll getMidiStatus() for progress. */
export async function startModelDownload(model: MuscriptorModel): Promise<void> {
  await jsonOrThrow(await fetch(`${API_BASE}/models/${model}/download`, { method: 'POST' }), 'Download');
}

/** Poll a transcription job's progress. */
export async function getMidiProgress(jobId: string): Promise<MidiJobProgress> {
  return jsonOrThrow(await fetch(`${API_BASE}/${jobId}/progress`), 'Progress');
}

/** SSE URL streaming live note events for an in-flight job (Phase 6 UI). */
export function getMidiStreamUrl(jobId: string): string {
  return `${API_BASE}/${jobId}/stream`;
}

/** Queue a transcription job. Returns the job ID. */
export async function submitTranscription(params: TranscribeParams): Promise<string> {
  const res = await fetch(`${API_BASE}/transcribe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  const data = await jsonOrThrow<{ id: string }>(res, 'Transcription');
  return data.id;
}

/** Fetch the parsed note data for the piano-roll preview. */
export async function getMidiNotes(jobId: string): Promise<ParsedMidi> {
  return jsonOrThrow(await fetch(`${API_BASE}/${jobId}/notes`), 'Notes');
}

/** List all transcription jobs (newest first). */
export async function listMidiJobs(): Promise<MidiJobSummary[]> {
  return jsonOrThrow(await fetch(`${API_BASE}/jobs`), 'List jobs');
}

/** Cancel and/or delete a job. */
export async function deleteMidiJob(jobId: string): Promise<void> {
  await jsonOrThrow(await fetch(`${API_BASE}/${jobId}`, { method: 'DELETE' }), 'Delete');
}

/** Direct URL for downloading the .mid file of a completed job. */
export function getMidiFileUrl(jobId: string): string {
  return `${API_BASE}/${jobId}/file`;
}

// ── GM instrument names (for channel legend) ─────────────────────────────

const GM_FAMILIES = [
  'Piano', 'Chromatic Perc.', 'Organ', 'Guitar', 'Bass', 'Strings',
  'Ensemble', 'Brass', 'Reed', 'Pipe', 'Synth Lead', 'Synth Pad',
  'Synth FX', 'Ethnic', 'Percussive', 'Sound FX',
];

/** Coarse GM name for a channel (family-level; channel 9 = drums). */
export function channelLabel(ch: MidiChannelInfo): string {
  if (ch.isDrums) return 'Drums';
  return GM_FAMILIES[Math.floor(ch.program / 8)] || `Program ${ch.program}`;
}
