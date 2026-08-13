// backends/minimax/client.ts — typed HTTP client for the engine's /mm3/* family
//
// MiniMax-Music3 lives INSIDE the same ace-server process as the ACE pipeline
// (plan §4.3: "same ace-server process, separate source tree"), so this client
// talks to the same base URL aceClient does — there is no second engine port.
// Only the family-scoped endpoints differ:
//
//   GET  /mm3/props           discovery + residency (see the BLOCKING warning)
//   POST /mm3/synth           production generation → { job_id, id, seed, ... }
//   GET  /mm3/job?id=         MM3-native stage detail (never takes the MM3 mutex)
//   POST /mm3/unload          free MM3 VRAM (idempotent)
//   POST /mm3/tokenize-check  cold-capable prompt-token pre-flight
//
// Progress / result / cancel for an MM3 job go through the STANDARD job
// endpoints (GET /job?id=, GET /job?id=&result=1, POST /job?id=&cancel=1) —
// i.e. aceClient.pollJob / getJobResult / cancelJob work unchanged. That is
// deliberate: the MM3 job rides the engine's one GPU worker queue like any ACE
// job, so the whole polling/cancel machinery is shared, not duplicated.
//
// NOTE on the base URL: aceClient.ts:12 snapshots `config.aceServer.url` at
// import time (a known latent staleness bug, plan §3.1). This module reads it
// per call instead, so a hot-reloaded host/port is picked up.

import { config } from '../../../config.js';

const base = () => config.aceServer.url;

/** GET /mm3/props takes the engine-side MM3 mutex, which an in-flight MM3
 *  generation holds for its ENTIRE run. A props call during a generation
 *  therefore blocks until the song finishes. Never wait on it — 2.5 s, then
 *  fall back to the last known-good manifest. */
const TIMEOUT_PROPS    = 2_500;
const TIMEOUT_QUICK    = 15_000;   // synth submit, unload — all cheap/synchronous
const TIMEOUT_TOKENIZE = 30_000;   // cold-capable: may load the tokenizer from the GGUF header
const TIMEOUT_JOB      = 10_000;   // /mm3/job never takes the MM3 mutex

// ── Wire shapes (mirrors engine/src/minimax/mm3-server.h + mm3-job.h) ──

export interface Mm3PropsFile {
  found?: boolean;
  path?: string;
  probe_ok?: boolean;
  [k: string]: unknown;
}

export interface Mm3Props {
  backend: string;                 // 'minimax-m3'
  model: string;                   // 'MiniMax-Music3'
  available: boolean;              // both GGUFs discovered
  loaded: boolean;                 // weights currently resident in VRAM
  /** Both GGUFs found AND their headers parsed clean — i.e. POST /mm3/synth
   *  will get as far as loading weights. NOT a VRAM/residency claim. */
  synth_ready: boolean;
  prompt_token_limit: number;      // 5000
  max_audio_frames_limit: number;  // 9000 (= 360 s @ 25 fps)
  models_dir?: string;
  search_dirs?: string[];
  files?: { lm?: Mm3PropsFile; synth?: Mm3PropsFile };
  vram?: Record<string, number>;
  errors?: string[];
  [k: string]: unknown;
}

export interface Mm3PropsResult {
  /** null only if /mm3/props has never answered since server start. */
  props: Mm3Props | null;
  /** true when `props` came from the last-known-good cache (probe timed out or
   *  failed), so callers can degrade honestly instead of reporting "down". */
  stale: boolean;
  error?: string;
  /** Epoch ms of the fetch that produced `props` (0 when never fetched). */
  fetchedAt: number;
}

export interface Mm3SynthRequest {
  /** REQUIRED, non-blank. The Structured Caption (see .claude/skills/mm3-captioning). */
  caption: string;
  /** "" (or omitted) → the engine substitutes its instrumental lyric. */
  lyrics?: string;
  /** Seconds. Required unless max_frames is given; engine maps to frames @25 fps. */
  duration?: number;
  /** Escape hatch — wins over duration. */
  max_frames?: number;
  /** -1 = engine draws one; the resolved value comes back in the response. */
  seed?: number;
  /** Default = the checkpoint's flow.cfg_scale (1.7). */
  cfg_flow?: number;
  /** Default = the checkpoint's flow.steps (30). */
  steps?: number;
  /** 16 | 24 | 32, default 16. */
  get_wav_bits?: number;
}

export interface Mm3SynthResponse {
  job_id: string;
  id: string;                 // the ACE /synth spelling — same value
  seed: number;               // RESOLVED seed (never -1)
  max_frames: number;
  duration: number;
  prompt_tokens: number;
  prompt_token_limit: number;
  instrumental: boolean;
  steps: number;
  cfg_flow: number;
  wav_bits: number;
}

/** MM3-native stage detail. `stage` is the real MM3 vocabulary
 *  (queued|arbitrating|warming|warm|ar|cond|flow|vocode|stitch|encoding|done|
 *  failed|cancelled), unlike GET /job's ACE-phase mapping. */
export interface Mm3JobDetail {
  id: string;
  status: string;             // running | done | failed | cancelled | unknown
  phase: string;              // the ACE-phase mapping
  stage: string;
  window: number;
  n_windows: number;
  step: number;
  n_steps: number;
  seed: number;
  max_frames: number;
  prompt_tokens: number;
  instrumental: boolean;
  evicted_modules?: number;
  evicted_mb?: number;
  arbitrate_ms?: number;
  warm_ms?: number;
  error?: string;
  result?: {
    frames: number;
    n_samples: number;
    sample_rate: number;
    channels: number;
    duration_sec: number;
    rms: number;
    peak: number;
    eos: boolean;
    has_nan: boolean;
    ms?: { ar: number; cond: number; flow: number; voc: number; total: number };
  };
}

export interface Mm3TokenizeCheck {
  tokens: number;
  limit: number;
  /** false is a 200, not an error — only POST /mm3/synth turns a "no" into a 400. */
  ok: boolean;
  instrumental: boolean;
  caption_clean?: string;
  lyrics_normalized?: string;
  prompt?: string;
}

export interface Mm3UnloadResult {
  unloaded: boolean;   // true only if something was actually resident
  loaded: boolean;     // always false on success
  freed_bytes?: number;
  freed_mb?: number;
}

// ── Helpers ──

/** The engine reports every /mm3 failure as {"error": "..."} with a real HTTP
 *  status. Surface that message verbatim — the validation errors are written
 *  for humans ("\"duration\" must be greater than 0 seconds"). */
async function mm3ErrorMessage(res: Response, endpoint: string): Promise<string> {
  const body = await res.text().catch(() => '');
  let msg = body;
  try {
    const parsed = JSON.parse(body) as { error?: string };
    if (parsed?.error) msg = parsed.error;
  } catch { /* non-JSON body — use it raw */ }
  return `ace-server ${endpoint} failed (${res.status}): ${msg || 'unknown error'}`;
}

async function mm3Post<T>(endpoint: string, body: unknown, timeoutMs: number): Promise<T> {
  const res = await fetch(`${base()}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new Error(await mm3ErrorMessage(res, `POST ${endpoint}`));
  return await res.json() as T;
}

// ── Props (with last-known-good cache) ──

let lastGoodProps: Mm3Props | null = null;
let lastGoodAt = 0;

/** GET /mm3/props with a short timeout and a last-known-good fallback.
 *
 *  A timeout here almost always means "an MM3 generation is running and holds
 *  the engine-side mutex" — which is the opposite of down. Treating it as down
 *  would make the backend toggle flicker to unavailable mid-song, so the cached
 *  manifest is returned with `stale: true` instead. */
export async function mm3Props(timeoutMs = TIMEOUT_PROPS): Promise<Mm3PropsResult> {
  try {
    const res = await fetch(`${base()}/mm3/props`, { signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) throw new Error(await mm3ErrorMessage(res, 'GET /mm3/props'));
    const props = await res.json() as Mm3Props;
    lastGoodProps = props;
    lastGoodAt = Date.now();
    return { props, stale: false, fetchedAt: lastGoodAt };
  } catch (err: any) {
    return {
      props: lastGoodProps,
      stale: true,
      error: err?.message || String(err),
      fetchedAt: lastGoodAt,
    };
  }
}

/** The cached manifest without touching the engine (never blocks). */
export function mm3PropsCached(): Mm3Props | null {
  return lastGoodProps;
}

// ── Generation ──

/** POST /mm3/synth — submit a generation. Every validation failure (blank
 *  caption, bad type, over-long prompt, bad duration) is a synchronous 400, so
 *  a resolved promise here means the job will actually run. */
export async function mm3Synth(req: Mm3SynthRequest): Promise<Mm3SynthResponse> {
  return mm3Post<Mm3SynthResponse>('/mm3/synth', req, TIMEOUT_QUICK);
}

/** GET /mm3/job?id= — MM3 stage detail. Soft-fails to null: this is progress
 *  garnish on top of the authoritative GET /job status, never a failure signal. */
export async function mm3JobDetail(jobId: string): Promise<Mm3JobDetail | null> {
  try {
    const res = await fetch(`${base()}/mm3/job?id=${encodeURIComponent(jobId)}`, {
      signal: AbortSignal.timeout(TIMEOUT_JOB),
    });
    if (!res.ok) return null;
    return await res.json() as Mm3JobDetail;
  } catch {
    return null;
  }
}

/** POST /mm3/tokenize-check — cold-capable prompt-token pre-flight against the
 *  5,000-token limit. Works without the weights resident. */
export async function mm3TokenizeCheck(
  caption: string, lyrics = '', echoPrompt = false,
): Promise<Mm3TokenizeCheck> {
  return mm3Post<Mm3TokenizeCheck>(
    '/mm3/tokenize-check', { caption, lyrics, prompt: echoPrompt }, TIMEOUT_TOKENIZE,
  );
}

// ── Residency ──

/** POST /mm3/unload — free MM3 weights + KV cache. Idempotent and cheap when
 *  cold. Soft-fails: VRAM arbitration must never break a generation or a
 *  backend switch, so failures are reported, not thrown. */
export async function mm3Unload(timeoutMs = TIMEOUT_QUICK): Promise<Mm3UnloadResult | null> {
  try {
    const res = await fetch(`${base()}/mm3/unload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return null;
    return await res.json() as Mm3UnloadResult;
  } catch {
    return null;
  }
}
