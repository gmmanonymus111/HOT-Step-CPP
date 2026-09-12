// seam-baseline.mjs — pinned-seed baseline renders for ACE and MM3
//
// Captures one pinned request through the running Node server. Saves raw,
// mastered and delivered audio, submitted/stored parameters and the actual
// engine environment. A shared GPU lock prevents overlapping campaigns.
//
// Usage:
//   node server/scripts/seam-baseline.mjs --series pre-seam --backend ace --run 1
//   node server/scripts/seam-baseline.mjs --series pre-seam --backend ace --run 2
// Add --variant NAME --params FILE for a pinned request override.
//
// Existing run directories are never overwritten. Every comparison series
// needs run1/run2 for each variant. Legacy manifests remain read-only;
// --backfill writes a separate copy with raw/mastered hashes.

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import { acquireGpuLock } from './seam-lock.mjs';

const NODE = process.env.SEAM_BASELINE_NODE_URL || 'http://127.0.0.1:3001';
const OUT_ROOT = process.env.SEAM_BASELINE_OUT || 'D:/Ace-Step-Latest/_experiments/yue2/seam-baseline';
const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const PINNED_SEED = 4242;
const DURATION_SEC = 30;

// ── Fixed prompts, embedded so every run is byte-for-byte the same request ──

const ACE_CAPTION =
  'warm acoustic indie folk, fingerpicked guitar, gentle brushed drums, soft ' +
  'female lead vocal, intimate late-night mood, light room reverb';

const ACE_LYRICS =
  '[Verse]\n' +
  'Morning light through the window pane\n' +
  'Quiet streets after the rain\n' +
  '[Chorus]\n' +
  'Hold on, hold on\n' +
  'To the good we have known';

const MM3_CAPTION = [
  'Global Metadata: genre: indie folk; tempo: mid-tempo, around 92 BPM; key: G major; '
  + 'mood: warm, nostalgic, intimate; production: close-mic acoustic, gentle tape warmth.',
  'Vocal Details: single female lead vocal, breathy alto register, close and '
  + 'conversational delivery, light natural reverb, no ad-libs.',
  'Arrangement: The song opens with a solo fingerpicked acoustic guitar establishing '
  + 'the chord progression, joined within a few bars by soft brushed percussion and a '
  + 'warm upright bass holding a simple root-note pulse. The lead vocal enters at the '
  + 'first verse, sitting close and intimate in the mix. A second guitar layer with a '
  + 'light melodic countermelody enters ahead of the chorus, and the drums shift from '
  + 'brushes to a gentle backbeat under the vocal lift into the chorus hook, doubled by '
  + 'a soft harmony line. The second verse thins back to guitar, bass and vocal before '
  + 'building into a final chorus with a slightly fuller harmony stack, then the '
  + 'arrangement falls back to solo guitar and a tail of room reverb for the ending.',
].join('\n');

const MM3_LYRICS =
  '[verse]\n' +
  'Morning light through the window pane\n' +
  'Quiet streets after the rain\n' +
  '[chorus]\n' +
  'Hold on, hold on\n' +
  'To the good we have known';

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--backend') out.backend = argv[++i];
    else if (a === '--run') out.run = argv[++i];
    else if (a.startsWith('--backend=')) out.backend = a.slice('--backend='.length);
    else if (a.startsWith('--run=')) out.run = a.slice('--run='.length);
    else if (['--series', '--variant', '--params', '--python', '--manifest'].includes(a)) out[a.slice(2)] = argv[++i];
    else if (a === '--backfill') out.backfill = true;
    else throw new Error(`Unknown argument: ${a}`);
  }
  return out;
}

function normalizeBackend(raw) {
  const v = String(raw || '').trim().toLowerCase();
  if (v === 'ace') return 'ace';
  if (v === 'mm3' || v === 'minimax' || v === 'minimax-m3') return 'minimax-m3';
  throw new Error(`Unknown --backend "${raw}" (expected ace | mm3 | minimax-m3)`);
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function getToken() {
  const r = await fetch(`${NODE}/api/auth/auto`);
  if (!r.ok) throw new Error(`/api/auth/auto ${r.status}: ${await r.text()}`);
  return (await r.json()).token;
}

async function getActiveBackend() {
  const r = await fetch(`${NODE}/api/backends`);
  if (!r.ok) throw new Error(`GET /api/backends ${r.status}: ${await r.text()}`);
  return (await r.json()).activeId;
}

async function setActiveBackend(id) {
  const r = await fetch(`${NODE}/api/backends/active`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id }),
  });
  if (!r.ok) throw new Error(`POST /api/backends/active ${r.status}: ${await r.text()}`);
  return (await r.json());
}

async function submit(token, body) {
  const r = await fetch(`${NODE}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`POST /api/generate ${r.status}: ${await r.text()}`);
  return (await r.json()).jobId;
}

async function pollStatus(token, jobId, { timeoutMs = 10 * 60 * 1000, intervalMs = 1500 } = {}) {
  const t0 = Date.now();
  for (;;) {
    const r = await fetch(`${NODE}/api/generate/status/${jobId}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!r.ok) throw new Error(`GET /api/generate/status/${jobId} ${r.status}: ${await r.text()}`);
    const s = await r.json();
    if (['succeeded', 'failed', 'cancelled'].includes(s.status)) return s;
    if (Date.now() - t0 > timeoutMs) throw new Error(`Job ${jobId} timed out after ${timeoutMs}ms (last status: ${s.status}/${s.stage})`);
    await sleep(intervalMs);
  }
}

async function getSong(songId) {
  const r = await fetch(`${NODE}/api/songs/${songId}`);
  if (!r.ok) throw new Error(`GET /api/songs/${songId} ${r.status}: ${await r.text()}`);
  return (await r.json()).song;
}

function sha256File(filePath) {
  const buf = fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(buf).digest('hex');
}

function buildParams(backend) {
  if (backend === 'ace') {
    return {
      backend: 'ace',
      prompt: ACE_CAPTION,
      lyrics: ACE_LYRICS,
      instrumental: false,
      duration: DURATION_SEC,
      seed: PINNED_SEED,
      randomSeed: false,
      lmSeed: PINNED_SEED,
      lmSeedFollowsDit: false,
      cacheLmCodes: false,
    };
  }
  return {
    backend: 'minimax-m3',
    prompt: MM3_CAPTION,
    lyrics: MM3_LYRICS,
    instrumental: false,
    duration: DURATION_SEC,
    seed: PINNED_SEED,
    randomSeed: false,
    mm3ReuseAr: false,
    mm3RequireEnding: false,
    mm3Takes: 1,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.backfill) return backfillLegacy(args);
  if (!args.series) throw new Error('--series is required; legacy captures remain read-only.');
  return runSeries(args);
}
async function json(url, init = {}) {
  const r = await fetch(url, { ...init, signal: AbortSignal.timeout(15_000) });
  if (!r.ok) throw new Error(`${init.method || 'GET'} ${url}: ${r.status}`);
  return r.json();
}

async function assertIdle() {
  const queue = await json(`${NODE}/api/generate/queue`);
  const jobs = await json('http://127.0.0.1:8085/jobs');
  if (queue.running || queue.pending || queue.depth || !Array.isArray(jobs) ||
      jobs.some(j => !['done', 'failed', 'cancelled'].includes(j.status))) {
    throw new Error('GPU work is active; refusing to submit or change the backend.');
  }
}

function fileIdentity(file) {
  const stat = fs.statSync(file);
  return { path: file, bytes: stat.size, mtimeMs: stat.mtimeMs };
}

function engineIdentity(python) {
  // Read only the numerical environment allowlist, never credentials or the
  // rest of another process's environment. psutil reads the actual child.
  const code = 'import psutil,json; ps=[p for p in psutil.process_iter(["name"]) if p.info["name"]=="ace-server.exe"]; assert len(ps)==1,"Expected exactly one engine"; p=ps[0]; print(json.dumps({"pid":p.pid,"created":p.create_time(),"exe":p.exe(),"environment":{k:v for k,v in p.environ().items() if k.startswith("MM3_") or k in ["CUDA_VISIBLE_DEVICES","CUBLAS_WORKSPACE_CONFIG"]}}))';
  return JSON.parse(execFileSync(python, ['-c', code], { encoding: 'utf8', windowsHide: true }));
}

async function environment(python) {
  const engine = engineIdentity(python);
  const health = await json(`${NODE}/api/health`);
  const ace = await json('http://127.0.0.1:8085/props');
  const props = await json('http://127.0.0.1:8085/mm3/props');
  const cache = path.join(REPO, 'models/mm3/mm3-trt-cache');
  const runtime = props.dit_runtime || {};
  const { gpu_mb, ...stableRuntime } = runtime; // residency is not a numerical setting
  return {
    engine: { ...engine, binarySha256: sha256File(engine.exe), version: health.aceServer?.version },
    ditRuntime: stableRuntime,
    aceDefaults: ace.default,
    modelFiles: [path.join(REPO, 'models'), path.join(REPO, 'models/mm3')].flatMap(dir =>
      fs.readdirSync(dir).filter(n => n.endsWith('.gguf')).sort().map(n => fileIdentity(path.join(dir, n)))),
    selection: Object.fromEntries(Object.entries(props.variants || {}).map(([role, v]) => [role, { requested: v.requested, selected: v.selected }])),
    trtFiles: fs.existsSync(cache) ? fs.readdirSync(cache).filter(n => /\.(engine|json|onnx)$/.test(n)).sort().map(n => fileIdentity(path.join(cache, n))) : [],
  };
}

async function audioIdentity(url, target) {
  if (!url) return null;
  const r = await fetch(`${NODE}${url}`, { signal: AbortSignal.timeout(60_000) });
  if (!r.ok) throw new Error(`Audio ${url}: ${r.status}`);
  const bytes = Buffer.from(await r.arrayBuffer());
  if (target) fs.writeFileSync(target, bytes, { flag: 'wx' });
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

async function backfillLegacy(args) {
  const filename = args.manifest || path.join(OUT_ROOT, 'manifest.json');
  const manifest = JSON.parse(fs.readFileSync(filename, 'utf8'));
  const records = structuredClone(manifest);
  for (const [backend, runs] of Object.entries(records)) {
    for (const [run, entry] of Object.entries(runs)) {
      const song = await getSong(entry.songId);
      entry.rawSha256 = await audioIdentity(song.audio_url);
      entry.masteredSha256 = await audioIdentity(song.mastered_audio_url);
      entry.environmentCoverage = 'Original process environment was not captured; no retroactive assertion.';
      console.log(`${backend}/${run}: raw=${entry.rawSha256} mastered=${entry.masteredSha256}`);
    }
  }
  fs.writeFileSync(filename.replace(/\.json$/, '-backfilled.json'), JSON.stringify(records, null, 2), { flag: 'wx' });
}

async function runSeries(args) {
  const backend = normalizeBackend(args.backend);
  const variant = args.variant || 'base';
  const runNum = Number(args.run);
  for (const name of [args.series, variant]) if (!/^[a-zA-Z0-9_-]+$/.test(name)) throw new Error('Invalid series/variant name');
  if (!Number.isInteger(runNum) || runNum < 1) throw new Error('Positive --run required');
  const filename = args.manifest || path.join(OUT_ROOT, 'manifest-v2.json');
  const output = path.join(OUT_ROOT, args.series, backend, variant, `run${runNum}`);
  const manifest = fs.existsSync(filename) ? JSON.parse(fs.readFileSync(filename, 'utf8')) : { version: 2, series: {} };
  if (manifest.version !== 2) throw new Error('Use a separate v2 manifest, preserving the legacy baseline');
  if (manifest.series?.[args.series]?.backends?.[backend]?.[variant]?.[`run${runNum}`]) throw new Error('Run already recorded; choose a new run number');
  if (fs.existsSync(output)) throw new Error('Run directory already exists; retain it and choose a new run number');
  const python = args.python || 'C:/Users/rob/AppData/Local/Programs/Python/Python313/python.exe';
  const params = { ...buildParams(backend), ...(args.params ? JSON.parse(fs.readFileSync(args.params, 'utf8')) : {}) };
  params.backend = backend;
  const lock = acquireGpuLock({ owner: 'Codex', reason: `seam ${args.series}/${backend}/${variant}/run${runNum}` });
  let switched = false, safeToRelease = false;
  let originalBackend;
  try {
    await assertIdle();
    originalBackend = await getActiveBackend();
    if (originalBackend !== backend) { await setActiveBackend(backend); switched = true; }
    const before = await environment(python);
    const series = manifest.series[args.series];
    if (series && !isDeepStrictEqual(series.environment, before)) throw new Error('Environment differs from this series; investigate before running.');
    fs.mkdirSync(output, { recursive: true });
    fs.writeFileSync(path.join(output, 'request.json'), JSON.stringify(params, null, 2));
    const token = await getToken();
    const submittedAt = new Date().toISOString();
    const jobId = await submit(token, params);
    fs.writeFileSync(path.join(output, 'job.json'), JSON.stringify({ jobId, submittedAt, before }, null, 2));
    console.log(`Submitted ${backend}/${variant} run${runNum}: ${jobId}`);
    const status = await pollStatus(token, jobId);
    fs.writeFileSync(path.join(output, 'status.json'), JSON.stringify(status, null, 2));
    await assertIdle();
    safeToRelease = true;
    if (status.status !== 'succeeded') throw new Error(`Job ${jobId}: ${status.status}: ${status.error}`);
    const songId = status.result?.songIds?.[0];
    const audioUrl = status.result?.audioUrls?.[0];
    if (!songId || !audioUrl) throw new Error('Successful job has no song/audio');
    const song = await getSong(songId);
    const sha256 = await audioIdentity(audioUrl, path.join(output, 'delivered.wav'));
    const rawSha256 = await audioIdentity(song.audio_url, path.join(output, 'raw.wav'));
    const masteredSha256 = await audioIdentity(song.mastered_audio_url, song.mastered_audio_url ? path.join(output, 'mastered.wav') : undefined);
    const after = await environment(python);
    const entry = { backend, variant, run: runNum, jobId, songId, audioUrl, path: output, submittedAt, sha256, rawSha256, masteredSha256,
      submittedParams: params, storedGenerationParams: JSON.parse(song.generation_params || '{}'), environment: before,
      ...(status.attempts ? { attempts: status.attempts } : {}), environmentAfter: after };
    fs.writeFileSync(path.join(output, 'record.json'), JSON.stringify(entry, null, 2));
    if (!isDeepStrictEqual(before, after)) throw new Error('Environment changed during the run; artifacts retained, no accepted manifest entry');
    manifest.series[args.series] ||= { environment: before, backends: {} };
    const backends = manifest.series[args.series].backends;
    backends[backend] ||= {};
    backends[backend][variant] ||= {};
    backends[backend][variant][`run${runNum}`] = entry;
    const temporary = `${filename}.tmp-${process.pid}`;
    fs.writeFileSync(temporary, JSON.stringify(manifest, null, 2), { flag: 'wx' });
    fs.renameSync(temporary, filename);
    console.log(JSON.stringify({ backend, variant, run: runNum, jobId, rawSha256, masteredSha256 }));
  } finally {
    if (!safeToRelease) {
      try { await assertIdle(); safeToRelease = true; } catch { /* retain ownership of uncertain work */ }
    }
    if (safeToRelease) {
      try {
        if (switched) {
          if (await getActiveBackend() === backend) await setActiveBackend(originalBackend);
          else console.error('Active backend changed during capture; preserving the newer selection.');
        }
      }
      finally {
        if (!lock.release()) throw new Error('GPU lock ownership changed; no foreign lock was removed.');
      }
    } else console.error('GPU completion is uncertain. Retaining GPU.lock; verify the owning work before recovery.');
  }
}

main().catch(err => {
  console.error('FATAL:', err?.stack || err?.message || err);
  process.exit(1);
});
