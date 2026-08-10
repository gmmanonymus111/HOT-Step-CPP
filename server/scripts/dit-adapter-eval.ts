#!/usr/bin/env npx tsx
/**
 * dit-adapter-eval.ts — objective eval of a DiT (sound) adapter (2026-08-10).
 *
 * The DiT twin of lm-adapter-eval.ts, in latent space instead of token space.
 * Claim under test: "the adapter moves the DiT's audio toward the artist".
 *
 * Method: for each dataset song, render through /synth CONDITIONED ON THE
 * SONG'S OWN GROUND-TRUTH CODES (from lm_codes.jsonl) — base DiT vs
 * adapter DiT, same seed — so planning is held fixed and the DiT adapter's
 * timbre contribution is the only variable. The rendered audio is encoded
 * back to ACE latents via /vae, and both populations are compared against
 * the dataset's own preprocess latents (target_latents [T,64] @25 Hz):
 *
 *   - Frechet distance over the 64-dim latent distribution (mean+covariance,
 *     the FAD construction in ACE's native latent space — no external
 *     embedding model)
 *   - per-channel mean/std profile distance (L2 over 64 channels)
 *   - channel-correlation structure distance (Frobenius norm of corr diff)
 *
 * The floor (artist's own variability) is the same metric between the two
 * halves of the ground truth. VALIDATION STATUS: new science — run the
 * validation study (base vs adapter vs floor on a few artists) before
 * trusting verdicts; no HTML report until the metric earns it.
 *
 *   npx tsx scripts/dit-adapter-eval.ts generate --dataset <slug> --adapter <path|artist> [opts]
 *   npx tsx scripts/dit-adapter-eval.ts report --run <dir>
 *
 *   --scale <x>       adapter_scale (default 1.0)
 *   --steps <n>       inference steps (default 8 — turbo base)
 *   --samples <n>     songs (default 8, evenly spaced)
 *   --duration <sec>  clip length (default 30; codes truncated to this)
 *   --seeds <n>       seeds per song per side (default 1)
 *   --out <dir>       default tensors/<slug>/<variant>/dit-eval/<stamp>
 *
 * IDEMPOTENT like the LM eval: completed (song, seed, side) triples are
 * skipped on re-run; base generations are adapter/scale-independent and are
 * reused by seeding runs.json (the calibrate pattern).
 */
import fs from 'fs';
import path from 'path';
import { aceClient, type AceRequest } from '../src/services/aceClient.js';
import { config } from '../src/config.js';
import { tensorsRoot } from '../src/services/training/aceTrain.js';
import { newestVariantKey } from '../src/services/training/trainLmStatus.js';
import { hasWeights, latestRunDir, runStamp } from '../src/services/training/adapterLayout.js';

const ENGINE = `http://127.0.0.1:${config.aceServer.port}`;
const LATENT_DIM = 64;
const JOB_DEADLINE_MS = 10 * 60_000;

function die(msg: string): never {
  console.error(`\nERROR: ${msg}`);
  process.exit(1);
}

function parseArgs(argv: string[]): Map<string, string> {
  const out = new Map<string, string>();
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const next = argv[i + 1];
    if (next !== undefined && !next.startsWith('--')) { out.set(a.slice(2), next); i++; }
    else out.set(a.slice(2), 'true');
  }
  return out;
}

// ── safetensors reading (target_latents from the preprocess cache) ────────

function readTensorF32(file: string, name: string): { data: Float32Array; rows: number; cols: number } | null {
  const raw = fs.readFileSync(file);
  const headerLen = Number(raw.readBigUInt64LE(0));
  const header = JSON.parse(raw.subarray(8, 8 + headerLen).toString('utf-8')) as Record<string, any>;
  const info = header[name];
  if (!info) return null;
  const [start, end] = info.data_offsets;
  const body = raw.subarray(8 + headerLen + start, 8 + headerLen + end);
  const rows = Number(info.shape[0]);
  const cols = Number(info.shape[1] ?? 1);
  if (info.dtype === 'F32') {
    return { data: new Float32Array(body.buffer.slice(body.byteOffset, body.byteOffset + body.length)), rows, cols };
  }
  if (info.dtype === 'BF16') {
    const u = new Uint16Array(body.buffer.slice(body.byteOffset, body.byteOffset + body.length));
    const f = new Float32Array(u.length);
    const scratch = new ArrayBuffer(4);
    const u32 = new Uint32Array(scratch);
    const f32 = new Float32Array(scratch);
    for (let i = 0; i < u.length; i++) { u32[0] = u[i] << 16; f[i] = f32[0]; }
    return { data: f, rows, cols };
  }
  return null;
}

// ── codes rows (captions + GT codes for conditioning) ─────────────────────

interface CodesRow {
  id: string; file: string; caption: string; lyrics: string;
  bpm: number; keyscale: string; timesignature: string; duration: number; codes: number[];
}

function readCodesFile(slug: string, variantKey: string): CodesRow[] {
  const codesPath = path.join(tensorsRoot(slug), variantKey, 'lm_codes.jsonl');
  if (!fs.existsSync(codesPath)) die(`no lm_codes.jsonl at ${codesPath}`);
  const rows: CodesRow[] = [];
  for (const line of fs.readFileSync(codesPath, 'utf-8').split('\n')) {
    const t = line.trim();
    if (!t) continue;
    try {
      const raw = JSON.parse(t) as Record<string, unknown>;
      const codes = Array.isArray(raw.codes)
        ? (raw.codes as unknown[]).map(v => Math.trunc(Number(v))).filter(v => v >= 0) : [];
      if (!codes.length) continue;
      rows.push({
        id: String(raw._id ?? ''), file: String(raw.file ?? ''), caption: String(raw.caption ?? ''),
        lyrics: String(raw.lyrics ?? ''), bpm: Math.trunc(Number(raw.bpm) || 0),
        keyscale: String(raw.keyscale ?? ''), timesignature: String(raw.timesignature ?? ''),
        duration: Math.trunc(Number(raw.duration) || 0), codes,
      });
    } catch { /* torn line */ }
  }
  if (!rows.length) die(`${codesPath} has no usable rows`);
  return rows;
}

/** The preprocess .st file for a codes row: <tensors>/<stem>-<id8>.safetensors
 *  (same match rule as findCodesRow's fallback). */
function stFileFor(tensorsDir: string, row: CodesRow): string {
  const m = /-([0-9a-f]{8})\.safetensors$/i.exec(row.file);
  const id8 = (m ? m[1] : row.id.slice(0, 8)).toLowerCase();
  for (const f of fs.readdirSync(tensorsDir)) {
    if (f.toLowerCase().endsWith(`-${id8}.safetensors`)) return path.join(tensorsDir, f);
  }
  return '';
}

// ── adapter resolution (dit-* roots) ──────────────────────────────────────

function resolveDitAdapter(arg: string): string {
  const asPath = path.resolve(arg);
  if (fs.existsSync(asPath) && fs.statSync(asPath).isDirectory()) {
    if (hasWeights(asPath)) return asPath;
    const run = latestRunDir(asPath);
    if (run) return run;
    die(`${asPath} holds no adapter weights`);
  }
  const root = config.aceServer.adapters;
  const lc = arg.toLowerCase();
  for (const sub of fs.readdirSync(root, { withFileTypes: true })) {
    if (!sub.isDirectory() || !/^dit-/i.test(sub.name)) continue;
    for (const e of fs.readdirSync(path.join(root, sub.name), { withFileTypes: true })) {
      if (e.isDirectory() && e.name.toLowerCase() === lc) {
        const run = latestRunDir(path.join(root, sub.name, e.name));
        if (run) return run;
      }
    }
  }
  die(`no DiT adapter found for "${arg}" under the dit-* roots`);
}

// ── engine calls ──────────────────────────────────────────────────────────

let cancelled = false;
process.on('SIGINT', () => { cancelled = true; console.log('\n  cancelling after the current job…'); });

async function awaitJob(jobId: string, what: string): Promise<void> {
  const deadline = Date.now() + JOB_DEADLINE_MS;
  for (;;) {
    if (cancelled) { await aceClient.cancelJob(jobId).catch(() => {}); throw new Error('cancelled'); }
    if (Date.now() > deadline) { await aceClient.cancelJob(jobId).catch(() => {}); throw new Error(`${what} timed out`); }
    const st = await aceClient.pollJob(jobId);
    if (st.status === 'done') return;
    if (st.status === 'failed') throw new Error(`${what} failed — see ace_engine.log`);
    if (st.status === 'cancelled') throw new Error(`${what} cancelled`);
    await new Promise(r => setTimeout(r, 300));
  }
}

/** /synth with GT codes; returns the rendered WAV bytes. */
async function synthClip(row: CodesRow, durSec: number, seed: number, adapterDir: string, scale: number,
                         steps: number, synthModel: string): Promise<Buffer> {
  const codesCsv = row.codes.slice(0, durSec * 5).join(',');
  const req: AceRequest = {
    caption: row.caption,
    lyrics: row.lyrics,
    duration: durSec,
    ...(row.bpm > 0 ? { bpm: row.bpm } : {}),
    ...(row.keyscale ? { keyscale: row.keyscale } : {}),
    ...(row.timesignature ? { timesignature: row.timesignature } : {}),
    seed,
    audio_codes: codesCsv,
    inference_steps: steps,
    task_type: 'text2music',
    ...(synthModel ? { synth_model: synthModel } : {}),
    ...(adapterDir ? { adapter: adapterDir, adapter_scale: scale } : {}),
  };
  const jobId = await aceClient.submitSynth(req, 'wav16');
  await awaitJob(jobId, `synth (${row.file}, seed ${seed})`);
  const res = await aceClient.getJobResult(jobId);
  if (!res.ok) throw new Error(`synth result fetch failed (${res.status})`);
  return Buffer.from(await res.arrayBuffer());
}

/** /vae encode: WAV in → raw F32 latents [T,64] out. */
async function encodeLatents(wav: Buffer): Promise<Float32Array> {
  const form = new FormData();
  form.append('request', JSON.stringify({}));
  form.append('audio', new Blob([new Uint8Array(wav)], { type: 'audio/wav' }), 'clip.wav');
  const res = await fetch(`${ENGINE}/vae`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`/vae encode submit failed (${res.status})`);
  const { id } = await res.json() as { id: string };
  await awaitJob(id, 'vae encode');
  const out = await aceClient.getJobResult(id);
  if (!out.ok) throw new Error(`/vae result fetch failed (${out.status})`);
  const buf = Buffer.from(await out.arrayBuffer());
  if (buf.length % (LATENT_DIM * 4) !== 0) throw new Error(`/vae returned ${buf.length} bytes — not [T,64] F32`);
  return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length));
}

// ── latent statistics ─────────────────────────────────────────────────────

class LatentStats {
  n = 0;
  mean = new Float64Array(LATENT_DIM);
  m2 = new Float64Array(LATENT_DIM * LATENT_DIM);   // sum of outer products (centered later)
  sumSq = new Float64Array(LATENT_DIM);

  add(latents: Float32Array): void {
    const T = latents.length / LATENT_DIM;
    for (let t = 0; t < T; t++) {
      const off = t * LATENT_DIM;
      for (let c = 0; c < LATENT_DIM; c++) {
        const v = latents[off + c];
        this.mean[c] += v;
        this.sumSq[c] += v * v;
      }
      for (let ci = 0; ci < LATENT_DIM; ci++) {
        const vi = latents[off + ci];
        const rowOff = ci * LATENT_DIM;
        for (let cj = 0; cj <= ci; cj++) {
          this.m2[rowOff + cj] += vi * latents[off + cj];
        }
      }
      this.n++;
    }
  }

  finalize(): { mu: Float64Array; cov: Float64Array; std: Float64Array } {
    const mu = new Float64Array(LATENT_DIM);
    const std = new Float64Array(LATENT_DIM);
    for (let c = 0; c < LATENT_DIM; c++) {
      mu[c] = this.mean[c] / this.n;
      std[c] = Math.sqrt(Math.max(0, this.sumSq[c] / this.n - mu[c] * mu[c]));
    }
    const cov = new Float64Array(LATENT_DIM * LATENT_DIM);
    for (let i = 0; i < LATENT_DIM; i++) {
      for (let j = 0; j <= i; j++) {
        const v = this.m2[i * LATENT_DIM + j] / this.n - mu[i] * mu[j];
        cov[i * LATENT_DIM + j] = v;
        cov[j * LATENT_DIM + i] = v;
      }
    }
    return { mu, cov, std };
  }
}

function matMul(a: Float64Array, b: Float64Array, n: number): Float64Array {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < n; k++) {
      const aik = a[i * n + k];
      if (aik === 0) continue;
      const ko = k * n, io = i * n;
      for (let j = 0; j < n; j++) out[io + j] += aik * b[ko + j];
    }
  }
  return out;
}

function trace(a: Float64Array, n: number): number {
  let t = 0;
  for (let i = 0; i < n; i++) t += a[i * n + i];
  return t;
}

/** PSD matrix square root via Denman–Beavers iteration (stable for our
 *  well-conditioned latent covariances; ~20 iterations converge). */
function sqrtmPSD(a: Float64Array, n: number): Float64Array {
  // scale to unit norm for convergence
  let norm = 0;
  for (let i = 0; i < n * n; i++) norm += a[i] * a[i];
  norm = Math.sqrt(norm) || 1;
  const y = Float64Array.from(a, v => v / norm);
  const z = new Float64Array(n * n);
  for (let i = 0; i < n; i++) z[i * n + i] = 1;
  let Y = y, Z = z;
  for (let it = 0; it < 25; it++) {
    const Yi = invPSD(Y, n);
    const Zi = invPSD(Z, n);
    const Yn = new Float64Array(n * n);
    const Zn = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) {
      Yn[i] = 0.5 * (Y[i] + Zi[i]);
      Zn[i] = 0.5 * (Z[i] + Yi[i]);
    }
    Y = Yn; Z = Zn;
  }
  return Float64Array.from(Y, v => v * Math.sqrt(norm));
}

/** Inverse via Gauss-Jordan with partial pivoting (64x64, fine). */
function invPSD(a: Float64Array, n: number): Float64Array {
  const m = Float64Array.from(a);
  const inv = new Float64Array(n * n);
  for (let i = 0; i < n; i++) inv[i * n + i] = 1;
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(m[r * n + col]) > Math.abs(m[piv * n + col])) piv = r;
    if (piv !== col) {
      for (let j = 0; j < n; j++) {
        [m[col * n + j], m[piv * n + j]] = [m[piv * n + j], m[col * n + j]];
        [inv[col * n + j], inv[piv * n + j]] = [inv[piv * n + j], inv[col * n + j]];
      }
    }
    const d = m[col * n + col] || 1e-12;
    for (let j = 0; j < n; j++) { m[col * n + j] /= d; inv[col * n + j] /= d; }
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = m[r * n + col];
      if (f === 0) continue;
      for (let j = 0; j < n; j++) { m[r * n + j] -= f * m[col * n + j]; inv[r * n + j] -= f * inv[col * n + j]; }
    }
  }
  return inv;
}

/** Frechet distance between two Gaussians fit to latent frames. */
function frechet(s1: { mu: Float64Array; cov: Float64Array }, s2: { mu: Float64Array; cov: Float64Array }): number {
  const n = LATENT_DIM;
  let d2 = 0;
  for (let i = 0; i < n; i++) d2 += (s1.mu[i] - s2.mu[i]) ** 2;
  // Tr(C1 + C2 - 2*sqrtm(C1*C2)) — C1*C2 of PSD matrices has a PSD-similar
  // sqrt; regularize the diagonal to keep the iteration stable.
  const c1 = Float64Array.from(s1.cov);
  const c2 = Float64Array.from(s2.cov);
  for (let i = 0; i < n; i++) { c1[i * n + i] += 1e-6; c2[i * n + i] += 1e-6; }
  const prod = matMul(c1, c2, n);
  const root = sqrtmPSD(prod, n);
  return d2 + trace(c1, n) + trace(c2, n) - 2 * trace(root, n);
}

function profileDistance(a: { mu: Float64Array; std: Float64Array }, b: { mu: Float64Array; std: Float64Array }): number {
  let d = 0;
  for (let c = 0; c < LATENT_DIM; c++) d += (a.mu[c] - b.mu[c]) ** 2 + (a.std[c] - b.std[c]) ** 2;
  return Math.sqrt(d);
}

function corrDistance(a: { cov: Float64Array; std: Float64Array }, b: { cov: Float64Array; std: Float64Array }): number {
  const n = LATENT_DIM;
  let d = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) {
      const ca = a.cov[i * n + j] / ((a.std[i] * a.std[j]) || 1e-9);
      const cb = b.cov[i * n + j] / ((b.std[i] * b.std[j]) || 1e-9);
      d += (ca - cb) ** 2;
    }
  }
  return Math.sqrt(d);
}

// ── run file ──────────────────────────────────────────────────────────────

interface GenRec { rowId: string; side: 'base' | 'adapter'; seed: number; latentsB64: string; frames: number }

interface DitRuns {
  version: number; createdAt: string; dataset: string; variant: string;
  adapterPath: string; adapterScale: number; synthModel: string;
  params: { steps: number; duration: number; seeds: number; samples: number; seedBase: number };
  rows: Array<{ id: string; file: string }>;
  gens: GenRec[];
}

function saveRuns(outDir: string, runs: DitRuns): void {
  const tmp = path.join(outDir, 'runs.json.tmp');
  fs.writeFileSync(tmp, JSON.stringify(runs));
  fs.renameSync(tmp, path.join(outDir, 'runs.json'));
}

// ── analysis ──────────────────────────────────────────────────────────────

function analyze(runs: DitRuns, tensorsDir: string, rowsFull: CodesRow[]): Record<string, unknown> {
  const durFrames = runs.params.duration * 25;
  const gtAll = new LatentStats(), gtA = new LatentStats(), gtB = new LatentStats();
  let gtSongs = 0;
  runs.rows.forEach((r, i) => {
    const row = rowsFull.find(x => x.id === r.id || x.file === r.file);
    if (!row) return;
    const stf = stFileFor(tensorsDir, row);
    if (!stf) return;
    const t = readTensorF32(stf, 'target_latents');
    if (!t) return;
    const take = Math.min(durFrames, t.rows) * LATENT_DIM;
    const cut = t.data.subarray(0, take);
    gtAll.add(cut as Float32Array);
    (i % 2 === 0 ? gtA : gtB).add(cut as Float32Array);
    gtSongs++;
  });
  if (gtSongs < 2) die('fewer than 2 songs with target_latents — cannot analyze');

  const sideStats = (side: 'base' | 'adapter') => {
    const s = new LatentStats();
    for (const g of runs.gens.filter(x => x.side === side)) {
      const buf = Buffer.from(g.latentsB64, 'base64');
      s.add(new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length)));
    }
    return s;
  };
  const bS = sideStats('base'), aS = sideStats('adapter');
  if (!bS.n || !aS.n) die('missing a side — generate first');

  const gt = gtAll.finalize(), gA = gtA.finalize(), gB = gtB.finalize();
  const base = bS.finalize(), adapter = aS.finalize();

  const result = {
    frames: { gt: gtAll.n, base: bS.n, adapter: aS.n, gtSongs },
    frechet: {
      base: frechet(gt, base), adapter: frechet(gt, adapter),
      floor: frechet(gA, gB), baseVsAdapter: frechet(base, adapter),
    },
    profile: {
      base: profileDistance(gt, base), adapter: profileDistance(gt, adapter), floor: profileDistance(gA, gB),
    },
    corr: {
      base: corrDistance(gt, base), adapter: corrDistance(gt, adapter), floor: corrDistance(gA, gB),
    },
  };
  const closer = (m: { base: number; adapter: number }) => m.adapter < m.base;
  const wins = [result.frechet, result.profile, result.corr].filter(closer).length;
  return {
    ...result,
    verdict: {
      wins,
      status: wins >= 2 ? 'toward' : wins === 0 ? 'away' : 'inconclusive',
      summary: `adapter closer on ${wins}/3 latent metrics ` +
        `(frechet ${result.frechet.base.toFixed(3)}→${result.frechet.adapter.toFixed(3)}, floor ${result.frechet.floor.toFixed(3)})`,
    },
  };
}

// ── main ──────────────────────────────────────────────────────────────────

async function cmdGenerate(args: Map<string, string>): Promise<void> {
  const slug = args.get('dataset') ?? '';
  const adapterArg = args.get('adapter') ?? '';
  if (!slug || !adapterArg) die('generate needs --dataset and --adapter');
  const variant = args.get('variant') || newestVariantKey(slug);
  if (!variant) die(`dataset "${slug}" has no tensor variants`);
  const tensorsDir = path.join(tensorsRoot(slug), variant);
  const rowsFull = readCodesFile(slug, variant);
  const adapterPath = resolveDitAdapter(adapterArg);
  // The variant IS the base the latents came from — render on it, not a
  // default. The variant key is the model name with the extension stripped
  // (variantKeyFor), so put it back for the engine's resolver.
  const synthModel = `${variant}.gguf`;

  const scale = Number(args.get('scale')) || 1.0;
  const steps = Math.trunc(Number(args.get('steps')) || 8);
  const durSec = Math.trunc(Number(args.get('duration')) || 30);
  const seeds = Math.max(1, Math.trunc(Number(args.get('seeds')) || 1));
  const seedBase = Math.trunc(Number(args.get('seed-base')) || 42);
  const nSamples = Math.trunc(Number(args.get('samples')) || 8);

  let rows = rowsFull;
  if (nSamples > 0 && nSamples < rowsFull.length) {
    const step = rowsFull.length / nSamples;
    rows = Array.from({ length: nSamples }, (_, i) => rowsFull[Math.floor(i * step)]);
  }

  const outDir = path.resolve(args.get('out') ?? path.join(tensorsDir, 'dit-eval', runStamp()));
  fs.mkdirSync(outDir, { recursive: true });

  let runs: DitRuns = {
    version: 1, createdAt: new Date().toISOString(), dataset: slug, variant,
    adapterPath, adapterScale: scale, synthModel,
    params: { steps, duration: durSec, seeds, samples: rows.length, seedBase },
    rows: rows.map(r => ({ id: r.id, file: r.file })),
    gens: [],
  };
  const runsPath = path.join(outDir, 'runs.json');
  if (fs.existsSync(runsPath)) {
    const prev = JSON.parse(fs.readFileSync(runsPath, 'utf-8')) as DitRuns;
    if (prev.adapterPath !== adapterPath || prev.adapterScale !== scale) {
      die(`${runsPath} belongs to a different adapter/scale — pick a fresh --out (or seed base gens only)`);
    }
    runs = prev;
    console.log(`  resuming: ${runs.gens.length} generations already done`);
  }
  const done = new Set(runs.gens.map(g => `${g.rowId}|${g.seed}|${g.side}`));

  const work: Array<{ row: CodesRow; seed: number; side: 'base' | 'adapter' }> = [];
  for (const side of ['base', 'adapter'] as const) {
    for (const row of rows) {
      for (let k = 0; k < seeds; k++) {
        const seed = seedBase + k * 101;
        if (!done.has(`${row.id}|${seed}|${side}`)) work.push({ row, seed, side });
      }
    }
  }
  console.log(`\nDiT adapter eval — ${slug} (${variant})`);
  console.log(`  adapter: ${adapterPath} (scale ${scale}) · steps ${steps} · ${durSec}s clips`);
  console.log(`  to run:  ${work.length} renders (${runs.gens.length} done)\n  out: ${outDir}\n`);

  for (let i = 0; i < work.length; i++) {
    if (cancelled) break;
    const { row, seed, side } = work[i];
    process.stdout.write(`  [${i + 1}/${work.length}] ${side.padEnd(7)} ${row.file.slice(0, 44).padEnd(46)} seed ${seed} … `);
    try {
      const t0 = Date.now();
      const wav = await synthClip(row, durSec, seed, side === 'adapter' ? adapterPath : '', scale, steps, synthModel);
      const lat = await encodeLatents(wav);
      runs.gens.push({
        rowId: row.id, side, seed,
        latentsB64: Buffer.from(lat.buffer, lat.byteOffset, lat.byteLength).toString('base64'),
        frames: lat.length / LATENT_DIM,
      });
      saveRuns(outDir, runs);
      console.log(`${lat.length / LATENT_DIM} frames in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
    } catch (err) {
      console.log(`FAILED: ${err instanceof Error ? err.message : String(err)}`);
      if (cancelled) break;
    }
  }
  if (cancelled) { console.log('\n  stopped — re-run to resume'); return; }

  const results = analyze(runs, tensorsDir, rowsFull);
  fs.writeFileSync(path.join(outDir, 'results.json'), JSON.stringify(results, null, 2));
  console.log(`\n${JSON.stringify(results, null, 2)}`);
  console.log(`\n  results: ${path.join(outDir, 'results.json')}`);
}

function cmdReport(args: Map<string, string>): void {
  const dir = path.resolve(args.get('run') ?? '');
  const runs = JSON.parse(fs.readFileSync(path.join(dir, 'runs.json'), 'utf-8')) as DitRuns;
  const tensorsDir = path.join(tensorsRoot(runs.dataset), runs.variant);
  const rowsFull = readCodesFile(runs.dataset, runs.variant);
  const results = analyze(runs, tensorsDir, rowsFull);
  fs.writeFileSync(path.join(dir, 'results.json'), JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results, null, 2));
}

const cmd = process.argv[2];
const args = parseArgs(process.argv.slice(3));
if (cmd === 'generate') {
  cmdGenerate(args).catch(err => die(err instanceof Error ? err.message : String(err)));
} else if (cmd === 'report') {
  cmdReport(args);
} else {
  console.log('usage: npx tsx scripts/dit-adapter-eval.ts generate --dataset <slug> --adapter <path|artist> [opts]');
  console.log('       npx tsx scripts/dit-adapter-eval.ts report --run <dir>');
  process.exit(cmd ? 1 : 0);
}
