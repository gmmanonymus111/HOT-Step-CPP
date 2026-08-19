#!/usr/bin/env npx tsx
/**
 * lm-adapter-eval.ts — objective eval of a planner-LM adapter (2026-08-08).
 *
 * Answers "did training actually move the LM toward the artist?" with data
 * instead of ears: generates audio-code sequences base-vs-adapter on the
 * dataset's own captions/lyrics (identical seeds — the audition A/B
 * discipline), then compares both populations against the ground-truth codes
 * in lm_codes.jsonl. Metrics + HTML report: lm-adapter-eval-report.ts.
 *
 * Needs the app running (ace-server reachable) — run from server/:
 *
 *   npx tsx scripts/lm-adapter-eval.ts generate --dataset <slug> --adapter <path|artist>
 *   npx tsx scripts/lm-adapter-eval.ts report --run <dir>
 *
 *   generate:
 *     --dataset <slug>        dataset (tensors/<slug>) — required
 *     --adapter <path|name>   PEFT dir, or an artist name resolved in the
 *                             lm-* adapter roots (newest run wins) — required
 *     --variant <key>         tensor variant (default: newest)
 *     --seeds <n>             seeds per song per side (default 2)
 *     --samples <n>           songs to use (default all, evenly spaced subset)
 *     --max-duration <sec>    per-song generation cap (default 120; 0 = full
 *                             song length, capped at 300)
 *     --scale <x>             lm_adapter_scale (default 1.0)
 *     --lm-model <name>       base LM override (default: derived from adapter size)
 *     --temp/--top-p/--cfg/--rep   sampling (defaults 0.85/0.9/2.0/1.0 — audition parity)
 *     --seed-base <n>         first seed (default 42; seeds step +101)
 *     --out <dir>             output dir (default tensors/<slug>/<variant>/lm-eval/<stamp>)
 *
 * IDEMPOTENT: re-running with the same --out resumes — completed
 * (song, seed, side) triples are read back from runs.json and skipped, so an
 * interrupted batch only generates the shortfall.
 *
 * Order of work: every base generation first, then every adapter generation —
 * at most two LM (re)load cycles instead of one per pair. The LM is pinned
 * with ?keep_loaded=1 for the batch and the evict policy restored in a
 * finally, mirroring auditionRunner.
 */
import fs from 'fs';
import path from 'path';
import { aceClient, type AceRequest } from '../src/services/aceClient.js';
import {
  getModelSnapshot, pickLmFor, refreshModelSnapshot, tensorsRoot,
} from '../src/services/training/aceTrain.js';
import { newestVariantKey, variantExists } from '../src/services/training/trainLmStatus.js';
import {
  hasWeights, latestRunDir, lmAdapterRoots, lmSizeOfAdapterPath, runStamp,
} from '../src/services/training/adapterLayout.js';
import type { LmSize } from '../src/services/training/types.js';
import {
  analyzeRuns, renderReport, type EvalGen, type EvalResults, type EvalRow, type EvalRuns, type EvalSide,
} from './lm-adapter-eval-report.js';

const LM_DEADLINE_MS = 10 * 60_000;   // audition parity: a wedged-engine tripwire

// ── args ──────────────────────────────────────────────────────────────────

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

function argNum(args: Map<string, string>, key: string, fallback: number): number {
  const n = Number(args.get(key));
  return Number.isFinite(n) ? n : fallback;
}

function die(msg: string): never {
  console.error(`\nERROR: ${msg}`);
  process.exit(1);
}

// ── lm_codes.jsonl ────────────────────────────────────────────────────────

interface CodesFileRow {
  id: string; file: string; caption: string; lyrics: string;
  bpm: number; keyscale: string; timesignature: string; duration: number; codes: number[];
}

function readCodesFile(slug: string, variantKey: string): CodesFileRow[] {
  const codesPath = path.join(tensorsRoot(slug), variantKey, 'lm_codes.jsonl');
  if (!fs.existsSync(codesPath)) {
    die(`no lm_codes.jsonl at ${codesPath} — run Preprocess + Extract for this variant first`);
  }
  const rows: CodesFileRow[] = [];
  for (const line of fs.readFileSync(codesPath, 'utf-8').split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let raw: Record<string, unknown>;
    try { raw = JSON.parse(trimmed) as Record<string, unknown>; } catch { continue; }
    const codes = Array.isArray(raw.codes)
      ? (raw.codes as unknown[]).map(v => Math.trunc(Number(v))).filter(v => v >= 0)
      : [];
    if (!codes.length) continue;
    rows.push({
      id: String(raw._id ?? '') || String(raw.file ?? '').slice(-20),
      file: String(raw.file ?? ''),
      caption: String(raw.caption ?? ''),
      lyrics: String(raw.lyrics ?? ''),
      bpm: Math.trunc(Number(raw.bpm) || 0),
      keyscale: String(raw.keyscale ?? ''),
      timesignature: String(raw.timesignature ?? ''),
      duration: Math.trunc(Number(raw.duration) || 0),
      codes,
    });
  }
  if (!rows.length) die(`lm_codes.jsonl at ${codesPath} has no usable rows`);
  return rows;
}

// ── adapter resolution ────────────────────────────────────────────────────

function resolveAdapter(arg: string): string {
  const asPath = path.resolve(arg);
  if (fs.existsSync(asPath) && fs.statSync(asPath).isDirectory()) {
    if (hasWeights(asPath)) return asPath;
    const run = latestRunDir(asPath);
    if (run) return run;
    die(`${asPath} exists but holds no adapter weights (adapter_model.safetensors)`);
  }
  // Artist-name lookup across the planner-adapter roots, newest run wins.
  const lc = arg.toLowerCase();
  const candidates: string[] = [];
  for (const root of lmAdapterRoots()) {
    try {
      for (const entry of fs.readdirSync(root.dir, { withFileTypes: true })) {
        if (entry.isDirectory() && entry.name.toLowerCase() === lc) {
          const run = latestRunDir(path.join(root.dir, entry.name));
          if (run) candidates.push(run);
        }
      }
    } catch { /* root absent */ }
  }
  if (!candidates.length) die(`no adapter found for "${arg}" — pass a PEFT dir path or an artist folder name from the lm-* adapter roots`);
  if (candidates.length > 1) {
    console.log(`  note: "${arg}" matched ${candidates.length} adapters; using ${candidates[0]}`);
  }
  return candidates[0];
}

// ── engine helpers ────────────────────────────────────────────────────────

let cancelled = false;
process.on('SIGINT', () => {
  cancelled = true;
  console.log('\n  cancelling after the current job…');
});

async function awaitEngineJob(jobId: string, what: string): Promise<void> {
  const deadline = Date.now() + LM_DEADLINE_MS;
  for (;;) {
    if (cancelled) {
      await aceClient.cancelJob(jobId).catch(() => { /* engine may be gone */ });
      throw new Error('cancelled');
    }
    if (Date.now() > deadline) {
      await aceClient.cancelJob(jobId).catch(() => { /* best effort */ });
      throw new Error(`${what} timed out after ${Math.round(LM_DEADLINE_MS / 60_000)} min`);
    }
    const status = await aceClient.pollJob(jobId);
    if (status.status === 'done') return;
    if (status.status === 'failed') throw new Error(`${what} failed — see ace_engine.log`);
    if (status.status === 'cancelled') throw new Error(`${what} cancelled`);
    await new Promise(r => setTimeout(r, 250));
  }
}

/** One /lm run. Every field written explicitly — never rebuilt from an echoed
 *  request (LM echo sideband rule; see auditionService.buildLmRequest). */
async function runLm(
  row: CodesFileRow, durUsed: number, seed: number, side: EvalSide,
  cfg: { lmModel: string; adapterPath: string; scale: number; temperature: number; topP: number; cfgScale: number; repPenalty: number },
): Promise<EvalGen> {
  const req: AceRequest & { lm_mode: string } = {
    caption: row.caption,          // pre-tagged by the trainer (lm_apply_tag)
    lyrics: row.lyrics,
    duration: durUsed,
    seed,
    lm_seed: seed,
    lm_mode: 'generate',
    lm_batch_size: 1,
    lm_temperature: cfg.temperature,
    lm_top_p: cfg.topP,
    lm_cfg_scale: cfg.cfgScale,
    lm_rep_penalty: cfg.repPenalty,
    lm_model: cfg.lmModel,
    lm_adapter: side === 'adapter' ? cfg.adapterPath : '',
    lm_adapter_scale: cfg.scale,
  };
  const t0 = Date.now();
  const jobId = await aceClient.submitLm(req, undefined, true);
  await awaitEngineJob(jobId, `LM ${side} (${row.file}, seed ${seed})`);
  const res = await aceClient.getJobResult(jobId);
  if (!res.ok) throw new Error(`LM result fetch failed (${res.status})`);
  const echoed = await res.json() as AceRequest[];
  const plan = Array.isArray(echoed) ? echoed[0] : undefined;
  const csv = typeof plan?.audio_codes === 'string' ? plan.audio_codes : '';
  const codes = csv.split(',').map(t => Math.trunc(Number(t))).filter(v => Number.isFinite(v) && v >= 0);
  if (!codes.length) throw new Error('LM returned no audio codes');
  return {
    rowId: row.id, side, seed, codes,
    lmMs: Date.now() - t0,
    bpm: Math.trunc(Number(plan?.bpm) || 0),
    keyscale: String(plan?.keyscale ?? ''),
  };
}

// ── output ────────────────────────────────────────────────────────────────

function saveRuns(outDir: string, runs: EvalRuns): void {
  const tmp = path.join(outDir, 'runs.json.tmp');
  fs.writeFileSync(tmp, JSON.stringify(runs));
  fs.renameSync(tmp, path.join(outDir, 'runs.json'));
}

function writeReport(outDir: string, runs: EvalRuns): EvalResults {
  const results = analyzeRuns(runs);
  fs.writeFileSync(path.join(outDir, 'results.json'), JSON.stringify(results, null, 2));
  fs.writeFileSync(path.join(outDir, 'report.html'), renderReport(runs, results));
  return results;
}

function printSummary(outDir: string, results: EvalResults): void {
  const v = results.verdict;
  console.log('\n── Results ─────────────────────────────────────────────');
  console.log(`  Unigram JS → GT:   base ${results.unigram.base.toFixed(4)}  adapter ${results.unigram.adapter.toFixed(4)}  (floor ${results.unigram.floor.toFixed(4)})`);
  console.log(`  Marginals (mean):  base ${results.marginalMean.base.toFixed(4)}  adapter ${results.marginalMean.adapter.toFixed(4)}  (floor ${results.marginalMean.floor.toFixed(4)})`);
  console.log(`  Transitions:       base ${results.transitionMean.base.toFixed(4)}  adapter ${results.transitionMean.adapter.toFixed(4)}  (floor ${results.transitionMean.floor.toFixed(4)})`);
  console.log(`  Dims won:          ${v.dimsWon}/6 marginals, ${v.transWon}/6 transitions`);
  console.log(`  OOV mass:          base ${(results.oovMass.base * 100).toFixed(1)}%  adapter ${(results.oovMass.adapter * 100).toFixed(1)}%`);
  console.log(`  Max replayed run:  adapter ${results.memorization.maxSec.adapter.toFixed(1)}s  base ${results.memorization.maxSec.base.toFixed(1)}s`);
  console.log(`\n  VERDICT: ${v.status.toUpperCase()} — ${v.summary}`);
  console.log(`\n  Report: ${path.join(outDir, 'report.html')}`);
}

// ── main ──────────────────────────────────────────────────────────────────

async function cmdReport(args: Map<string, string>): Promise<void> {
  const dir = args.get('run') ?? '';
  if (!dir) die('report needs --run <dir containing runs.json>');
  const runsPath = path.join(path.resolve(dir), 'runs.json');
  if (!fs.existsSync(runsPath)) die(`no runs.json in ${dir}`);
  const runs = JSON.parse(fs.readFileSync(runsPath, 'utf-8')) as EvalRuns;
  const results = writeReport(path.dirname(runsPath), runs);
  printSummary(path.dirname(runsPath), results);
}

async function cmdGenerate(args: Map<string, string>): Promise<void> {
  const slug = args.get('dataset') ?? '';
  const adapterArg = args.get('adapter') ?? '';
  if (!slug || !adapterArg) die('generate needs --dataset <slug> and --adapter <path|artist>');

  const variant = args.get('variant') || newestVariantKey(slug);
  if (!variant) die(`dataset "${slug}" has no preprocessed tensor variants (${tensorsRoot(slug)})`);
  if (!variantExists(slug, variant)) die(`variant "${variant}" not found under ${tensorsRoot(slug)}`);

  const allRows = readCodesFile(slug, variant);
  const adapterPath = resolveAdapter(adapterArg);

  // Base LM: explicit override, else derived from the adapter's size — never
  // the engine's sticky resolve_name fallback (the 4B-on-0.6B failure).
  await refreshModelSnapshot();
  const lms = getModelSnapshot().lm;
  let lmModel = args.get('lm-model') ?? '';
  if (!lmModel) {
    let size: LmSize | '' = '';
    let probe = adapterPath;
    for (let hops = 0; hops < 4 && probe && !size; hops++) {
      size = lmSizeOfAdapterPath(probe);
      probe = path.dirname(probe);
    }
    if (!size) die(`cannot derive the base-LM size from ${adapterPath} — pass --lm-model explicitly`);
    if (!lms.length) die('engine unreachable (no /props) — start the app (dev.bat) first');
    lmModel = pickLmFor(size, lms);
    if (!lmModel) die(`adapter needs a ${size} LM base but none is installed (engine lists: ${lms.join(', ') || 'none'})`);
  }

  const seeds = Math.max(1, Math.trunc(argNum(args, 'seeds', 2)));
  const seedBase = Math.trunc(argNum(args, 'seed-base', 42));
  const maxDuration = Math.trunc(argNum(args, 'max-duration', 120));
  const nSamples = Math.trunc(argNum(args, 'samples', allRows.length));
  const cfg = {
    lmModel, adapterPath,
    scale: argNum(args, 'scale', 1.0),
    temperature: argNum(args, 'temp', 0.85),
    topP: argNum(args, 'top-p', 0.9),
    cfgScale: argNum(args, 'cfg', 2.0),
    repPenalty: argNum(args, 'rep', 1.0),
  };

  // Evenly-spaced subset when --samples < rows: keeps album coverage.
  let rows = allRows;
  if (nSamples > 0 && nSamples < allRows.length) {
    const step = allRows.length / nSamples;
    rows = Array.from({ length: nSamples }, (_, i) => allRows[Math.floor(i * step)]);
  }

  const durFor = (row: CodesFileRow): number => {
    const full = row.duration > 0 ? row.duration : Math.round(row.codes.length / 5);
    return Math.max(10, Math.min(full, maxDuration > 0 ? maxDuration : 300));
  };

  const outDir = path.resolve(args.get('out') ?? path.join(tensorsRoot(slug), variant, 'lm-eval', runStamp()));
  fs.mkdirSync(outDir, { recursive: true });

  // Resume: completed (row, seed, side) triples from a previous partial run.
  let runs: EvalRuns = {
    version: 1, createdAt: new Date().toISOString(),
    dataset: slug, variant, adapterPath, adapterScale: cfg.scale, lmModel,
    params: { temperature: cfg.temperature, topP: cfg.topP, cfgScale: cfg.cfgScale, repPenalty: cfg.repPenalty, maxDuration, seeds, seedBase },
    rows: rows.map((r): EvalRow => ({
      id: r.id, file: r.file, caption: r.caption, lyrics: r.lyrics,
      bpm: r.bpm, duration: r.duration, durUsed: durFor(r), gtCodes: r.codes,
    })),
    gens: [],
  };
  const runsPath = path.join(outDir, 'runs.json');
  if (fs.existsSync(runsPath)) {
    try {
      const prev = JSON.parse(fs.readFileSync(runsPath, 'utf-8')) as EvalRuns;
      if (prev.adapterPath === adapterPath && prev.lmModel === lmModel) {
        runs = prev;
        console.log(`  resuming: ${runs.gens.length} generations already in ${runsPath}`);
      } else {
        die(`${runsPath} exists for a different adapter/base (${prev.adapterPath}) — pick a fresh --out`);
      }
    } catch (e) {
      if (e instanceof SyntaxError) die(`${runsPath} is unreadable — delete it or pick a fresh --out`);
      throw e;
    }
  }
  const done = new Set(runs.gens.map(g => `${g.rowId}|${g.seed}|${g.side}`));

  // Work plan: all base runs first, then all adapter runs (2 LM loads, not 2N).
  const work: Array<{ row: CodesFileRow; seed: number; side: EvalSide }> = [];
  for (const side of ['base', 'adapter'] as const) {
    for (const row of rows) {
      for (let k = 0; k < seeds; k++) {
        const seed = seedBase + k * 101;
        if (!done.has(`${row.id}|${seed}|${side}`)) work.push({ row, seed, side });
      }
    }
  }

  console.log(`\nPlanner-LM adapter eval — ${slug} (${variant})`);
  console.log(`  adapter:  ${adapterPath} (scale ${cfg.scale})`);
  console.log(`  base LM:  ${lmModel}`);
  console.log(`  songs:    ${rows.length}/${allRows.length} · seeds ${seeds} · cap ${maxDuration || 300}s`);
  console.log(`  to run:   ${work.length} generations (${runs.gens.length} already done)`);
  console.log(`  out:      ${outDir}\n`);

  let failures = 0;
  try {
    for (let i = 0; i < work.length; i++) {
      if (cancelled) break;
      const { row, seed, side } = work[i];
      const durUsed = durFor(row);
      process.stdout.write(`  [${i + 1}/${work.length}] ${side.padEnd(7)} ${row.file.slice(0, 48).padEnd(50)} seed ${seed} … `);
      try {
        const gen = await runLm(row, durUsed, seed, side, cfg);
        runs.gens.push(gen);
        saveRuns(outDir, runs);
        console.log(`${gen.codes.length} codes in ${(gen.lmMs / 1000).toFixed(1)}s`);
      } catch (err) {
        failures++;
        console.log(`FAILED: ${err instanceof Error ? err.message : String(err)}`);
        if (cancelled) break;
        if (failures >= 3 && runs.gens.length === 0) die('first three generations all failed — is the engine up and the adapter path valid?');
      }
    }
  } finally {
    await aceClient.restoreEvictPolicy().catch(() => { /* best effort */ });
  }

  if (cancelled) {
    console.log(`\n  stopped: ${runs.gens.length} generations saved — re-run the same command to resume, or run report --run "${outDir}"`);
    return;
  }
  if (failures) console.log(`\n  ${failures} generation(s) failed — analysis uses what completed`);

  const results = writeReport(outDir, runs);
  printSummary(outDir, results);
}

const args = parseArgs(process.argv.slice(3));
const cmd = process.argv[2];
if (cmd === 'generate') {
  cmdGenerate(args).catch(err => die(err instanceof Error ? err.message : String(err)));
} else if (cmd === 'report') {
  cmdReport(args).catch(err => die(err instanceof Error ? err.message : String(err)));
} else {
  console.log('usage: npx tsx scripts/lm-adapter-eval.ts generate --dataset <slug> --adapter <path|artist> [options]');
  console.log('       npx tsx scripts/lm-adapter-eval.ts report --run <dir>');
  console.log('(see the header comment for all options)');
  process.exit(cmd ? 1 : 0);
}
