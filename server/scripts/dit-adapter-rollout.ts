#!/usr/bin/env npx tsx
/**
 * dit-adapter-rollout.ts — calibrate + repoint for DiT (sound) adapters
 * (2026-08-11). The DiT twin of lm-adapter-rollout.ts.
 *
 * Metric: latent-space Frechet distance (dit-adapter-eval.ts) — validated
 * 2026-08-10 with positive/negative controls AND Rob's ears (matched adapter
 * better than base, mismatched adapter audibly bad + 73% worse on the
 * metric). Frechet is the ranking score; the channel profile is reported as
 * a sanity column; corr is excluded (weak — mildly favored the wrong adapter
 * in the negative control).
 *
 *   calibrate --dataset <slug> --new-run <dir> [--old-run <dir>]
 *             [--scales 0.75,1] [--samples 8] [--duration 30] [--seeds 1]
 *             [--tag ditcal]
 *       evals {old?, new} x scales around a freshly-trained run, picks by
 *       Frechet under the strict-win rule, bakes non-1.0 winners, writes
 *       hot_step_eval.json sidecars. Emits CALIBRATE_RESULT {json}.
 *       No training, no DB writes (the caller owns preset repoints).
 *   repoint [--apply]
 *       album_presets.adapter_path -> served DiT adapters (dry-run default,
 *       SQLite online backup before writes).
 *   report
 *       summarize sidecars across the dit-* roots.
 *
 * Deliberately NO `run` (batch deep-training) subcommand yet: a DiT resume
 * costs real GPU time per artist (unlike the LM's ~2 min), so back-catalogue
 * deepening is a scheduling decision, not a default. Studio wiring queues
 * `calibrate` after every train-dit job instead.
 */
import { spawnSync } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';
import { config } from '../src/config.js';
import { tensorsRoot } from '../src/services/training/aceTrain.js';
import { newestVariantKey } from '../src/services/training/trainLmStatus.js';
import { hasWeights, runStamp } from '../src/services/training/adapterLayout.js';
import { bakeFile } from './lm-adapter-calibrate.js';

const SERVER_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require2 = createRequire(path.join(SERVER_DIR, 'package.json'));
const SIDECAR = 'hot_step_eval.json';

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

function sha1File(p: string): string {
  return crypto.createHash('sha1').update(fs.readFileSync(p)).digest('hex');
}

function weightsFileIn(dir: string): string {
  for (const f of ['lokr_weights.safetensors', 'adapter_model.safetensors']) {
    const p = path.join(dir, f);
    if (fs.existsSync(p)) return p;
  }
  return '';
}

// ── candidates + eval ─────────────────────────────────────────────────────

interface Candidate { label: string; adapterDir: string; scale: number; outDir: string }

interface CandScore {
  label: string; adapterDir: string; scale: number;
  frechet: number; profile: number; baseFrechet: number; floor: number;
  wins: number; verdict: string;
}

function evalCandidate(dataset: string, variant: string, cand: Candidate, baseCacheRuns: string,
                       lean: { seeds: number; duration: number; samples: number }): boolean {
  if (fs.existsSync(path.join(cand.outDir, 'results.json'))) return true;
  if (baseCacheRuns && !fs.existsSync(path.join(cand.outDir, 'runs.json'))) {
    const runs = JSON.parse(fs.readFileSync(baseCacheRuns, 'utf-8'));
    runs.gens = runs.gens.filter((g: any) => g.side === 'base');
    runs.adapterPath = cand.adapterDir;
    runs.adapterScale = cand.scale;
    fs.mkdirSync(cand.outDir, { recursive: true });
    fs.writeFileSync(path.join(cand.outDir, 'runs.json'), JSON.stringify(runs));
  }
  const r = spawnSync('npx', [
    'tsx', 'scripts/dit-adapter-eval.ts', 'generate',
    '--dataset', dataset, '--variant', variant,
    '--adapter', cand.adapterDir, '--scale', String(cand.scale),
    '--seeds', String(lean.seeds), '--duration', String(lean.duration), '--samples', String(lean.samples),
    '--out', cand.outDir,
  ], { stdio: ['ignore', 'pipe', 'pipe'], encoding: 'utf-8', shell: process.platform === 'win32', cwd: SERVER_DIR, timeout: 30 * 60_000 });
  if (r.status !== 0) {
    const tail = ((r.stdout ?? '') + (r.stderr ?? '')).split('\n').filter(l => l.trim()).slice(-3).join(' | ');
    console.log(`      eval ${cand.label} FAILED: ${tail}`);
    return false;
  }
  return fs.existsSync(path.join(cand.outDir, 'results.json'));
}

function scoreOf(cand: Candidate): CandScore | null {
  try {
    const r = JSON.parse(fs.readFileSync(path.join(cand.outDir, 'results.json'), 'utf-8'));
    return {
      label: cand.label, adapterDir: cand.adapterDir, scale: cand.scale,
      frechet: r.frechet.adapter, profile: r.profile.adapter,
      baseFrechet: r.frechet.base, floor: r.frechet.floor,
      wins: r.verdict.wins, verdict: r.verdict.status,
    };
  } catch { return null; }
}

function writeSidecar(dir: string, dataset: string, variant: string, s: CandScore,
                      extra: Record<string, unknown>, lean: { seeds: number; duration: number; samples: number }): void {
  fs.writeFileSync(path.join(dir, SIDECAR), JSON.stringify({
    version: 1,
    evaluatedAt: new Date().toISOString(),
    dataset, variant,
    /** DiT scores are latent-space Frechet — LOWER = closer. NOT comparable
     *  to LM adapters' JS-divergence scores; `metric` says which this is. */
    metric: 'latent-frechet',
    score: Number(s.frechet.toFixed(3)),
    profile: Number(s.profile.toFixed(3)),
    baseScore: Number(s.baseFrechet.toFixed(3)),
    floor: Number(s.floor.toFixed(3)),
    verdict: s.verdict,
    measuredAtScale: s.scale,
    protocol: { seeds: lean.seeds, duration: lean.duration, samples: lean.samples, steps: 8 },
    ...extra,
  }, null, 2));
}

// ── calibrate ─────────────────────────────────────────────────────────────

interface Outcome {
  artist: string;
  status: 'repointed' | 'kept-old' | 'baked-old' | 'failed';
  servedDir: string;
  oldScore: number | null;
  newScore: number | null;
  winner: string;
  detail: string;
  minutes: number;
}

async function cmdCalibrate(args: Map<string, string>): Promise<void> {
  const t0 = Date.now();
  const dataset = args.get('dataset') ?? '';
  const newRun = path.resolve(args.get('new-run') ?? '');
  if (!dataset || !newRun) die('calibrate needs --dataset <slug> and --new-run <dir>');
  if (!hasWeights(newRun)) die(`${newRun} holds no adapter weights`);
  const variant = args.get('variant') || newestVariantKey(dataset);
  const tensorsDir = variant ? path.join(tensorsRoot(dataset), variant) : '';
  if (!variant || !fs.existsSync(path.join(tensorsDir, 'lm_codes.jsonl'))) {
    die(`dataset "${dataset}" has no extracted lm_codes.jsonl — the eval conditions on GT codes`);
  }
  const scales = (args.get('scales') ?? '0.75,1').split(',').map(Number).filter(Number.isFinite);
  const lean = {
    seeds: Math.max(1, Math.trunc(Number(args.get('seeds')) || 1)),
    duration: Math.max(15, Math.trunc(Number(args.get('duration')) || 30)),
    samples: Math.max(4, Math.trunc(Number(args.get('samples')) || 8)),
  };
  const tag = args.get('tag') || 'ditcal';
  const artistDir = path.dirname(newRun);
  const artist = path.basename(artistDir);

  // Old rival: explicit, else newest other non-calibrated run.
  let oldRun = args.get('old-run') ? path.resolve(args.get('old-run') as string) : '';
  if (!args.get('old-run')) {
    try {
      const runs = fs.readdirSync(artistDir, { withFileTypes: true })
        .filter(e => e.isDirectory() && !/-calibrated$/i.test(e.name)
          && hasWeights(path.join(artistDir, e.name))
          && path.resolve(path.join(artistDir, e.name)) !== newRun)
        .map(e => e.name)
        .sort();
      if (runs.length) oldRun = path.join(artistDir, runs[runs.length - 1]);
    } catch { /* no siblings */ }
  }

  const evalRoot = path.join(tensorsDir, 'dit-eval');
  const out = (status: Outcome['status'], servedDir: string, oldS: CandScore | null, newS: CandScore | null,
               winner: string, detail: string): Outcome => ({
    artist, status, servedDir,
    oldScore: oldS ? Number(oldS.frechet.toFixed(3)) : null,
    newScore: newS ? Number(newS.frechet.toFixed(3)) : null,
    winner, detail, minutes: Number(((Date.now() - t0) / 60_000).toFixed(1)),
  });
  const emit = (o: Outcome): void => {
    console.log(`CALIBRATE_RESULT ${JSON.stringify(o)}`);
    if (o.status === 'failed') process.exit(1);
  };

  // Snapshots: old (if any) + new final + distinct milestones.
  const newWeights = weightsFileIn(newRun);
  const finalHash = newWeights ? sha1File(newWeights) : '';
  const snapshots: Array<{ label: string; dir: string }> = [];
  if (oldRun && path.resolve(oldRun) !== path.resolve(newRun)) snapshots.push({ label: 'old', dir: oldRun });
  snapshots.push({ label: 'new', dir: newRun });
  const msRoot = path.join(newRun, 'milestones');
  if (fs.existsSync(msRoot)) {
    for (const e of fs.readdirSync(msRoot, { withFileTypes: true })) {
      if (!e.isDirectory()) continue;
      const w = weightsFileIn(path.join(msRoot, e.name));
      if (w && sha1File(w) !== finalHash) snapshots.push({ label: e.name, dir: path.join(msRoot, e.name) });
    }
  }
  const cands: Candidate[] = [];
  for (const snap of snapshots) {
    for (const s of scales) {
      cands.push({
        label: `${snap.label}@${s}`, adapterDir: snap.dir, scale: s,
        outDir: path.join(evalRoot, `${tag}-${snap.label}-s${String(s).replace('.', '_')}`),
      });
    }
  }

  let baseCache = '';
  const scores: CandScore[] = [];
  for (const cand of cands) {
    console.log(`   eval ${cand.label}`);
    if (!evalCandidate(dataset, variant, cand, baseCache, lean)) continue;
    if (!baseCache) baseCache = path.join(cand.outDir, 'runs.json');
    const s = scoreOf(cand);
    if (s) scores.push(s);
  }
  for (const s of scores) {
    console.log(`      ${s.label.padEnd(14)} frechet ${s.frechet.toFixed(3)} (base ${s.baseFrechet.toFixed(3)}, floor ${s.floor.toFixed(3)}) profile ${s.profile.toFixed(3)}`);
  }
  const oldBest = scores.filter(s => s.label.startsWith('old')).sort((x, y) => x.frechet - y.frechet)[0] ?? null;
  const newBest = scores.filter(s => !s.label.startsWith('old')).sort((x, y) => x.frechet - y.frechet)[0] ?? null;
  if (!oldBest && !newBest) { emit(out('failed', '', null, null, '', 'every candidate failed eval')); return; }

  const bake = (s: CandScore): string => {
    if (s.scale === 1.0) return s.adapterDir;
    const dstDir = path.join(artistDir, `${runStamp()}-calibrated`);
    fs.mkdirSync(dstDir, { recursive: true });
    let scaled = 0;
    for (const f of fs.readdirSync(s.adapterDir)) {
      const src = path.join(s.adapterDir, f);
      if (!fs.statSync(src).isFile()) continue;
      if (f.endsWith('.safetensors')) scaled += bakeFile(src, path.join(dstDir, f), s.scale).scaled;
      else fs.copyFileSync(src, path.join(dstDir, f));
    }
    console.log(`   baked ${s.label} (${scaled} tensors x ${s.scale}) -> ${path.basename(dstDir)}`);
    return dstDir;
  };

  if (oldBest) {
    writeSidecar(oldRun, dataset, variant, oldBest,
      { role: 'previous', bestScale: oldBest.scale, note: 'pre-calibration adapter; score is its best candidate' }, lean);
  }
  if (newBest && (!oldBest || newBest.frechet < oldBest.frechet)) {
    const servedDir = bake(newBest);
    writeSidecar(servedDir, dataset, variant, newBest,
      { role: 'served', winner: newBest.label, bakedScale: newBest.scale !== 1.0 ? newBest.scale : undefined }, lean);
    emit(out('repointed', servedDir, oldBest, newBest, newBest.label, 'new adapter wins — repoint'));
    return;
  }
  const old1 = scores.find(s => s.label === 'old@1');
  if (oldBest && oldBest.scale !== 1.0 && old1 && oldBest.frechet < old1.frechet) {
    const servedDir = bake(oldBest);
    writeSidecar(servedDir, dataset, variant, oldBest,
      { role: 'served', winner: oldBest.label, bakedScale: oldBest.scale }, lean);
    emit(out('baked-old', servedDir, oldBest, newBest, oldBest.label, 'old adapter at rescaled optimum'));
    return;
  }
  emit(out('kept-old', oldRun, oldBest, newBest, oldBest?.label ?? '', 'no new candidate beat the old adapter'));
}

// ── repoint (album presets, DiT column) ───────────────────────────────────

function cmdRepoint(args: Map<string, string>): void {
  const apply = args.get('apply') === 'true';
  const root = config.aceServer.adapters;
  const served = new Map<string, string>();
  for (const sub of fs.readdirSync(root, { withFileTypes: true })) {
    if (!sub.isDirectory() || !/^dit-/i.test(sub.name)) continue;
    for (const artist of fs.readdirSync(path.join(root, sub.name), { withFileTypes: true })) {
      if (!artist.isDirectory()) continue;
      const adir = path.join(root, sub.name, artist.name);
      let best: { dir: string; at: string } | null = null;
      for (const run of fs.readdirSync(adir, { withFileTypes: true })) {
        if (!run.isDirectory()) continue;
        const scp = path.join(adir, run.name, SIDECAR);
        if (!fs.existsSync(scp)) continue;
        try {
          const sc = JSON.parse(fs.readFileSync(scp, 'utf-8'));
          if (sc.role !== 'served') continue;
          if (!best || String(sc.evaluatedAt) > best.at) best = { dir: path.join(adir, run.name), at: String(sc.evaluatedAt) };
        } catch { /* torn */ }
      }
      if (best) served.set(artist.name.toLowerCase(), best.dir);
    }
  }
  console.log(`\n${served.size} artist(s) with a served DiT adapter`);

  const Database = require2('better-sqlite3');
  const dbPath = path.join(SERVER_DIR, 'data', 'hotstep.db');
  const db = new Database(dbPath);
  const rows = db.prepare(
    `SELECT ap.lyrics_set_id AS id, ls.album, ap.adapter_path
     FROM album_presets ap LEFT JOIN lyrics_sets ls ON ls.id = ap.lyrics_set_id
     WHERE ap.adapter_path IS NOT NULL AND ap.adapter_path != ''`).all() as
    Array<{ id: number; album: string | null; adapter_path: string }>;

  const updates: Array<{ id: number; album: string; from: string; to: string }> = [];
  for (const r of rows) {
    const m = /[\\/]dit-[^\\/]+[\\/]([^\\/]+)/i.exec(r.adapter_path);
    if (!m) continue;
    const dest = served.get(m[1].toLowerCase());
    if (!dest) continue;
    if (path.resolve(r.adapter_path).toLowerCase() === path.resolve(dest).toLowerCase()) continue;
    updates.push({ id: r.id, album: r.album ?? `(set ${r.id})`, from: r.adapter_path, to: dest });
  }
  console.log(`${rows.length} preset(s) with a DiT adapter, ${updates.length} to repoint (${apply ? 'APPLY' : 'DRY RUN'})\n`);
  for (const u of updates.slice(0, 10)) console.log(`  ${u.album}\n    ${u.from}\n    -> ${u.to}`);
  if (!apply) { console.log('\nDRY RUN — re-run with --apply to write.'); db.close(); return; }

  const stampNow = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
  const bak = dbPath.replace(/\.db$/i, `_backup_ditrollout_${stampNow}.db`);
  db.backup(bak).then(() => {
    console.log(`\nbacked up to ${path.basename(bak)}`);
    const upd = db.prepare('UPDATE album_presets SET adapter_path = ? WHERE lyrics_set_id = ?');
    let n = 0;
    for (const u of updates) { upd.run(u.to, u.id); n++; }
    db.close();
    console.log(`updated ${n} preset(s).`);
  }).catch((e: unknown) => die(`backup failed: ${e instanceof Error ? e.message : String(e)}`));
}

// ── report ────────────────────────────────────────────────────────────────

function cmdReport(): void {
  const root = config.aceServer.adapters;
  let served = 0, previous = 0;
  const lines: string[] = [];
  for (const sub of fs.readdirSync(root, { withFileTypes: true })) {
    if (!sub.isDirectory() || !/^dit-/i.test(sub.name)) continue;
    for (const artist of fs.readdirSync(path.join(root, sub.name), { withFileTypes: true })) {
      if (!artist.isDirectory()) continue;
      const adir = path.join(root, sub.name, artist.name);
      for (const run of fs.readdirSync(adir, { withFileTypes: true })) {
        if (!run.isDirectory()) continue;
        const scp = path.join(adir, run.name, SIDECAR);
        if (!fs.existsSync(scp)) continue;
        try {
          const sc = JSON.parse(fs.readFileSync(scp, 'utf-8'));
          if (sc.role === 'served') { served++; lines.push(`  ${artist.name}: frechet ${sc.score} (base ${sc.baseScore}, floor ${sc.floor}) @ scale ${sc.measuredAtScale}${sc.bakedScale ? ` baked ${sc.bakedScale}` : ''}`); }
          else previous++;
        } catch { /* torn */ }
      }
    }
  }
  console.log(`\nDiT adapters evaluated: ${served} served, ${previous} previous`);
  for (const l of lines.sort()) console.log(l);
}

// ── main ──────────────────────────────────────────────────────────────────

const cmd = process.argv[2];
const args = parseArgs(process.argv.slice(3));
if (cmd === 'calibrate') {
  cmdCalibrate(args).catch(err => die(err instanceof Error ? err.message : String(err)));
} else if (cmd === 'repoint') {
  cmdRepoint(args);
} else if (cmd === 'report') {
  cmdReport();
} else {
  console.log('usage: npx tsx scripts/dit-adapter-rollout.ts <calibrate|repoint|report> [options]');
  console.log('  calibrate --dataset <slug> --new-run <dir> [--old-run <dir>] [--scales 0.75,1] [--samples 8] [--duration 30] [--seeds 1] [--tag ditcal]');
  console.log('  repoint   [--apply]');
  console.log('  report');
  process.exit(cmd ? 1 : 0);
}
