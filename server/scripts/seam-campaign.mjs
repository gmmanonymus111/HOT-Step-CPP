// seam-campaign.mjs — fail-closed post-build YuE2 seam capture.
//
// This is an orchestration wrapper. Each render is delegated to
// seam-baseline.mjs, which owns the per-render GPU lock and backend restore.
// --dry-run is CPU-only and prints the exact command order.

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { compareManifest } from './seam-compare.mjs';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const BASELINE = path.join(REPO, 'server/scripts/seam-baseline.mjs');
const INPUTS = path.join(REPO, 'docs/plans/yue2/baseline-inputs');
const DEFAULT_OUT = 'D:/Ace-Step-Latest/_experiments/yue2/seam-baseline';
const NODE_URL = process.env.SEAM_BASELINE_NODE_URL || 'http://127.0.0.1:3001';
const GPU_LOCK = 'D:/Ace-Step-Latest/_experiments/yue2/GPU.lock';
function parseArgs(argv) {
  const args = { series: 'post-engine', baselineSeries: 'pre-seam', out: DEFAULT_OUT, dryRun: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--dry-run') args.dryRun = true;
    else if (a === '--series') args.series = argv[++i];
    else if (a === '--baseline-series') args.baselineSeries = argv[++i];
    else if (a === '--out') args.out = argv[++i];
    else if (a.startsWith('--series=')) args.series = a.slice(9);
    else if (a.startsWith('--baseline-series=')) args.baselineSeries = a.slice(18);
    else if (a.startsWith('--out=')) args.out = a.slice(6);
    else if (a === '--help' || a === '-h') { args.help = true; }
    else throw new Error(`Unknown argument: ${a}`);
  }
  return args;
}

function usage() {
  return 'Usage: node server/scripts/seam-campaign.mjs [--series post-engine] [--baseline-series pre-seam] [--out PATH] [--dry-run]';
}
function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(k => [k, stable(value[k])]));
  return value;
}
function equal(left, right) { return JSON.stringify(stable(left)) === JSON.stringify(stable(right)); }
function sha256(file) {
  return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}
function readJson(file) { return JSON.parse(fs.readFileSync(file, 'utf8')); }
function contextSnapshot(context) {
  if (!Array.isArray(context.inputFiles) || context.inputFiles.length === 0) throw new Error('campaign-context.json inputFiles must be a non-empty array');
  if (!context.vstChain || typeof context.vstChain !== 'object' || !Array.isArray(context.vstChain.plugins)) throw new Error('campaign-context.json vstChain must be an object with plugins');
  const files = (context.inputFiles || []).map(item => {
    if (!item || typeof item.path !== 'string' || !item.path || typeof item.sha256 !== 'string' || !/^[a-f0-9]{64}$/i.test(item.sha256)) throw new Error('campaign-context.json has an invalid pinned input hash');
    if (!fs.existsSync(item.path)) throw new Error(`Pinned input is missing: ${item.path}`);
    const actual = sha256(item.path);
    if (actual !== item.sha256) throw new Error(`Pinned input hash changed: ${item.path}`);
    return { path: item.path, sha256: actual };
  });
  return fetch(`${NODE_URL}/api/vst/chain`, { signal: AbortSignal.timeout(15_000) })
    .then(async response => {
      if (!response.ok) throw new Error(`GET /api/vst/chain ${response.status}`);
      const chain = await response.json();
      if (!equal(chain, context.vstChain)) throw new Error('VST chain differs from campaign-context.json');
      return { files, chain };
    });
}

function withoutRuntimeBackend(environment) {
  const copy = structuredClone(environment);
  if (copy.ditRuntime && typeof copy.ditRuntime === 'object') delete copy.ditRuntime.backend;
  return copy;
}

function outputDir(out, series, backend, variant, run) {
  return path.join(out, series, backend, variant, `run${run}`);
}

function commandFor({ manifest, series, backend, variant, run, params }) {
  const args = [BASELINE, '--series', series, '--backend', backend, '--variant', variant, '--run', String(run), '--manifest', manifest];
  if (params) args.push('--params', params);
  return { args, text: [process.execPath, ...args].map(a => JSON.stringify(a)).join(' ') };
}

export function verifyRecord(dir, backend, expectedRuntime) {
  const recordPath = path.join(dir, 'record.json');
  const statusPath = path.join(dir, 'status.json');
  if (!fs.existsSync(recordPath) || !fs.existsSync(statusPath)) throw new Error(`Capture did not retain record/status: ${dir}`);
  const record = readJson(recordPath);
  const status = readJson(statusPath);
  if (status.status !== 'succeeded') throw new Error(`Capture status is ${status.status}: ${dir}`);
  if (record.backend !== backend || record.submittedParams?.backend !== backend) throw new Error(`Capture backend mismatch: ${dir}`);
  if (expectedRuntime && record.environmentAfter?.ditRuntime?.backend !== expectedRuntime) {
    throw new Error(`Capture runtime mismatch (wanted ${expectedRuntime}): ${dir}`);
  }
  return { ...record, status: status.status };
}

export function validateWarmupTransition({ runnerStatus, stderr = '', lockPresent, record, expectedRuntime }) {
  if (lockPresent) throw new Error('Warmup still has a GPU owner; expected runtime transition is not complete');
  if (!record || record.status !== 'succeeded') throw new Error('Warmup record status is not succeeded');
  if (record.backend !== 'minimax-m3' || record.submittedParams?.backend !== 'minimax-m3') throw new Error('Warmup backend was not minimax-m3');
  if (expectedRuntime && record.environmentAfter?.ditRuntime?.backend !== expectedRuntime) throw new Error(`Warmup runtime is not ${expectedRuntime}`);
  const runtimeChanged = record.environment?.ditRuntime?.backend !== record.environmentAfter?.ditRuntime?.backend;
  if (runnerStatus !== 0 && !(runnerStatus === 1
    && /Environment changed during the run; artifacts retained, no accepted manifest entry/.test(stderr)
    && !lockPresent
    && runtimeChanged)) throw new Error('Warmup runner failed without the expected runtime transition');
  if (!equal(withoutRuntimeBackend(record.environment), withoutRuntimeBackend(record.environmentAfter))) {
    throw new Error('Warmup environment changed beyond ditRuntime.backend');
  }
  return true;
}

export function validateContextUnchanged(before, after) {
  if (!equal(before?.files, after?.files) || !equal(before?.chain, after?.chain)) throw new Error('Pinned campaign context changed during capture');
  return true;
}

function runCapture(spec, { dryRun = false, warmup = false } = {}) {
  const dir = outputDir(spec.out, spec.series, spec.backend, spec.variant, spec.run);
  const command = commandFor(spec);
  if (dryRun) { console.log(command.text); return; }
  if (fs.existsSync(dir)) throw new Error(`Refusing existing capture directory: ${dir}`);
  const result = spawnSync(process.execPath, command.args, {
    cwd: REPO, stdio: ['inherit', 'pipe', 'pipe'], encoding: 'utf8', windowsHide: true,
    env: { ...process.env, SEAM_BASELINE_OUT: spec.out },
  });
  if (result.error) throw result.error;
  if (result.stdout) process.stdout.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);
  if (result.status !== 0 && !warmup) throw new Error(`Baseline runner failed (${result.status}) for ${spec.series}/${spec.backend}/${spec.variant}/run${spec.run}`);
  const record = verifyRecord(dir, spec.backend, spec.expectedRuntime);
  return { dir, status: result.status, stderr: result.stderr || '', record };
}

function expectedPath(out, series) { return path.join(out, `${series}-expected-environment.json`); }

function plan(args) {
  const ggmlSeries = `${args.series}-ggml`;
  return {
    warmTrt: { out: args.out, manifest: path.join(args.out, `manifest-v2-warmups-${args.series}.json`), series: `${args.series}-warm-trt`, backend: 'minimax-m3', variant: 'base', run: 1, expectedRuntime: 'tensorrt' },
    aceAndMm3: [
      { series: args.series, backend: 'ace', variant: 'base' },
      { series: args.series, backend: 'ace', variant: 'lmcache', params: path.join(INPUTS, 'lmcache.json') },
      { series: args.series, backend: 'ace', variant: 'adapter', params: path.join(INPUTS, 'adapter.json') },
      { series: args.series, backend: 'ace', variant: 'cover', params: path.join(INPUTS, 'cover.json') },
      { series: args.series, backend: 'ace', variant: 'pp', params: path.join(INPUTS, 'pp.json') },
      { series: args.series, backend: 'minimax-m3', variant: 'base' },
    ],
    warmGgml: { out: args.out, manifest: path.join(args.out, `manifest-v2-warmups-${args.series}.json`), series: `${args.series}-warm-ggml`, backend: 'minimax-m3', variant: 'ggml', run: 1, params: path.join(INPUTS, 'ggml.json'), expectedRuntime: 'ggml' },
    ggml: { series: ggmlSeries, backend: 'minimax-m3', variant: 'ggml', params: path.join(INPUTS, 'ggml.json') },
  };
}

function printDryRun(args) {
  const p = plan(args);
  runCapture(p.warmTrt, { dryRun: true });
  for (const capture of p.aceAndMm3) for (const run of [1, 2]) {
    runCapture({ ...capture, out: args.out, manifest: path.join(args.out, 'manifest-v2.json'), run }, { dryRun: true });
  }
  runCapture(p.warmGgml, { dryRun: true });
  for (const run of [1, 2]) runCapture({ ...p.ggml, out: args.out, manifest: path.join(args.out, 'manifest-v2.json'), run }, { dryRun: true });
  console.log(`compare ${args.baselineSeries} -> ${args.series}`);
  console.log(`compare ${args.baselineSeries}-ggml -> ${args.series}-ggml`);
}
function runWarmup(spec) {
  const dir = outputDir(spec.out, spec.series, spec.backend, spec.variant, spec.run);
  const capture = runCapture(spec, { warmup: true });
  const record = capture.record;
  validateWarmupTransition({ runnerStatus: capture.status, stderr: capture.stderr, lockPresent: fs.existsSync(GPU_LOCK), record, expectedRuntime: spec.expectedRuntime });
}

function preflight(args, p, manifest) {
  if (!fs.existsSync(manifest)) throw new Error(`Missing baseline manifest: ${manifest}`);
  const existing = readJson(manifest);
  if (existing.version !== 2 || !existing.series?.[args.baselineSeries] || !existing.series?.[`${args.baselineSeries}-ggml`]) {
    throw new Error(`Baseline manifest must contain ${args.baselineSeries} and ${args.baselineSeries}-ggml`);
  }
  for (const name of [args.series, p.ggml.series]) {
    if (existing.series?.[name]) throw new Error(`Candidate series already exists: ${name}`);
    if (fs.existsSync(expectedPath(args.out, name))) throw new Error(`Refusing existing expected environment: ${expectedPath(args.out, name)}`);
  }
  const warmManifest = p.warmTrt.manifest;
  if (fs.existsSync(warmManifest)) throw new Error(`Refusing existing warmup manifest: ${warmManifest}`);
  const specs = [p.warmTrt, p.warmGgml, ...p.aceAndMm3, p.ggml];
  for (const spec of specs) {
    const runs = spec.run ? [spec.run] : [1, 2];
    for (const run of runs) if (fs.existsSync(outputDir(args.out, spec.series, spec.backend, spec.variant, run))) {
      throw new Error(`Refusing existing capture directory: ${outputDir(args.out, spec.series, spec.backend, spec.variant, run)}`);
    }
  }
}

async function campaign(args) {
  if (!/^[A-Za-z0-9_-]+$/.test(args.series) || !/^[A-Za-z0-9_-]+$/.test(args.baselineSeries)) throw new Error('Invalid series name');
  if (!fs.existsSync(args.out)) throw new Error(`Output root is missing: ${args.out}`);
  const contextPath = path.join(args.out, 'campaign-context.json');
  if (!fs.existsSync(contextPath)) throw new Error(`Missing campaign context: ${contextPath}`);
  const context = readJson(contextPath);
  const p = plan(args);
  const manifest = path.join(args.out, 'manifest-v2.json');
  preflight(args, p, manifest);
  const before = await contextSnapshot(context);
  runWarmup(p.warmTrt);
  for (const capture of p.aceAndMm3) for (const run of [1, 2]) {
    runCapture({ ...capture, out: args.out, manifest, run });
  }
  runWarmup(p.warmGgml);
  for (const run of [1, 2]) runCapture({ ...p.ggml, out: args.out, manifest, run });
  const ggmlSeries = p.ggml.series;

  const after = await contextSnapshot(context);
  validateContextUnchanged(before, after);
  const candidate = readJson(manifest);
  const expected = [args.series, ggmlSeries].map(name => {
    const environment = candidate.series?.[name]?.environment;
    if (!environment) throw new Error(`Candidate series has no environment: ${name}`);
    const file = expectedPath(args.out, name);
    if (fs.existsSync(file)) throw new Error(`Refusing existing expected environment: ${file}`);
    fs.writeFileSync(file, JSON.stringify(environment, null, 2), { flag: 'wx' });
    return { name, environment, file };
  });
  for (const item of expected) {
    const baselineSeries = item.name === ggmlSeries ? `${args.baselineSeries}-ggml` : args.baselineSeries;
    const report = compareManifest(candidate, {
      baselineSeries,
      series: item.name,
      expectedEnvironment: item.environment,
    });
    console.log(JSON.stringify(report, null, 2));
    if (!report.ok) throw new Error(`Seam comparison failed: ${baselineSeries} vs ${item.name}`);
  }
  console.log(`Campaign context preserved (${context.coverage || 'context coverage not labelled'}).`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) { console.log(usage()); return; }
  if (args.dryRun) {
    // Keep dry-run entirely offline: no context reads, fetches, subprocesses,
    // lock acquisition, or directory creation.
    printDryRun(args);
    return;
  }
  await campaign(args);
}

if (fileURLToPath(import.meta.url) === path.resolve(process.argv[1] || '')) {
  main().catch(error => { console.error(`FATAL: ${error?.stack || error}`); process.exitCode = 1; });
}
