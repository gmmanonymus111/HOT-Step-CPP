// seam-compare.mjs — fail-closed comparison for pinned seam renders.
//
// This script intentionally has no HOT-Step runtime dependencies.  It compares
// the JSON manifest produced by seam-baseline.mjs (or its version 2 successor)
// so it can also be used in a clean release checkout.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const DEFAULT_MANIFEST = 'D:/Ace-Step-Latest/_experiments/yue2/seam-baseline/manifest-v2.json';

// These are transformations visible in the currently captured MM3 manifests.
// They are reported as notes, rather than being removed from the comparison.
// The stored object is still compared exactly between runs/series.
const KNOWN_STORED_NOTES = new Set([
  'duration',
  'seed',
  'mm3Request',
  'mm3',
]);

const hasOwn = (value, key) => Object.prototype.hasOwnProperty.call(value, key);

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function sortValue(value) {
  if (Array.isArray(value)) return value.map(sortValue);
  if (isObject(value)) {
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, sortValue(value[key])]));
  }
  return value;
}

function stableStringify(value) {
  return JSON.stringify(sortValue(value));
}

function deepEqual(a, b) {
  return stableStringify(a) === stableStringify(b);
}

function diffValues(expected, actual, at = '') {
  if (deepEqual(expected, actual)) return [];
  if (Array.isArray(expected) || Array.isArray(actual)) {
    if (!Array.isArray(expected) || !Array.isArray(actual)) {
      return [{ path: at || '$', expected, actual }];
    }
    const diffs = [];
    const length = Math.max(expected.length, actual.length);
    for (let i = 0; i < length; i++) {
      diffs.push(...diffValues(expected[i], actual[i], `${at}[${i}]`));
    }
    return diffs;
  }
  if (isObject(expected) || isObject(actual)) {
    if (!isObject(expected) || !isObject(actual)) {
      return [{ path: at || '$', expected, actual }];
    }
    const diffs = [];
    const keys = [...new Set([...Object.keys(expected), ...Object.keys(actual)])].sort();
    for (const key of keys) {
      const childPath = at ? `${at}.${key}` : key;
      if (!hasOwn(expected, key) || !hasOwn(actual, key)) {
        diffs.push({ path: childPath, expected: expected[key], actual: actual[key] });
      } else {
        diffs.push(...diffValues(expected[key], actual[key], childPath));
      }
    }
    return diffs;
  }
  return [{ path: at || '$', expected, actual }];
}

function parseArgs(argv) {
  const args = {
    manifest: DEFAULT_MANIFEST,
    baselineSeries: undefined,
    series: undefined,
    expectedEnvironment: undefined,
    allowLegacy: false,
    help: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const take = name => {
      const equals = `${name}=`;
      if (arg.startsWith(equals)) return arg.slice(equals.length);
      if (arg === name) {
        if (i + 1 >= argv.length) throw new Error(`${name} requires a value`);
        return argv[++i];
      }
      return undefined;
    };
    const manifest = take('--manifest');
    if (manifest !== undefined) { args.manifest = manifest; continue; }
    const baseline = take('--baseline-series');
    if (baseline !== undefined) { args.baselineSeries = baseline; continue; }
    const series = take('--series');
    if (series !== undefined) { args.series = series; continue; }
    const expected = take('--expected-environment');
    if (expected !== undefined) { args.expectedEnvironment = expected; continue; }
    if (arg === '--allow-legacy') { args.allowLegacy = true; continue; }
    if (arg === '--help' || arg === '-h') { args.help = true; continue; }
    throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function attemptCount(value) {
  return Array.isArray(value) ? value.length : NaN;
}

function makeError(code, message, details = {}) {
  return { code, message, ...details };
}

function addStoredNotes(report, identity, entry) {
  if (!isObject(entry.submittedParams) || !isObject(entry.storedGenerationParams)) return;
  const diffs = diffValues(entry.submittedParams, entry.storedGenerationParams);
  if (diffs.length === 0) return;
  report.storedParamNotes.push({
    ...identity,
    fields: diffs.map(diff => ({
      ...diff,
      classification: KNOWN_STORED_NOTES.has(diff.path.split('.')[0]) ? 'known-derived-or-volatile' : 'unclassified',
    })),
  });
}

function validateEntry(report, entry, identity, schema, seriesEnvironment) {
  if (!isObject(entry)) {
    report.errors.push(makeError('invalid-entry', 'Entry must be an object', identity));
    return false;
  }

  // The field was added after the legacy manifest was written, but if an old
  // or hand-edited entry has it, it must still describe one render attempt.
  if (hasOwn(entry, 'attempts') && attemptCount(entry.attempts) !== 1) {
    report.errors.push(makeError('invalid-attempt-count', 'attempts must represent exactly one attempt', {
      ...identity, attempts: entry.attempts,
    }));
  }

  if (schema === 'v2') {
    for (const field of ['sha256', 'rawSha256', 'masteredSha256', 'submittedParams', 'storedGenerationParams', 'environment']) {
      if (!hasOwn(entry, field)) {
        report.errors.push(makeError('missing-field', `v2 entry is missing ${field}`, { ...identity, field }));
      }
    }
    for (const field of ['sha256', 'rawSha256']) {
      if (hasOwn(entry, field) && (typeof entry[field] !== 'string' || entry[field].length === 0)) {
        report.errors.push(makeError('invalid-field', `v2 ${field} must be a non-empty string`, { ...identity, field }));
      }
    }
    if (hasOwn(entry, 'masteredSha256') && entry.masteredSha256 !== null &&
        (typeof entry.masteredSha256 !== 'string' || entry.masteredSha256.length === 0)) {
      report.errors.push(makeError('invalid-field', 'v2 masteredSha256 must be null or a non-empty string', { ...identity, field: 'masteredSha256' }));
    }
    for (const field of ['submittedParams', 'storedGenerationParams', 'environment']) {
      if (hasOwn(entry, field) && !isObject(entry[field])) {
        report.errors.push(makeError('invalid-field', `v2 ${field} must be an object`, { ...identity, field }));
      }
    }
    if (hasOwn(entry, 'environment') && isObject(seriesEnvironment) && !deepEqual(entry.environment, seriesEnvironment)) {
      report.blocked.push(makeError('environment-conflict', 'Entry environment differs from its series environment', identity));
    }
    if (hasOwn(entry, 'environmentAfter') &&
        (!isObject(entry.environmentAfter) || !isObject(entry.environment) || !deepEqual(entry.environmentAfter, entry.environment))) {
      report.blocked.push(makeError('environment-changed-during-run', 'environmentAfter differs from the captured entry environment', identity));
    }
    addStoredNotes(report, identity, entry);
    return report.errors.length === 0;
  }

  for (const field of ['sha256', 'submittedParams', 'storedGenerationParams']) {
    if (!hasOwn(entry, field)) {
      report.errors.push(makeError('missing-field', `Legacy entry is missing ${field}`, { ...identity, field }));
    }
  }
  if (hasOwn(entry, 'sha256') && (typeof entry.sha256 !== 'string' || entry.sha256.length === 0)) {
    report.errors.push(makeError('invalid-field', 'Legacy sha256 must be a non-empty string', { ...identity, field: 'sha256' }));
  }
  if (hasOwn(entry, 'submittedParams') && !isObject(entry.submittedParams)) {
    report.errors.push(makeError('invalid-field', 'Legacy submittedParams must be an object', { ...identity, field: 'submittedParams' }));
  }
  if (hasOwn(entry, 'storedGenerationParams') && !isObject(entry.storedGenerationParams)) {
    report.errors.push(makeError('invalid-field', 'Legacy storedGenerationParams must be an object', { ...identity, field: 'storedGenerationParams' }));
  }
  addStoredNotes(report, identity, entry);
  return report.errors.length === 0;
}

function collectLegacy(manifest) {
  const result = new Map();
  for (const [backend, runs] of Object.entries(manifest || {})) {
    if (!isObject(runs)) continue;
    result.set(`${backend}\u0000default`, { backend, variant: 'default', runs });
  }
  return result;
}

function collectV2Series(manifest, seriesName, report) {
  const series = manifest.series?.[seriesName];
  if (!isObject(series)) {
    report.errors.push(makeError('missing-series', `Manifest has no v2 series named ${seriesName}`, { series: seriesName }));
    return new Map();
  }
  if (!isObject(series.backends)) {
    report.errors.push(makeError('invalid-series', `Series ${seriesName} has no backends object`, { series: seriesName }));
    return new Map();
  }
  const result = new Map();
  for (const [backend, backendValue] of Object.entries(series.backends)) {
    if (!isObject(backendValue)) {
      report.errors.push(makeError('invalid-backend', 'Backend value must be an object', { series: seriesName, backend }));
      continue;
    }
    const directRuns = Object.keys(backendValue).some(key => /^run\d+$/.test(key));
    const variants = directRuns ? { default: backendValue } : backendValue;
    for (const [variant, runs] of Object.entries(variants)) {
      const identity = { series: seriesName, backend, variant };
      if (!isObject(runs)) {
        report.errors.push(makeError('invalid-variant', 'Variant value must be an object', identity));
        continue;
      }
      const key = `${backend}\u0000${variant}`;
      result.set(key, { ...identity, runs, seriesEnvironment: series.environment });
      for (const [run, entry] of Object.entries(runs)) {
        if (!/^run\d+$/.test(run)) continue;
        validateEntry(report, entry, { ...identity, run }, 'v2', series.environment);
      }
    }
  }
  if (result.size === 0) {
    report.errors.push(makeError('empty-series', `v2 series ${seriesName} has no backend variants`, { series: seriesName }));
  }
  if (!hasOwn(series, 'environment') || !isObject(series.environment)) {
    report.errors.push(makeError('missing-field', `v2 series ${seriesName} is missing environment`, { series: seriesName, field: 'environment' }));
  }
  return result;
}

function compareEntry(report, left, right, identity, label, schema) {
  const mismatches = [];
  const fields = schema === 'v2'
    ? ['sha256', 'rawSha256', 'masteredSha256', 'submittedParams', 'storedGenerationParams']
    : ['sha256', 'submittedParams', 'storedGenerationParams'];
  for (const field of fields) {
    const diffs = diffValues(left?.[field], right?.[field]);
    if (diffs.length) mismatches.push({ field, differences: diffs });
  }
  const comparison = { ...identity, label, outcome: mismatches.length ? 'FAIL' : 'PASS', mismatches };
  report.comparisons.push(comparison);
  return mismatches.length === 0;
}

const ALLOWED_IDENTITY_PATHS = new Set([
  'engine.pid',
  'engine.created',
  'engine.binarySha256',
  'engine.version',
]);

function isAllowedIdentityDifference(diff) {
  return ALLOWED_IDENTITY_PATHS.has(diff.path);
}

function compareEnvironments(report, left, right, identity, label, expectedEnvironment) {
  const leftEnv = left?.environment;
  const rightEnv = right?.environment;
  if (!isObject(leftEnv) || !isObject(rightEnv)) return false;
  if (expectedEnvironment !== undefined && !deepEqual(rightEnv, expectedEnvironment)) {
    report.blocked.push(makeError('unexpected-environment', 'Candidate environment does not exactly match --expected-environment', {
      ...identity, label, expected: expectedEnvironment, actual: rightEnv,
    }));
    return false;
  }
  const differences = diffValues(leftEnv, rightEnv);
  if (differences.length === 0) return true;
  const stableDifferences = differences.filter(diff => !isAllowedIdentityDifference(diff));
  if (stableDifferences.length > 0) {
    report.blocked.push(makeError('unexpected-environment', 'Stable environment differs from baseline', {
      ...identity, label, differences: stableDifferences, baseline: leftEnv, candidate: rightEnv,
    }));
    return false;
  }
  if (expectedEnvironment !== undefined) {
    report.environmentNotes.push({
      ...identity, label, outcome: 'ALLOWED_IDENTITY_DIFFERENCE', differences,
      baseline: leftEnv, candidate: rightEnv,
    });
    return true;
  }
  report.blocked.push(makeError('unexpected-environment', 'Engine identity differs without an expected environment', {
    ...identity, label, differences, baseline: leftEnv, candidate: rightEnv,
  }));
  return false;
}

function compareRuns(report, left, right, identity, label, schema, expectedEnvironment) {
  if (!left || !right) return false;
  const paramsPass = compareEntry(report, left, right, identity, label, schema);
  const envPass = schema === 'v2'
    ? compareEnvironments(report, left, right, identity, label, expectedEnvironment)
    : true;
  return paramsPass && envPass;
}

/**
 * Compare a parsed seam manifest.
 *
 * `expectedEnvironment`, when supplied, must be the parsed JSON object.  The
 * CLI's --expected-environment option accepts a path to that JSON file.
 */
export function compareManifest(manifest, options = {}) {
  const report = {
    ok: false,
    pass: false,
    passed: false,
    checksPass: false,
    status: 'FAIL',
    schema: manifest?.version === 2 || hasOwn(manifest || {}, 'series') ? 'v2' : 'legacy',
    limited: false,
    baselineSeries: undefined,
    series: undefined,
    coverage: { checked: [], unchecked: [] },
    comparisons: [],
    errors: [],
    blocked: [],
    environmentNotes: [],
    storedParamNotes: [],
  };
  const schema = report.schema;
  let expectedEnvironment = options.expectedEnvironment;
  if (expectedEnvironment === undefined && options.expectedEnvironmentPath !== undefined) {
    expectedEnvironment = options.expectedEnvironmentPath;
  }
  if (typeof expectedEnvironment === 'string') {
    try {
      expectedEnvironment = JSON.parse(fs.readFileSync(expectedEnvironment, 'utf8'));
    } catch (error) {
      report.errors.push(makeError('invalid-expected-environment', `Could not read --expected-environment: ${error?.message || error}`));
      expectedEnvironment = undefined;
    }
  }

  if (!isObject(manifest)) {
    report.errors.push(makeError('invalid-manifest', 'Manifest root must be an object'));
    return report;
  }

  if (schema === 'legacy') {
    report.limited = true;
    report.coverage = {
      checked: ['sha256', 'submittedParams', 'storedGenerationParams'],
      unchecked: ['rawSha256', 'masteredSha256', 'environment'],
    };
    const groups = collectLegacy(manifest);
    for (const group of groups.values()) {
      const runs = group.runs;
      const identity = { backend: group.backend, variant: group.variant };
      for (const [run, entry] of Object.entries(runs)) {
        if (/^run\d+$/.test(run)) validateEntry(report, entry, { ...identity, run }, 'legacy');
      }
      if (!hasOwn(runs, 'run1')) {
        report.errors.push(makeError('missing-entry', 'Legacy backend is missing run1', { ...identity, run: 'run1' }));
      }
      const runNames = Object.keys(runs)
        .filter(run => /^run\d+$/.test(run))
        .sort((a, b) => Number(a.slice(3)) - Number(b.slice(3)));
      for (const run of runNames) {
        if (run !== 'run1') compareRuns(report, runs.run1, runs[run], identity, `run1-vs-${run}`, 'legacy');
      }
    }
    report.checksPass = report.errors.length === 0 && report.blocked.length === 0 && report.comparisons.every(c => c.outcome === 'PASS');
    report.status = 'LIMITED';
    report.ok = report.checksPass && Boolean(options.allowLegacy);
    report.pass = report.ok;
    report.passed = report.ok;
    return report;
  }

  if (manifest.version !== 2 || !isObject(manifest.series)) {
    report.errors.push(makeError('invalid-version', 'v2 manifest must have version: 2 and a series object'));
    return report;
  }
  const names = Object.keys(manifest.series);
  if (!names.length) {
    report.errors.push(makeError('missing-series', 'v2 manifest has no series'));
    return report;
  }
  const baselineName = options.baselineSeries || (names.includes('baseline') ? 'baseline' : names[0]);
  const candidateName = options.series || (names.includes('candidate') ? 'candidate' : baselineName);
  report.baselineSeries = baselineName;
  report.series = candidateName;
  if (!hasOwn(manifest.series, baselineName)) {
    report.errors.push(makeError('missing-series', `Manifest has no v2 series named ${baselineName}`, { series: baselineName }));
  }
  if (!hasOwn(manifest.series, candidateName)) {
    report.errors.push(makeError('missing-series', `Manifest has no v2 series named ${candidateName}`, { series: candidateName }));
  }
  // Validate every recorded series before comparing the selected pair.  A
  // malformed unselected series is still a malformed v2 manifest and must not
  // quietly become a future baseline.
  const allSeries = new Map();
  for (const name of names) allSeries.set(name, collectV2Series(manifest, name, report));
  const baseline = allSeries.get(baselineName) || new Map();
  const candidate = allSeries.get(candidateName) || new Map();
  const baselineSeriesObject = manifest.series[baselineName];
  const candidateSeriesObject = manifest.series[candidateName];
  if (isObject(baselineSeriesObject?.environment) && isObject(candidateSeriesObject?.environment)) {
    const seriesIdentity = { backend: '*', variant: '*', label: 'baseline-series-vs-candidate-series' };
    const baselineEnvironment = { environment: baselineSeriesObject.environment };
    const candidateEnvironment = { environment: candidateSeriesObject.environment };
    compareEnvironments(report, baselineEnvironment, candidateEnvironment, seriesIdentity, seriesIdentity.label, expectedEnvironment);
  }
  const keys = [...new Set([...baseline.keys(), ...candidate.keys()])].sort();
  for (const key of keys) {
    const leftGroup = baseline.get(key);
    const rightGroup = candidate.get(key);
    const identity = {
      backend: leftGroup?.backend || rightGroup?.backend,
      variant: leftGroup?.variant || rightGroup?.variant,
    };
    if (!leftGroup) {
      report.errors.push(makeError('missing-entry', 'Candidate backend/variant is absent from baseline series', { ...identity, baselineSeries: baselineName }));
      continue;
    }
    if (!rightGroup) {
      report.errors.push(makeError('missing-entry', 'Baseline backend/variant is absent from candidate series', { ...identity, series: candidateName }));
      continue;
    }
    if (!hasOwn(leftGroup.runs, 'run1')) report.errors.push(makeError('missing-entry', 'Baseline backend/variant is missing run1', { ...identity, series: baselineName, run: 'run1' }));
    if (!hasOwn(rightGroup.runs, 'run1')) report.errors.push(makeError('missing-entry', 'Candidate backend/variant is missing run1', { ...identity, series: candidateName, run: 'run1' }));
    if (!hasOwn(leftGroup.runs, 'run2')) report.errors.push(makeError('missing-entry', 'Baseline backend/variant is missing run2 for repeatability comparison', { ...identity, series: baselineName, run: 'run2' }));
    if (!hasOwn(rightGroup.runs, 'run2')) report.errors.push(makeError('missing-entry', 'Candidate backend/variant is missing run2 for repeatability comparison', { ...identity, series: candidateName, run: 'run2' }));
    if (leftGroup.runs.run1 && rightGroup.runs.run1) {
      compareRuns(report, leftGroup.runs.run1, rightGroup.runs.run1, identity, 'baseline-run1-vs-candidate-run1', 'v2', expectedEnvironment);
    }
    const baselineRunNames = Object.keys(leftGroup.runs)
      .filter(run => /^run\d+$/.test(run) && run !== 'run1')
      .sort((a, b) => Number(a.slice(3)) - Number(b.slice(3)));
    for (const run of baselineRunNames) {
      compareRuns(report, leftGroup.runs.run1, leftGroup.runs[run], identity, `baseline-run1-vs-baseline-${run}`, 'v2');
    }
    const candidateRunNames = Object.keys(rightGroup.runs)
      .filter(run => /^run\d+$/.test(run) && run !== 'run1')
      .sort((a, b) => Number(a.slice(3)) - Number(b.slice(3)));
    for (const run of candidateRunNames) {
      // For a candidate-only repeatability check, expectedEnvironment applies
      // to both candidate runs and the first run is the candidate side.
      compareRuns(report, rightGroup.runs.run1, rightGroup.runs[run], identity, `candidate-run1-vs-candidate-${run}`, 'v2', expectedEnvironment);
    }
  }
  report.checksPass = report.errors.length === 0 && report.blocked.length === 0 && report.comparisons.every(c => c.outcome === 'PASS');
  report.status = report.blocked.length ? 'BLOCKED' : report.checksPass ? 'PASS' : 'FAIL';
  report.ok = report.checksPass;
  report.pass = report.ok;
  report.passed = report.ok;
  return report;
}

function usage() {
  return [
    'Usage: node server/scripts/seam-compare.mjs [options]',
    '',
    `  --manifest PATH                 Manifest (default: ${DEFAULT_MANIFEST})`,
    '  --baseline-series NAME         v2 baseline series (default: baseline or first)',
    '  --series NAME                  v2 candidate series (default: candidate or baseline)',
    '  --expected-environment PATH    Exact candidate environment; only engine identity fields may differ',
    '  --allow-legacy                 Permit a legacy LIMITED report to exit 0',
  ].join('\n');
}

function isMain() {
  if (!process.argv[1]) return false;
  try {
    return fileURLToPath(import.meta.url) === path.resolve(process.argv[1]);
  } catch {
    return import.meta.url === pathToFileURL(process.argv[1]).href;
  }
}

function main() {
  let args;
  try {
    args = parseArgs(process.argv.slice(2));
    if (args.help) { console.log(usage()); return 0; }
    const manifest = JSON.parse(fs.readFileSync(args.manifest, 'utf8'));
    let expectedEnvironment;
    if (args.expectedEnvironment) expectedEnvironment = JSON.parse(fs.readFileSync(args.expectedEnvironment, 'utf8'));
    const report = compareManifest(manifest, {
      baselineSeries: args.baselineSeries,
      series: args.series,
      expectedEnvironment,
      allowLegacy: args.allowLegacy,
    });
    console.log(JSON.stringify(report, null, 2));
    return report.ok ? 0 : 1;
  } catch (error) {
    console.error(`FATAL: ${error?.message || error}`);
    return 1;
  }
}

if (isMain()) process.exitCode = main();
