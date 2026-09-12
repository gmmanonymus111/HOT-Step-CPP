import assert from 'node:assert/strict';
import test from 'node:test';
import { compareManifest } from './seam-compare.mjs';

const environment = {
  engine: {
    pid: 100,
    created: 1234,
    exe: 'D:/hot-step/ace-server.exe',
    environment: { MM3_DEPTH_FUSED: '1' },
    binarySha256: 'engine-hash-a',
    version: '1.0.0',
  },
  ditRuntime: { backend: 'tensorrt', available: true },
  aceDefaults: { steps: 8 },
  modelFiles: [{ path: 'models/ace.gguf', bytes: 10, mtimeMs: 1 }],
  selection: { dit: { requested: 'default', selected: 'ace.gguf' } },
  trtFiles: [],
};

function entry(overrides = {}) {
  return {
    sha256: 'audio-hash',
    rawSha256: 'raw-hash',
    masteredSha256: null,
    submittedParams: { backend: 'ace', seed: 4242, duration: 30 },
    storedGenerationParams: { backend: 'ace', seed: 4242, duration: 30 },
    environment: structuredClone(environment),
    attempts: [{ attempt: 1, endReason: 'completed' }],
    ...overrides,
  };
}

function v2Manifest(overrides = {}) {
  const baselineEntry = entry();
  const candidateEntry = entry();
  return {
    version: 2,
    series: {
      baseline: {
        environment: structuredClone(environment),
        backends: { ace: { default: { run1: baselineEntry, run2: entry() } } },
      },
      candidate: {
        environment: structuredClone(environment),
        backends: { ace: { default: { run1: candidateEntry, run2: entry() } } },
      },
    },
    ...overrides,
  };
}

function candidateRun1(manifest) {
  return manifest.series.candidate.backends.ace.default.run1;
}

test('passes a complete v2 manifest with exact repeat and cross-series matches', () => {
  const report = compareManifest(v2Manifest());
  assert.equal(report.status, 'PASS');
  assert.equal(report.ok, true);
  assert.equal(report.comparisons.length, 3);
});

test('fails on an audio hash mismatch', () => {
  const manifest = v2Manifest();
  candidateRun1(manifest).sha256 = 'different-audio';
  const report = compareManifest(manifest);
  assert.equal(report.status, 'FAIL');
  assert.equal(report.ok, false);
  assert.ok(report.comparisons.some(c => c.mismatches.some(m => m.field === 'sha256')));
});

test('fails closed when a v2 entry is missing', () => {
  const manifest = v2Manifest();
  delete manifest.series.candidate.backends.ace.default.run1;
  const report = compareManifest(manifest);
  assert.equal(report.status, 'FAIL');
  assert.ok(report.errors.some(error => error.code === 'missing-entry'));
});

test('fails closed when a v2 protected field is missing', () => {
  const manifest = v2Manifest();
  delete candidateRun1(manifest).rawSha256;
  const report = compareManifest(manifest);
  assert.equal(report.status, 'FAIL');
  assert.ok(report.errors.some(error => error.code === 'missing-field' && error.field === 'rawSha256'));
});

test('blocks unexpected environment drift', () => {
  const manifest = v2Manifest();
  candidateRun1(manifest).environment.build = 'new-build';
  manifest.series.candidate.environment.build = 'new-build';
  const report = compareManifest(manifest);
  assert.equal(report.status, 'BLOCKED');
  assert.equal(report.ok, false);
  assert.ok(report.blocked.some(error => error.code === 'unexpected-environment'));
});

test('expected environment permits engine identity drift but never an audio mismatch', () => {
  const manifest = v2Manifest();
  const candidate = candidateRun1(manifest);
  candidate.environment.engine.pid = 101;
  manifest.series.candidate.backends.ace.default.run2.environment.engine.pid = 101;
  manifest.series.candidate.environment.engine.pid = 101;
  candidate.sha256 = 'different-audio';
  const report = compareManifest(manifest, {
    expectedEnvironment: {
      ...environment,
      engine: { ...environment.engine, pid: 101 },
    },
  });
  assert.equal(report.status, 'FAIL');
  assert.equal(report.ok, false);
  assert.ok(report.environmentNotes.some(note => note.outcome === 'ALLOWED_IDENTITY_DIFFERENCE'));
  assert.ok(report.comparisons.some(c => c.mismatches.some(m => m.field === 'sha256')));
});

test('expected environment does not permit model or stable runtime drift', () => {
  const manifest = v2Manifest();
  const changed = {
    ...environment,
    modelFiles: [{ path: 'models/changed.gguf', bytes: 11, mtimeMs: 2 }],
  };
  candidateRun1(manifest).environment = changed;
  manifest.series.candidate.backends.ace.default.run2.environment = changed;
  manifest.series.candidate.environment = changed;
  const report = compareManifest(manifest, { expectedEnvironment: changed });
  assert.equal(report.status, 'BLOCKED');
  assert.equal(report.ok, false);
  assert.ok(report.blocked.some(error => error.code === 'unexpected-environment'));
  assert.ok(report.blocked.some(error => error.differences?.some(diff => diff.path.startsWith('modelFiles'))));
});

test('legacy reports LIMITED coverage and requires explicit opt-in', () => {
  const legacy = { ace: { run1: entry(), run2: entry() } };
  const limited = compareManifest(legacy);
  assert.equal(limited.status, 'LIMITED');
  assert.equal(limited.limited, true);
  assert.equal(limited.ok, false);
  assert.deepEqual(limited.coverage.unchecked, ['rawSha256', 'masteredSha256', 'environment']);
  const allowed = compareManifest(legacy, { allowLegacy: true });
  assert.equal(allowed.status, 'LIMITED');
  assert.equal(allowed.ok, true);
});

test('rejects an attempts count other than one', () => {
  const manifest = v2Manifest();
  candidateRun1(manifest).attempts = [{ attempt: 1 }, { attempt: 2 }];
  const report = compareManifest(manifest);
  assert.equal(report.status, 'FAIL');
  assert.ok(report.errors.some(error => error.code === 'invalid-attempt-count'));
});

test('fails closed when a selected v2 series has no backend variants', () => {
  const manifest = v2Manifest();
  manifest.series.candidate.backends = {};
  const report = compareManifest(manifest);
  assert.equal(report.status, 'FAIL');
  assert.ok(report.errors.some(error => error.code === 'empty-series' && error.series === 'candidate'));
});

test('fails closed when the requested series is unknown', () => {
  const report = compareManifest(v2Manifest(), { series: 'missing-series' });
  assert.equal(report.status, 'FAIL');
  assert.ok(report.errors.some(error => error.code === 'missing-series' && error.series === 'missing-series'));
});

test('requires run2 for repeatability in every selected variant', () => {
  const manifest = v2Manifest();
  delete manifest.series.candidate.backends.ace.default.run2;
  const report = compareManifest(manifest);
  assert.equal(report.status, 'FAIL');
  assert.ok(report.errors.some(error => error.code === 'missing-entry' && error.run === 'run2' && error.series === 'candidate'));
});

test('compares every additional run against run1', () => {
  const manifest = v2Manifest();
  manifest.series.candidate.backends.ace.default.run3 = entry({ sha256: 'different-run3' });
  const report = compareManifest(manifest);
  assert.equal(report.status, 'FAIL');
  assert.ok(report.comparisons.some(c => c.label === 'candidate-run1-vs-candidate-run3' && c.mismatches.some(m => m.field === 'sha256')));
});

test('blocks a v2 entry whose recorded environment changed during the run', () => {
  const manifest = v2Manifest();
  manifest.series.candidate.backends.ace.default.run1.environmentAfter = { ...environment, build: 'changed-after-start' };
  const report = compareManifest(manifest);
  assert.equal(report.status, 'BLOCKED');
  assert.ok(report.blocked.some(error => error.code === 'environment-changed-during-run'));
});
