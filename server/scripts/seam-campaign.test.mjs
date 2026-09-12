import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { verifyRecord, validateContextUnchanged, validateWarmupTransition, validatePinnedSelection } from './seam-campaign.mjs';

const transitionError = 'Environment changed during the run; artifacts retained, no accepted manifest entry';

test('blocks capture when engine restart lost the pinned model selection', () => {
  const pinned = { lm: { requested: 'q8_0', selected: 'q8_0' }, voc: { requested: 'f16', selected: 'f16' } };
  assert.doesNotThrow(() => validatePinnedSelection(pinned, structuredClone(pinned)));
  assert.throws(() => validatePinnedSelection(pinned, { ...pinned, lm: { requested: '', selected: 'f16' } }), /model selection differs/);
  assert.throws(() => validatePinnedSelection(pinned, {}), /model selection differs/);
  assert.throws(() => validatePinnedSelection({}, {}), /model selection differs/);
});

function record(overrides = {}) {
  return {
    status: 'succeeded',
    backend: 'minimax-m3',
    submittedParams: { backend: 'minimax-m3' },
    environment: { stable: 'same', ditRuntime: { backend: 'tensorrt' } },
    environmentAfter: { stable: 'same', ditRuntime: { backend: 'ggml' } },
    ...overrides,
  };
}

test('accepts only the documented warmup transition failure', () => {
  assert.equal(validateWarmupTransition({
    runnerStatus: 1, stderr: transitionError, lockPresent: false, record: record(), expectedRuntime: 'ggml',
  }), true);
  assert.throws(() => validateWarmupTransition({
    runnerStatus: 2, stderr: transitionError, lockPresent: false, record: record(), expectedRuntime: 'ggml',
  }), /expected runtime transition/);
  assert.throws(() => validateWarmupTransition({
    runnerStatus: 0, stderr: '', lockPresent: false, record: record({ status: 'failed' }), expectedRuntime: 'ggml',
  }), /record status/);
  assert.throws(() => validateWarmupTransition({
    runnerStatus: 1, stderr: transitionError, lockPresent: true, record: record(), expectedRuntime: 'ggml',
  }), /expected runtime transition/);
});

test('rejects pinned input hash drift', () => {
  const before = { files: [{ path: 'input.wav', sha256: 'a'.repeat(64) }], chain: { plugins: [] } };
  const after = { files: [{ path: 'input.wav', sha256: 'b'.repeat(64) }], chain: { plugins: [] } };
  assert.throws(() => validateContextUnchanged(before, after), /context changed/);
  assert.equal(validateContextUnchanged(before, structuredClone(before)), true);
});

test('loads job status from the separate baseline status artifact', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'seam-campaign-'));
  try {
    const raw = record();
    delete raw.status; // Production record.json has no status field.
    fs.writeFileSync(path.join(dir, 'record.json'), JSON.stringify(raw));
    fs.writeFileSync(path.join(dir, 'status.json'), JSON.stringify({ status: 'succeeded' }));
    const loaded = verifyRecord(dir, 'minimax-m3', 'ggml');
    assert.equal(validateWarmupTransition({ runnerStatus: 1, stderr: transitionError,
      lockPresent: false, record: loaded, expectedRuntime: 'ggml' }), true);
    const changed = structuredClone(loaded);
    changed.environmentAfter.stable = 'changed';
    assert.throws(() => validateWarmupTransition({ runnerStatus: 1, stderr: transitionError,
      lockPresent: false, record: changed, expectedRuntime: 'ggml' }), /beyond/);
    assert.throws(() => validateWarmupTransition({ runnerStatus: 1, stderr: 'backend restore failed',
      lockPresent: false, record: loaded, expectedRuntime: 'ggml' }), /expected runtime transition/);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
