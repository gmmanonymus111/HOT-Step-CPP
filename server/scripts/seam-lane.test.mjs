// CPU-only regression tests for production GPU-lane ownership and reset.
// No engine, server, or GPU process is involved.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const require = createRequire(path.join(root, 'server/package.json'));
const ts = require('typescript');
const laneFile = 'server/src/services/generation/gpuLane.ts';
const compilerOptions = { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS };

function loadLane() {
  const source = fs.readFileSync(path.join(root, laneFile), 'utf8');
  const output = ts.transpileModule(source, {
    fileName: laneFile,
    compilerOptions,
  }).outputText;
  const module = { exports: {} };
  vm.runInNewContext(output, {
    module,
    exports: module.exports,
    console: { log() {}, error() {}, warn() {} },
    Object,
    Error,
    Promise,
  }, { filename: laneFile });
  return module.exports;
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

async function flush(rounds = 8) {
  for (let i = 0; i < rounds; i++) await Promise.resolve();
}

function isResetError(error) {
  return error?.name === 'LaneResetError' && error?.message === 'Queue reset by user';
}

test('runs queued work FIFO and hands ownership over after settlement', async () => {
  const lane = loadLane();
  const order = [];
  const gates = [deferred(), deferred(), deferred()];
  const promises = gates.map((gate, index) => lane.runOnGpuLane(async lease => {
    order.push({ label: String(index + 1), lease: lease.id });
    await gate.promise;
    return index + 1;
  }, { label: `fifo-${index + 1}` }));

  await flush();
  assert.deepEqual(order.map(item => item.label), ['1']);
  assert.equal(lane.gpuLaneDepth(), 2);
  gates[0].resolve();
  await flush();
  assert.deepEqual(order.map(item => item.label), ['1', '2']);
  gates[1].resolve();
  await flush();
  assert.deepEqual(order.map(item => item.label), ['1', '2', '3']);
  gates[2].resolve();
  assert.deepEqual(await Promise.all(promises), [1, 2, 3]);
  assert.equal(lane.gpuLaneBusy(), false);
  assert.equal(lane.gpuLaneOwner(), null);
});

test('reset rejects every pending promise while the current owner drains', async () => {
  const lane = loadLane();
  const running = deferred();
  let owner;
  let entries = 0;
  const currentPromise = lane.runOnGpuLane(async lease => {
    owner = lease;
    entries++;
    await running.promise;
    return 'current';
  }, { label: 'current' });
  const pendingOne = lane.runOnGpuLane(async () => { entries++; return 'one'; }, { label: 'one' });
  const pendingTwo = lane.runOnGpuLane(async () => { entries++; return 'two'; }, { label: 'two' });

  await flush();
  assert.equal(entries, 1);
  assert.equal(lane.resetGpuLane(), 2);
  assert.equal(lane.gpuLaneBusy(), true);
  assert.equal(lane.gpuLaneOwner(), owner);
  assert.equal(Object.isFrozen(owner), true);
  assert.equal(owner.draining, true);
  assert.equal(owner.isCurrent(), false);
  await assert.rejects(pendingOne, isResetError);
  await assert.rejects(pendingTwo, isResetError);

  const afterReset = lane.runOnGpuLane(async () => { entries++; return 'after-reset'; }, { label: 'after-reset' });
  assert.equal(lane.gpuLaneDepth(), 1);
  running.resolve();
  assert.equal(await currentPromise, 'current');
  assert.equal(await afterReset, 'after-reset');
  assert.equal(entries, 2);
  assert.equal(lane.gpuLaneBusy(), false);
});

test('reset before the first microtask prevents callback entry', async () => {
  const lane = loadLane();
  let entered = false;
  const promise = lane.runOnGpuLane(async () => {
    entered = true;
    return 'must not run';
  }, { label: 'pre-microtask' });

  assert.equal(lane.resetGpuLane(), 0);
  assert.equal(lane.gpuLaneBusy(), true);
  assert.equal(lane.gpuLaneOwner().draining, true);
  await assert.rejects(promise, isResetError);
  await flush();
  assert.equal(entered, false);
  assert.equal(lane.gpuLaneBusy(), false);
});

test('repeated reset rejects work added between resets and drains only once', async () => {
  const lane = loadLane();
  const running = deferred();
  let entries = 0;
  const current = lane.runOnGpuLane(async () => {
    entries++;
    await running.promise;
  }, { label: 'long-running' });
  const firstPending = lane.runOnGpuLane(async () => { entries++; }, { label: 'first-pending' });
  await flush();
  assert.equal(lane.resetGpuLane(), 1);
  const secondPending = lane.runOnGpuLane(async () => { entries++; }, { label: 'second-pending' });
  assert.equal(lane.resetGpuLane(), 1);
  await assert.rejects(firstPending, isResetError);
  await assert.rejects(secondPending, isResetError);
  running.resolve();
  await current;
  await flush();
  assert.equal(entries, 1);
  assert.equal(lane.gpuLaneBusy(), false);
  assert.equal(lane.gpuLaneDepth(), 0);
});

test('a rejected task releases the lane and admits the next task', async () => {
  const lane = loadLane();
  let secondEntered = false;
  const first = lane.runOnGpuLane(async () => {
    throw new Error('expected failure');
  }, { label: 'rejecting' });
  const second = lane.runOnGpuLane(async () => {
    secondEntered = true;
    return 'next';
  }, { label: 'next' });

  await assert.rejects(first, /expected failure/);
  assert.equal(await second, 'next');
  assert.equal(secondEntered, true);
  assert.equal(lane.gpuLaneBusy(), false);
});

test('a stale lease cannot clear the newer owner after reset handoff', async () => {
  const lane = loadLane();
  const running = deferred();
  const nextGate = deferred();
  let oldLease;
  const old = lane.runOnGpuLane(async lease => {
    oldLease = lease;
    await running.promise;
  }, { label: 'old-owner' });
  await flush();
  assert.equal(lane.resetGpuLane(), 0);
  let newLease;
  const next = lane.runOnGpuLane(async lease => {
    newLease = lease;
    await nextGate.promise;
    return lease.label;
  }, { label: 'new-owner' });
  running.resolve();
  await old;
  await flush();
  assert.equal(lane.gpuLaneOwner(), newLease);
  assert.equal(lane.gpuLaneOwner().label, 'new-owner');
  assert.equal(oldLease.isCurrent(), false);
  nextGate.resolve();
  assert.equal(await next, 'new-owner');
});

test('external GPU work keeps the lane until the callback promise unwinds', async () => {
  const lane = loadLane();
  const external = deferred();
  let nextEntered = false;
  const first = lane.runOnGpuLane(async () => {
    await external.promise;
    await Promise.resolve(); // model the external PP cleanup/finally
  }, { label: 'external-pp' });
  const next = lane.runOnGpuLane(async () => {
    nextEntered = true;
    return 'next';
  }, { label: 'next' });

  await flush();
  assert.equal(nextEntered, false);
  assert.equal(lane.gpuLaneBusy(), true);
  external.resolve();
  await first;
  assert.equal(await next, 'next');
  assert.equal(nextEntered, true);
  assert.equal(lane.gpuLaneBusy(), false);
});
