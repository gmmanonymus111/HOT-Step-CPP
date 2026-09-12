import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { acquireGpuLock } from './seam-lock.mjs';

const testFile = fileURLToPath(import.meta.url);

function temporaryLockPath() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hot-step-seam-lock-'));
  return { dir, lockPath: path.join(dir, 'GPU.lock') };
}

function removeTemporary({ dir }) {
  fs.rmSync(dir, { recursive: true, force: true });
}

async function waitForChildReady(child) {
  return new Promise((resolve, reject) => {
    let output = '';
    const onData = (chunk) => {
      output += chunk;
      const line = output.split(/\r?\n/, 1)[0];
      if (line === 'READY') {
        child.stdout.off('data', onData);
        resolve();
      }
    };
    child.stdout.on('data', onData);
    child.once('error', reject);
    child.once('exit', (code) => {
      if (code !== null && code !== 0) reject(new Error(`lock child exited before ready (${code}): ${output}`));
    });
  });
}

function waitForExit(child) {
  return new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', (code, signal) => resolve({ code, signal }));
  });
}

// A child mode gives the contention test a genuine independent Node process.
if (process.argv[2] === '--child-hold') {
  const lockPath = process.argv[3];
  const lock = acquireGpuLock({ path: lockPath, owner: 'child-owner', reason: 'contention test' });
  console.log('READY');
  process.stdin.setEncoding('utf8');
  process.stdin.once('data', () => {
    lock.release();
    process.exit(0);
  });
  setInterval(() => {}, 1000);
} else {
  test('writes token and owner metadata and releases idempotently', () => {
    const temp = temporaryLockPath();
    try {
      const lock = acquireGpuLock({ path: temp.lockPath, owner: 'test-owner', reason: 'metadata test', token: 'test-token' });
      const metadata = JSON.parse(fs.readFileSync(temp.lockPath, 'utf8'));
      assert.equal(metadata.token, 'test-token');
      assert.equal(metadata.owner, 'test-owner');
      assert.equal(metadata.reason, 'metadata test');
      assert.equal(metadata.pid, process.pid);
      assert.match(metadata.since, /^\d{4}-\d\d-\d\dT/);
      assert.equal(lock.release(), true);
      assert.equal(lock.release(), false);
      assert.equal(fs.existsSync(temp.lockPath), false);
    } finally {
      removeTemporary(temp);
    }
  });

  test('reports an existing lock without taking it over', () => {
    const temp = temporaryLockPath();
    try {
      const first = acquireGpuLock({ path: temp.lockPath, owner: 'first-owner', reason: 'first reason' });
      assert.throws(
        () => acquireGpuLock({ path: temp.lockPath, owner: 'second-owner', reason: 'second reason' }),
        (error) => {
          assert.equal(error.code, 'ELOCKED');
          assert.match(error.message, /GPU lock already held/);
          assert.match(error.message, /first-owner/);
          assert.match(error.message, /first reason/);
          return true;
        },
      );
      assert.equal(JSON.parse(fs.readFileSync(temp.lockPath, 'utf8')).owner, 'first-owner');
      first.release();
    } finally {
      removeTemporary(temp);
    }
  });

  test('does not unlink a replacement with a wrong token or owner', () => {
    const temp = temporaryLockPath();
    try {
      const first = acquireGpuLock({ path: temp.lockPath, owner: 'first-owner', reason: 'replacement test', token: 'first-token' });
      fs.writeFileSync(temp.lockPath, JSON.stringify({
        token: 'foreign-token',
        owner: 'foreign-owner',
        reason: 'foreign reason',
        since: new Date().toISOString(),
        pid: 999,
      }));
      assert.equal(first.release(), false);
      assert.equal(JSON.parse(fs.readFileSync(temp.lockPath, 'utf8')).token, 'foreign-token');
      fs.unlinkSync(temp.lockPath);
    } finally {
      removeTemporary(temp);
    }
  });

  test('contends with an independent Node process', async () => {
    const temp = temporaryLockPath();
    const child = spawn(process.execPath, [testFile, '--child-hold', temp.lockPath], {
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    try {
      await waitForChildReady(child);
      assert.throws(
        () => acquireGpuLock({ path: temp.lockPath, owner: 'parent-owner', reason: 'while child holds' }),
        (error) => error.code === 'ELOCKED' && /child-owner/.test(error.message),
      );
      child.stdin.write('release\n');
      const exit = await waitForExit(child);
      assert.equal(exit.code, 0);
      const parentLock = acquireGpuLock({ path: temp.lockPath, owner: 'parent-owner', reason: 'after child release' });
      assert.equal(parentLock.release(), true);
    } finally {
      if (!child.killed && child.exitCode === null) child.kill();
      removeTemporary(temp);
    }
  });
}
