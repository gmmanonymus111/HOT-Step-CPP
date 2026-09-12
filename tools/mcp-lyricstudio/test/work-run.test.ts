import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { realpathSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';
import { loadWorkStore, runGuardedCommand, type WorkStoreLike } from '../src/work-run.js';

const childSource = `
import { writeFile } from 'node:fs/promises';
const mode = process.argv[2];
const marker = process.argv[3];
await writeFile(marker, 'started');
if (mode === 'sleep') await new Promise(resolve => setTimeout(resolve, 350));
if (mode === 'fail') process.exit(7);
if (mode === 'signal') process.kill(process.pid, 'SIGTERM');
`;
const execFileAsync = promisify(execFile);

async function waitForFile(path: string): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt++) {
    try { await readFile(path); return; } catch { await new Promise(resolve => setTimeout(resolve, 10)); }
  }
  throw new Error(`child did not create ${path}`);
}

async function makeFixture() {
  const directory = await mkdtemp(join(tmpdir(), 'hotstep-work-run-'));
  const script = join(directory, 'child.mjs');
  const db = join(directory, 'work.db');
  await writeFile(script, childSource, 'utf8');
  return { directory, script, db };
}

async function removeFixture(directory: string): Promise<void> {
  // Windows can keep SQLite's last handle briefly after a child process has
  // exited.  Give the OS time to release it so cleanup does not obscure the
  // reservation assertions.
  for (let attempt = 0; attempt < 40; attempt++) {
    try { await rm(directory, { recursive: true, force: true }); return; }
    catch (error) { if (attempt === 39) throw error; await new Promise(resolve => setTimeout(resolve, 50)); }
  }
}

function command(script: string, mode: string, marker: string): string[] {
  return [process.execPath, script, mode, marker];
}

async function run(options: {
  db: string; script: string; name: string; resource: string; mode: string; marker: string;
  store?: WorkStoreLike; signal?: AbortSignal;
}) {
  return runGuardedCommand({
    dbPath: options.db,
    channel: 'HOT-Step',
    name: options.name,
    resources: [options.resource],
    reason: 'work-run test',
    command: command(options.script, options.mode, options.marker),
    heartbeatMs: 25,
    signal: options.signal,
  }, options.store);
}

test('holds a resource for the complete child lifetime and releases after normal close', async () => {
  const fixture = await makeFixture();
  try {
    const marker = join(fixture.directory, 'first.started');
    const first = run({ ...fixture, name: 'runner-a', resource: 'engine-build', mode: 'sleep', marker });
    await waitForFile(marker);

    const secondMarker = join(fixture.directory, 'second.started');
    await assert.rejects(
      run({ ...fixture, name: 'runner-b', resource: 'engine-build', mode: 'sleep', marker: secondMarker }),
      /resource|reservation|reserved|held|conflict|busy/i,
    );
    assert.equal((await first).released, true);

    const afterMarker = join(fixture.directory, 'after.started');
    const after = await run({ ...fixture, name: 'runner-c', resource: 'engine-build', mode: 'fail', marker: afterMarker });
    assert.equal(after.exitCode, 7);
    assert.equal(after.released, true);
  } finally {
    await removeFixture(fixture.directory);
  }
});

test('releases a normally observed nonzero child failure', async () => {
  const fixture = await makeFixture();
  try {
    const first = await run({ ...fixture, name: 'runner-failure', resource: 'cpu-test', mode: 'fail', marker: join(fixture.directory, 'failure.started') });
    assert.equal(first.exitCode, 7);
    assert.equal(first.uncertain, false);

    const second = await run({ ...fixture, name: 'runner-after-failure', resource: 'cpu-test', mode: 'fail', marker: join(fixture.directory, 'second.started') });
    assert.equal(second.released, true);
  } finally {
    await removeFixture(fixture.directory);
  }
});

test('does not clear a reservation when the child closes by signal', async () => {
  const fixture = await makeFixture();
  try {
    const controller = new AbortController();
    const firstPromise = run({ ...fixture, name: 'runner-signal', resource: 'gpu0', mode: 'sleep', marker: join(fixture.directory, 'signal.started'), signal: controller.signal });
    await waitForFile(join(fixture.directory, 'signal.started'));
    controller.abort();
    const first = await firstPromise;
    assert.equal(first.uncertain, true);
    assert.equal(first.released, false);

    await assert.rejects(
      run({ ...fixture, name: 'runner-blocked', resource: 'gpu0', mode: 'fail', marker: join(fixture.directory, 'blocked.started') }),
      /resource|reservation|reserved|held|conflict|busy|uncertain/i,
    );
  } finally {
    await removeFixture(fixture.directory);
  }
});

test('loads the real WorkStore from the project implementation', async () => {
  const fixture = await makeFixture();
  try {
    const store = await loadWorkStore(fixture.db, 'work-run-test');
    assert.equal(typeof store.join, 'function');
    store.close();
  } finally {
    await removeFixture(fixture.directory);
  }
});

test('does not join or start a child when already aborted', async () => {
  const fixture = await makeFixture();
  let joins = 0;
  const controller = new AbortController();
  controller.abort();
  const store: WorkStoreLike = {
    join() { joins++; throw new Error('join should not run'); },
    update() { throw new Error('update should not run'); },
    release() { throw new Error('release should not run'); },
    heartbeat() {}, disconnect() {}, close() {},
  };
  const marker = join(fixture.directory, 'aborted.started');
  try {
    await assert.rejects(runGuardedCommand({
      dbPath: fixture.db, channel: 'HOT-Step', name: 'runner-aborted', resources: ['cpu-test'],
      reason: 'already aborted', command: command(fixture.script, 'sleep', marker), signal: controller.signal,
    }, store), /aborted/i);
    assert.equal(joins, 0);
    await assert.rejects(readFile(marker));
  } finally {
    await removeFixture(fixture.directory);
  }
});

test('passes guard markers to the child and quotes a batch path with spaces', { skip: process.platform !== 'win32' }, async () => {
  const fixture = await makeFixture();
  const batchDirectory = join(fixture.directory, 'batch path');
  await mkdir(batchDirectory);
  const batch = join(batchDirectory, 'guard test.cmd');
  const marker = join(fixture.directory, 'batch result.txt');
  await writeFile(batch, '@echo off\r\n> "%~2" echo %~1\r\nif "%HOTSTEP_WORK_GUARD_ACTIVE%"=="1" if not "%HOTSTEP_WORK_GUARD_RESOURCES%"=="batch-test" exit /b 12\r\nexit /b 9\r\n', 'utf8');
  try {
    const result = await runGuardedCommand({
      dbPath: fixture.db, channel: 'HOT-Step', name: 'runner-batch', resources: ['batch-test'],
      reason: 'batch quoting test', command: [batch, 'hello world\\', marker],
    });
    assert.equal(result.exitCode, 9);
    assert.equal(result.released, true);
    assert.equal((await readFile(marker, 'utf8')).trim(), 'hello world\\');
  } finally {
    await removeFixture(fixture.directory);
  }
});

test('PowerShell 5.1 -File launcher runs a spaced child with the pinned runtime', { skip: process.platform !== 'win32', timeout: 30000 }, async () => {
  const fixture = await makeFixture();
  const spaced = join(fixture.directory, 'launcher path with spaces');
  await mkdir(spaced);
  const script = join(spaced, 'child script.mjs');
  const db = join(spaced, 'work database.db');
  const started = join(spaced, 'child started.marker');
  const completed = join(spaced, 'child completed.marker');
  const child = `import { writeFile } from 'node:fs/promises';\nconst [started, completed] = process.argv.slice(2);\nawait writeFile(started, 'started');\nawait writeFile(completed, 'completed');\n`;
  await writeFile(script, child, 'utf8');
  // Resolve from the physical test location because the dedicated runtime
  // contains the source tree but intentionally has no launcher junction.
  const physicalTest = realpathSync(fileURLToPath(import.meta.url));
  const launcher = realpathSync(fileURLToPath(new URL('../work-run.ps1', pathToFileURL(physicalTest))));
  try {
    await execFileAsync('powershell.exe', [
      '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', launcher,
      '--db', db, '--channel', 'HOT-Step', '--name', 'powershell-regression',
      '--resource', 'powershell-regression', '--reason', 'PowerShell launcher regression', '--',
      process.execPath, script, started, completed,
    ], { cwd: spaced, windowsHide: true, timeout: 20000 });
    assert.equal(await readFile(started, 'utf8'), 'started');
    assert.equal(await readFile(completed, 'utf8'), 'completed');
    assert.equal((await readFile(db)).length > 0, true);
  } finally {
    await removeFixture(fixture.directory);
  }
});
