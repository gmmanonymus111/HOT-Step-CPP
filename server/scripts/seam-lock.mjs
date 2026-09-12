// seam-lock.mjs — synchronous cross-process lock for the YuE2 GPU campaign.
//
// The lock is deliberately a create-only file. There is no stale-lock timeout:
// a human or the owning process must release it. The open descriptor is held
// for the lifetime of the lock so the original inode remains held; release
// checks the complete ownership metadata before unlinking. Creation-failure
// cleanup additionally checks descriptor file identity before unlinking.
// Node permits delete sharing on Windows, and POSIX permits unlinking an open
// file, so a replacement with different ownership metadata is always retained.

import fs from 'node:fs';
import pathModule from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { spawnSync } from 'node:child_process';

export const DEFAULT_GPU_LOCK_PATH = 'D:/Ace-Step-Latest/_experiments/yue2/GPU.lock';

function requireText(value, name) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new TypeError(`${name} must be a non-empty string`);
  }
  return value;
}

function sameFile(left, right) {
  // Windows exposes a useful file index through ino. Keeping dev in the
  // comparison handles platforms where inode numbers are only unique per
  // device. A failed identity check is intentionally treated as unsafe.
  return left.dev === right.dev && left.ino === right.ino;
}

function parseMetadata(text) {
  try {
    const value = JSON.parse(text);
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    return value;
  } catch {
    return null;
  }
}

function readMetadata(lockPath) {
  return parseMetadata(fs.readFileSync(lockPath, 'utf8'));
}

function describeExisting(lockPath) {
  let metadata = null;
  try {
    metadata = readMetadata(lockPath);
  } catch {
    // The lock exists but may be unreadable or may have disappeared. The
    // create-only failure remains an acquisition failure either way.
  }

  const details = metadata
    ? ` (owner=${JSON.stringify(metadata.owner ?? 'unknown')}, `
      + `reason=${JSON.stringify(metadata.reason ?? 'unknown')}, `
      + `since=${JSON.stringify(metadata.since ?? 'unknown')}, `
      + `pid=${JSON.stringify(metadata.pid ?? 'unknown')})`
    : ' (metadata unavailable or invalid)';
  const error = new Error(`GPU lock already held at ${lockPath}${details}`);
  error.code = 'ELOCKED';
  error.lockPath = lockPath;
  error.metadata = metadata;
  return error;
}

/**
 * Acquire the shared GPU lock synchronously.
 *
 * @param {{path?: string, owner: string, reason: string, token?: string}} options
 * @returns {{token: string, release: () => boolean}}
 */
export function acquireGpuLock({
  path: lockPath = DEFAULT_GPU_LOCK_PATH,
  owner,
  reason,
  token,
} = {}) {
  lockPath = requireText(lockPath, 'path');
  owner = requireText(owner, 'owner');
  reason = requireText(reason, 'reason');
  token = token === undefined ? randomUUID() : requireText(token, 'token');

  const metadata = {
    token,
    owner,
    reason,
    since: new Date().toISOString(),
    pid: process.pid,
  };
  const contents = `${JSON.stringify(metadata, null, 2)}\n`;
  let fd;

  try {
    // wx is the only acquisition primitive: never inspect/delete/retry an
    // existing lock, since that creates a race and enables accidental takeover.
    fd = fs.openSync(lockPath, 'wx');
  } catch (error) {
    if (error && error.code === 'EEXIST') throw describeExisting(lockPath);
    throw error;
  }

  let createdIdentity;
  try {
    fs.writeFileSync(fd, contents, 'utf8');
    fs.fsyncSync(fd);
    createdIdentity = fs.fstatSync(fd);
  } catch (error) {
    // Only remove the file if the path still points to the file this call
    // created. If identity cannot be proven, leave it for manual inspection.
    try {
      const currentIdentity = fs.statSync(lockPath);
      const ownIdentity = createdIdentity || fs.fstatSync(fd);
      if (sameFile(ownIdentity, currentIdentity)) fs.unlinkSync(lockPath);
    } catch {
      // Preserve the original write/fsync error and never risk another lock.
    }
    try { fs.closeSync(fd); } catch { /* preserve original error */ }
    throw error;
  }

  let holderFd = fd;
  let released = false;

  function closeHolder() {
    if (holderFd === undefined) return;
    const currentFd = holderFd;
    holderFd = undefined;
    try { fs.closeSync(currentFd); } catch { /* already closed */ }
  }

  function release() {
    if (released) return false;

    let currentMetadata;
    try {
      currentMetadata = readMetadata(lockPath);
    } catch (error) {
      closeHolder();
      if (error && error.code === 'ENOENT') {
        released = true;
        return false;
      }
      throw error;
    }

    // Verify the complete record, including both ownership fields requested
    // by the caller. A foreign or replaced lock must survive this release call.
    const metadataMatches = currentMetadata
      && currentMetadata.token === metadata.token
      && currentMetadata.owner === metadata.owner
      && currentMetadata.reason === metadata.reason
      && currentMetadata.since === metadata.since
      && currentMetadata.pid === metadata.pid;
    // Some Windows runtimes have reported a different fstat/stat identity for
    // the same lock path after a long-running operation. Complete metadata is
    // the fallback identity in that case; a foreign owner/token still cannot
    // be removed.
    if (!metadataMatches) {
      closeHolder();
      released = true;
      return false;
    }

    try {
      fs.unlinkSync(lockPath);
      released = true;
      return true;
    } catch (error) {
      // Do not mark this complete on failure; a caller may retry, and every
      // retry re-reads and re-verifies the current metadata first.
      throw error;
    } finally {
      // Unlink while the holder descriptor is still open. This is atomic on
      // POSIX and Node opens Windows files with delete sharing; closing first
      // would leave a larger pathname replacement race.
      closeHolder();
    }
  }

  return { token, release };
}

function status(lockPath) {
  try {
    const metadata = readMetadata(lockPath);
    return { held: true, path: lockPath, metadata };
  } catch (error) {
    if (error && error.code === 'ENOENT') return { held: false, path: lockPath };
    return {
      held: true,
      path: lockPath,
      metadata: null,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function parseCli(argv) {
  const command = argv[0];
  const result = { command, path: DEFAULT_GPU_LOCK_PATH };
  let i = 1;
  while (i < argv.length && argv[i] !== '--') {
    const arg = argv[i++];
    if (arg === '--path') result.path = argv[i++];
    else if (arg.startsWith('--path=')) result.path = arg.slice('--path='.length);
    else if (arg === '--owner') result.owner = argv[i++];
    else if (arg.startsWith('--owner=')) result.owner = arg.slice('--owner='.length);
    else if (arg === '--reason') result.reason = argv[i++];
    else if (arg.startsWith('--reason=')) result.reason = arg.slice('--reason='.length);
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (argv[i] === '--') result.args = argv.slice(i + 1);
  return result;
}

function printUsage() {
  console.error('Usage: seam-lock.mjs status [--path PATH]');
  console.error('   or: seam-lock.mjs run --owner OWNER --reason REASON [--path PATH] -- COMMAND [ARGS...]');
}

function main(argv) {
  const options = parseCli(argv);
  if (options.command === 'status') {
    console.log(JSON.stringify(status(options.path), null, 2));
    return 0;
  }
  if (options.command !== 'run' || !options.owner || !options.reason || !options.args?.length) {
    printUsage();
    return 2;
  }

  const lock = acquireGpuLock(options);
  try {
    const result = spawnSync(options.args[0], options.args.slice(1), { stdio: 'inherit', shell: false });
    if (result.error) throw result.error;
    if (result.signal) return 1;
    return result.status ?? 1;
  } finally {
    lock.release();
  }
}

const invokedPath = process.argv[1] ? pathModule.resolve(process.argv[1]) : '';
if (invokedPath && fileURLToPath(import.meta.url) === invokedPath) {
  try {
    process.exitCode = main(process.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
