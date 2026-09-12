/**
 * Run one foreground command while holding Work channel reservations.
 *
 * This module deliberately knows only the small WorkStore contract.  The
 * store is opened by the CLI through loadWorkStore(), while tests and callers
 * may inject an already-created store.  A reservation is released only after
 * Node has observed a normal child close; all other endings become an
 * uncertain/disconnected lease for human recovery.
 */
import { spawn, type ChildProcess, type SpawnOptions } from 'node:child_process';
import { existsSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { extname, resolve } from 'node:path';
import { WorkStore } from './work-store.js';
import type { JoinInput, ReleaseInput, UpdateInput } from './work-contract.js';

export const DEFAULT_WORK_DB = fileURLToPath(new URL('../../../data/collaboration.db', import.meta.url));
export const DEFAULT_WORK_CHANNEL = 'HOT-Step';

export type WorkGrant = { id: string; resource: string; token: string };

export interface WorkStoreLike {
  join(input: JoinInput, human?: boolean): { agent: string; channel: string; protocol: string };
  update(input: UpdateInput): { id: number; grants?: WorkGrant[] };
  release(input: ReleaseInput): { released: string };
  heartbeat(): void;
  disconnect(): void;
  close(): void;
}

export type GuardedCommandOptions = {
  dbPath?: string;
  channel?: string;
  name: string;
  resources: string[];
  reason: string;
  command: string[];
  session?: string;
  heartbeatMs?: number;
  signal?: AbortSignal;
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  stdio?: SpawnOptions['stdio'];
};

export type GuardedCommandResult = {
  exitCode: number | null;
  signal: NodeJS.Signals | null;
  uncertain: boolean;
  released: boolean;
};

const CMD_UNSAFE = /[\r\n&|<>^%!()]/;

/**
 * cmd.exe has no argv API.  We therefore accept only a deliberately small
 * argument alphabet for .cmd/.bat commands and quote every argument.  This
 * supports normal paths, spaces, dashes, dots and equals signs.  Callers that
 * need shell metacharacters must use an executable with shell:false instead.
 */
export function quoteCmdArgument(value: string): string {
  if (CMD_UNSAFE.test(value) || value.includes('"')) {
    throw new Error('.cmd/.bat arguments may not contain quotes, %, !, &, |, <, >, parentheses, or newlines');
  }
  if (value.length === 0) return '""';
  // This is cmd.exe syntax, where backslash is an ordinary character.  Do not
  // apply CommandLineToArgvW's trailing-backslash doubling: cmd would pass the
  // doubled slashes through to the batch file.
  return `"${value}"`;
}

function isBatchCommand(command: string): boolean {
  const suffix = extname(command).toLowerCase();
  return suffix === '.bat' || suffix === '.cmd';
}

function spawnForeground(command: string[], options: GuardedCommandOptions): ChildProcess {
  const [executable, ...args] = command;
  const spawnOptions: SpawnOptions = {
    cwd: options.cwd,
    env: {
      ...process.env,
      ...options.env,
      HOTSTEP_WORK_GUARD_ACTIVE: '1',
      HOTSTEP_WORK_GUARD_RESOURCES: options.resources.join(','),
    },
    stdio: options.stdio ?? 'inherit',
    shell: false,
  };

  if (!isBatchCommand(executable)) {
    return spawn(executable, args, spawnOptions);
  }

  if (process.platform !== 'win32') {
    throw new Error('Cannot execute .cmd/.bat through a POSIX shell; use the executable directly.');
  }
  const commandLine = [executable, ...args].map(quoteCmdArgument).join(' ');
  // The outer pair is required by cmd /s /c when the batch path is quoted.
  return spawn('cmd.exe', ['/d', '/s', '/c', `"${commandLine}"`], {
    ...spawnOptions,
    windowsVerbatimArguments: true,
  });
}

function safeClose(store: WorkStoreLike): void {
  try { store.close(); } catch { /* preserve the original command result */ }
}

function markDisconnected(store: WorkStoreLike): void {
  try { store.disconnect(); } catch { /* recovery state is best effort */ }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Run a command with an injected WorkStore, or open the configured store. */
export async function runGuardedCommand(
  options: GuardedCommandOptions,
  injectedStore?: WorkStoreLike,
): Promise<GuardedCommandResult> {
  if (!options.name.trim()) throw new Error('--name is required');
  if (!options.reason.trim()) throw new Error('--reason is required');
  if (!options.resources.length) throw new Error('at least one --resource is required');
  if (!options.command.length || !options.command[0]) throw new Error('a command is required after --');
  if (options.signal?.aborted) throw new Error('command was aborted before reservation');

  const channel = options.channel?.trim() || DEFAULT_WORK_CHANNEL;
  const store = injectedStore ?? await loadWorkStore(options.dbPath ?? DEFAULT_WORK_DB, options.session);
  let connected = true;
  let joined = false;
  let grants: WorkGrant[] = [];
  let released = false;
  let uncertain = false;
  let child: ChildProcess | undefined;
  let heartbeat: NodeJS.Timeout | undefined;
  const requestId = `work-run:${randomUUID()}`;
  const signalHandlers: Array<[NodeJS.Signals, () => void]> = [];
  let abortHandler: (() => void) | undefined;

  const disconnect = () => {
    if (!connected) return;
    connected = false;
    uncertain = true;
    markDisconnected(store);
  };

  try {
    if (options.signal?.aborted) throw new Error('command was aborted before reservation');
    const participant = store.join({ channel, name: options.name });
    joined = true;
    if (options.signal?.aborted) throw new Error('command was aborted before reservation');
    const update: UpdateInput = {
      channel,
      agent: participant.agent,
      request_id: requestId,
      text: `Running guarded command: ${options.command[0]}`,
      activity: 'working',
      state: 'doing',
      reserve: options.resources.map(resource => ({
        resource,
        mode: 'exclusive' as const,
        reason: options.reason,
      })),
    };
    const result = store.update(update);
    grants = result.grants ?? [];
    if (grants.length !== options.resources.length) {
      throw new Error('WorkStore did not grant every requested resource');
    }

    const interval = options.heartbeatMs ?? 30_000;
    if (interval > 0) {
      heartbeat = setInterval(() => {
        try { store.heartbeat(); } catch { disconnect(); }
      }, interval);
      heartbeat.unref();
    }

    const onParentSignal = (signal: NodeJS.Signals) => {
      disconnect();
      try { child?.kill(signal); } catch { /* uncertain state is retained */ }
    };
    for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP'] as NodeJS.Signals[]) {
      const handler = () => onParentSignal(signal);
      process.once(signal, handler);
      signalHandlers.push([signal, handler]);
    }
    if (options.signal) {
      abortHandler = () => {
        disconnect();
        try { child?.kill(); } catch { /* uncertain state is retained */ }
      };
      if (options.signal.aborted) abortHandler();
      else options.signal.addEventListener('abort', abortHandler, { once: true });
    }

    let spawnError: Error | undefined;
    let closeCode: number | null = null;
    let closeSignal: NodeJS.Signals | null = null;
    const closePromise = new Promise<void>((resolveClose, rejectClose) => {
      try {
        if (options.signal?.aborted) {
          rejectClose(new Error('command was aborted before spawn'));
          return;
        }
        child = spawnForeground(options.command, options);
      } catch (error) {
        rejectClose(error);
        return;
      }
      child.once('error', error => { spawnError = error; });
      child.once('close', (code, signal) => {
        closeCode = code;
        closeSignal = signal;
        resolveClose();
      });
    });
    await closePromise;
    if (spawnError) {
      disconnect();
      throw spawnError;
    }
    if (closeSignal !== null || closeCode === null || uncertain) {
      disconnect();
    }

    if (!uncertain && joined) {
      try {
        store.update({
          channel,
          agent: participant.agent,
          request_id: `${requestId}:done`,
          text: `Guarded command finished: ${options.command[0]}`,
          activity: 'done',
          state: 'done',
        });
        for (const grant of grants) {
          // WorkStore request ids are idempotency keys.  Each token is a
          // separate release operation, so sharing the update key would make
          // the second release look like a conflicting replay.
          store.release({ channel, agent: participant.agent, request_id: `${requestId}:release:${grant.id}`, token: grant.token, note: 'foreground command closed normally' });
        }
        released = true;
      } catch (error) {
        disconnect();
        throw new Error(`command completed but WorkStore release failed: ${errorMessage(error)}`);
      }
    }
    return { exitCode: closeCode, signal: closeSignal, uncertain, released };
  } catch (error) {
    if (joined && !released) disconnect();
    throw error;
  } finally {
    if (heartbeat) clearInterval(heartbeat);
    for (const [signal, handler] of signalHandlers) process.removeListener(signal, handler);
    if (abortHandler && options.signal) options.signal.removeEventListener('abort', abortHandler);
    safeClose(store);
  }
}

/** Load the WorkStore implementation without choosing its filename for the MCP tier. */
export async function loadWorkStore(dbPath: string, session?: string): Promise<WorkStoreLike> {
  return new WorkStore(dbPath, session ? { session } : {});
}

export type ParsedWorkRun = Omit<GuardedCommandOptions, 'dbPath'> & { dbPath: string };

function requiredFlag(argv: string[], index: number, flag: string): [string, number] {
  const value = argv[index + 1];
  if (!value || value === '--') throw new Error(`${flag} requires a value`);
  return [value, index + 2];
}

export function parseWorkRunArgs(argv: string[]): ParsedWorkRun | { help: true } {
  const separator = argv.indexOf('--');
  if (separator < 0) {
    if (argv.includes('--help') || argv.includes('-h')) return { help: true };
    throw new Error('command must follow --');
  }
  const flags = argv.slice(0, separator);
  const command = argv.slice(separator + 1);
  let dbPath = DEFAULT_WORK_DB;
  let channel = DEFAULT_WORK_CHANNEL;
  let name = '';
  let reason = '';
  let cwd = '';
  const resources: string[] = [];
  for (let index = 0; index < flags.length;) {
    const flag = flags[index];
    if (flag === '--db') [dbPath, index] = requiredFlag(flags, index, flag);
    else if (flag === '--channel') [channel, index] = requiredFlag(flags, index, flag);
    else if (flag === '--name') [name, index] = requiredFlag(flags, index, flag);
    else if (flag === '--reason') [reason, index] = requiredFlag(flags, index, flag);
    else if (flag === '--cwd') [cwd, index] = requiredFlag(flags, index, flag);
    else if (flag === '--resource') {
      let value: string; [value, index] = requiredFlag(flags, index, flag); resources.push(value);
    } else throw new Error(`unknown option: ${flag}`);
  }
  return { dbPath: resolve(dbPath), channel, name, reason, resources, command, cwd: cwd ? resolve(cwd) : undefined };
}

export const WORK_RUN_HELP = `Usage: work-run.ps1 [options] -- COMMAND [ARG...]

Options:
  --db PATH          Work database (default: project data/collaboration.db)
  --channel NAME     Work channel (default: HOT-Step)
  --name NAME        Honest agent/worker name
  --resource NAME    Resource to reserve (repeatable; exclusive)
  --reason TEXT      Human-readable reservation reason
  --cwd PATH         Actual working directory for COMMAND

Batch files are supported on Windows with a restricted argument alphabet;
arguments containing shell metacharacters (including parentheses) must use an executable directly.`;

async function main(): Promise<void> {
  const parsed = parseWorkRunArgs(process.argv.slice(2));
  if ('help' in parsed) { console.log(WORK_RUN_HELP); return; }
  const result = await runGuardedCommand(parsed);
  if (result.uncertain) process.exitCode = 125;
  else if (result.exitCode !== null) process.exitCode = result.exitCode;
}

const thisFile = fileURLToPath(import.meta.url);
const invokedFile = process.argv[1] ? resolve(process.argv[1]) : '';
if (invokedFile === resolve(thisFile) || existsSync(invokedFile) && resolve(invokedFile) === resolve(thisFile)) {
  main().catch(error => {
    console.error(`[work-run] ${errorMessage(error)}`);
    process.exitCode = 1;
  });
}
