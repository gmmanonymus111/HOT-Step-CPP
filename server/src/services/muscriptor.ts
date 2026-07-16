// muscriptor.ts — MuScriptor audio→MIDI transcription runner
//
// Manages a self-contained Python venv under data/muscriptor/ and spawns the
// MuScriptor CLI for multi-instrument transcription (audio in → .mid out).
//
// MuScriptor is developed by Kyutai & Mirelo (Rouard, Krause, Roebel,
// Simon-Gabriel, Défossez — arXiv:2607.08168). Code is MIT; model weights are
// CC BY-NC 4.0 (non-commercial). https://github.com/muscriptor/muscriptor
//
// Requires a system Python 3.10–3.12 to bootstrap the venv. Model weights are
// downloaded automatically from Hugging Face on first transcription.

import { spawn, spawnSync, ChildProcess } from 'child_process';
import path from 'path';
import fs from 'fs';
import { config } from '../config.js';

const IS_WIN = process.platform === 'win32';

export const MUSCRIPTOR_DIR = path.join(config.data.dir, 'muscriptor');
const VENV_DIR = path.join(MUSCRIPTOR_DIR, 'venv');

export const MUSCRIPTOR_MODELS = ['small', 'medium', 'large'] as const;
export type MuscriptorModel = typeof MUSCRIPTOR_MODELS[number];

function venvPython(): string {
  return IS_WIN
    ? path.join(VENV_DIR, 'Scripts', 'python.exe')
    : path.join(VENV_DIR, 'bin', 'python');
}

function muscriptorBin(): string {
  return IS_WIN
    ? path.join(VENV_DIR, 'Scripts', 'muscriptor.exe')
    : path.join(VENV_DIR, 'bin', 'muscriptor');
}

export function isInstalled(): boolean {
  return fs.existsSync(muscriptorBin());
}

// ── Hugging Face access token ────────────────────────────────────────────
// The MuScriptor model weights are GATED on Hugging Face — users must
// request access on the model page, then authenticate downloads. We store
// an optional user-provided read token in data/muscriptor/hf_token and pass
// it to the CLI via env. (A global `huggingface-cli login` also works, since
// huggingface_hub reads its cached token regardless of venv.)

const HF_TOKEN_PATH = path.join(MUSCRIPTOR_DIR, 'hf_token');

export function getHfToken(): string | null {
  try {
    const t = fs.readFileSync(HF_TOKEN_PATH, 'utf-8').trim();
    return t || null;
  } catch { return null; }
}

export function setHfToken(token: string): void {
  fs.mkdirSync(MUSCRIPTOR_DIR, { recursive: true });
  const t = (token || '').trim();
  if (!t) {
    fs.rmSync(HF_TOKEN_PATH, { force: true });
  } else {
    fs.writeFileSync(HF_TOKEN_PATH, t, { encoding: 'utf-8' });
  }
}

/** Heuristic: does this failure look like a gated-model / auth problem? */
export function looksLikeGatedError(text: string): boolean {
  return /gated|401|403|unauthorized|forbidden|restricted|access to model|awaiting a review|accept the conditions|not authenticated|invalid (user )?token|authentication/i.test(text);
}

// ── System Python discovery ──────────────────────────────────────────────

interface PythonCandidate {
  cmd: string;
  args: string[];
  version: string;
}

// MuScriptor supports Python 3.10–3.12 (not 3.13+).
const PYTHON_CANDIDATES: Array<{ cmd: string; args: string[] }> = [
  { cmd: 'py', args: ['-3.12'] },
  { cmd: 'py', args: ['-3.11'] },
  { cmd: 'py', args: ['-3.10'] },
  { cmd: 'python3.12', args: [] },
  { cmd: 'python3.11', args: [] },
  { cmd: 'python3.10', args: [] },
  { cmd: 'python3', args: [] },
  { cmd: 'python', args: [] },
];

export function findSystemPython(): PythonCandidate | null {
  for (const c of PYTHON_CANDIDATES) {
    try {
      const r = spawnSync(c.cmd, [...c.args, '--version'], { encoding: 'utf-8', timeout: 10_000 });
      if (r.status !== 0) continue;
      const out = `${r.stdout || ''}${r.stderr || ''}`;
      const m = out.match(/Python (3\.(\d+)\.\d+)/);
      if (!m) continue;
      const minor = parseInt(m[2], 10);
      if (minor >= 10 && minor <= 12) {
        return { cmd: c.cmd, args: c.args, version: m[1] };
      }
    } catch { /* candidate not on PATH */ }
  }
  return null;
}

// ── Install (venv bootstrap + pip install) ───────────────────────────────

interface InstallState {
  installing: boolean;
  step: string;        // human-readable current step
  lastLine: string;    // last output line from the running step (pip progress)
  error?: string;
}

const installState: InstallState = { installing: false, step: '', lastLine: '' };

export interface MuscriptorStatus {
  installed: boolean;
  installing: boolean;
  installStep: string;
  installLine: string;
  installError?: string;
  pythonVersion: string | null;
  hfTokenSet: boolean;
}

export function getStatus(): MuscriptorStatus {
  const installed = isInstalled();
  return {
    installed,
    installing: installState.installing,
    installStep: installState.step,
    installLine: installState.lastLine,
    installError: installState.error,
    hfTokenSet: getHfToken() !== null,
    // Probing system python is cheap-ish but not free; only do it when the
    // answer matters (i.e. not yet installed).
    pythonVersion: installed ? null : (findSystemPython()?.version ?? null),
  };
}

/** Run a child process, streaming output lines into installState.lastLine. */
function runStep(cmd: string, args: string[], step: string): Promise<void> {
  installState.step = step;
  installState.lastLine = '';
  console.log(`[MidiStudio] Setup: ${step} — ${cmd} ${args.join(' ')}`);
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { windowsHide: true });
    let lastChunk = '';
    const onData = (buf: Buffer) => {
      // pip uses \r progress bars — treat \r like \n when extracting lines
      const lines = (lastChunk + buf.toString('utf-8')).split(/[\r\n]+/);
      lastChunk = lines.pop() || '';
      const line = lines.filter(l => l.trim()).pop();
      if (line) installState.lastLine = line.trim().slice(0, 300);
    };
    child.stdout.on('data', onData);
    child.stderr.on('data', onData);
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${step} failed (exit ${code}): ${installState.lastLine}`));
    });
  });
}

/**
 * Bootstrap the MuScriptor venv. Returns immediately if already running.
 * Progress is exposed via getStatus(); this is a fire-and-poll API.
 */
export function startInstall(): { started: boolean; error?: string } {
  if (installState.installing) return { started: false, error: 'Install already in progress' };
  if (isInstalled()) return { started: false, error: 'Already installed' };

  const python = findSystemPython();
  if (!python) {
    const msg = 'No suitable Python found. MuScriptor requires Python 3.10–3.12 on PATH (or the Windows "py" launcher).';
    installState.error = msg;
    return { started: false, error: msg };
  }

  installState.installing = true;
  installState.error = undefined;

  (async () => {
    try {
      fs.mkdirSync(MUSCRIPTOR_DIR, { recursive: true });
      // Recreate a half-built venv from scratch so a previously failed
      // install can't leave us wedged.
      if (fs.existsSync(VENV_DIR) && !isInstalled()) {
        fs.rmSync(VENV_DIR, { recursive: true, force: true });
      }
      await runStep(python.cmd, [...python.args, '-m', 'venv', VENV_DIR], `Creating venv (Python ${python.version})`);
      await runStep(venvPython(), ['-m', 'pip', 'install', '--upgrade', 'pip'], 'Upgrading pip');
      await runStep(venvPython(), ['-m', 'pip', 'install', 'muscriptor'], 'Installing MuScriptor (this can take several minutes)');
      if (!isInstalled()) {
        throw new Error('pip finished but the muscriptor CLI was not found in the venv');
      }
      installState.step = 'done';
      console.log('[MidiStudio] Setup complete — MuScriptor installed');
    } catch (err: any) {
      installState.error = err.message || 'Install failed';
      console.error(`[MidiStudio] Setup FAILED: ${installState.error}`);
    } finally {
      installState.installing = false;
    }
  })();

  return { started: true };
}

// ── Transcription ────────────────────────────────────────────────────────

export interface TranscribeHandle {
  child: ChildProcess;
  done: Promise<void>;
}

/**
 * Spawn `muscriptor transcribe <audio> -o <mid> --model <m>`.
 * onLine receives progress lines (weight download + transcription progress).
 * The first run of a given model downloads its weights from Hugging Face.
 */
export function transcribe(
  audioPath: string,
  outMidPath: string,
  model: MuscriptorModel,
  onLine: (line: string) => void,
): TranscribeHandle {
  const args = ['transcribe', audioPath, '-o', outMidPath, '--model', model];
  console.log(`[MidiStudio] Transcribe: muscriptor ${args.join(' ')}`);
  // Weights are gated on HF — pass the stored token (both env names:
  // huggingface_hub accepts HF_TOKEN, older versions HUGGING_FACE_HUB_TOKEN)
  const hfToken = getHfToken();
  const child = spawn(muscriptorBin(), args, {
    windowsHide: true,
    env: hfToken
      ? { ...process.env, HF_TOKEN: hfToken, HUGGING_FACE_HUB_TOKEN: hfToken }
      : undefined,
  });

  const tail: string[] = [];
  let lastChunk = '';
  const onData = (buf: Buffer) => {
    const lines = (lastChunk + buf.toString('utf-8')).split(/[\r\n]+/);
    lastChunk = lines.pop() || '';
    for (const raw of lines) {
      const line = raw.trim();
      if (!line) continue;
      tail.push(line);
      if (tail.length > 20) tail.shift();
      onLine(line.slice(0, 300));
    }
  };
  child.stdout?.on('data', onData);
  child.stderr?.on('data', onData);

  const done = new Promise<void>((resolve, reject) => {
    child.on('error', (err) => reject(new Error(`Failed to launch muscriptor: ${err.message}`)));
    child.on('close', (code, signal) => {
      if (code === 0) resolve();
      else if (signal) reject(new Error(`Cancelled (${signal})`));
      else reject(new Error(`muscriptor exited with code ${code}: ${tail.slice(-5).join(' | ')}`));
    });
  });

  return { child, done };
}
