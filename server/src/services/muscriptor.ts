// muscriptor.ts — MuScriptor (audio→MIDI) shared helpers
//
// MuScriptor is developed by Kyutai & Mirelo (Rouard, Krause, Roebel,
// Simon-Gabriel, Défossez — arXiv:2607.08168). Code MIT; model weights
// CC BY-NC 4.0 and GATED on Hugging Face (users request access + read token).
//
// NOTE (2026-07-16): the original integration ran the upstream Python CLI in
// a managed venv. That approach was removed — transcription is being ported
// to a native GGML binary (`ace-midi`). Design: docs/plans/muscriptor-cpp-port.md.
// What remains here is the part the C++ path also needs: the Hugging Face
// token store used to download the gated weights.
//
// Rob's local venv at data/muscriptor/venv is intentionally NOT deleted —
// it is the numerics-validation oracle for the port (see design doc §6).

import path from 'path';
import fs from 'fs';
import { config } from '../config.js';

export const MUSCRIPTOR_DIR = path.join(config.data.dir, 'muscriptor');

export const MUSCRIPTOR_MODELS = ['small', 'medium', 'large'] as const;
export type MuscriptorModel = typeof MUSCRIPTOR_MODELS[number];

export const HF_MODEL_REPOS: Record<MuscriptorModel, string> = {
  small: 'MuScriptor/muscriptor-small',
  medium: 'MuScriptor/muscriptor-medium',
  large: 'MuScriptor/muscriptor-large',
};

// ── Hugging Face access token ────────────────────────────────────────────

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
