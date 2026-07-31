// training/activeModels.ts — server-side mirror of the model the user has
// selected in the Models tab.
//
// WHY THIS EXISTS
//
// The active DiT lives in the UI's globalParamsStore, which persists to
// localStorage (`hs-ditModel`) and nowhere else. The server therefore had no
// idea what the user had loaded, and the preprocess route fell back to
// `pickBf16(snap.dit)` — the FIRST BF16 name in the model catalogue, which is
// the stock base.
//
// That is how an overnight bulk run trained 18 datasets' worth of DiT adapters
// against the base model instead of the user's fine-tune: the adapter binds to
// whichever DiT produced the cached latents, so the whole run was wasted
// (2026-07-31).
//
// The engine's own `g_loaded_dit` is NOT a usable source here — training stops
// ace-server (stopEngine defaults true), so at the moment a training job needs
// the answer there is no engine to ask. A persisted setting is the only thing
// that survives that.

import { getSetting, setSetting } from '../../db/lireekDb.js';

const KEY = 'active_models_v1';

export interface ActiveModels {
  ditModel: string;
  lmModel: string;
  vaeModel: string;
  textEncoder: string;
}

const EMPTY: ActiveModels = { ditModel: '', lmModel: '', vaeModel: '', textEncoder: '' };

export function getActiveModels(): ActiveModels {
  try {
    const raw = getSetting(KEY, '');
    if (!raw) return { ...EMPTY };
    const p = JSON.parse(raw) as Partial<ActiveModels>;
    return {
      ditModel: typeof p.ditModel === 'string' ? p.ditModel : '',
      lmModel: typeof p.lmModel === 'string' ? p.lmModel : '',
      vaeModel: typeof p.vaeModel === 'string' ? p.vaeModel : '',
      textEncoder: typeof p.textEncoder === 'string' ? p.textEncoder : '',
    };
  } catch {
    return { ...EMPTY };
  }
}

/** Shallow-merge: the UI reports whichever field changed, not the whole set. */
export function setActiveModels(patch: Partial<ActiveModels>): ActiveModels {
  const merged = { ...getActiveModels() };
  for (const k of Object.keys(EMPTY) as Array<keyof ActiveModels>) {
    const v = patch[k];
    if (typeof v === 'string') merged[k] = v.trim();
  }
  setSetting(KEY, JSON.stringify(merged));
  return merged;
}

/**
 * The DiT to preprocess/train against, given the engine's catalogue.
 *
 * Order:
 *   1. the user's active DiT, if it is BF16 and still present
 *   2. its BF16 SIBLING — same model family, BF16 build. Training needs BF16
 *      (the mirror refuses a quantized base), so a user running a Q8 copy of
 *      their fine-tune should still train against THAT fine-tune, not the stock
 *      base. Matched by stripping a trailing quant token and looking for a
 *      BF16-suffixed name with the same stem.
 *   3. the previous behaviour — first BF16 in the catalogue
 *
 * Returns '' when the catalogue has no BF16 at all, which the caller already
 * treats as a hard error.
 */
export function resolveTrainingDit(active: string, catalogue: string[], fallback: string): string {
  const a = (active || '').trim();
  if (!a) return fallback;

  const isBf16 = (n: string) => /bf16/i.test(n);
  const present = (n: string) => catalogue.length === 0 || catalogue.includes(n);

  if (isBf16(a) && present(a)) return a;

  // Stem = the name minus extension and minus a trailing quant/precision token,
  // e.g. 'acestep-v15-mytune-Q8_0.gguf' -> 'acestep-v15-mytune'.
  const stem = a
    .replace(/\.(gguf|safetensors)$/i, '')
    .replace(/[-_](bf16|f16|fp16|f32|q\d+(_\w+)?|int8)$/i, '')
    .toLowerCase();
  if (stem) {
    const sibling = catalogue.find(n => isBf16(n)
      && n.replace(/\.(gguf|safetensors)$/i, '').replace(/[-_]bf16$/i, '').toLowerCase() === stem);
    if (sibling) return sibling;
  }

  return fallback;
}
