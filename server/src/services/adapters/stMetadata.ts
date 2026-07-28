/**
 * Read the `__metadata__` header of a safetensors file — the slot HOT-Step's
 * trainers use to record an adapter's trigger word.
 *
 * The format is: 8 bytes little-endian u64 header length, then that many bytes
 * of JSON, then the tensor payload. Unknown metadata keys are ignored by every
 * other consumer (PEFT, ComfyUI, LyCORIS, Side-Step, diffusers), which is why
 * the trigger lives here and not in `adapter_config.json`.
 *
 * docs/plans/2026-07-28-adapter-trigger-embedding.md §2.1 / §4
 *
 * Nothing in this module throws. An unreadable, truncated or non-safetensors
 * file degrades to "no metadata" — an adapter without a trigger is a normal
 * state, not an error.
 */
import fs from 'fs';
import path from 'path';

/** Refuse to buffer a bogus header length. Real headers are well under a MB. */
const MAX_HEADER_BYTES = 64 * 1024 * 1024;

export interface AdapterTrigger {
  /** '' when the adapter carries no embedded trigger. */
  trigger: string;
  /** 'prepend' | 'append', or '' when there is no trigger. */
  position: 'prepend' | 'append' | '';
}

const EMPTY: AdapterTrigger = { trigger: '', position: '' };

/** `path -> {size, mtimeMs, value}` so a directory scan reads each file once. */
const cache = new Map<string, { size: number; mtimeMs: number; value: AdapterTrigger }>();

/** Read only the `__metadata__` object. Returns `{}` on any failure. */
export function readSafetensorsMetadata(file: string): Record<string, string> {
  let fd: number | undefined;
  try {
    fd = fs.openSync(file, 'r');
    const lenBuf = Buffer.allocUnsafe(8);
    if (fs.readSync(fd, lenBuf, 0, 8, 0) !== 8) return {};
    const headerLen = Number(lenBuf.readBigUInt64LE(0));
    if (!Number.isSafeInteger(headerLen) || headerLen <= 0 || headerLen > MAX_HEADER_BYTES) return {};

    const hdr = Buffer.allocUnsafe(headerLen);
    if (fs.readSync(fd, hdr, 0, headerLen, 8) !== headerLen) return {};

    const parsed: unknown = JSON.parse(hdr.toString('utf8'));
    if (!parsed || typeof parsed !== 'object') return {};
    const md = (parsed as Record<string, unknown>).__metadata__;
    if (!md || typeof md !== 'object') return {};

    const out: Record<string, string> = {};
    for (const [k, v] of Object.entries(md as Record<string, unknown>)) {
      if (typeof v === 'string') out[k] = v;
    }
    return out;
  } catch {
    return {};
  } finally {
    if (fd !== undefined) {
      try { fs.closeSync(fd); } catch { /* ignore */ }
    }
  }
}

/**
 * Resolve an adapter's embedded trigger. Accepts either a bare `.safetensors`
 * file or an adapter directory (in which case `adapter_model.safetensors` inside
 * it is read, falling back to `lokr_weights.safetensors` — the LyCORIS leaf a
 * DiT LoKR export writes instead, carrying the same trigger keys). Memoised on
 * path + size + mtime, so re-scanning a folder of 200 adapters costs one stat
 * each after the first pass.
 */
export function readAdapterTrigger(pathOrDir: string): AdapterTrigger {
  try {
    let file = pathOrDir;
    const st = fs.statSync(pathOrDir);
    if (st.isDirectory()) {
      file = path.join(pathOrDir, 'adapter_model.safetensors');
      if (!fs.existsSync(file)) file = path.join(pathOrDir, 'lokr_weights.safetensors');
    }

    const fst = file === pathOrDir ? st : fs.statSync(file);
    if (!fst.isFile()) return EMPTY;

    const hit = cache.get(file);
    if (hit && hit.size === fst.size && hit.mtimeMs === fst.mtimeMs) return hit.value;

    const md = readSafetensorsMetadata(file);
    // `hot_step_trigger` is authoritative; `modelspec.trigger_phrase` is the
    // ecosystem convention we also write, and lets us read adapters stamped by
    // kohya-style tooling that never heard of HOT-Step.
    const trigger = (md.hot_step_trigger || md['modelspec.trigger_phrase'] || '').trim();
    const raw = (md.hot_step_trigger_position || '').trim();
    const position: AdapterTrigger['position'] =
      !trigger ? '' : raw === 'append' ? 'append' : 'prepend';

    const value: AdapterTrigger = trigger ? { trigger, position } : EMPTY;
    cache.set(file, { size: fst.size, mtimeMs: fst.mtimeMs, value });
    return value;
  } catch {
    return EMPTY;
  }
}
