// training/auditionStore.ts — the codes-audition preview filesystem layer
//
// data/training/previews/<previewId>/{base.wav, adapter.wav, preview.json}
//
// There is NO SQLite row (C9). A preview's whole state is its preview.json,
// exactly as a training run's state is its lm_train_log.json. The json is
// written LAST, so "preview.json exists" is the definition of a complete dir —
// anything else is garbage and is eligible for pruning at any time.
//
// Never throws upward. Every function is total: a corrupt preview.json is
// skipped, a locked file is left for the next prune, and a failed write is
// logged at console.warn. A preview is a convenience, never a job's success
// criterion.
//
// Spec: docs/plans/2026-07-28-codes-preview.md §5.2, C9, C11

import fs from 'fs';
import path from 'path';
import { config } from '../../config.js';
import type { AuditionPreview, AuditionSlot } from './types.js';

/** C11, FROZEN: keep 40 previews / 1.0 GB, whichever bites first. */
const MAX_PREVIEWS = 40;
const MAX_BYTES = 1024 * 1024 * 1024;

const PREVIEW_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const SLOT_RE = /^(base|adapter)$/;
// A slot's stored audio: the codes sketch (`base.wav`) or its opt-in DiT
// render (`base-render.wav`). Both are addressed by the same static route.
const FILE_KEY_RE = /^(base|adapter)(-render)?$/;

export function previewsRoot(): string {
  return path.join(config.training.dir, 'previews');
}

/** `<previews>/<previewId>`. The id is uuid-validated here as well as by the
 *  caller — this path is handed to fs.rmSync and to a static file route. */
export function previewDir(previewId: string): string {
  if (!isPreviewId(previewId)) return '';
  return path.join(previewsRoot(), previewId);
}

export function isPreviewId(value: unknown): value is string {
  return typeof value === 'string' && PREVIEW_ID_RE.test(value);
}

export function isPreviewSlot(value: unknown): value is AuditionSlot {
  return typeof value === 'string' && SLOT_RE.test(value);
}

/** slot OR slot-render — everything the static preview route may serve. */
export function isPreviewFileKey(value: unknown): value is string {
  return typeof value === 'string' && FILE_KEY_RE.test(value);
}

/** `<previews>/<id>/<key>.<ext>` — both tokens validated, '' when either is not. */
export function previewFile(previewId: string, key: string, ext = 'wav'): string {
  const dir = previewDir(previewId);
  if (!dir || !isPreviewFileKey(key)) return '';
  const safeExt = ext === 'mp3' ? 'mp3' : 'wav';
  return path.join(dir, `${key}.${safeExt}`);
}

/** The file actually on disk for a slot/key, whichever extension it was
 *  written with. '' when nothing is there. */
export function resolvePreviewFile(previewId: string, key: string): string {
  for (const ext of ['wav', 'mp3']) {
    const p = previewFile(previewId, key, ext);
    if (p && fs.existsSync(p)) return p;
  }
  return '';
}

export interface PreviewAudio {
  slot: AuditionSlot;
  buf: Buffer;
  ext: 'wav' | 'mp3';
  /** True for the opt-in DiT render — stored as `<slot>-render.<ext>`. */
  render?: boolean;
}

/**
 * Write one preview dir: audio files first, `preview.json` LAST, then prune.
 *
 * Returns false when nothing could be written — the caller still has its
 * in-memory AuditionPreview and reports the run honestly; it simply has no
 * playable URL.
 */
export function writePreview(rec: AuditionPreview, audio: PreviewAudio[]): boolean {
  const dir = previewDir(rec.previewId);
  if (!dir) {
    console.warn(`[Training] Refusing to write preview with a non-uuid id: ${rec.previewId}`);
    return false;
  }
  try {
    fs.mkdirSync(dir, { recursive: true });
    for (const a of audio) {
      const file = previewFile(rec.previewId, a.render ? `${a.slot}-render` : a.slot, a.ext);
      if (!file) continue;
      fs.writeFileSync(file, a.buf);
    }
    // Last — the completeness marker.
    fs.writeFileSync(path.join(dir, 'preview.json'), JSON.stringify(rec, null, 2), 'utf-8');
  } catch (err: any) {
    console.warn(`[Training] Could not write preview ${rec.previewId}: ${err?.message || err}`);
    return false;
  }
  prunePreviews();
  return true;
}

export function readPreview(previewId: string): AuditionPreview | null {
  const dir = previewDir(previewId);
  if (!dir) return null;
  try {
    const raw = fs.readFileSync(path.join(dir, 'preview.json'), 'utf-8');
    const parsed = JSON.parse(raw) as AuditionPreview;
    if (!parsed || typeof parsed !== 'object' || !Array.isArray(parsed.sides)) return null;
    return parsed;
  } catch {
    return null;
  }
}

interface PreviewDirInfo {
  id: string;
  dir: string;
  mtimeMs: number;
  bytes: number;
  rec: AuditionPreview | null;
}

/** One readdir + stat pass. A dir with no/corrupt preview.json gets rec: null
 *  and mtimeMs 0, which sorts it to the end and makes it prune-eligible. */
function scanPreviewDirs(): PreviewDirInfo[] {
  const root = previewsRoot();
  const out: PreviewDirInfo[] = [];
  let entries: fs.Dirent[];
  try {
    if (!fs.existsSync(root)) return out;
    entries = fs.readdirSync(root, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const dir = path.join(root, entry.name);
    let mtimeMs = 0;
    let rec: AuditionPreview | null = null;
    try {
      mtimeMs = fs.statSync(path.join(dir, 'preview.json')).mtimeMs;
      rec = readPreview(entry.name);
    } catch { /* garbage dir — mtime 0, rec null */ }
    if (!rec) mtimeMs = 0;
    out.push({ id: entry.name, dir, mtimeMs, bytes: dirBytes(dir), rec });
  }
  return out;
}

function dirBytes(dir: string): number {
  let total = 0;
  try {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (!entry.isFile()) continue;
      try { total += fs.statSync(path.join(dir, entry.name)).size; } catch { /* vanished */ }
    }
  } catch { /* unreadable */ }
  return total;
}

function removeDir(dir: string): void {
  // preview.json goes FIRST, as its own operation. rmSync walks entries in
  // readdir order and throws on the first file it cannot unlink, so a browser
  // holding a read handle on `base.wav` (Windows opens it without
  // FILE_SHARE_DELETE) aborts the walk PART WAY THROUGH — alphabetically
  // `adapter.wav` is already gone and `preview.json` has not been reached. The
  // dir then still parses, still passes listPreviews' filter, and still renders
  // a player whose audioUrl 404s.
  //
  // Removing the completeness marker first makes that partial failure degrade
  // to a garbage dir: invisible to listPreviews, mtime 0, and prune-eligible on
  // the next pass — which is what the "skipped and retried next prune" comment
  // always claimed happened.
  try {
    fs.rmSync(path.join(dir, 'preview.json'), { force: true });
  } catch { /* already gone, or locked — the rmSync below still tries */ }
  try {
    fs.rmSync(dir, { recursive: true, force: true });
  } catch (err: any) {
    console.warn(`[Training] Could not prune preview dir ${dir}: ${err?.message || err}`);
  }
}

/**
 * C11, FROZEN. Sort by preview.json mtime DESC; delete every dir past index 39;
 * then walk the survivors accumulating bytes and delete every dir past the point
 * where the running total exceeds 1.0 GB.
 *
 * Called lazily on every preview write and on every GET …/audition — there is
 * deliberately no boot hook (that would mean editing index.ts for nothing).
 */
export function prunePreviews(): void {
  try {
    const dirs = scanPreviewDirs();
    dirs.sort((a, b) => b.mtimeMs - a.mtimeMs);

    const survivors: PreviewDirInfo[] = [];
    for (let i = 0; i < dirs.length; i++) {
      if (i >= MAX_PREVIEWS) { removeDir(dirs[i].dir); continue; }
      survivors.push(dirs[i]);
    }

    let running = 0;
    let overflowed = false;
    for (const info of survivors) {
      if (overflowed) { removeDir(info.dir); continue; }
      running += info.bytes;
      // "past the point where the running total exceeds 1.0 GB" — the dir that
      // crosses the line is kept, everything after it goes.
      if (running > MAX_BYTES) overflowed = true;
    }
  } catch (err: any) {
    console.warn(`[Training] Preview prune failed: ${err?.message || err}`);
  }
}

/** Newest-first previews for one dataset. Corrupt dirs are silently skipped. */
export function listPreviews(datasetId: string, limit = 20): AuditionPreview[] {
  try {
    const out: AuditionPreview[] = [];
    for (const info of scanPreviewDirs()) {
      if (!info.rec) continue;
      if (info.rec.datasetId !== datasetId) continue;
      out.push(info.rec);
    }
    out.sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));
    const n = Number.isFinite(limit) ? Math.max(1, Math.min(100, Math.trunc(limit))) : 20;
    return out.slice(0, n);
  } catch (err: any) {
    console.warn(`[Training] Preview list failed: ${err?.message || err}`);
    return [];
  }
}

/** Swept by DELETE /datasets/:id — a deleted dataset leaves no orphan audio. */
export function deleteDatasetPreviews(datasetId: string): void {
  try {
    for (const info of scanPreviewDirs()) {
      if (!info.rec || info.rec.datasetId !== datasetId) continue;
      removeDir(info.dir);
    }
  } catch (err: any) {
    console.warn(`[Training] Preview sweep for dataset ${datasetId} failed: ${err?.message || err}`);
  }
}
