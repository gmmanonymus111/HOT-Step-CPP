// adapters.ts — Adapter filesystem browsing and scanning
//
// Provides server-side endpoints for:
//   1. GET /browse — directory navigation with file type filtering
//   2. POST /scan  — flat listing of .safetensors files in a folder

import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { config } from '../config.js';
import { readAdapterTrigger } from '../services/adapters/stMetadata.js';

const router = Router();

/** Extensions accepted per filter category */
const FILTER_EXTENSIONS: Record<string, string[]> = {
  adapters: ['.safetensors'],
  audio: ['.wav', '.mp3', '.flac', '.ogg', '.opus'],
  // Training Studio folder picker — deliberately separate from `audio`, which
  // other callers depend on staying as-is.
  trainingAudio: ['.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a', '.aac'],
};

/**
 * GET /api/adapters/browse?path=...&filter=adapters
 *
 * Lists the contents of a directory, returning sub-directories and
 * files that match the optional filter.  Always includes a '..'
 * parent entry unless already at a filesystem root.
 *
 * Response: { current: string, entries: BrowseEntry[] }
 */
router.get('/browse', (req, res) => {
  const rawPath = (req.query.path as string) || '';
  const filter = (req.query.filter as string) || '';
  const allowedExts = FILTER_EXTENSIONS[filter] || [];

  // Resolve to an absolute path
  let dirPath: string;
  try {
    dirPath = rawPath ? path.resolve(rawPath) : path.resolve('.');
  } catch {
    res.status(400).json({ error: 'Invalid path' });
    return;
  }

  // Verify path exists and is a directory
  if (!fs.existsSync(dirPath) || !fs.statSync(dirPath).isDirectory()) {
    res.status(404).json({ error: 'Directory not found', current: dirPath, entries: [] });
    return;
  }

  try {
    const rawEntries = fs.readdirSync(dirPath, { withFileTypes: true });
    const entries: Array<{ name: string; path: string; type: 'dir' | 'file'; size?: number }> = [];

    // Parent directory (unless at root)
    const parent = path.dirname(dirPath);
    if (parent !== dirPath) {
      entries.push({ name: '..', path: parent, type: 'dir' });
    }

    // Directories first (skip hidden)
    for (const entry of rawEntries) {
      if (entry.name.startsWith('.')) continue;
      const fullPath = path.join(dirPath, entry.name);
      if (entry.isDirectory()) {
        entries.push({ name: entry.name, path: fullPath, type: 'dir' });
      }
    }

    // Then files (filtered, skip hidden)
    for (const entry of rawEntries) {
      if (entry.name.startsWith('.')) continue;
      if (!entry.isFile()) continue;
      const fullPath = path.join(dirPath, entry.name);
      const ext = path.extname(entry.name).toLowerCase();
      if (allowedExts.length === 0 || allowedExts.includes(ext)) {
        try {
          const stat = fs.statSync(fullPath);
          entries.push({ name: entry.name, path: fullPath, type: 'file', size: stat.size });
        } catch {
          // Skip files we can't stat (locked, permissions, etc.)
        }
      }
    }

    res.json({ current: dirPath, entries });
  } catch (err: any) {
    res.status(500).json({ error: err.message, current: dirPath, entries: [] });
  }
});

/**
 * POST /api/adapters/scan
 *
 * Flat scan of a single directory for adapters, in two forms:
 *   * bare `.safetensors` files (the hand-installed convention), and
 *   * PEFT sub-directories — `adapter_model.safetensors` + `adapter_config.json`
 *     — which is what the DiT trainer writes.
 *
 * The PEFT half is what makes a freshly trained adapter appear in the Create
 * view's dropdown WITHOUT an engine restart: `path` is the directory, and the
 * engine's path-fallback resolver already accepts a PEFT dir (adapter-merge.h).
 * Without it a just-finished training run would be invisible until relaunch.
 *
 * Returns an empty array if the folder doesn't exist or is empty.
 *
 * Body: { folder: string }
 * Response: { files: AdapterFile[] }
 */
router.post('/scan', (req, res) => {
  const folder = req.body?.folder;
  if (!folder || typeof folder !== 'string') {
    res.json({ files: [] });
    return;
  }

  const dirPath = path.resolve(folder);
  if (!fs.existsSync(dirPath) || !fs.statSync(dirPath).isDirectory()) {
    res.json({ files: [] });
    return;
  }

  try {
    const rawEntries = fs.readdirSync(dirPath, { withFileTypes: true });
    const files = rawEntries
      .filter(e => e.isFile() && e.name.endsWith('.safetensors'))
      .map(e => {
        const fullPath = path.join(dirPath, e.name);
        const stat = fs.statSync(fullPath);
        const tg = readAdapterTrigger(fullPath);
        return { name: e.name, path: fullPath, size: stat.size, trigger: tg.trigger, triggerPosition: tg.position };
      });

    // PEFT directories, mirroring GET /adapters/lm. Both files are required:
    // a dir with weights but no adapter_config.json loads with its alpha/rank
    // scale silently degraded to 1.0, so it must not be offered as ready.
    for (const e of rawEntries) {
      if (!e.isDirectory() || e.name.startsWith('.')) continue;
      const dir = path.join(dirPath, e.name);
      const model = path.join(dir, 'adapter_model.safetensors');
      if (!fs.existsSync(model) || !fs.existsSync(path.join(dir, 'adapter_config.json'))) continue;
      try {
        const tg = readAdapterTrigger(dir);
        files.push({ name: e.name, path: dir, size: fs.statSync(model).size, trigger: tg.trigger,
                     triggerPosition: tg.position });
      } catch { /* skip */ }
    }
    files.sort((a, b) => a.name.localeCompare(b.name));

    res.json({ files });
  } catch {
    res.json({ files: [] });
  }
});

/**
 * GET /api/adapters/lm?folder=...
 *
 * Lists planner-LM adapters (local HOT-Step feature): PEFT directories
 * (adapter_model.safetensors + adapter_config.json) and bare .safetensors
 * files.  Scans `folder` when given (the user's archive, like the DiT
 * adapter folder), else the adapters root's `lm/` subtree.  Filesystem-based
 * so freshly trained adapters appear WITHOUT an engine restart — the UI
 * sends the absolute path and the engine's path-fallback resolver loads it.
 *
 * Response: { root: string, adapters: { name, path, kind, size, mtime }[] }
 */
router.get('/lm', (req, res) => {
  const folderParam = (req.query.folder as string) || '';
  const root = folderParam ? path.resolve(folderParam) : path.join(config.aceServer.adapters, 'lm');
  const adapters: Array<{
    name: string; path: string; kind: 'peft' | 'safetensors'; size: number; mtime: number;
    trigger: string; triggerPosition: 'prepend' | 'append' | '';
  }> = [];
  try {
    if (fs.existsSync(root) && fs.statSync(root).isDirectory()) {
      for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
        if (entry.name.startsWith('.')) continue;
        const fullPath = path.join(root, entry.name);
        try {
          if (entry.isDirectory()) {
            const model = path.join(fullPath, 'adapter_model.safetensors');
            if (fs.existsSync(model)) {
              const stat = fs.statSync(model);
              const tg = readAdapterTrigger(fullPath);
              adapters.push({ name: entry.name, path: fullPath, kind: 'peft', size: stat.size, mtime: stat.mtimeMs,
                              trigger: tg.trigger, triggerPosition: tg.position });
            }
          } else if (entry.isFile() && entry.name.endsWith('.safetensors')) {
            const stat = fs.statSync(fullPath);
            const tg = readAdapterTrigger(fullPath);
            adapters.push({ name: entry.name, path: fullPath, kind: 'safetensors', size: stat.size,
                            mtime: stat.mtimeMs, trigger: tg.trigger, triggerPosition: tg.position });
          }
        } catch { /* skip unreadable entries */ }
      }
    }
    adapters.sort((a, b) => a.name.localeCompare(b.name));
    res.json({ root, adapters });
  } catch (err: any) {
    res.json({ root, adapters: [], error: err.message });
  }
});

export default router;
