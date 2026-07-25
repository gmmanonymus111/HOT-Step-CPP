// adapters.ts — Adapter filesystem browsing and scanning
//
// Provides server-side endpoints for:
//   1. GET /browse — directory navigation with file type filtering
//   2. POST /scan  — flat listing of .safetensors files in a folder

import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { config } from '../config.js';

const router = Router();

/** Extensions accepted per filter category */
const FILTER_EXTENSIONS: Record<string, string[]> = {
  adapters: ['.safetensors'],
  audio: ['.wav', '.mp3', '.flac', '.ogg', '.opus'],
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
 * Flat scan of a single directory for .safetensors adapter files.
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
        return { name: e.name, path: fullPath, size: stat.size };
      });
    res.json({ files });
  } catch {
    res.json({ files: [] });
  }
});

/**
 * GET /api/adapters/lm
 *
 * Lists planner-LM adapters (local HOT-Step feature) from the adapters
 * root's `lm/` subtree: PEFT directories (adapter_model.safetensors +
 * adapter_config.json) and bare .safetensors files.  Filesystem-based so
 * freshly trained adapters appear WITHOUT an engine restart — the UI sends
 * the absolute path and the engine's path-fallback resolver loads it.
 *
 * Response: { root: string, adapters: { name, path, kind, size, mtime }[] }
 */
router.get('/lm', (_req, res) => {
  const root = path.join(config.aceServer.adapters, 'lm');
  const adapters: Array<{ name: string; path: string; kind: 'peft' | 'safetensors'; size: number; mtime: number }> = [];
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
              adapters.push({ name: entry.name, path: fullPath, kind: 'peft', size: stat.size, mtime: stat.mtimeMs });
            }
          } else if (entry.isFile() && entry.name.endsWith('.safetensors')) {
            const stat = fs.statSync(fullPath);
            adapters.push({ name: entry.name, path: fullPath, kind: 'safetensors', size: stat.size, mtime: stat.mtimeMs });
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
