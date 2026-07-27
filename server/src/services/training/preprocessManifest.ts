// training/preprocessManifest.ts — writes the job's preprocess_manifest.json
//
// The manifest is the ONLY thing ace-train reads about the dataset: it carries
// the resolved absolute audio paths, the labelled fields, and the cache file
// stem for every song. `id` is taken from the built dataset.json so a rebuild
// never renames an existing cache file.
//
// Spec: docs/plans/2026-07-27-preprocess-implementation.md §2.3, §4.3

import fs from 'fs';
import path from 'path';
import { datasetSampleId, jobsDir } from './paths.js';
import type { TagPosition, TrainingDatasetRow, TrainingSample } from './types.js';

export interface PreprocessManifestSample {
  id: string;
  audioPath: string;
  filename: string;
  relPath: string;
  outStem: string;
  srcBytes: number;
  srcMtimeMs: number;
  caption: string;
  genre: string;
  lyrics: string;
  bpm: number;
  keyscale: string;
  timesignature: string;
  duration: number;
  language: string;
  isInstrumental: boolean;
  customTag: string;
  tagPosition: string;
  promptOverride: string | null;
  repeat: number;
}

export interface PreprocessManifestFile {
  version: number;
  datasetId: string;
  slug: string;
  name: string;
  sourceDir: string;
  datasetJsonPath: string;
  customTag: string;
  tagPosition: TagPosition;
  genreRatio: number;
  samples: PreprocessManifestSample[];
}

/**
 * P6 cache-stem rule — MUST stay byte-identical to `pm_safe_stem` in
 * engine/src/train/preprocess-io.h (§6.1 has the six shared test vectors).
 *
 * relPath → extension stripped → '/' becomes '__' → every char outside
 * [A-Za-z0-9._-] becomes '_' → truncated to 96 chars.
 */
export function safeStem(relPath: string): string {
  const s = String(relPath ?? '').replace(/\\/g, '/');
  const slash = s.lastIndexOf('/');
  const dir = slash >= 0 ? s.slice(0, slash + 1) : '';
  const base = slash >= 0 ? s.slice(slash + 1) : s;
  const dot = base.lastIndexOf('.');
  const stem = dot > 0 ? base.slice(0, dot) : base;
  return (dir + stem)
    .replace(/\//g, '__')
    .replace(/[^A-Za-z0-9._-]/g, '_')
    .slice(0, 96);
}

/** `id` for a sample: the built dataset.json row wins, so cache names are
 *  stable across rebuilds; otherwise the deterministic relPath hash (D9). */
function idIndex(datasetJson: unknown): Map<string, string> {
  const byPath = new Map<string, string>();
  const root = datasetJson as { samples?: unknown } | null;
  const rows = root && Array.isArray(root.samples) ? root.samples : [];
  for (const row of rows) {
    if (!row || typeof row !== 'object') continue;
    const r = row as Record<string, unknown>;
    const audioPath = typeof r.audio_path === 'string' ? r.audio_path : '';
    const id = typeof r.id === 'string' ? r.id : '';
    if (audioPath && id) byPath.set(audioPath.toLowerCase(), id);
  }
  return byPath;
}

export function buildPreprocessManifest(
  ds: TrainingDatasetRow,
  samples: TrainingSample[],
  datasetJson: unknown,
): PreprocessManifestFile {
  const ids = idIndex(datasetJson);

  const rows: PreprocessManifestSample[] = samples.map((s) => {
    const id = ids.get(s.audioPath.toLowerCase()) || datasetSampleId(s.relPath);

    let srcBytes = s.sizeBytes;
    let srcMtimeMs = 0;
    try {
      const st = fs.statSync(s.audioPath);
      srcBytes = st.size;
      srcMtimeMs = Math.round(st.mtimeMs);
    } catch { /* file vanished between the scan and now — the engine reports it */ }

    return {
      id,
      audioPath: s.audioPath,
      filename: s.filename,
      relPath: s.relPath,
      outStem: `${safeStem(s.relPath)}-${id}`,
      srcBytes,
      srcMtimeMs,
      caption: s.caption,
      genre: s.genre,
      // Mirrors datasetBuilder: an empty lyric field is the instrumental marker.
      lyrics: s.lyrics.trim() ? s.lyrics : '[Instrumental]',
      bpm: s.bpm ?? 0,
      keyscale: s.key,
      timesignature: s.signature,
      duration: Math.round(s.duration),
      language: s.language,
      isInstrumental: s.isInstrumental,
      customTag: s.customTag,
      tagPosition: '',              // '' = inherit the dataset-level value
      promptOverride: s.promptOverride,
      repeat: s.repeat,
    };
  });

  return {
    version: 1,
    datasetId: ds.id,
    slug: ds.slug,
    name: ds.name,
    sourceDir: ds.sourceDir,
    datasetJsonPath: ds.datasetJsonPath,
    customTag: ds.customTag,
    tagPosition: ds.tagPosition,
    genreRatio: ds.genreRatio,
    samples: rows,
  };
}

/** Writes <jobsDir()>/<jobId>/preprocess_manifest.json; returns the absolute path. */
export function writePreprocessManifest(jobId: string, manifest: PreprocessManifestFile): string {
  const dir = path.join(jobsDir(), jobId);
  fs.mkdirSync(dir, { recursive: true });
  const target = path.join(dir, 'preprocess_manifest.json');
  fs.writeFileSync(target, JSON.stringify(manifest, null, 2), 'utf-8');
  return target;
}
