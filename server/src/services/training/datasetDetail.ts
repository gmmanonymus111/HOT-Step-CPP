// training/datasetDetail.ts — fresh-scan dataset detail + counter sync
//
// Lifted verbatim out of routes/training.ts: the batch pipeline needs the same
// payload GET /datasets/:id returns (its post-label quality gate reads
// labeledCount), and creating a dataset returns it too — neither can go back
// through HTTP for it.
//
// Spec: docs/plans/2026-07-27-dataset-studio-implementation.md §2.0

import { buildSamples } from './datasetScan.js';
import * as repo from './datasetsRepo.js';
import * as queue from './labelingQueue.js';
import { countPreprocessedVariants } from './preprocessStatus.js';
import type { TrainingDatasetDetail, TrainingDatasetRow, TrainingSample } from './types.js';

/**
 * The dataset status the fresh scan implies (§2.0). Both the response body and
 * the DB cache are derived from this, so a GET can never report the pre-sync
 * value (e.g. 'labeling' for one call after a job ended).
 */
export function computeStatus(
  ds: TrainingDatasetRow,
  samples: TrainingSample[],
): TrainingDatasetRow['status'] {
  if (ds.status === 'built') return 'built';
  if (queue.activeJobForDataset(ds.id)) return 'labeling';
  return samples.some(s => !!s.caption.trim()) ? 'labeled' : 'draft';
}

/** Build the full detail payload — always a fresh scan (D3). */
export async function detailFor(ds: TrainingDatasetRow): Promise<TrainingDatasetDetail> {
  const warnings: string[] = [];
  const samples = await buildSamples(ds, { warnings });

  const included = samples.filter(s => !s.excluded);
  if (included.length > 0 && included.length < 10) {
    warnings.push(`Only ${included.length} samples — 10+ recommended`);
  }
  const missing = included.filter(s => s.fileMissing).length;
  if (missing > 0) warnings.push(`${missing} file(s) are missing from disk`);

  const active = queue.activeJobForDataset(ds.id);

  let preprocessedVariants = 0;
  try { preprocessedVariants = countPreprocessedVariants(ds.slug); } catch { /* stays 0 */ }

  return {
    ...ds,
    // The DB counters are a cache; the fresh scan is what the client sees.
    sampleCount: samples.length,
    labeledCount: samples.filter(s => !!s.caption.trim()).length,
    excludedCount: samples.filter(s => s.excluded).length,
    status: computeStatus(ds, samples),
    samples,
    warnings,
    activeJobId: active ? active.id : null,
    preprocessedVariants,
  };
}

/** Refresh the cached counters after a scan. */
export function syncCounters(ds: TrainingDatasetRow, samples: TrainingSample[]): void {
  const labeled = samples.filter(s => !!s.caption.trim()).length;
  const excluded = samples.filter(s => s.excluded).length;
  const status = computeStatus(ds, samples);
  try {
    repo.updateCounters(ds.id, {
      sampleCount: samples.length,
      labeledCount: labeled,
      excludedCount: excluded,
      status,
    });
  } catch (err: any) {
    console.warn(`[Training] Counter sync failed: ${err.message}`);
  }
}
