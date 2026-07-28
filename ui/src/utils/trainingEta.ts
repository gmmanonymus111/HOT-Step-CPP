// trainingEta.ts — "when will this run hit the target loss?"
//
// JobProgress's generic ETA is a per-ITEM pace estimate (elapsed / completed),
// which is meaningless for a training job: `done`/`total` there count optimizer
// steps or nothing at all, and what a user actually wants to know is whether the
// loss is still falling fast enough to reach the target before the epoch cap.
//
// Everything here is derived from the epoch series the SSE `metric:'epoch'`
// events already carry (epoch, loss, ms) — no new engine or server field.

/** The slice of TrainLmEpoch / TrainDitEpoch this module needs. Both are
 *  structurally identical, so one shape covers LM and DiT. */
export interface EtaEpochPoint {
  epoch: number;
  loss: number;
  ms: number;
}

export type TrainingEta =
  /** Fewer than MIN_EPOCHS epochs, or no usable epoch duration yet. */
  | { kind: 'estimating' }
  /** Loss is descending fast enough to reach `target` before the cap. */
  | { kind: 'target'; etaMs: number; target: number }
  /** Descent has stalled (or is too slow to make the cap) — show the cap instead. */
  | { kind: 'stalled'; etaMs: number; target: number }
  /** No target set, or the target is already met: time to the epoch cap. */
  | { kind: 'cap'; etaMs: number }
  /** Nothing sensible to say (no cap known and no target to chase). */
  | { kind: 'none' };

/** Epochs needed before a rate is worth showing. Two epochs is one delta. */
const MIN_EPOCHS = 3;
/** How many trailing epoch deltas the descent rate is averaged over. */
const RATE_WINDOW = 5;
/** Below this per-epoch descent the curve counts as flat. */
const STALL_RATE = 1e-3;

function mean(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length;
}

/**
 * Mean per-epoch descent over the last RATE_WINDOW deltas.
 *
 * DEVIATION from a literal "mean of the last 5 POSITIVE deltas": the window is
 * the last 5 deltas and negatives are dropped rather than searched past. Taking
 * the last 5 positive deltas wherever they occur would keep reporting a healthy
 * rate for a curve that plateaued twenty epochs ago — exactly the case the
 * stalled state exists to catch. Positives are still summed over the full window
 * size, so an all-descending window gives the plain mean of its five deltas.
 */
export function descentRate(losses: number[]): number {
  if (losses.length < 2) return 0;
  const deltas: number[] = [];
  for (let i = 1; i < losses.length; i++) deltas.push(losses[i - 1] - losses[i]);
  const window = deltas.slice(-RATE_WINDOW);
  const gained = window.reduce((a, d) => a + (d > 0 ? d : 0), 0);
  return gained / window.length;
}

/** Mean wall-clock duration of the last RATE_WINDOW epochs that reported one. */
export function avgEpochMs(epochs: EtaEpochPoint[]): number {
  const durations = epochs
    .map(e => e.ms)
    .filter(ms => Number.isFinite(ms) && ms > 0)
    .slice(-RATE_WINDOW);
  return mean(durations);
}

/**
 * Estimate remaining time from the live epoch series.
 *
 * @param epochs     live epoch series, ascending by epoch (the store keeps it sorted)
 * @param targetLoss target loss; <= 0 means auto-stop is disabled
 * @param maxEpochs  the run's epoch cap; 0 = unknown
 */
export function computeTrainingEta(
  epochs: EtaEpochPoint[],
  targetLoss: number,
  maxEpochs: number,
): TrainingEta {
  const usable = epochs.filter(e => Number.isFinite(e.loss));
  if (usable.length < MIN_EPOCHS) return { kind: 'estimating' };

  const perEpochMs = avgEpochMs(usable);
  if (perEpochMs <= 0) return { kind: 'estimating' };

  const last = usable[usable.length - 1];
  const epochsLeft = maxEpochs > 0 ? Math.max(0, maxEpochs - last.epoch) : 0;
  const capMs = epochsLeft * perEpochMs;

  // No target (or already met) — the epoch cap is the only honest deadline.
  if (!(targetLoss > 0) || last.loss <= targetLoss) {
    return capMs > 0 ? { kind: 'cap', etaMs: capMs } : { kind: 'none' };
  }

  const rate = descentRate(usable.map(e => e.loss));
  if (rate > STALL_RATE) {
    const epochsToTarget = (last.loss - targetLoss) / rate;
    // A target the cap will arrive at first is not a target this run will hit,
    // and quoting a time past the cap would be a promise the job cannot keep.
    if (maxEpochs <= 0 || epochsToTarget <= epochsLeft) {
      return { kind: 'target', etaMs: epochsToTarget * perEpochMs, target: targetLoss };
    }
  }
  return { kind: 'stalled', etaMs: capMs, target: targetLoss };
}

/** Coarse duration label — same buckets as JobProgress's own formatEta. */
export function formatEtaMs(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return '';
  const m = Math.round(ms / 60000);
  if (m < 1) return '<1m';
  if (m < 60) return `${m}m`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}
