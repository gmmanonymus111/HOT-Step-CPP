// training/rateLimit.ts — per-service concurrency + pacing gates for labeling
//
// The Dataset Studio label pipeline talks to three services with wildly
// different cost profiles:
//
//   essentia — local ffmpeg + extractor, CPU-bound, no quota
//   genius   — HTML scrape, no published rate limit but trivially bannable
//   caption  — Gemini/OpenAI/etc, hard per-minute quotas, 5-15 s per audio call
//
// Before this existed every one of them ran strictly serially with a hardcoded
// courtesy sleep, so a 300-file folder cost 300 x (genius + gemini) in wall
// clock — hours, almost all of it spent waiting on someone else's server.
//
// Each service gets ONE process-wide limiter so the standalone Genius/Caption
// jobs and a full Label job cannot collectively exceed the budget by running at
// the same time (they now can — the queue has a network lane, see enqueue()).
//
// Two independent knobs per service:
//   concurrency   — how many calls may be IN FLIGHT at once
//   minIntervalMs — minimum spacing between call STARTS (pacing, not spacing
//                   between completions), which is what quota buckets measure
//
// concurrency 1 + minInterval N reproduces the old serial-with-sleep behaviour
// exactly, so it stays available as the safe fallback.

import { config, configReloadListeners } from '../../config.js';

export interface RateLimitSpec {
  concurrency: number;
  minIntervalMs: number;
}

export class RateLimiter {
  private concurrency: number;
  private minIntervalMs: number;
  private active = 0;
  private nextSlotAt = 0;
  private waiters: Array<() => void> = [];

  constructor(spec: RateLimitSpec, readonly name = '') {
    this.concurrency = Math.max(1, Math.trunc(spec.concurrency));
    this.minIntervalMs = Math.max(0, Math.trunc(spec.minIntervalMs));
  }

  /** Live-adjust. In-flight calls keep their slot; the new cap applies next. */
  update(spec: Partial<RateLimitSpec>): void {
    if (typeof spec.concurrency === 'number' && Number.isFinite(spec.concurrency)) {
      this.concurrency = Math.max(1, Math.trunc(spec.concurrency));
    }
    if (typeof spec.minIntervalMs === 'number' && Number.isFinite(spec.minIntervalMs)) {
      this.minIntervalMs = Math.max(0, Math.trunc(spec.minIntervalMs));
    }
    this.drain();
  }

  get spec(): RateLimitSpec {
    return { concurrency: this.concurrency, minIntervalMs: this.minIntervalMs };
  }

  /** Run fn under the gate. Rejections propagate; the slot is always released. */
  async run<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.active--;
      this.drain();
    }
  }

  private acquire(): Promise<void> {
    return new Promise<void>(resolve => {
      this.waiters.push(resolve);
      this.drain();
    });
  }

  private drain(): void {
    while (this.waiters.length > 0 && this.active < this.concurrency) {
      const now = Date.now();
      if (this.minIntervalMs > 0 && now < this.nextSlotAt) {
        // Too soon for the next START. Re-check when the window opens rather
        // than spinning; a later update() shortening the interval also calls
        // drain(), so a shortened interval is picked up without waiting out
        // the original timer.
        setTimeout(() => this.drain(), this.nextSlotAt - now);
        return;
      }
      this.nextSlotAt = now + this.minIntervalMs;
      this.active++;
      this.waiters.shift()!();
    }
  }
}

// ── Process-wide service gates ───────────────────────────────────────────
//
// Defaults are deliberately conservative: they reproduce the pre-2026-07-30
// behaviour for the two network services (serial + the old courtesy delay), so
// enabling the network queue lane cannot silently multiply anyone's API spend
// or get them blocked. Raise them from Settings once you know your quota.

const num = (v: number, fallback: number) => (Number.isFinite(v) && v > 0 ? v : fallback);
const nonNeg = (v: number, fallback: number) => (Number.isFinite(v) && v >= 0 ? v : fallback);

export const essentiaLimiter = new RateLimiter({
  concurrency: num(config.labeling.essentiaConcurrency, 2),
  minIntervalMs: 0,
}, 'essentia');

export const geniusLimiter = new RateLimiter({
  concurrency: num(config.labeling.geniusConcurrency, 1),
  minIntervalMs: nonNeg(config.labeling.geniusMinIntervalMs, 400),
}, 'genius');

export const captionLimiter = new RateLimiter({
  concurrency: num(config.labeling.captionConcurrency, 1),
  minIntervalMs: nonNeg(config.labeling.captionMinIntervalMs, 250),
}, 'caption');

export interface LabelRateSpecs {
  essentia?: Partial<RateLimitSpec>;
  genius?: Partial<RateLimitSpec>;
  caption?: Partial<RateLimitSpec>;
}

export function applyLabelRates(rates: LabelRateSpecs | undefined): void {
  if (!rates) return;
  if (rates.essentia) essentiaLimiter.update(rates.essentia);
  if (rates.genius) geniusLimiter.update(rates.genius);
  if (rates.caption) captionLimiter.update(rates.caption);
}

/** Pull the limiters back in line with config — called on every .env reload. */
export function syncLimitersFromConfig(): void {
  essentiaLimiter.update({ concurrency: num(config.labeling.essentiaConcurrency, 2) });
  geniusLimiter.update({
    concurrency: num(config.labeling.geniusConcurrency, 1),
    minIntervalMs: nonNeg(config.labeling.geniusMinIntervalMs, 400),
  });
  captionLimiter.update({
    concurrency: num(config.labeling.captionConcurrency, 1),
    minIntervalMs: nonNeg(config.labeling.captionMinIntervalMs, 250),
  });
}

configReloadListeners.push(syncLimitersFromConfig);

export function currentLabelRates(): Record<string, RateLimitSpec> {
  return {
    essentia: essentiaLimiter.spec,
    genius: geniusLimiter.spec,
    caption: captionLimiter.spec,
  };
}
