// LossSparkline.tsx — the live epoch-loss curve
//
// Dependency-free inline SVG: the studio ships no charting library and one is
// not worth 40 kB for a polyline. The viewBox is a fixed 100×32 grid stretched
// to the container (preserveAspectRatio="none"), so strokes carry
// vector-effect="non-scaling-stroke" to stay 1 px regardless of the stretch.
//
// Colours come from Tailwind text utilities via `currentColor` — never a
// hard-coded hex, so the curve reads correctly in both themes.

import React from 'react';
import type { TrainLmEpoch } from '../../services/trainingApi';

const VB_W = 100;
const VB_H = 32;

interface Props {
  epochs: TrainLmEpoch[];
  /** Target loss; <= 0 hides the target line. */
  target: number;
}

export const LossSparkline: React.FC<Props> = ({ epochs, target }) => {
  if (epochs.length < 2) return null;

  const losses = epochs.map(e => e.loss).filter(v => Number.isFinite(v));
  if (losses.length < 2) return null;

  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const hasTarget = target > 0;

  // Keep the target line inside the frame so "how far left to go" is readable.
  let lo = Math.min(hasTarget ? target : minLoss, minLoss) * 0.95;
  let hi = Math.max(maxLoss, hasTarget ? target : maxLoss) * 1.02;
  if (!(hi > lo)) { lo = lo - 0.5; hi = hi + 0.5; }   // flat curve → give it room

  const xFor = (i: number) => (epochs.length === 1 ? VB_W / 2 : (i / (epochs.length - 1)) * VB_W);
  const yFor = (v: number) => {
    const f = (v - lo) / (hi - lo);
    return VB_H - Math.min(1, Math.max(0, f)) * VB_H;
  };

  const points = epochs.map((e, i) => `${xFor(i).toFixed(2)},${yFor(e.loss).toFixed(2)}`).join(' ');
  const targetY = hasTarget ? yFor(target) : 0;
  const colW = VB_W / epochs.length;

  return (
    <div className="flex flex-col gap-1">
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          preserveAspectRatio="none"
          role="img"
          aria-label="Epoch loss curve"
          className="w-full h-24 min-w-[240px] rounded-lg bg-zinc-100 dark:bg-black/30"
        >
          {hasTarget && (
            <line
              x1={0} y1={targetY} x2={VB_W} y2={targetY}
              className="text-emerald-500"
              stroke="currentColor"
              strokeWidth={1}
              strokeDasharray="3 3"
              vectorEffect="non-scaling-stroke"
            />
          )}

          <polyline
            points={points}
            fill="none"
            className="text-amber-500"
            stroke="currentColor"
            strokeWidth={1.5}
            strokeLinejoin="round"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />

          {/* Transparent hit columns carry the per-epoch tooltip. Circles would
              be squashed into ellipses by preserveAspectRatio="none"; a rect
              is a rect either way. */}
          {epochs.map((e, i) => (
            <rect key={e.epoch} x={i * colW} y={0} width={colW} height={VB_H} fill="transparent">
              <title>{`#${e.epoch} · ${e.loss.toFixed(4)}`}</title>
            </rect>
          ))}
        </svg>
      </div>

      <div className="flex items-center justify-between text-[10px] text-zinc-500 tabular-nums">
        <span>{maxLoss.toFixed(3)}</span>
        {hasTarget && <span className="text-emerald-500">{target.toFixed(2)}</span>}
        <span>{minLoss.toFixed(3)}</span>
      </div>
    </div>
  );
};

export default LossSparkline;
