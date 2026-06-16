// StreamPlayer.tsx — Progressive audio preview player for streaming generation
//
// Shows during stream_mode generation:
// - Waveform visualization of received preview chunks
// - Play/Pause controls for progressive playback
// - Progress bar showing DiT denoising steps
// - Step count and preview chunk indicators

import React, { useMemo } from 'react';
import { Play, Pause, Square, Radio, Loader2 } from 'lucide-react';
import type { StreamPreview, StreamStatus } from '../../hooks/useStreamGeneration';

interface StreamPlayerProps {
  connected: boolean;
  status: StreamStatus | null;
  previews: StreamPreview[];
  playing: boolean;
  done: boolean;
  error: string | null;
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
}

export const StreamPlayer: React.FC<StreamPlayerProps> = ({
  connected,
  status,
  previews,
  playing,
  done,
  error,
  onPlay,
  onPause,
  onStop,
}) => {
  // Count loaded previews (those with decoded audio buffers)
  const loadedCount = useMemo(
    () => previews.filter(p => p.buffer !== null).length,
    [previews]
  );

  // Progress percentage from the latest preview's step info
  const progress = useMemo(() => {
    if (!status?.progress) return 0;
    // status.progress is already 0-100 from the server
    return Math.min(100, Math.max(0, status.progress));
  }, [status?.progress]);

  const stageLabel = status?.stage || (connected ? 'Connecting…' : 'Waiting…');

  // Step info from latest preview
  const latestPreview = previews[previews.length - 1];
  const stepLabel = latestPreview
    ? `Step ${latestPreview.step}/${latestPreview.totalSteps}`
    : '';

  return (
    <div className="bg-zinc-900/80 backdrop-blur-sm border border-emerald-500/20 rounded-xl p-3 space-y-2">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${
            error ? 'bg-red-500' :
            done ? 'bg-emerald-500' :
            connected ? 'bg-emerald-400 animate-pulse' :
            'bg-zinc-600'
          }`} />
          <span className="text-xs font-medium text-zinc-300 flex items-center gap-1.5">
            <Radio size={12} className="text-emerald-400" />
            Stream Preview
          </span>
        </div>
        {stepLabel && (
          <span className="text-[10px] text-zinc-500 font-mono">{stepLabel}</span>
        )}
      </div>

      {/* Stage / Status */}
      <div className="flex items-center gap-2">
        {!done && !error && (
          <Loader2 size={12} className="text-emerald-400 animate-spin flex-shrink-0" />
        )}
        <span className="text-xs text-zinc-400 truncate">{stageLabel}</span>
      </div>

      {/* Progress bar */}
      <div className="relative h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 bg-gradient-to-r from-emerald-500 to-teal-400 rounded-full transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
        {/* Preview chunk markers */}
        {previews.map((p, i) => {
          const pos = p.totalSteps > 0 ? (p.step / p.totalSteps) * 100 : 0;
          return (
            <div
              key={i}
              className={`absolute top-0 w-0.5 h-full ${
                p.buffer ? 'bg-emerald-300' : 'bg-zinc-600'
              }`}
              style={{ left: `${pos}%` }}
              title={`Preview at step ${p.step}/${p.totalSteps}`}
            />
          );
        })}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {/* Play/Pause */}
          <button
            onClick={playing ? onPause : onPlay}
            disabled={loadedCount === 0}
            className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ${
              loadedCount === 0
                ? 'bg-zinc-800 text-zinc-600 cursor-not-allowed'
                : playing
                  ? 'bg-emerald-500 text-white hover:bg-emerald-400 shadow-lg shadow-emerald-500/20'
                  : 'bg-zinc-700 text-zinc-200 hover:bg-emerald-500 hover:text-white'
            }`}
            title={playing ? 'Pause' : 'Play preview'}
          >
            {playing
              ? <Pause size={14} fill="currentColor" />
              : <Play size={14} fill="currentColor" className="ml-0.5" />
            }
          </button>

          {/* Stop */}
          <button
            onClick={onStop}
            disabled={!playing && loadedCount === 0}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            title="Stop"
          >
            <Square size={12} fill="currentColor" />
          </button>
        </div>

        {/* Preview count */}
        <div className="flex items-center gap-3">
          <span className="text-[10px] text-zinc-500 font-mono">
            {loadedCount}/{previews.length} previews
          </span>

          {/* Error indicator */}
          {error && (
            <span className="text-[10px] text-red-400 font-medium">
              {error}
            </span>
          )}

          {/* Done indicator */}
          {done && !error && (
            <span className="text-[10px] text-emerald-400 font-medium">
              ✓ Complete
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
