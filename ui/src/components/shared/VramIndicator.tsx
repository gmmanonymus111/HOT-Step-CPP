// VramIndicator.tsx — Standalone GPU VRAM usage display
//
// Polls /api/logs/vram every 5 seconds. Shows used/total GB with a
// color-coded progress bar (green → yellow → red).
// Used in both the GlobalParamBar and TerminalPanel.

import React, { useState, useEffect } from 'react';
import { Cpu } from 'lucide-react';

interface VramInfo {
  used_mb: number;
  total_mb: number;
  free_mb: number;
}

interface VramIndicatorProps {
  /** Polling interval in ms (default 5000) */
  pollInterval?: number;
  /** Compact mode — hides the label and uses smaller text */
  compact?: boolean;
}

export const VramIndicator: React.FC<VramIndicatorProps> = ({
  pollInterval = 5000,
  compact = false,
}) => {
  const [vram, setVram] = useState<VramInfo | null>(null);

  useEffect(() => {
    const fetchVram = async () => {
      try {
        const res = await fetch('/api/logs/vram');
        if (res.ok) {
          const data = await res.json();
          if (data.total_mb > 0) setVram(data);
        }
      } catch { /* ignore */ }
    };
    fetchVram();
    const interval = setInterval(fetchVram, pollInterval);
    return () => clearInterval(interval);
  }, [pollInterval]);

  if (!vram || vram.total_mb <= 0) return null;

  const percent = Math.round((vram.used_mb / vram.total_mb) * 100);
  const colorText = percent > 90 ? 'text-red-400' : percent > 70 ? 'text-yellow-400' : 'text-emerald-400';
  const colorBg = percent > 90 ? 'bg-red-500' : percent > 70 ? 'bg-yellow-500' : 'bg-emerald-500';

  return (
    <div className="flex items-center gap-1.5">
      <Cpu size={compact ? 10 : 11} className="text-zinc-500 flex-shrink-0" />
      <div className="flex items-center gap-1">
        {!compact && <span className="text-[10px] text-zinc-500">VRAM</span>}
        <span className={`text-[10px] font-mono font-medium ${colorText}`}>
          {(vram.used_mb / 1024).toFixed(1)}/{(vram.total_mb / 1024).toFixed(1)}
        </span>
        <div className="w-10 h-1 rounded-full bg-zinc-200 dark:bg-zinc-700 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${colorBg}`}
            style={{ width: `${percent}%` }}
          />
        </div>
      </div>
    </div>
  );
};
