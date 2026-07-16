// PianoRoll.tsx — read-only SVG piano-roll preview of a transcribed MIDI file
//
// Renders notes fetched from /api/midi-studio/:jobId/notes. Colors are
// per-MIDI-channel so multi-instrument transcriptions read at a glance.

import React, { useEffect, useMemo, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { getMidiNotes, channelLabel, type ParsedMidi } from '../../services/midiStudioApi';

const PX_PER_SEC = 40;
const ROLL_HEIGHT = 240;

/** Stable, readable color per MIDI channel (0-15) */
function channelColor(channel: number): string {
  if (channel === 9) return 'hsl(0, 70%, 55%)'; // drums = red
  const hue = (channel * 67) % 360;
  return `hsl(${hue}, 70%, 55%)`;
}

export const PianoRoll: React.FC<{ jobId: string }> = ({ jobId }) => {
  const { t } = useTranslation();
  const [data, setData] = useState<ParsedMidi | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let alive = true;
    getMidiNotes(jobId)
      .then(d => { if (alive) setData(d); })
      .catch(err => { if (alive) setError(err.message); });
    return () => { alive = false; };
  }, [jobId]);

  const layout = useMemo(() => {
    if (!data || data.notes.length === 0) return null;
    let minPitch = 127, maxPitch = 0;
    for (const n of data.notes) {
      if (n.pitch < minPitch) minPitch = n.pitch;
      if (n.pitch > maxPitch) maxPitch = n.pitch;
    }
    minPitch = Math.max(0, minPitch - 2);
    maxPitch = Math.min(127, maxPitch + 2);
    const rows = maxPitch - minPitch + 1;
    const rowH = ROLL_HEIGHT / rows;
    const width = Math.max(200, Math.ceil(data.durationSec * PX_PER_SEC) + 20);
    return { minPitch, maxPitch, rowH, width };
  }, [data]);

  if (error) {
    return <div className="text-xs text-red-500 dark:text-red-400 py-2">{t('midiStudio.previewError')}: {error}</div>;
  }
  if (!data) {
    return (
      <div className="flex items-center gap-2 text-xs text-zinc-500 dark:text-zinc-400 py-3">
        <Loader2 size={14} className="animate-spin" /> {t('midiStudio.loadingPreview')}
      </div>
    );
  }
  if (!layout) {
    return <div className="text-xs text-zinc-500 dark:text-zinc-400 py-2">{t('midiStudio.noNotes')}</div>;
  }

  const gridLines: React.ReactNode[] = [];
  for (let s = 0; s <= data.durationSec; s += 10) {
    const x = s * PX_PER_SEC;
    gridLines.push(
      <g key={s}>
        <line x1={x} y1={0} x2={x} y2={ROLL_HEIGHT} stroke="currentColor" strokeOpacity={0.08} />
        <text x={x + 3} y={12} fontSize={9} fill="currentColor" fillOpacity={0.4}>{s}s</text>
      </g>
    );
  }

  return (
    <div className="mt-2">
      {/* Channel legend */}
      <div className="flex flex-wrap gap-x-3 gap-y-1 mb-2">
        {data.channels.map(ch => (
          <span key={ch.channel} className="flex items-center gap-1.5 text-[11px] text-zinc-600 dark:text-zinc-400">
            <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ backgroundColor: channelColor(ch.channel) }} />
            {channelLabel(ch)} <span className="opacity-50">({ch.noteCount})</span>
          </span>
        ))}
      </div>
      {/* Scrollable roll */}
      <div className="overflow-x-auto rounded-lg border border-zinc-200 dark:border-white/5 bg-zinc-50 dark:bg-black/30 text-zinc-900 dark:text-zinc-100">
        <svg width={layout.width} height={ROLL_HEIGHT} className="block">
          {gridLines}
          {data.notes.map((n, i) => (
            <rect
              key={i}
              x={n.start * PX_PER_SEC}
              y={(layout.maxPitch - n.pitch) * layout.rowH}
              width={Math.max(1.5, n.duration * PX_PER_SEC)}
              height={Math.max(1.5, layout.rowH - 0.5)}
              rx={1}
              fill={channelColor(n.channel)}
              fillOpacity={0.35 + (n.velocity / 127) * 0.6}
            />
          ))}
        </svg>
      </div>
    </div>
  );
};

export default PianoRoll;
