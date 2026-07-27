// SampleDrawer.tsx — full caption / lyrics editor with audio preview
//
// Same optimistic + debounced edit path as the grid cells. The <audio>
// element is local to the drawer: preview never touches the global playback
// store because these files are not library songs.

import React, { useMemo } from 'react';
import { ChevronLeft, ChevronRight, Loader2, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { sampleAudioUrl } from '../../services/trainingApi';
import { mergedSample, useTrainingStore } from '../../stores/trainingStore';

interface SampleDrawerProps {
  sampleId: string;
}

export const SampleDrawer: React.FC<SampleDrawerProps> = ({ sampleId }) => {
  const { t } = useTranslation();
  const datasetId = useTrainingStore(s => s.selectedDatasetId);
  const sampleRaw = useTrainingStore(s => s.samplesById[sampleId]);
  const pending = useTrainingStore(s => s.pendingEdits[sampleId]);
  const sampleOrder = useTrainingStore(s => s.sampleOrder);
  const saving = useTrainingStore(s => s.savingSampleIds.has(sampleId));
  const editSample = useTrainingStore(s => s.editSample);
  const flushSample = useTrainingStore(s => s.flushSample);
  const revertSampleField = useTrainingStore(s => s.revertSampleField);
  const setOpenSampleId = useTrainingStore(s => s.setOpenSampleId);

  const sample = useMemo(() => mergedSample(sampleRaw, pending), [sampleRaw, pending]);
  const idx = sampleOrder.indexOf(sampleId);

  if (!sample || !datasetId) return null;

  const readOnly = sample.labelStatus === 'processing' || sample.fileMissing;
  const close = () => { void flushSample(sampleId); setOpenSampleId(null); };
  const goto = (delta: number) => {
    const next = sampleOrder[idx + delta];
    if (!next) return;
    void flushSample(sampleId);
    setOpenSampleId(next);
  };

  const area = `w-full rounded-lg px-3 py-2 text-sm bg-zinc-100 dark:bg-black/20 border border-zinc-300 dark:border-white/10 text-zinc-800 dark:text-zinc-200 placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-amber-500 disabled:opacity-50`;

  return (
    <div className="fixed inset-0 z-[140] flex justify-end">
      <div className="absolute inset-0 bg-black/30 dark:bg-black/60 backdrop-blur-sm" onClick={close} />
      <div className="relative w-[560px] max-w-full h-full overflow-y-auto bg-white dark:bg-suno-card border-l border-zinc-200 dark:border-white/10 shadow-2xl flex flex-col gap-4 p-5">
        {/* Header */}
        <div className="flex items-center gap-2">
          <div className="flex-1 min-w-0">
            <div className="text-sm font-bold text-zinc-900 dark:text-white truncate" title={sample.relPath}>{sample.filename}</div>
            <div className="text-[11px] font-mono text-zinc-500 truncate">{sample.relPath}</div>
          </div>
          {saving && <Loader2 size={14} className="animate-spin text-amber-500" />}
          <button
            onClick={() => goto(-1)}
            disabled={idx <= 0}
            title={t('trainingStudio.drawer.prev')}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-800 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-30 transition-colors"
          >
            <ChevronLeft size={16} />
          </button>
          <button
            onClick={() => goto(1)}
            disabled={idx < 0 || idx >= sampleOrder.length - 1}
            title={t('trainingStudio.drawer.next')}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-800 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-30 transition-colors"
          >
            <ChevronRight size={16} />
          </button>
          <button
            onClick={close}
            title={t('trainingStudio.drawer.close')}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-800 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        {/* Preview */}
        {!sample.fileMissing && (
          <audio src={sampleAudioUrl(datasetId, sampleId)} controls className="w-full h-9" preload="none" />
        )}

        {/* Caption */}
        <label className="flex flex-col gap-1.5">
          <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">{t('trainingStudio.drawer.caption')}</span>
          <textarea
            rows={5}
            value={sample.caption}
            disabled={readOnly}
            placeholder={t('trainingStudio.drawer.captionPlaceholder')}
            onChange={(e) => void editSample(sampleId, { caption: e.target.value })}
            onBlur={() => void flushSample(sampleId)}
            onKeyDown={(e) => { if (e.key === 'Escape') revertSampleField(sampleId, 'caption'); }}
            className={area}
          />
        </label>

        {/* Instrumental */}
        <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
          <input
            type="checkbox"
            checked={sample.isInstrumental}
            disabled={readOnly}
            onChange={(e) => { void editSample(sampleId, { isInstrumental: e.target.checked }); void flushSample(sampleId); }}
            className="accent-amber-500"
          />
          {t('trainingStudio.drawer.instrumental')}
        </label>

        {/* Lyrics */}
        <label className="flex flex-col gap-1.5 flex-1">
          <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">{t('trainingStudio.drawer.lyrics')}</span>
          <textarea
            rows={16}
            value={sample.lyrics}
            disabled={readOnly}
            placeholder={t('trainingStudio.drawer.lyricsPlaceholder')}
            onChange={(e) => void editSample(sampleId, { lyrics: e.target.value })}
            onBlur={() => void flushSample(sampleId)}
            onKeyDown={(e) => { if (e.key === 'Escape') revertSampleField(sampleId, 'lyrics'); }}
            className={`${area} font-mono text-xs leading-relaxed`}
          />
        </label>

        {/* Meta strip */}
        <div className="flex items-center gap-3 flex-wrap text-[11px] text-zinc-500">
          {sample.bpm != null && <span>{sample.bpm} BPM</span>}
          {sample.key && <span>{sample.key}</span>}
          {sample.signature && <span>{sample.signature}</span>}
          {sample.language && <span>{sample.language}</span>}
          {sample.labeledAt && <span>{t('trainingStudio.drawer.saved')} · {sample.labeledAt.slice(0, 10)}</span>}
        </div>
      </div>
    </div>
  );
};

export default SampleDrawer;
