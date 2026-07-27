// LabelPanel.tsx — one-click labeling: Essentia + Genius + AI caption
//
// 2026-07-27 pivot: ace-understand is out of the default flow (weak captions,
// hallucinated lyrics). One button runs, per track: local BPM/key (Essentia,
// parallel CPU lane) → Genius lyrics → audio-grounded LLM caption+genre.
// No engine/GPU involvement at all. Each step degrades to disabled + an
// explanation when its capability is missing.

import React, { useState } from 'react';
import { Loader2, Play } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { MergePolicy } from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { EnhancePanel } from './EnhancePanel';
import { JobProgress } from './JobProgress';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';

const MERGE_OPTIONS: Array<{ id: MergePolicy; labelKey: string }> = [
  { id: 'fill_missing',      labelKey: 'trainingStudio.label.mergeFill' },
  { id: 'overwrite_caption', labelKey: 'trainingStudio.label.mergeCaption' },
  { id: 'overwrite_lyrics',  labelKey: 'trainingStudio.label.mergeLyrics' },
  { id: 'overwrite_all',     labelKey: 'trainingStudio.label.mergeAll' },
];

export const LabelPanel: React.FC = () => {
  const { t } = useTranslation();
  const caps = useTrainingStore(s => s.capabilities);
  const activeJob = useTrainingStore(s => s.activeJob);
  const startLabel = useTrainingStore(s => s.startLabel);
  const selectedSampleIds = useTrainingStore(s => s.selectedSampleIds);

  const essentiaOk = caps?.essentia.available !== false;
  const geniusOk = !!caps?.genius.configured;
  const captionOk = !!caps?.llm.configured;

  const [scope, setScope] = useState<'unlabeled' | 'all' | 'selected'>('unlabeled');
  const [useEssentia, setUseEssentia] = useState(true);
  const [useGenius, setUseGenius] = useState(true);
  const [useCaption, setUseCaption] = useState(true);
  const [mergePolicy, setMergePolicy] = useState<MergePolicy>('fill_missing');
  const [starting, setStarting] = useState(false);

  const selCount = selectedSampleIds.size;
  const jobRunning = !!activeJob && (activeJob.status === 'queued' || activeJob.status === 'running');
  const effectiveEssentia = useEssentia && essentiaOk;
  const effectiveGenius = useGenius && geniusOk;
  const effectiveCaption = useCaption && captionOk;
  const anyStep = effectiveEssentia || effectiveGenius || effectiveCaption;

  const handleStart = async () => {
    setStarting(true);
    try {
      await startLabel({
        ...(scope === 'selected' ? { sampleIds: Array.from(selectedSampleIds) } : { scope: scope === 'all' ? 'all' : 'unlabeled' }),
        useEssentia: effectiveEssentia,
        useGenius: effectiveGenius,
        useCaption: effectiveCaption,
        mergePolicy,
      });
    } finally {
      setStarting(false);
    }
  };

  const radio = 'flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300 cursor-pointer';

  return (
    <div className="flex flex-col gap-4">
      <div className={`${CARD} flex flex-col gap-4`}>
        <div>
          <h2 className="text-sm font-bold text-zinc-900 dark:text-white">{t('trainingStudio.label.title')}</h2>
          <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-1">{t('trainingStudio.label.subtitle')}</p>
        </div>

        {/* Scope */}
        <div className="flex flex-col gap-2">
          <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">{t('trainingStudio.label.scope')}</span>
          <div className="flex flex-col gap-1.5">
            <label className={radio}>
              <input type="radio" checked={scope === 'unlabeled'} onChange={() => setScope('unlabeled')} className="accent-amber-500" />
              {t('trainingStudio.label.scopeUnlabeled')}
            </label>
            <label className={radio}>
              <input type="radio" checked={scope === 'all'} onChange={() => setScope('all')} className="accent-amber-500" />
              {t('trainingStudio.label.scopeAll')}
            </label>
            <label className={`${radio} ${selCount === 0 ? 'opacity-40 cursor-not-allowed' : ''}`}>
              <input
                type="radio"
                checked={scope === 'selected'}
                disabled={selCount === 0}
                onChange={() => setScope('selected')}
                className="accent-amber-500"
              />
              {t('trainingStudio.label.scopeSelected', { count: selCount })}
            </label>
          </div>
        </div>

        {/* Steps */}
        <div className="flex flex-col gap-2">
          <label className={`${radio} ${!essentiaOk ? 'opacity-50' : ''}`}>
            <input
              type="checkbox"
              checked={effectiveEssentia}
              disabled={!essentiaOk}
              onChange={(e) => setUseEssentia(e.target.checked)}
              className="accent-amber-500"
            />
            {t('trainingStudio.label.useEssentia')}
          </label>
          {!essentiaOk && (
            <div className="ml-6 text-[11px] text-amber-600 dark:text-amber-400">{t('trainingStudio.caps.essentiaMissing')}</div>
          )}

          <label className={`${radio} ${!geniusOk ? 'opacity-50' : ''}`}>
            <input
              type="checkbox"
              checked={effectiveGenius}
              disabled={!geniusOk}
              onChange={(e) => setUseGenius(e.target.checked)}
              className="accent-amber-500"
            />
            {t('trainingStudio.label.useGenius')}
          </label>
          {!geniusOk && (
            <div className="ml-6 text-[11px] text-amber-600 dark:text-amber-400">{t('trainingStudio.enhance.geniusMissing')}</div>
          )}

          <label className={`${radio} ${!captionOk ? 'opacity-50' : ''}`}>
            <input
              type="checkbox"
              checked={effectiveCaption}
              disabled={!captionOk}
              onChange={(e) => setUseCaption(e.target.checked)}
              className="accent-amber-500"
            />
            {t('trainingStudio.label.useCaption')}
          </label>
          {captionOk ? (
            <div className="ml-6 text-[11px] text-zinc-500">{t('trainingStudio.label.useCaptionHint')}</div>
          ) : (
            <div className="ml-6 text-[11px] text-amber-600 dark:text-amber-400">{t('trainingStudio.enhance.captionMissing')}</div>
          )}
        </div>

        {/* Merge policy */}
        <label className="flex flex-col gap-1.5">
          <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">{t('trainingStudio.label.mergePolicy')}</span>
          <select
            value={mergePolicy}
            onChange={(e) => setMergePolicy(e.target.value as MergePolicy)}
            className="self-start rounded-lg px-2.5 py-1.5 text-xs bg-zinc-100 dark:bg-black/20 border border-zinc-300 dark:border-white/10 text-zinc-800 dark:text-zinc-200 focus:outline-none focus:border-amber-500"
          >
            {MERGE_OPTIONS.map(o => <option key={o.id} value={o.id}>{t(o.labelKey)}</option>)}
          </select>
        </label>

        {/* `error` is rendered once by DatasetDetail so every step sees it. */}

        <button
          onClick={() => void handleStart()}
          disabled={jobRunning || starting || !anyStep}
          className="self-start flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {starting ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
          {starting ? t('trainingStudio.label.starting') : t('trainingStudio.label.start')}
        </button>
      </div>

      <JobProgress />

      <EnhancePanel selectedSampleIds={Array.from(selectedSampleIds)} disabled={jobRunning} />
    </div>
  );
};

export default LabelPanel;
