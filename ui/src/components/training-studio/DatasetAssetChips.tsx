// DatasetAssetChips.tsx — the pipeline-progress pill row for one dataset
//
// Label → Build → Preprocess → Train DiT → Train LM, in stage order, lit when
// the artefact is actually on disk and dimmed when it is not. The server reads
// every flag fresh (services/training/datasetAssets.ts), so a deleted tensors
// folder or adapter goes dark on the next refresh.
//
// Shared by the dataset cards and the batch-import wizard so both windows onto
// the same corpus never disagree about what is done.

import React from 'react';
import { useTranslation } from 'react-i18next';
import type { DatasetAssets } from '../../services/trainingApi';

const ON = 'text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 border-emerald-500/25';
const OFF = 'text-zinc-400 dark:text-zinc-600 bg-transparent border-zinc-300/60 dark:border-white/10';

function formatWhen(iso: string): string {
  if (!iso) return '';
  const d = new Date(iso.includes('T') || iso.includes('Z') ? iso : iso.replace(' ', 'T') + 'Z');
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

interface ChipProps {
  on: boolean;
  label: string;
  title: string;
  suffix?: string;
}

const Chip: React.FC<ChipProps> = ({ on, label, title, suffix }) => (
  <span
    title={title}
    className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full border whitespace-nowrap ${on ? ON : OFF}`}
  >
    {label}{on && suffix ? <span className="font-normal opacity-80"> {suffix}</span> : null}
  </span>
);

interface DatasetAssetChipsProps {
  assets: DatasetAssets | undefined;
  /** Sample counts for the Labeled chip's tooltip — the row's own cached ones. */
  labeledCount?: number;
  sampleCount?: number;
  className?: string;
}

export const DatasetAssetChips: React.FC<DatasetAssetChipsProps> = ({
  assets, labeledCount = 0, sampleCount = 0, className = '',
}) => {
  const { t } = useTranslation();
  if (!assets) return null;

  return (
    <div className={`flex items-center gap-1 flex-wrap ${className}`}>
      <Chip
        on={assets.labeled}
        label={t('trainingStudio.assets.labeled')}
        title={assets.labeled
          ? t('trainingStudio.assets.labeledOn', { done: labeledCount, total: sampleCount })
          : t('trainingStudio.assets.labeledOff')}
      />
      <Chip
        on={assets.built}
        label={t('trainingStudio.assets.built')}
        title={assets.built ? t('trainingStudio.assets.builtOn') : t('trainingStudio.assets.builtOff')}
      />
      <Chip
        on={assets.tensorVariants > 0}
        label={t('trainingStudio.assets.tensors')}
        suffix={assets.tensorSamples > 0 ? String(assets.tensorSamples) : ''}
        title={assets.tensorVariants > 0
          ? t('trainingStudio.assets.tensorsOn', {
              count: assets.tensorSamples,
              variant: assets.tensorVariantKey || '?',
            })
          : t('trainingStudio.assets.tensorsOff')}
      />
      <Chip
        on={!!assets.dit}
        label={t('trainingStudio.assets.dit')}
        title={assets.dit
          ? t('trainingStudio.assets.ditOn', {
              base: assets.dit.detail || '?',
              when: formatWhen(assets.dit.trainedAt) || '—',
            })
          : t('trainingStudio.assets.ditOff')}
      />
      <Chip
        on={!!assets.lm}
        label={t('trainingStudio.assets.lm')}
        suffix={assets.lm?.detail ?? ''}
        title={assets.lm
          ? t('trainingStudio.assets.lmOn', {
              size: assets.lm.detail || '?',
              when: formatWhen(assets.lm.trainedAt) || '—',
            })
          : t('trainingStudio.assets.lmOff')}
      />
    </div>
  );
};

export default DatasetAssetChips;
