// TrainingStudio.tsx — Training Studio orchestrator (Dataset phase)
//
// Phase 1 ships dataset creation: import audio → label locally
// (/understand + Essentia) → optionally enhance in the cloud → review/edit →
// build dataset.json. Phase 2 adds Preprocess (dataset.json → tensor caches).
// Phase 3 adds Train (tensor caches → an LM LoRA). Monitor is still a stepper
// chip only.

import React, { useEffect } from 'react';
import { AlertTriangle, GraduationCap } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useTrainingStore } from '../../stores/trainingStore';
import { CapabilityBanner } from './CapabilityBanner';
import { DatasetDetail } from './DatasetDetail';
import { DatasetList } from './DatasetList';
import { PhaseStepper } from './PhaseStepper';
import { PreprocessPanel } from './PreprocessPanel';
import { TrainPanel } from './TrainPanel';

export const TrainingStudio: React.FC = () => {
  const { t } = useTranslation();
  const phase = useTrainingStore(s => s.phase);
  const selectedDatasetId = useTrainingStore(s => s.selectedDatasetId);
  const error = useTrainingStore(s => s.error);
  const detail = useTrainingStore(s => s.detail);
  const loadCapabilities = useTrainingStore(s => s.loadCapabilities);
  const loadDatasets = useTrainingStore(s => s.loadDatasets);

  useEffect(() => {
    void loadCapabilities();
    void loadDatasets();
  }, [loadCapabilities, loadDatasets]);

  const fatalError = error && !detail;

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-6xl mx-auto px-6 py-8 flex flex-col gap-6">
        {/* Title */}
        <div className="flex items-start gap-3">
          <GraduationCap size={26} className="text-amber-500 mt-0.5 flex-shrink-0" />
          <div>
            <h1 className="text-xl font-bold text-zinc-900 dark:text-white">{t('trainingStudio.title')}</h1>
            <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-0.5">{t('trainingStudio.subtitle')}</p>
          </div>
        </div>

        <CapabilityBanner />
        <PhaseStepper />

        {fatalError ? (
          <div className="rounded-xl border border-red-500/25 bg-red-500/10 p-5 flex flex-col items-start gap-3">
            <div className="flex items-start gap-2 text-sm text-red-500 dark:text-red-400">
              <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
              {t('trainingStudio.error.load', { message: error })}
            </div>
            <button
              onClick={() => { void loadCapabilities(); void loadDatasets(); }}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-red-500/15 border border-red-500/25 text-red-500 hover:bg-red-500/25 transition-colors"
            >
              {t('trainingStudio.error.retry')}
            </button>
          </div>
        ) : phase === 'dataset' ? (
          selectedDatasetId ? <DatasetDetail /> : <DatasetList />
        ) : phase === 'preprocess' ? (
          <PreprocessPanel />
        ) : phase === 'train' ? (
          <TrainPanel />
        ) : (
          <div className="rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-8 text-center text-sm text-zinc-500">
            {t('trainingStudio.phase.comingSoon')}
          </div>
        )}
      </div>
    </div>
  );
};

export default TrainingStudio;
