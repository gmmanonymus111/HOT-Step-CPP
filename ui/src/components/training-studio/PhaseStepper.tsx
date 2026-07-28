// PhaseStepper.tsx — Dataset · Preprocess · Train · Monitor
//
// Dataset, Preprocess and Train ship; Monitor renders as a disabled chip with
// a "coming soon" tooltip so it slots in without a restructure.

import React from 'react';
import { Database, Layers, Cpu, Activity, Lock } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useTrainingStore } from '../../stores/trainingStore';

type Phase = 'dataset' | 'preprocess' | 'train' | 'monitor';

const PHASES: Array<{ id: Phase; icon: React.ReactNode; labelKey: string; enabled: boolean }> = [
  { id: 'dataset',    icon: <Database size={14} />, labelKey: 'trainingStudio.phase.dataset',    enabled: true },
  { id: 'preprocess', icon: <Layers size={14} />,   labelKey: 'trainingStudio.phase.preprocess', enabled: true },
  { id: 'train',      icon: <Cpu size={14} />,      labelKey: 'trainingStudio.phase.train',      enabled: true },
  { id: 'monitor',    icon: <Activity size={14} />, labelKey: 'trainingStudio.phase.monitor',    enabled: false },
];

export const PhaseStepper: React.FC = () => {
  const { t } = useTranslation();
  const phase = useTrainingStore(s => s.phase);
  const setPhase = useTrainingStore(s => s.setPhase);

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {PHASES.map((p, i) => {
        const active = phase === p.id;
        return (
          <React.Fragment key={p.id}>
            {i > 0 && <div className="w-4 h-px bg-zinc-300 dark:bg-white/10" />}
            <button
              type="button"
              disabled={!p.enabled}
              onClick={() => p.enabled && setPhase(p.id)}
              title={p.enabled ? undefined : t('trainingStudio.phase.comingSoon')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border transition-all ${
                active
                  ? 'text-amber-500 bg-amber-500/10 border-amber-500/30'
                  : p.enabled
                    ? 'text-zinc-500 border-transparent hover:text-zinc-700 dark:hover:text-zinc-300 hover:bg-black/5 dark:hover:bg-white/5'
                    : 'text-zinc-400 dark:text-zinc-600 border-transparent cursor-not-allowed'
              }`}
            >
              {p.enabled ? p.icon : <Lock size={12} />}
              {t(p.labelKey)}
            </button>
          </React.Fragment>
        );
      })}
    </div>
  );
};

export default PhaseStepper;
