// Mm3TrainCard.tsx — Training Studio phase 3, MiniMax-Music3 branch.
//
// Rendered instead of the ACE LM/DiT cards when the active backend is
// MiniMax-Music3: codes + the MM3-native captions -> an LM LoRA.
//
// The CODES export is phase 2, not here (Mm3CodesCard) — for MM3, codes are
// what preprocessing means. This card only GATES on them, exactly as the ACE
// train panel gates on a preprocess variant, so there is one place to run the
// export and one place to run training.
//
// Everything below the form is the SHARED job machinery: the same SSE stream,
// the same JobProgress, the same loss chart.
//
// THE DEFAULTS ARE NOT DUPLICATED HERE. They arrive in the `mm3` status
// payload from services/training/mm3Train.ts, which is the single place the
// validated recipe lives. This form seeds itself from that response, so a
// change on the server reaches the UI without an edit here.

import React, { useState } from 'react';
import {
  AlertTriangle, ChevronDown, ChevronRight, Cpu, Loader2, Play, XCircle,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';

import type { Mm3TrainLmRequest } from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { JobProgress } from './JobProgress';
import { useMm3Status } from './useMm3Status';
import { TrainingChart } from './TrainingChart';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';
const INPUT = 'w-full px-2.5 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 '
            + 'dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 outline-none '
            + 'focus:border-amber-500/50';

interface FormState {
  steps: number;
  saveEvery: number;
  rank: number;
  alpha: number;
  lr: number;
  maxFrames: number;
  cropMode: 'random' | 'beginning';
  optimizer: 'muon' | 'adamw';
  muonLrScale: number;
  gradAccum: number;
  seed: number;
  trigger: string;
}

const NumField: React.FC<{
  label: string; value: number; onChange: (v: number) => void; step?: number; hint?: string;
}> = ({ label, value, onChange, step = 1, hint }) => (
  <label className="flex flex-col gap-1">
    <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">{label}</span>
    <input
      type="number" className={INPUT} value={value} step={step}
      onChange={e => onChange(Number(e.target.value))}
    />
    {hint && <span className="text-[10px] text-zinc-500 leading-snug">{hint}</span>}
  </label>
);

export const Mm3TrainCard: React.FC<{ datasetId: string; trigger?: string }> = ({ datasetId, trigger }) => {
  const { t } = useTranslation();
  const activeJob = useTrainingStore(s => s.activeJob);
  const startMm3TrainLm = useTrainingStore(s => s.startMm3TrainLm);
  const storeError = useTrainingStore(s => s.error);
  const setPhase = useTrainingStore(s => s.setPhase);
  const trainStepSeries = useTrainingStore(s => s.trainStepSeries);
  const trainMilestones = useTrainingStore(s => s.trainMilestones);

  const { status, error: statusError } = useMm3Status(datasetId);
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  // The form is DERIVED, not seeded: server defaults underneath, the user's
  // edits on top. No effect copies one into the other, so there is no window
  // where the form holds stale numbers and no cascading render — and the
  // validated recipe still has exactly one home (mm3Train.ts).
  const [edits, setEdits] = useState<Partial<FormState>>({});

  const jobKind = activeJob?.kind;
  const jobStatus = activeJob?.status;
  const jobRunning = jobStatus === 'queued' || jobStatus === 'running';
  const mine = jobKind === 'mm3-train-lm';

  const form: FormState | null = status ? {
    steps: status.defaults.steps ?? 800,
    saveEvery: status.defaults.saveEvery ?? 100,
    rank: status.defaults.rank ?? 256,
    alpha: status.defaults.alpha ?? 256,
    lr: status.defaults.lr ?? 8e-5,
    maxFrames: status.defaults.maxFrames ?? 1500,
    cropMode: (status.defaults.cropMode as 'random' | 'beginning') ?? 'random',
    optimizer: status.defaults.optimizer ?? 'muon',
    muonLrScale: status.defaults.muonLrScale ?? 64,
    gradAccum: status.defaults.gradAccum ?? 1,
    seed: status.defaults.seed ?? 42,
    trigger: trigger ?? '',
    ...edits,
  } : null;

  const startTrain = async () => {
    if (!form) return;
    setBusy(true);
    try {
      const body: Mm3TrainLmRequest = {
        steps: form.steps, saveEvery: form.saveEvery, rank: form.rank, alpha: form.alpha,
        lr: form.lr, maxFrames: form.maxFrames, cropMode: form.cropMode,
        optimizer: form.optimizer, muonLrScale: form.muonLrScale,
        gradAccum: form.gradAccum, seed: form.seed,
        ...(form.trigger.trim() ? { trigger: form.trigger.trim() } : {}),
      };
      await startMm3TrainLm(body);
    } finally {
      setBusy(false);
    }
  };

  if (!status && !statusError) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <Loader2 size={18} className="animate-spin mr-2" /> …
      </div>
    );
  }

  const set = <K extends keyof FormState>(k: K, v: FormState[K]) =>
    setEdits(e => ({ ...e, [k]: v }));

  const hasCodes = (status?.codes ?? 0) > 0;
  const trainBlocked = (status?.missingForTrain.length ?? 0) > 0;

  return (
    <div className="flex flex-col gap-4">
      {(statusError || storeError) && (
        <div className="rounded-xl border border-red-500/25 bg-red-500/10 p-3 flex items-start gap-2 text-sm text-red-500">
          <XCircle size={16} className="mt-0.5 flex-shrink-0" />
          <span className="min-w-0 break-words">{statusError || storeError}</span>
        </div>
      )}

      {/* ── LM LoRA ── */}
      <div className={CARD}>
        <div className="flex items-center gap-2 mb-2">
          <Cpu size={15} className="text-amber-500" />
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
            {t('trainingStudio.mm3.trainTitle', 'LM LoRA training')}
          </h3>
        </div>
        <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
          {t('trainingStudio.mm3.trainBlurb',
            'Trains the planner LM on this dataset. Checkpoints are written straight to the MiniMax-Music3 '
            + 'adapter folder, so they appear in the generation panel\'s adapter picker as soon as they are '
            + 'saved — there is no install step. The engine is paused for the run.')}
        </p>

        {trainBlocked ? (
          <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
            <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
            <span>
              {t('trainingStudio.mm3.missing', 'Missing model files')}: {status?.missingForTrain.join(', ')}
            </span>
          </div>
        ) : !hasCodes ? (
          <div className="flex flex-col items-start gap-3">
            <div className="flex items-start gap-2 text-xs text-zinc-500">
              <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
              {t('trainingStudio.mm3.needsCodes',
                'Export the RVQ codes first — training reads them, not the audio.')}
            </div>
            <button
              onClick={() => setPhase('preprocess')}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/15 border border-amber-500/25 text-amber-600 dark:text-amber-400 hover:bg-amber-500/25 transition-colors"
            >
              {t('trainingStudio.mm3.goToCodes', 'Go to Codes')}
            </button>
          </div>
        ) : form && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <NumField label={t('trainingStudio.mm3.steps', 'Steps')} value={form.steps}
                onChange={v => set('steps', v)} />
              <NumField label={t('trainingStudio.mm3.saveEvery', 'Checkpoint every')} value={form.saveEvery}
                onChange={v => set('saveEvery', v)} />
              <NumField label={t('trainingStudio.mm3.rank', 'Rank')} value={form.rank}
                onChange={v => set('rank', v)} />
              <NumField label={t('trainingStudio.mm3.maxFrames', 'Crop (frames)')} value={form.maxFrames}
                onChange={v => set('maxFrames', v)} step={50} />
            </div>
            <label className="flex flex-col gap-1 mt-3">
              <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                {t('trainingStudio.mm3.trigger', 'Trigger word')}
              </span>
              <input className={INPUT} value={form.trigger} onChange={e => set('trigger', e.target.value)} />
              <span className="text-[10px] text-zinc-500 leading-snug">
                {t('trainingStudio.mm3.triggerHint',
                  'Recorded in the adapter sidecar so the picker can show it. Your captions should already '
                  + 'contain it — this does not add it to them.')}
              </span>
            </label>

            <button
              onClick={() => setAdvanced(v => !v)}
              className="flex items-center gap-1 mt-3 text-[11px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
            >
              {advanced ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              {t('trainingStudio.mm3.advanced', 'Advanced')}
            </button>

            {advanced && (
              <div className="mt-3 pl-3 border-l-2 border-zinc-200 dark:border-white/10 flex flex-col gap-3">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <NumField label={t('trainingStudio.mm3.alpha', 'Alpha')} value={form.alpha}
                    onChange={v => set('alpha', v)} />
                  <NumField label={t('trainingStudio.mm3.lr', 'Learning rate')} value={form.lr}
                    onChange={v => set('lr', v)} step={1e-5} />
                  <NumField label={t('trainingStudio.mm3.gradAccum', 'Grad accum')} value={form.gradAccum}
                    onChange={v => set('gradAccum', v)} />
                  <NumField label={t('trainingStudio.mm3.seed', 'Seed')} value={form.seed}
                    onChange={v => set('seed', v)} />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <label className="flex flex-col gap-1">
                    <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                      {t('trainingStudio.mm3.optimizer', 'Optimizer')}
                    </span>
                    <select className={INPUT} value={form.optimizer}
                      onChange={e => set('optimizer', e.target.value as 'muon' | 'adamw')}>
                      <option value="muon">Muon</option>
                      <option value="adamw">AdamW</option>
                    </select>
                    <span className="text-[10px] text-zinc-500 leading-snug">
                      {t('trainingStudio.mm3.optimizerHint',
                        'Muon is the default for two reasons: at rank 256 AdamW\'s second momentum buffer '
                        + 'does not fit on a 32 GB card, and Muon\'s normalised update means an AdamW '
                        + 'learning rate does not apply to it.')}
                    </span>
                  </label>
                  {form.optimizer === 'muon' && (
                    <NumField label={t('trainingStudio.mm3.muonLrScale', 'Muon LR scale')}
                      value={form.muonLrScale} onChange={v => set('muonLrScale', v)}
                      hint={t('trainingStudio.mm3.muonLrScaleHint',
                        '64 is the best of the values measured so far, not a tuned optimum.') as string} />
                  )}
                </div>
                <label className="flex flex-col gap-1">
                  <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                    {t('trainingStudio.mm3.cropMode', 'Crop mode')}
                  </span>
                  <select className={INPUT} value={form.cropMode}
                    onChange={e => set('cropMode', e.target.value as 'random' | 'beginning')}>
                    <option value="random">random</option>
                    <option value="beginning">beginning</option>
                  </select>
                  <span className="text-[10px] text-amber-600/80 dark:text-amber-400/80 leading-snug">
                    {t('trainingStudio.mm3.cropModeHint',
                      '`beginning` trains on song intros only — it exists to reproduce a known failure, '
                      + 'not to be used. Leave this on random.')}
                  </span>
                </label>
              </div>
            )}

            <button
              onClick={() => void startTrain()}
              disabled={busy || jobRunning}
              className="mt-4 px-4 py-2 rounded-lg text-sm font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 transition-colors flex items-center gap-2"
            >
              <Play size={14} />
              {t('trainingStudio.mm3.start', 'Start training')}
            </button>
          </>
        )}
      </div>

      {/* ── Live run: the shared job machinery, unchanged ── */}
      {mine && activeJob && (
        <div className={CARD}>
          <JobProgress />
          {jobKind === 'mm3-train-lm' && trainStepSeries.length > 1 && (
            <div className="mt-3">
              {/* No epoch series: MM3 trains in STEPS, so the step layer is the
                  whole chart and there is no target line to draw. */}
              <TrainingChart
                epochs={[]}
                steps={trainStepSeries}
                milestones={trainMilestones}
                target={0}
                maxEpochs={0}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Mm3TrainCard;
