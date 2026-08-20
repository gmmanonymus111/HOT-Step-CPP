// Mm3TrainCard.tsx — Training Studio phase 3, MiniMax-Music3 branch.
//
// Rendered instead of the ACE LM/DiT cards when the active backend is
// MiniMax-Music3. Two stages, in order, because the second cannot run without
// the first:
//
//   1. CODES   audio -> RVQ codes (`ace-train mm3-codes`). MM3 ships decode-side
//              only, so the codes an LM trains on come from a community encoder;
//              this is that export, run natively.
//   2. TRAIN   codes + the MM3-native captions -> an LM LoRA.
//
// Everything below the buttons is the SHARED job machinery: the same SSE
// stream, the same JobProgress, the same loss chart. That is why this file is
// a form and two status cards rather than a studio.
//
// THE DEFAULTS ARE NOT DUPLICATED HERE. They arrive in the `mm3` status
// payload from services/training/mm3Train.ts, which is the single place the
// validated recipe lives. This form seeds itself from that response, so a
// change on the server reaches the UI without an edit here.

import React, { useEffect, useState } from 'react';
import {
  AlertTriangle, ChevronDown, ChevronRight, Cpu, FileCode2, Loader2, Play, XCircle,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';

import * as trainingApi from '../../services/trainingApi';
import type { Mm3Status, Mm3TrainLmRequest } from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { JobProgress } from './JobProgress';
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
  const startMm3Codes = useTrainingStore(s => s.startMm3Codes);
  const startMm3TrainLm = useTrainingStore(s => s.startMm3TrainLm);
  const trainStepSeries = useTrainingStore(s => s.trainStepSeries);
  const trainMilestones = useTrainingStore(s => s.trainMilestones);

  const [status, setStatus] = useState<Mm3Status | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [form, setForm] = useState<FormState | null>(null);

  const jobKind = activeJob?.kind;
  const jobStatus = activeJob?.status;
  const jobRunning = jobStatus === 'queued' || jobStatus === 'running';
  const mine = jobKind === 'mm3-codes' || jobKind === 'mm3-train-lm';

  // ONE fetch effect, re-keyed rather than re-called. `finishedKey` changes when
  // one of OUR jobs reaches a terminal state, which is exactly when the codes
  // cache on disk may have changed — so a finished export refreshes the counts
  // without a second effect calling setState synchronously to trigger it.
  const finishedKey = mine && !jobRunning ? `${activeJob?.id ?? ''}:${jobStatus ?? ''}` : '';
  useEffect(() => {
    let cancelled = false;
    trainingApi.getMm3Status(datasetId)
      .then(s => {
        if (cancelled) return;
        setStatus(s);
        setError(null);
        // Seed ONCE, from the server's own defaults — never from constants
        // here, so the recipe has exactly one home.
        setForm(prev => prev ?? {
          steps: s.defaults.steps ?? 800,
          saveEvery: s.defaults.saveEvery ?? 100,
          rank: s.defaults.rank ?? 256,
          alpha: s.defaults.alpha ?? 256,
          lr: s.defaults.lr ?? 8e-5,
          maxFrames: s.defaults.maxFrames ?? 1500,
          cropMode: (s.defaults.cropMode as 'random' | 'beginning') ?? 'random',
          optimizer: s.defaults.optimizer ?? 'muon',
          muonLrScale: s.defaults.muonLrScale ?? 64,
          gradAccum: s.defaults.gradAccum ?? 1,
          seed: s.defaults.seed ?? 42,
          trigger: trigger ?? '',
        });
      })
      .catch(err => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => { cancelled = true; };
  }, [datasetId, trigger, finishedKey]);

  const startCodes = async () => {
    setBusy(true);
    try {
      await startMm3Codes();
    } finally {
      setBusy(false);
    }
  };

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

  if (!status && !error) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <Loader2 size={18} className="animate-spin mr-2" /> …
      </div>
    );
  }

  const set = <K extends keyof FormState>(k: K, v: FormState[K]) =>
    setForm(f => (f ? { ...f, [k]: v } : f));

  const hasCodes = (status?.codes ?? 0) > 0;
  const codesBlocked = (status?.missingForCodes.length ?? 0) > 0;
  const trainBlocked = (status?.missingForTrain.length ?? 0) > 0;

  return (
    <div className="flex flex-col gap-4">
      {error && (
        <div className="rounded-xl border border-red-500/25 bg-red-500/10 p-3 flex items-start gap-2 text-sm text-red-500">
          <XCircle size={16} className="mt-0.5 flex-shrink-0" />
          <span className="min-w-0 break-words">{error}</span>
        </div>
      )}

      {/* ── Stage 1: codes ── */}
      <div className={CARD}>
        <div className="flex items-center gap-2 mb-2">
          <FileCode2 size={15} className="text-amber-500" />
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
            {t('trainingStudio.mm3.codesTitle', 'RVQ codes')}
          </h3>
        </div>
        <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
          {t('trainingStudio.mm3.codesBlurb',
            'MiniMax-Music3 ships no audio-to-code encoder, so the tokens the LM learns from are produced '
            + 'here, natively, by the adopted community encoder. Run this once per dataset; re-running '
            + 'overwrites with whichever encoder is installed now.')}
        </p>
        {codesBlocked ? (
          <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
            <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
            <span>
              {t('trainingStudio.mm3.missing', 'Missing model files')}: {status?.missingForCodes.join(', ')}
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={() => void startCodes()}
              disabled={busy || jobRunning}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/15 border border-amber-500/25 text-amber-600 dark:text-amber-400 hover:bg-amber-500/25 disabled:opacity-40 transition-colors"
            >
              {hasCodes
                ? t('trainingStudio.mm3.reexport', 'Re-export codes')
                : t('trainingStudio.mm3.export', 'Export codes')}
            </button>
            <span className="text-xs text-zinc-500">
              {hasCodes
                ? `${status?.codes} ${t('trainingStudio.mm3.codesReady', 'tracks encoded')}`
                  + (status?.encoder ? ` · ${status.encoder}` : '')
                : t('trainingStudio.mm3.noCodes', 'no codes yet')}
            </span>
          </div>
        )}
      </div>

      {/* ── Stage 2: train ── */}
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
          <div className="flex items-start gap-2 text-xs text-zinc-500">
            <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
            {t('trainingStudio.mm3.needsCodes', 'Export the RVQ codes first — training reads them, not the audio.')}
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
