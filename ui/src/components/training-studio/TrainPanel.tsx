// TrainPanel.tsx — Training Studio phase 3
//
// Tensor cache → 5 Hz FSQ codes → a Qwen3 LM LoRA in adapters/lm/<name>-<size>.
// The whole run happens in the standalone ace-train binary, which needs the GPU
// to itself: the server stops ace-server for the duration and restarts it
// afterwards, so this panel is explicit about the engine being paused.
//
// LM only in this phase. The DiT card below is a deliberate dead sibling so the
// Train phase does not read as "LoRA planner adapters, forever".

import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle, Check, Copy, Cpu, Database, FileCode2, Flag, Loader2, Lock, PauseCircle, Target, XCircle,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { TrainLmOptions } from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { JobProgress } from './JobProgress';
import { LossSparkline } from './LossSparkline';
import { TRAIN_LM_DEFAULTS, TrainLmForm, type TrainLmFormState } from './TrainLmForm';
import { useTrainingStream } from './useTrainingStream';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';
const STATUS_RELOAD_DEBOUNCE_MS = 400;

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v >= 100 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
}

function formatEtaMs(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return '';
  const m = Math.round(ms / 60000);
  if (m < 1) return '<1m';
  if (m < 60) return `${m}m`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

export const TrainPanel: React.FC = () => {
  const { t } = useTranslation();
  const capabilities = useTrainingStore(s => s.capabilities);
  const datasets = useTrainingStore(s => s.datasets);
  const detail = useTrainingStore(s => s.detail);
  const selectedDatasetId = useTrainingStore(s => s.selectedDatasetId);
  const activeJob = useTrainingStore(s => s.activeJob);
  const error = useTrainingStore(s => s.error);
  const trainLmStatus = useTrainingStore(s => s.trainLmStatus);
  const trainLmLoading = useTrainingStore(s => s.trainLmLoading);
  const trainLmEpochs = useTrainingStore(s => s.trainLmEpochs);
  const trainLmLast = useTrainingStore(s => s.trainLmLast);
  const trainLmSkippedLong = useTrainingStore(s => s.trainLmSkippedLong);
  const openDataset = useTrainingStore(s => s.openDataset);
  const setPhase = useTrainingStore(s => s.setPhase);
  const loadCapabilities = useTrainingStore(s => s.loadCapabilities);
  const loadTrainLmStatus = useTrainingStore(s => s.loadTrainLmStatus);
  const startTrainLm = useTrainingStore(s => s.startTrainLm);

  const [form, setForm] = useState<TrainLmFormState>(TRAIN_LM_DEFAULTS);
  const [starting, setStarting] = useState(false);
  const [copied, setCopied] = useState(false);
  // True from the instant the status debounce is armed until that read settles.
  // `trainLmLoading` only turns true INSIDE loadTrainLmStatus, i.e. 400 ms after
  // mount, and until then both it and `trainLmStatus` are falsy — the spinner and
  // the needs-preprocess gate below are both skipped and the fully enabled form
  // renders for that window. Seeded true so the hole never exists.
  const [statusPending, setStatusPending] = useState(true);

  const tl = capabilities?.trainLm;
  const jobKind = activeJob?.kind;
  const jobStatus = activeJob?.status;
  const jobId = activeJob?.id;
  const jobActive = jobStatus === 'queued' || jobStatus === 'running';
  const trainJobActive = jobKind === 'train-lm' && jobActive;

  // The dataset phase mounts the stream inside DatasetDetail, which is not
  // rendered here — without this the progress bar and loss curve never move.
  useTrainingStream(jobActive && jobId ? jobId : null);

  // The adapter defaults to the dataset's own slug (L12), re-seeded whenever a
  // different dataset is opened.
  const slug = detail?.slug ?? '';
  useEffect(() => {
    if (slug) setForm(f => ({ ...f, adapterName: slug }));
  }, [slug]);

  // Adapter name and size both change where the run would write, so the status
  // card follows them. Debounced — the name is a text input.
  const { lmSize, adapterName } = form;
  // Read by the terminal-job effect below without joining its dep array.
  const statusQueryRef = useRef({ lmSize, adapterName });
  statusQueryRef.current = { lmSize, adapterName };
  const statusQuery = () => {
    const { lmSize: sz, adapterName: nm } = statusQueryRef.current;
    return { lmSize: sz, ...(nm.trim() ? { adapterName: nm.trim() } : {}) };
  };

  useEffect(() => {
    if (!selectedDatasetId) return;
    setStatusPending(true);
    const handle = window.setTimeout(() => {
      void loadTrainLmStatus({
        lmSize,
        ...(adapterName.trim() ? { adapterName: adapterName.trim() } : {}),
      }).finally(() => setStatusPending(false));
    }, STATUS_RELOAD_DEBOUNCE_MS);
    return () => window.clearTimeout(handle);
  }, [selectedDatasetId, lmSize, adapterName, loadTrainLmStatus]);

  // A finished job wrote the adapter dir and un-suspended the engine.
  //
  // lmSize/adapterName are deliberately NOT in the dep array (the proven sibling
  // PreprocessPanel does the same): `activeJob` is never cleared when a job ends,
  // so after any run jobKind/jobStatus stay terminal and every keystroke in the
  // adapter-name box would re-fire this body — an undebounced status GET plus a
  // /capabilities call, and /capabilities does a live /props round-trip to
  // ace-server. They are read through a ref instead.
  useEffect(() => {
    if (jobKind !== 'train-lm' || !jobStatus) return;
    if (jobStatus === 'queued' || jobStatus === 'running') return;
    void loadTrainLmStatus(statusQuery());
    void loadCapabilities();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, jobKind, jobStatus, loadTrainLmStatus, loadCapabilities]);

  const patchForm = (patch: Partial<TrainLmFormState>) => setForm(f => ({ ...f, ...patch }));

  const handleStart = async () => {
    const opts: TrainLmOptions = {
      lmSize: form.lmSize,
      // §2.7: OMITTED means "let the server pick the BF16 base for this size".
      // Sending '' would be forwarded as `--lm ''` and ace-train exits 2.
      ...(form.lmModel ? { lmModel: form.lmModel } : {}),
      ...(trainLmStatus?.variantKey ? { variantKey: trainLmStatus.variantKey } : {}),
      adapterName: form.adapterName.trim(),
      targetLoss: form.targetLoss,
      epochs: form.epochs,
      rank: form.rank,
      alpha: form.alpha,
      learningRate: form.learningRate,
      gradAccum: form.gradAccum,
      gradClip: form.gradClip,
      warmupRatio: form.warmupRatio,
      weightDecay: form.weightDecay,
      maxLen: form.maxLen,
      seed: form.seed,
      lossOnCot: form.lossOnCot,
      order: form.order,
      milestoneStep: form.milestoneStep,
      milestoneKeep: form.milestoneKeep,
      stages: form.stages,
      overwrite: form.overwrite,
      stopEngine: form.stopEngine,
    };
    setStarting(true);
    try { await startTrainLm(opts); } finally { setStarting(false); }
  };

  const copyAdapterDir = () => {
    const dir = trainLmStatus?.adapterDir;
    if (!dir) return;
    void navigator.clipboard?.writeText(dir).then(
      () => { setCopied(true); window.setTimeout(() => setCopied(false), 1500); },
      () => { /* clipboard blocked — the path is selectable on screen anyway */ },
    );
  };

  const header = (
    <div className="flex items-start gap-3">
      <Cpu size={18} className="text-amber-500 mt-0.5 flex-shrink-0" />
      <div className="min-w-0">
        <h2 className="text-sm font-bold text-zinc-900 dark:text-white">{t('trainingStudio.train.title')}</h2>
        <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-0.5">{t('trainingStudio.train.subtitle')}</p>
      </div>
    </div>
  );

  // ── Gate 1: no dataset selected ─────────────────────────────────────────
  if (!selectedDatasetId) {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <div className={`${CARD} flex flex-col gap-3`}>
          <p className="text-sm text-zinc-600 dark:text-zinc-400">{t('trainingStudio.train.pickDataset')}</p>
          <div className="grid grid-cols-[repeat(auto-fill,minmax(240px,1fr))] gap-2">
            {datasets.map(ds => (
              <button
                key={ds.id}
                onClick={() => void openDataset(ds.id)}
                className="flex items-center gap-2 px-3 py-2 rounded-lg border border-zinc-200 dark:border-white/10 hover:border-amber-500/40 text-left transition-colors"
              >
                <Database size={14} className="text-amber-500 flex-shrink-0" />
                <span className="flex-1 min-w-0">
                  <span className="block text-sm font-semibold text-zinc-900 dark:text-white truncate">{ds.name}</span>
                  <span className="block text-[11px] text-zinc-500 truncate">
                    {ds.builtAt
                      ? t('trainingStudio.list.samples', { count: ds.sampleCount })
                      : t('trainingStudio.list.notBuilt')}
                  </span>
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!detail || !capabilities) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <Loader2 size={18} className="animate-spin mr-2" /> …
      </div>
    );
  }

  // ── Gate 2: no ace-train binary ─────────────────────────────────────────
  if (!tl?.available) {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <div className="rounded-xl border border-red-500/25 bg-red-500/10 p-5 flex items-start gap-2 text-sm text-red-500 dark:text-red-400">
          <XCircle size={16} className="mt-0.5 flex-shrink-0" />
          {t('trainingStudio.train.noBinary')}
        </div>
      </div>
    );
  }

  // ── Gate 3: not built ───────────────────────────────────────────────────
  if (!detail.builtAt || !detail.datasetJsonPath) {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <div className="rounded-xl border border-amber-500/25 bg-amber-500/10 p-5 flex flex-col items-start gap-3">
          <div className="flex items-start gap-2 text-sm text-amber-600 dark:text-amber-400">
            <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
            {t('trainingStudio.train.needsBuild')}
          </div>
          <button
            onClick={() => setPhase('dataset')}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/15 border border-amber-500/25 text-amber-600 dark:text-amber-400 hover:bg-amber-500/25 transition-colors"
          >
            {t('trainingStudio.preprocess.goToDataset')}
          </button>
        </div>
      </div>
    );
  }

  // Status still in flight on first open — deciding gate 4 needs it. Covers the
  // debounce window too (statusPending), not just the request itself.
  if (!trainLmStatus && (trainLmLoading || statusPending)) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <Loader2 size={18} className="animate-spin mr-2" /> …
      </div>
    );
  }

  // ── Gate 4: no preprocessed tensors ─────────────────────────────────────
  // Only when the status actually loaded — a failed read must not masquerade
  // as "you never preprocessed".
  if (trainLmStatus && trainLmStatus.variantKey === '') {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <div className="rounded-xl border border-amber-500/25 bg-amber-500/10 p-5 flex flex-col items-start gap-3">
          <div className="flex items-start gap-2 text-sm text-amber-600 dark:text-amber-400">
            <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
            {t('trainingStudio.train.needsPreprocess')}
          </div>
          <button
            onClick={() => setPhase('preprocess')}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/15 border border-amber-500/25 text-amber-600 dark:text-amber-400 hover:bg-amber-500/25 transition-colors"
          >
            {t('trainingStudio.train.goToPreprocess')}
          </button>
        </div>
      </div>
    );
  }

  // ── Gate 5: another kind of job owns this dataset ───────────────────────
  if (jobActive && jobKind !== 'train-lm') {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
          <PauseCircle size={13} className="mt-0.5 flex-shrink-0" />
          {t('trainingStudio.train.otherJobRunning')}
        </div>
        <JobProgress />
      </div>
    );
  }

  const enginePaused = form.stopEngine || !!capabilities.preprocess?.engineSuspended;
  const status = trainLmStatus;
  const etaLabel = formatEtaMs(trainLmLast?.etaMs ?? 0);

  return (
    <div className="flex flex-col gap-4">
      {header}

      <div className="flex items-center gap-2 text-[11px] text-zinc-500">
        <Database size={12} className="text-amber-500 flex-shrink-0" />
        <span className="font-semibold text-zinc-700 dark:text-zinc-300 truncate">{detail.name}</span>
        {status?.tensorsDir && (
          <span className="font-mono truncate" title={status.tensorsDir}>{status.tensorsDir}</span>
        )}
      </div>

      {/* Codes freshness. The server computes these at real cost (every line of
          lm_codes.jsonl plus one statSync per row); without surfacing them the
          user has no signal that the extract stage is about to re-encode the
          whole dataset — or that they should tick "Re-extract codes". */}
      {status?.variantKey && (
        <div className="flex items-center gap-2 flex-wrap text-[11px] text-zinc-500 -mt-2">
          <FileCode2 size={12} className="text-amber-500 flex-shrink-0" />
          <span className="tabular-nums">
            {status.codesExists
              ? t('trainingStudio.train.codes', { count: status.codesCount })
              : t('trainingStudio.train.codesNone')}
          </span>
          {status.codesStale > 0 && (
            <span className="tabular-nums px-1.5 py-0.5 rounded-full border text-amber-600 dark:text-amber-400 bg-amber-500/10 border-amber-500/20">
              {t('trainingStudio.train.codesStale', { count: status.codesStale })}
            </span>
          )}
          {status.codesMissing > 0 && (
            <span className="tabular-nums px-1.5 py-0.5 rounded-full border text-amber-600 dark:text-amber-400 bg-amber-500/10 border-amber-500/20">
              {t('trainingStudio.train.codesMissing', { count: status.codesMissing })}
            </span>
          )}
        </div>
      )}

      {error && (
        <div className="px-3 py-2 rounded-lg border border-red-500/25 bg-red-500/10 text-xs text-red-500 dark:text-red-400">{error}</div>
      )}

      {/* ── LM card ──────────────────────────────────────────────────────── */}
      <div className={`${CARD} flex flex-col gap-4`}>
        <div className="flex items-start gap-2">
          <Cpu size={15} className="text-amber-500 mt-0.5 flex-shrink-0" />
          <div className="min-w-0">
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white">{t('trainingStudio.train.lmTitle')}</h3>
            <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-0.5">{t('trainingStudio.train.lmSubtitle')}</p>
          </div>
        </div>

        {trainJobActive && enginePaused && (
          <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
            <PauseCircle size={13} className="mt-0.5 flex-shrink-0" />
            {t('trainingStudio.preprocess.enginePaused')}
          </div>
        )}

        <TrainLmForm
          capabilities={capabilities}
          value={form}
          onChange={patchForm}
          disabled={jobActive || starting}
          starting={starting}
          onStart={() => void handleStart()}
        />
      </div>

      {/* ── Job area ─────────────────────────────────────────────────────── */}
      {(jobKind === 'train-lm' || jobActive) && (
        <div className="flex flex-col gap-3">
          <JobProgress />

          {trainLmSkippedLong > 0 && (
            <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
              <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
              {t('trainingStudio.train.skippedLong', { count: trainLmSkippedLong })}
            </div>
          )}

          {trainLmEpochs.length >= 2 && (
            <div className={`${CARD} flex flex-col gap-2`}>
              <LossSparkline epochs={trainLmEpochs} target={form.targetLoss} />
            </div>
          )}

          {trainLmLast && (
            <div className={`${CARD} grid grid-cols-2 sm:grid-cols-4 gap-3`}>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wide text-zinc-500">{t('trainingStudio.train.epochLoss')}</span>
                <span className="text-sm font-bold tabular-nums text-zinc-900 dark:text-white">{trainLmLast.loss.toFixed(4)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wide text-zinc-500">{t('trainingStudio.train.learningRateShort')}</span>
                <span className="text-sm font-bold tabular-nums text-zinc-900 dark:text-white">{trainLmLast.lr.toExponential(2)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wide text-zinc-500">{t('trainingStudio.train.gradNorm')}</span>
                <span className="text-sm font-bold tabular-nums text-zinc-900 dark:text-white">{trainLmLast.gradNorm.toFixed(3)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wide text-zinc-500">{t('trainingStudio.train.eta')}</span>
                <span className="text-sm font-bold tabular-nums text-zinc-900 dark:text-white">{etaLabel || '—'}</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Done state ───────────────────────────────────────────────────── */}
      {status?.adapterExists && (
        <div className={`${CARD} flex flex-col gap-3 border-emerald-500/30`}>
          <div className="flex items-center gap-2">
            <Check size={15} className="text-emerald-500 flex-shrink-0" />
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white">{t('trainingStudio.train.done')}</h3>
            {status.stoppedOnTarget && (
              <span className="flex items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded-full border text-emerald-500 bg-emerald-500/10 border-emerald-500/20">
                <Target size={10} /> {t('trainingStudio.train.stoppedOnTarget')}
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <code className="flex-1 min-w-0 truncate font-mono text-[11px] text-zinc-600 dark:text-zinc-400" title={status.adapterDir}>
              {status.adapterDir}
            </code>
            <button
              onClick={copyAdapterDir}
              className="p-1.5 rounded-lg text-zinc-500 hover:text-amber-500 hover:bg-amber-500/10 transition-colors flex-shrink-0"
              title={status.adapterDir}
            >
              {copied ? <Check size={13} className="text-emerald-500" /> : <Copy size={13} />}
            </button>
          </div>

          <div className="flex items-center gap-3 flex-wrap text-[11px] text-zinc-600 dark:text-zinc-400">
            {status.finalLoss >= 0 && (
              <span className="tabular-nums">
                {t('trainingStudio.train.finalLoss')}: <strong>{status.finalLoss.toFixed(4)}</strong>
              </span>
            )}
            {status.epochsRun > 0 && (
              <>
                <span className="text-zinc-400 dark:text-zinc-600">·</span>
                <span className="tabular-nums">
                  {t('trainingStudio.train.epochsRun')}: <strong>{status.epochsRun}</strong>
                </span>
              </>
            )}
            {status.adapterBytes > 0 && (
              <>
                <span className="text-zinc-400 dark:text-zinc-600">·</span>
                <span className="tabular-nums">{formatBytes(status.adapterBytes)}</span>
              </>
            )}
            {status.lmSize && (
              <>
                <span className="text-zinc-400 dark:text-zinc-600">·</span>
                <span>{status.lmSize}</span>
              </>
            )}
          </div>

          {status.epochs.length >= 2 && (
            <LossSparkline epochs={status.epochs} target={status.targetLoss} />
          )}

          {status.milestones.length > 0 && (
            <div className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wide text-zinc-500">{t('trainingStudio.train.milestones')}</span>
              <div className="flex items-center gap-2 flex-wrap">
                {status.milestones.map(m => (
                  <span
                    key={m.path}
                    title={m.path}
                    className="flex items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded-full border text-sky-500 bg-sky-500/10 border-sky-500/20 tabular-nums"
                  >
                    <Flag size={10} /> {m.loss.toFixed(1)} · #{m.epoch}
                  </span>
                ))}
              </div>
            </div>
          )}

          <p className="text-xs text-emerald-600 dark:text-emerald-400">{t('trainingStudio.train.visibleNow')}</p>
        </div>
      )}

      {/* ── DiT sub-card (deliberately inert) ────────────────────────────── */}
      <div className={`${CARD} flex items-start gap-3 opacity-60`}>
        <Lock size={15} className="text-zinc-400 dark:text-zinc-600 mt-0.5 flex-shrink-0" />
        <div className="min-w-0">
          <h3 className="text-sm font-bold text-zinc-700 dark:text-zinc-300">{t('trainingStudio.train.dit.title')}</h3>
          <p className="text-xs text-zinc-500 mt-0.5">{t('trainingStudio.train.dit.comingSoon')}</p>
        </div>
      </div>
    </div>
  );
};

export default TrainPanel;
