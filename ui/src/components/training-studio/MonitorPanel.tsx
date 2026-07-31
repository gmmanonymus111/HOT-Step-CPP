// MonitorPanel.tsx — batch pipeline queue view (PhaseStepper's "monitor" tab)
//
// The LIST is polling-only: while a pipeline is running this re-fetches every
// 2s, and on completion it also refreshes the dataset list so newly-created
// datasets appear without a manual reload.
//
// DEVIATION from plan §5.2, which forbade wiring useTrainingStream here to keep
// that SSE hook per-dataset. The stage CHIPS say which stage an item is on and
// nothing about how far through it is, so a multi-hour train-dit inside a bulk
// run had no visible progress anywhere in the app (Rob, 2026-07-31). The
// subscription is delegated to PipelineStageProgress, which owns exactly one
// stream — the single running stage's — and unsubscribes when it changes. The
// per-dataset invariant §5.2 was protecting still holds; what changed is that
// the bulk runner is now also a legitimate owner of a job worth watching.

import React, { useEffect, useRef, useState } from 'react';
import { AlertTriangle, CheckCircle2, ChevronDown, ChevronRight, Clock, Loader2, XCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useTrainingStore } from '../../stores/trainingStore';
import { PIPELINE_STAGES, type PipelineItem, type PipelineStage, type PipelineSummary } from '../../services/trainingApi';
import { ConfirmDialog } from '../shared/ConfirmDialog';
import { PipelineStageProgress } from './PipelineStageProgress';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';
const POLL_MS = 2000;

const STAGE_LABEL_KEYS: Record<PipelineStage, string> = {
  label: 'trainingStudio.batch.stageLabel',
  build: 'trainingStudio.batch.stageBuild',
  preprocess: 'trainingStudio.batch.stagePreprocess',
  'train-dit': 'trainingStudio.batch.stageTrainDit',
  'train-lm': 'trainingStudio.batch.stageTrainLm',
  'lyric-studio': 'trainingStudio.batch.stageLyricStudio',
};

const CHIP_STYLE: Record<string, string> = {
  pending:   'text-zinc-500 bg-zinc-500/10 border-zinc-500/20',
  creating:  'text-sky-500 bg-sky-500/10 border-sky-500/20 animate-pulse',
  running:   'text-amber-500 bg-amber-500/10 border-amber-500/20 animate-pulse',
  done:      'text-emerald-500 bg-emerald-500/10 border-emerald-500/20',
  failed:    'text-rose-500 bg-rose-500/10 border-rose-500/20',
  cancelled: 'text-zinc-500 bg-zinc-500/10 border-zinc-500/20 line-through',
};

function folderBasename(dir: string): string {
  return dir.replace(/[\\/]+$/, '').split(/[\\/]/).pop() ?? dir;
}

function formatWhen(ts: number): string {
  if (!ts) return '';
  return new Date(ts).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

const PipelineItemRow: React.FC<{ item: PipelineItem; stages: PipelineStage[] }> = ({ item, stages }) => {
  const { t } = useTranslation();
  const openDataset = useTrainingStore(s => s.openDataset);
  const setPhase = useTrainingStore(s => s.setPhase);
  const stageByName = new Map(item.stages.map(s => [s.stage, s]));

  return (
    <div className="flex flex-col gap-1.5 py-2">
      <div className="flex items-center gap-2 flex-wrap">
        {item.datasetId ? (
          <button
            onClick={() => { void openDataset(item.datasetId); setPhase('dataset'); }}
            className="text-xs font-semibold text-zinc-800 dark:text-zinc-200 hover:text-amber-500 transition-colors truncate max-w-[240px]"
            title={item.sourceDir}
          >
            {folderBasename(item.sourceDir)} → {item.name}
          </button>
        ) : (
          <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200 truncate max-w-[240px]" title={item.sourceDir}>
            {folderBasename(item.sourceDir)}
          </span>
        )}
        {item.reusedExisting && (
          <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full border text-sky-500 bg-sky-500/10 border-sky-500/20">
            {t('trainingStudio.monitor.reused')}
          </span>
        )}
        <div className="flex items-center gap-1 flex-wrap">
          {stages.map(stage => {
            const result = stageByName.get(stage);
            const status = result?.status ?? 'pending';
            return (
              <span
                key={stage}
                title={result?.error || undefined}
                className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full border ${CHIP_STYLE[status] || CHIP_STYLE.pending}`}
              >
                {t(STAGE_LABEL_KEYS[stage])}
              </span>
            );
          })}
        </div>
      </div>
      {item.error && <div className="text-[11px] text-rose-500 dark:text-rose-400">{item.error}</div>}
    </div>
  );
};

const PipelineCard: React.FC<{ pipeline: PipelineSummary; active: boolean }> = ({ pipeline, active }) => {
  const { t } = useTranslation();
  const cancelPipeline = useTrainingStore(s => s.cancelPipeline);
  const [confirmCancel, setConfirmCancel] = useState(false);

  const doneItems = pipeline.items.filter(i => i.status === 'done').length;
  const failedItems = pipeline.items.filter(i => i.status === 'failed').length;

  return (
    <div className={`${CARD} flex flex-col gap-3 ${pipeline.status === 'failed' ? 'border-rose-500/30' : ''}`}>
      <div className="flex items-center gap-2 flex-wrap">
        {pipeline.status === 'running' && <Loader2 size={15} className="animate-spin text-amber-500 flex-shrink-0" />}
        {pipeline.status === 'done' && <CheckCircle2 size={15} className="text-emerald-500 flex-shrink-0" />}
        {pipeline.status === 'failed' && <XCircle size={15} className="text-rose-500 flex-shrink-0" />}
        {pipeline.status === 'cancelled' && <AlertTriangle size={15} className="text-amber-500 flex-shrink-0" />}
        <div className="text-sm font-semibold text-zinc-800 dark:text-zinc-200 flex-1 min-w-0">
          {t('trainingStudio.monitor.progress', { done: doneItems, total: pipeline.items.length })}
          {failedItems > 0 && (
            <span className="text-rose-500 ml-2">{t('trainingStudio.monitor.failedCount', { count: failedItems })}</span>
          )}
        </div>
        <span className="text-[11px] text-zinc-500 flex items-center gap-1 flex-shrink-0">
          <Clock size={11} /> {formatWhen(pipeline.createdAt)}
        </span>
        {active && (
          <button
            onClick={() => setConfirmCancel(true)}
            className="px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-red-500/10 border border-red-500/20 text-red-500 hover:bg-red-500/20 transition-colors flex-shrink-0"
          >
            {t('trainingStudio.monitor.cancel')}
          </button>
        )}
      </div>

      {/* The stage actually executing. The chips below say WHICH stage each item
          is on; this is the only place that shows how far through it is. */}
      <PipelineStageProgress item={pipeline.items.find(i => i.status === 'running') ?? null} />

      <div className="divide-y divide-zinc-200 dark:divide-white/5">
        {pipeline.items.map(item => (
          <PipelineItemRow key={item.sourceDir} item={item} stages={pipeline.stages.length > 0 ? pipeline.stages : PIPELINE_STAGES.slice()} />
        ))}
      </div>

      <ConfirmDialog
        isOpen={confirmCancel}
        title={t('trainingStudio.monitor.cancel')}
        message={t('trainingStudio.monitor.cancelConfirm')}
        danger
        onConfirm={() => { setConfirmCancel(false); void cancelPipeline(pipeline.id); }}
        onCancel={() => setConfirmCancel(false)}
      />
    </div>
  );
};

export const MonitorPanel: React.FC = () => {
  const { t } = useTranslation();
  const pipelines = useTrainingStore(s => s.pipelines);
  const pipelinesLoading = useTrainingStore(s => s.pipelinesLoading);
  const loadPipelines = useTrainingStore(s => s.loadPipelines);
  const loadDatasets = useTrainingStore(s => s.loadDatasets);
  const [recentOpen, setRecentOpen] = useState(false);

  useEffect(() => {
    void loadPipelines();
  }, [loadPipelines]);

  const activePipeline = pipelines.find(p => p.status === 'running') ?? null;
  const recent = pipelines.filter(p => p.id !== activePipeline?.id);

  // Poll only while something is running; once it drops out, catch up the
  // dataset list once so freshly-created datasets appear without a reload.
  const wasRunningRef = useRef(false);
  useEffect(() => {
    if (!activePipeline) {
      if (wasRunningRef.current) { wasRunningRef.current = false; void loadDatasets(); }
      return;
    }
    wasRunningRef.current = true;
    const handle = window.setInterval(() => { void loadPipelines(); }, POLL_MS);
    return () => window.clearInterval(handle);
  }, [activePipeline, loadPipelines, loadDatasets]);

  if (pipelinesLoading && pipelines.length === 0) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <Loader2 size={18} className="animate-spin mr-2" /> …
      </div>
    );
  }

  if (pipelines.length === 0) {
    return (
      <div className={`${CARD} flex flex-col items-center text-center gap-2 py-14`}>
        <div className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">{t('trainingStudio.monitor.empty')}</div>
        <p className="text-xs text-zinc-500 max-w-md">{t('trainingStudio.monitor.emptyHint')}</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {activePipeline ? (
        <PipelineCard pipeline={activePipeline} active />
      ) : (
        <div className={`${CARD} text-xs text-zinc-500`}>{t('trainingStudio.monitor.noActive')}</div>
      )}

      {recent.length > 0 && (
        <div className="flex flex-col gap-2">
          <button
            onClick={() => setRecentOpen(o => !o)}
            className="flex items-center gap-1.5 text-xs font-semibold text-zinc-600 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200 transition-colors self-start"
          >
            {recentOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
            {t('trainingStudio.monitor.recent', { count: recent.length })}
          </button>
          {recentOpen && (
            <div className="flex flex-col gap-3">
              {recent.map(p => <PipelineCard key={p.id} pipeline={p} active={false} />)}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default MonitorPanel;
