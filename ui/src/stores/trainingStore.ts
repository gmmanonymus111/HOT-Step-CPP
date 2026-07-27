// trainingStore.ts — Zustand store for the Training Studio (Dataset phase)
//
// Disk is the source of truth for sample data; this store mirrors what the
// server reports. `samplesById` is the authoritative row map — SSE `sample`
// events REPLACE entries by sampleId (never append), so an EventSource
// reconnect that replays the whole buffer is idempotent.

import { create } from 'zustand';
import * as trainingApi from '../services/trainingApi';
import type {
  BulkSetInput,
  CaptionOptions,
  CreateDatasetInput,
  GeniusOptions,
  LabelOptions,
  PatchDatasetInput,
  PatchSampleInput,
  TrainingCapabilities,
  TrainingDatasetDetail,
  TrainingDatasetSummary,
  TrainingJobSummary,
  TrainingSample,
  TrainingStreamEvent,
} from '../services/trainingApi';

const JOB_LOG_CAP = 200;
const EDIT_DEBOUNCE_MS = 500;

/** Fields the grid/drawer may edit inline. */
export type EditableField = keyof PatchSampleInput;

// Debounce timers live outside React state — one per (sampleId, field).
const editTimers = new Map<string, number>();
// Jobs whose terminal `status` event already triggered the authoritative
// refresh. The SSE buffer is replayed on reconnect, so this must be guarded.
const refreshedJobs = new Set<string>();

function timerKey(sampleId: string, field: string): string {
  return `${sampleId}:${field}`;
}

function clearTimer(key: string): void {
  const t = editTimers.get(key);
  if (t !== undefined) { window.clearTimeout(t); editTimers.delete(key); }
}

interface TrainingState {
  // navigation
  phase: 'dataset' | 'preprocess' | 'train' | 'monitor';   // only 'dataset' selectable in v1
  step: 'label' | 'review' | 'build';
  selectedDatasetId: string | null;

  // data
  capabilities: TrainingCapabilities | null;
  datasets: TrainingDatasetSummary[];
  detail: TrainingDatasetDetail | null;
  samplesById: Record<string, TrainingSample>;   // authoritative row map, merged from SSE
  sampleOrder: string[];                          // display order (relPath asc)

  // job
  activeJob: TrainingJobSummary | null;
  jobLog: Array<{ level: string; message: string; ts: number }>;   // capped at 200

  // grid
  selectedSampleIds: Set<string>;
  openSampleId: string | null;
  pendingEdits: Record<string, PatchSampleInput>;  // optimistic, keyed by sampleId
  savingSampleIds: Set<string>;
  saveError: { sampleId: string; message: string } | null;
  loading: boolean;
  error: string | null;

  // actions
  setPhase(phase: TrainingState['phase']): void;
  setStep(step: TrainingState['step']): void;
  closeDataset(): void;
  loadCapabilities(): Promise<void>;
  loadDatasets(): Promise<void>;
  openDataset(id: string): Promise<void>;
  createDataset(input: CreateDatasetInput): Promise<string>;
  patchDataset(patch: PatchDatasetInput): Promise<void>;
  deleteDataset(id: string): Promise<void>;
  rescan(): Promise<void>;
  editSample(sampleId: string, patch: PatchSampleInput): Promise<void>;   // debounced PATCH
  toggleExcluded(sampleId: string, excluded: boolean): Promise<void>;      // works on missing files too
  flushSample(sampleId: string): Promise<void>;                           // blur / Enter
  revertSampleField(sampleId: string, field: EditableField): void;         // Escape
  clearSaveError(): void;
  bulkSet(set: BulkSetInput): Promise<void>;
  toggleSampleSelected(sampleId: string): void;
  setSelectedSampleIds(ids: string[]): void;
  clearSelection(): void;
  setOpenSampleId(sampleId: string | null): void;
  startLabel(opts: LabelOptions): Promise<void>;
  startGenius(opts: GeniusOptions): Promise<void>;
  startCaption(opts: CaptionOptions): Promise<void>;
  startBuild(outputPath?: string): Promise<void>;
  cancelJob(): Promise<void>;
  applyStreamEvent(ev: TrainingStreamEvent): void;   // called by useTrainingStream
}

/** Split a detail payload into the row map + display order. */
function indexSamples(samples: TrainingSample[]): { byId: Record<string, TrainingSample>; order: string[] } {
  const byId: Record<string, TrainingSample> = {};
  for (const s of samples) byId[s.sampleId] = s;
  const order = [...samples]
    .sort((a, b) => a.relPath.localeCompare(b.relPath))
    .map(s => s.sampleId);
  return { byId, order };
}

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  phase: 'dataset',
  step: 'label',
  selectedDatasetId: null,

  capabilities: null,
  datasets: [],
  detail: null,
  samplesById: {},
  sampleOrder: [],

  activeJob: null,
  jobLog: [],

  selectedSampleIds: new Set<string>(),
  openSampleId: null,
  pendingEdits: {},
  savingSampleIds: new Set<string>(),
  saveError: null,
  loading: false,
  error: null,

  setPhase: (phase) => set({ phase }),
  setStep: (step) => set({ step }),

  closeDataset: () => set({
    selectedDatasetId: null,
    detail: null,
    samplesById: {},
    sampleOrder: [],
    activeJob: null,
    jobLog: [],
    selectedSampleIds: new Set<string>(),
    openSampleId: null,
    pendingEdits: {},
    savingSampleIds: new Set<string>(),
    saveError: null,
    // Leaving a dataset must not carry its error back to the list, where
    // TrainingStudio would read it as a fatal load failure.
    error: null,
    step: 'label',
  }),

  loadCapabilities: async () => {
    try {
      const capabilities = await trainingApi.getCapabilities();
      set({ capabilities });
    } catch (err) {
      // Capabilities are advisory — a failure must never blank the studio.
      console.warn('[Training] capabilities failed:', errMessage(err));
      set({ capabilities: null });
    }
  },

  loadDatasets: async () => {
    set({ loading: true, error: null });
    try {
      const datasets = await trainingApi.listDatasets();
      set({ datasets, loading: false });
    } catch (err) {
      set({ error: errMessage(err), loading: false });
    }
  },

  openDataset: async (id) => {
    const isSwitch = get().selectedDatasetId !== id;
    set({ loading: true, error: null });
    try {
      const detail = await trainingApi.getDataset(id);
      const { byId, order } = indexSamples(detail.samples);
      set({
        detail,
        selectedDatasetId: id,
        samplesById: byId,
        sampleOrder: order,
        loading: false,
        ...(isSwitch
          ? {
              step: detail.labeledCount > 0 ? ('review' as const) : ('label' as const),
              selectedSampleIds: new Set<string>(),
              openSampleId: null,
              pendingEdits: {},
              jobLog: [],
            }
          : {}),
      });
      if (detail.activeJobId) {
        try { set({ activeJob: await trainingApi.getJob(detail.activeJobId) }); } catch { /* job may have just ended */ }
      } else if (isSwitch) {
        set({ activeJob: null });
      }
    } catch (err) {
      set({ error: errMessage(err), loading: false });
    }
  },

  createDataset: async (input) => {
    const detail = await trainingApi.createDataset(input);
    const { byId, order } = indexSamples(detail.samples);
    set({
      detail,
      selectedDatasetId: detail.id,
      samplesById: byId,
      sampleOrder: order,
      step: 'label',
      selectedSampleIds: new Set<string>(),
      openSampleId: null,
      pendingEdits: {},
      jobLog: [],
      activeJob: null,
    });
    await get().loadDatasets();
    return detail.id;
  },

  patchDataset: async (patch) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const summary = await trainingApi.patchDataset(id, patch);
      const detail = get().detail;
      set({
        detail: detail ? { ...detail, ...summary } : detail,
        datasets: get().datasets.map(d => (d.id === id ? { ...d, ...summary } : d)),
      });
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  deleteDataset: async (id) => {
    await trainingApi.deleteDataset(id);
    if (get().selectedDatasetId === id) get().closeDataset();
    await get().loadDatasets();
  },

  rescan: async () => {
    const id = get().selectedDatasetId;
    if (!id) return;
    set({ loading: true, error: null });
    try {
      const detail = await trainingApi.rescanDataset(id);
      const { byId, order } = indexSamples(detail.samples);
      set({ detail, samplesById: byId, sampleOrder: order, loading: false });
    } catch (err) {
      set({ error: errMessage(err), loading: false });
    }
  },

  editSample: async (sampleId, patch) => {
    const sample = get().samplesById[sampleId];
    // Rows the job is currently rewriting are read-only.
    if (sample && sample.labelStatus === 'processing') return;

    // 1. Optimistic — the cell renders from samplesById merged with pendingEdits.
    set({
      pendingEdits: { ...get().pendingEdits, [sampleId]: { ...get().pendingEdits[sampleId], ...patch } },
      saveError: null,
    });

    // 2. Debounce 500 ms per (sampleId, field).
    for (const field of Object.keys(patch) as EditableField[]) {
      const key = timerKey(sampleId, field);
      clearTimer(key);
      const handle = window.setTimeout(() => {
        editTimers.delete(key);
        void flushFields(set, get, sampleId, [field]);
      }, EDIT_DEBOUNCE_MS);
      editTimers.set(key, handle);
    }
  },

  toggleExcluded: async (sampleId, excluded) => {
    const id = get().selectedDatasetId;
    const sample = get().samplesById[sampleId];
    if (!id || !sample) return;

    // PATCH 409s on a sample whose audio has vanished (§2.4), but excluding it
    // is the documented way to unblock Build (§5.5) — and the bulk endpoint
    // allows an exclude-only change on a missing file. Route it there.
    if (sample.fileMissing) {
      try {
        const result = await trainingApi.bulkSetSamples(id, [sampleId], { excluded });
        if (result.failed.length > 0) throw new Error(result.failed[0].error);
        set({ samplesById: { ...get().samplesById, [sampleId]: { ...sample, excluded } }, saveError: null });
      } catch (err) {
        set({ saveError: { sampleId, message: errMessage(err) } });
      }
      return;
    }

    await get().editSample(sampleId, { excluded });
    await get().flushSample(sampleId);
  },

  flushSample: async (sampleId) => {
    const pending = get().pendingEdits[sampleId];
    const fields = pending ? (Object.keys(pending) as EditableField[]) : [];
    for (const f of fields) clearTimer(timerKey(sampleId, f));
    if (fields.length === 0) return;
    await flushFields(set, get, sampleId, fields);
  },

  revertSampleField: (sampleId, field) => {
    clearTimer(timerKey(sampleId, field));
    const pending = get().pendingEdits[sampleId];
    if (!pending) return;
    const next = { ...pending };
    delete next[field];
    const edits = { ...get().pendingEdits };
    if (Object.keys(next).length === 0) delete edits[sampleId];
    else edits[sampleId] = next;
    set({ pendingEdits: edits, saveError: null });
  },

  clearSaveError: () => set({ saveError: null }),

  bulkSet: async (bulk) => {
    const id = get().selectedDatasetId;
    const ids = Array.from(get().selectedSampleIds);
    if (!id || ids.length === 0) return;
    try {
      await trainingApi.bulkSetSamples(id, ids, bulk);
      await get().openDataset(id);
    } catch (err) {
      set({ saveError: { sampleId: '', message: errMessage(err) } });
    }
  },

  toggleSampleSelected: (sampleId) => {
    const next = new Set(get().selectedSampleIds);
    if (next.has(sampleId)) next.delete(sampleId);
    else next.add(sampleId);
    set({ selectedSampleIds: next });
  },

  setSelectedSampleIds: (ids) => set({ selectedSampleIds: new Set(ids) }),
  clearSelection: () => set({ selectedSampleIds: new Set<string>() }),
  setOpenSampleId: (sampleId) => set({ openSampleId: sampleId }),

  startLabel: async (opts) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const { jobId } = await trainingApi.startLabel(id, opts);
      set({ jobLog: [], error: null });
      await adoptJob(set, get, jobId);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  startGenius: async (opts) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const { jobId } = await trainingApi.startGenius(id, opts);
      set({ jobLog: [], error: null });
      await adoptJob(set, get, jobId);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  startCaption: async (opts) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const { jobId } = await trainingApi.startCaption(id, opts);
      set({ jobLog: [], error: null });
      await adoptJob(set, get, jobId);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  startBuild: async (outputPath) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const { jobId } = await trainingApi.startBuild(id, outputPath);
      set({ jobLog: [], error: null });
      await adoptJob(set, get, jobId);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  cancelJob: async () => {
    const job = get().activeJob;
    if (!job) return;
    try {
      await trainingApi.cancelJob(job.id);
      set({ activeJob: { ...job, status: 'cancelled' } });
      // That status change unmounts the EventSource, so the server's terminal
      // `status` frame may never arrive — run the authoritative refresh here.
      refreshAfterJob(get, job.id);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  applyStreamEvent: (ev) => {
    switch (ev.type) {
      case 'job':
        set({ activeJob: ev.job });
        break;

      case 'progress': {
        const job = get().activeJob;
        if (!job) break;
        set({
          activeJob: {
            ...job,
            done: ev.done,
            total: ev.total,
            failed: ev.failed,
            phase: ev.phase,
            currentSampleId: ev.currentSampleId,
            engineQueueDepth: ev.engineQueueDepth,
          },
        });
        break;
      }

      case 'sample': {
        // Idempotent: REPLACE the row keyed by sampleId. The SSE buffer is
        // replayed on every reconnect, so appending would duplicate rows.
        const prev = get().samplesById[ev.sampleId];
        const merged: TrainingSample | undefined = ev.sample
          ? ev.sample
          : prev
            ? { ...prev, labelStatus: ev.status, error: ev.error ?? null }
            : undefined;
        if (!merged) break;
        const order = get().sampleOrder.includes(ev.sampleId)
          ? get().sampleOrder
          : [...get().sampleOrder, ev.sampleId];
        // A finished row is authoritative — drop any optimistic edit for it.
        const edits = { ...get().pendingEdits };
        if (ev.sample) delete edits[ev.sampleId];
        set({
          samplesById: { ...get().samplesById, [ev.sampleId]: merged },
          sampleOrder: order,
          pendingEdits: edits,
        });
        break;
      }

      case 'log': {
        // The whole buffer is replayed on every reconnect, so an append-only
        // handler would duplicate the log each time. Identity is (ts, level, message).
        const cur = get().jobLog;
        if (cur.some(l => l.ts === ev.ts && l.level === ev.level && l.message === ev.message)) break;
        const log = [...cur, { level: ev.level, message: ev.message, ts: ev.ts }];
        set({ jobLog: log.length > JOB_LOG_CAP ? log.slice(log.length - JOB_LOG_CAP) : log });
        break;
      }

      case 'status': {
        const job = get().activeJob;
        if (job) set({ activeJob: { ...job, status: ev.status, error: ev.error ?? job.error } });
        // Authoritative refresh — exactly once per job, replay-safe.
        if (job) refreshAfterJob(get, job.id);
        break;
      }
    }
  },
}));

/** Re-read the dataset once a job reaches a terminal state. Guarded: the SSE
 *  buffer is replayed on reconnect and cancel calls this too. */
function refreshAfterJob(get: () => TrainingState, jobId: string): void {
  const id = get().selectedDatasetId;
  if (!id || refreshedJobs.has(jobId)) return;
  refreshedJobs.add(jobId);
  void get().openDataset(id);
  void get().loadDatasets();
}

/**
 * Adopt a freshly started job. A quick label over a handful of files (or an
 * immediate failure) can finish before this GET resolves — the stream is then
 * never opened, so the terminal refresh has to happen here instead.
 */
async function adoptJob(
  set: (partial: Partial<TrainingState>) => void,
  get: () => TrainingState,
  jobId: string,
): Promise<void> {
  const job = await trainingApi.getJob(jobId);
  set({ activeJob: job });
  if (job.status !== 'queued' && job.status !== 'running') refreshAfterJob(get, job.id);
}

/**
 * PATCH the given fields of one sample. Optimistic values are kept in
 * `pendingEdits` until the server confirms; on failure they are reverted so
 * the cell snaps back to the on-disk value.
 */
async function flushFields(
  set: (partial: Partial<TrainingState>) => void,
  get: () => TrainingState,
  sampleId: string,
  fields: EditableField[],
): Promise<void> {
  const state = get();
  const datasetId = state.selectedDatasetId;
  const pending = state.pendingEdits[sampleId];
  if (!datasetId || !pending) return;

  const patch: PatchSampleInput = {};
  let any = false;
  for (const f of fields) {
    if (f in pending) { (patch as Record<string, unknown>)[f] = pending[f]; any = true; }
  }
  if (!any) return;

  const saving = new Set(state.savingSampleIds);
  saving.add(sampleId);
  set({ savingSampleIds: saving });

  // Drop only what this request actually sent. If the user kept typing while it
  // was in flight, the newer value is still pending and must survive — clearing
  // it blindly loses those keystrokes and no-ops the queued flush.
  const dropFields = () => {
    const edits = { ...get().pendingEdits };
    const cur = edits[sampleId];
    if (cur) {
      const next = { ...cur } as Record<string, unknown>;
      for (const f of fields) {
        if (Object.is(next[f], (patch as Record<string, unknown>)[f])) delete next[f];
      }
      if (Object.keys(next).length === 0) delete edits[sampleId];
      else edits[sampleId] = next as PatchSampleInput;
    }
    return edits;
  };

  try {
    const sample = await trainingApi.patchSample(datasetId, sampleId, patch);
    const done = new Set(get().savingSampleIds);
    done.delete(sampleId);
    set({
      samplesById: { ...get().samplesById, [sampleId]: sample },
      pendingEdits: dropFields(),
      savingSampleIds: done,
    });
  } catch (err) {
    const done = new Set(get().savingSampleIds);
    done.delete(sampleId);
    set({
      pendingEdits: dropFields(),      // revert the optimistic value
      savingSampleIds: done,
      saveError: { sampleId, message: errMessage(err) },
    });
  }
}

/** Row as the UI should render it: server truth + any un-saved optimistic edit. */
export function mergedSample(
  sample: TrainingSample | undefined,
  pending: PatchSampleInput | undefined,
): TrainingSample | undefined {
  if (!sample) return undefined;
  return pending ? { ...sample, ...pending } : sample;
}
