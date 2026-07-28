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
  LmSize,
  PreprocessOptions,
  PreprocessStatus,
  TrainDitEpoch,
  TrainDitOptions,
  TrainDitStatus,
  TrainLmEpoch,
  TrainLmOptions,
  TrainLmStatus,
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
// The last EXPLICIT train-lm status query. §5.5 spells the terminal refresh as
// `loadTrainLmStatus()` with no argument, but the panel carries its own
// lmSize/adapterName: a no-arg call defaults server-side to 0.6B + dataset slug,
// i.e. a DIFFERENT adapter directory, and it races the panel's own correctly
// parameterised refresh. Remembering the last query makes the no-arg form mean
// "refresh what is on screen".
let lastTrainLmQuery: { variantKey?: string; adapterName?: string; lmSize?: LmSize } | undefined;
// Monotonic request id — the older of two in-flight status reads must not win.
let trainLmSeq = 0;
// Same two devices for train-dit.
let lastTrainDitQuery: { variantKey?: string; adapterName?: string } | undefined;
let trainDitSeq = 0;
// The `vram` metric carries crop/layers exactly ONCE, before the first step, and
// `trainDitLast` does not exist yet at that point — creating it there would put a
// fabricated 0.0000 loss on the metric strip. Park them here instead and fold them
// into every later trainDitLast write.
let ditVram = { crop: 0, layers: 0 };

function timerKey(sampleId: string, field: string): string {
  return `${sampleId}:${field}`;
}

function clearTimer(key: string): void {
  const t = editTimers.get(key);
  if (t !== undefined) { window.clearTimeout(t); editTimers.delete(key); }
}

interface TrainingState {
  // navigation
  phase: 'dataset' | 'preprocess' | 'train' | 'monitor';   // 'dataset' and 'preprocess' selectable
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

  // preprocess
  preprocessStatus: PreprocessStatus | null;
  preprocessLoading: boolean;

  // train (LM)
  trainLmStatus: TrainLmStatus | null;
  trainLmLoading: boolean;
  trainLmEpochs: TrainLmEpoch[];          // live, from `metric` events
  trainLmLast: { loss: number; lr: number; gradNorm: number; etaMs: number } | null;
  /** From the one `data` metric — songs skipped for exceeding max sequence
   *  length. Surfaced in the panel; without it the only trace is a warn line
   *  in the scrolling log tail (§5.6 mandates the string). */
  trainLmSkippedLong: number;
  /** From the one `vram` metric (4B plan §1.3/§2.2). Drives the per-size VRAM
   *  hint line in TrainLmForm; null until the engine has reported. */
  trainLmVram: {
    mode: string; maxLen: number; estMb: number; freeMb: number;
    baseMb: number; ckptMb: number; segPeakMb: number;
  } | null;

  // train (DiT)
  trainDitStatus: TrainDitStatus | null;
  trainDitLoading: boolean;
  trainDitEpochs: TrainDitEpoch[];        // live, from `metric` events
  trainDitLast: {
    loss: number; ma5: number; lr: number; gradNorm: number;
    crop: number; layers: number; etaMs: number;
  } | null;

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
  loadPreprocessStatus(): Promise<void>;
  startPreprocess(opts: PreprocessOptions): Promise<void>;
  deletePreprocessVariant(variantKey: string): Promise<void>;
  loadTrainLmStatus(q?: { variantKey?: string; adapterName?: string; lmSize?: LmSize }): Promise<void>;
  startTrainLm(opts: TrainLmOptions): Promise<void>;
  loadTrainDitStatus(q?: { variantKey?: string; adapterName?: string }): Promise<void>;
  startTrainDit(opts: TrainDitOptions): Promise<void>;
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

  preprocessStatus: null,
  preprocessLoading: false,

  trainLmStatus: null,
  trainLmLoading: false,
  trainLmEpochs: [],
  trainLmLast: null,
  trainLmSkippedLong: 0,
  trainLmVram: null,

  trainDitStatus: null,
  trainDitLoading: false,
  trainDitEpochs: [],
  trainDitLast: null,

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
    // Variant cards are per-dataset; keeping them would attribute one dataset's
    // caches to the next one opened.
    preprocessStatus: null,
    preprocessLoading: false,
    // Same reasoning: the adapter dir is keyed on the dataset slug, so B would
    // otherwise render A's trained-adapter card.
    trainLmStatus: null,
    trainLmLoading: false,
    trainLmEpochs: [],
    trainLmLast: null,
    trainLmSkippedLong: 0,
    trainLmVram: null,
    trainDitStatus: null,
    trainDitLoading: false,
    trainDitEpochs: [],
    trainDitLast: null,
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
              // Per-dataset, exactly like closeDataset. The Preprocess panel's
              // own picker calls openDataset(), so without this A's variant
              // cards stay rendered under B — and Delete would then post B's id
              // with A's variantKey (identical keys across datasets, since the
              // key is just the DiT base name) and rm -rf the wrong cache.
              preprocessStatus: null,
              preprocessLoading: false,
              trainLmStatus: null,
              trainLmLoading: false,
              trainLmEpochs: [],
              trainLmLast: null,
              trainLmSkippedLong: 0,
              trainLmVram: null,
              trainDitStatus: null,
              trainDitLoading: false,
              trainDitEpochs: [],
              trainDitLast: null,
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

  loadPreprocessStatus: async () => {
    const id = get().selectedDatasetId;
    if (!id) { set({ preprocessStatus: null }); return; }
    set({ preprocessLoading: true });
    try {
      const preprocessStatus = await trainingApi.getPreprocessStatus(id);
      // A slow request must not overwrite a newer dataset's status — but it must
      // still clear the spinner, or a stale response landing last leaves it
      // spinning forever next to the "Tensor caches" heading.
      if (get().selectedDatasetId !== id) { set({ preprocessLoading: false }); return; }
      set({ preprocessStatus, preprocessLoading: false });
    } catch (err) {
      // Advisory — a failed status read must never blank the panel's controls,
      // but it must not leave ANOTHER dataset's cards on screen either.
      console.warn('[Training] preprocess status failed:', errMessage(err));
      if (get().selectedDatasetId === id) set({ preprocessStatus: null, preprocessLoading: false });
      else set({ preprocessLoading: false });
    }
  },

  startPreprocess: async (opts) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const { jobId } = await trainingApi.startPreprocess(id, opts);
      set({ jobLog: [], error: null });
      await adoptJob(set, get, jobId);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  deletePreprocessVariant: async (variantKey) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      await trainingApi.deletePreprocessVariant(id, variantKey);
      set({ error: null });
      await get().loadPreprocessStatus();
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  loadTrainLmStatus: async (q) => {
    const id = get().selectedDatasetId;
    if (!id) { set({ trainLmStatus: null }); return; }
    // No argument = "refresh whatever the panel last asked for" (see
    // lastTrainLmQuery), not "read the 0.6B/<slug> directory".
    const query = q ?? lastTrainLmQuery;
    if (q) lastTrainLmQuery = q;
    const seq = ++trainLmSeq;
    set({ trainLmLoading: true });
    try {
      const trainLmStatus = await trainingApi.getTrainLmStatus(id, query);
      // A superseded request must not win: a newer one is in flight and owns the
      // spinner. A slow request must not overwrite a newer dataset's status, but
      // it must still clear the spinner — same rule as loadPreprocessStatus.
      if (seq !== trainLmSeq) return;
      if (get().selectedDatasetId !== id) { set({ trainLmLoading: false }); return; }
      set({ trainLmStatus, trainLmLoading: false });
    } catch (err) {
      // Advisory: a failed status read must never blank the panel's controls.
      console.warn('[Training] train-lm status failed:', errMessage(err));
      if (seq !== trainLmSeq) return;
      if (get().selectedDatasetId === id) set({ trainLmStatus: null, trainLmLoading: false });
      else set({ trainLmLoading: false });
    }
  },

  startTrainLm: async (opts) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const { jobId } = await trainingApi.startTrainLm(id, opts);
      set({ jobLog: [], error: null, trainLmEpochs: [], trainLmLast: null, trainLmSkippedLong: 0,
        trainLmVram: null });
      await adoptJob(set, get, jobId);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  loadTrainDitStatus: async (q) => {
    const id = get().selectedDatasetId;
    if (!id) { set({ trainDitStatus: null }); return; }
    // No argument = "refresh whatever the panel last asked for", not "read the
    // <slug> directory" — same rule as loadTrainLmStatus.
    const query = q ?? lastTrainDitQuery;
    if (q) lastTrainDitQuery = q;
    const seq = ++trainDitSeq;
    set({ trainDitLoading: true });
    try {
      const trainDitStatus = await trainingApi.getTrainDitStatus(id, query);
      if (seq !== trainDitSeq) return;
      if (get().selectedDatasetId !== id) { set({ trainDitLoading: false }); return; }
      set({ trainDitStatus, trainDitLoading: false });
    } catch (err) {
      // Advisory: a failed status read must never blank the panel's controls.
      console.warn('[Training] train-dit status failed:', errMessage(err));
      if (seq !== trainDitSeq) return;
      if (get().selectedDatasetId === id) set({ trainDitStatus: null, trainDitLoading: false });
      else set({ trainDitLoading: false });
    }
  },

  startTrainDit: async (opts) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const { jobId } = await trainingApi.startTrainDit(id, opts);
      ditVram = { crop: 0, layers: 0 };
      set({ jobLog: [], error: null, trainDitEpochs: [], trainDitLast: null });
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

      case 'metric': {
        // train-dit writes its own slices. Same event shape, same omit-vs-zero
        // discipline as the LM path below; only the destination differs.
        if (get().activeJob?.kind === 'train-dit') {
          if (ev.metric === 'vram') {
            // One-shot, before the first step. Parked in `ditVram` rather than
            // written into trainDitLast — see the note at the top of this file.
            ditVram = {
              crop: typeof ev.crop === 'number' ? ev.crop : ditVram.crop,
              layers: typeof ev.layers === 'number' ? ev.layers : ditVram.layers,
            };
            break;
          }
          if (ev.metric !== 'epoch' && ev.metric !== 'step') break;
          const prevDit = get().trainDitLast;
          const crop = typeof ev.crop === 'number' ? ev.crop : (prevDit?.crop ?? ditVram.crop);
          const layers = prevDit?.layers ?? ditVram.layers;
          if (ev.metric === 'epoch' && typeof ev.epoch === 'number' && typeof ev.loss === 'number') {
            // REPLACE by epoch number, never append — the SSE buffer is replayed
            // on reconnect and the curve must be idempotent.
            const epoch = ev.epoch;
            const next = [...get().trainDitEpochs.filter(e => e.epoch !== epoch), {
              epoch, loss: ev.loss, lr: ev.lr ?? 0,
              gradNorm: ev.gradNorm ?? 0, ms: ev.ms ?? 0,
            }].sort((a, b) => a.epoch - b.epoch);
            set({
              trainDitEpochs: next,
              trainDitLast: {
                loss: ev.loss,
                ma5: ev.ma5 ?? prevDit?.ma5 ?? 0,
                lr: ev.lr ?? prevDit?.lr ?? 0,
                gradNorm: ev.gradNorm ?? prevDit?.gradNorm ?? 0,
                crop, layers,
                etaMs: ev.etaMs ?? prevDit?.etaMs ?? 0,
              },
            });
          } else if (
            typeof ev.loss === 'number' || typeof ev.lr === 'number'
            || typeof ev.gradNorm === 'number' || typeof ev.crop === 'number'
          ) {
            set({
              trainDitLast: {
                loss: ev.loss ?? prevDit?.loss ?? 0,
                ma5: ev.ma5 ?? prevDit?.ma5 ?? 0,
                lr: ev.lr ?? prevDit?.lr ?? 0,
                gradNorm: ev.gradNorm ?? prevDit?.gradNorm ?? 0,
                crop, layers,
                etaMs: ev.etaMs ?? prevDit?.etaMs ?? 0,
              },
            });
          }
          break;
        }

        // §5.5 spells these as `ev.loss ?? 0`, but the runner deliberately OMITS
        // a metric field the engine did not send (trainLmRunner optNum: "a metric
        // field defaulted to 0 is indistinguishable from a real 0 in the loss
        // curve"). Defaulting here re-introduces exactly that: a loss-less epoch
        // frame would plot at 0.0000, drag LossSparkline's y-range and the target
        // line to the floor, and read as "converged". So an epoch with no loss is
        // not plotted, and trainLmLast keeps its previous value per field.
        const prev = get().trainLmLast;
        if (ev.metric === 'epoch' && typeof ev.epoch === 'number' && typeof ev.loss === 'number') {
          // REPLACE by epoch number, never append — the SSE buffer is replayed
          // on reconnect and the curve must be idempotent (same rule as
          // samplesById).
          const epoch = ev.epoch;
          const next = [...get().trainLmEpochs.filter(e => e.epoch !== epoch), {
            epoch, loss: ev.loss, lr: ev.lr ?? 0,
            gradNorm: ev.gradNorm ?? 0, ms: ev.ms ?? 0,
          }].sort((a, b) => a.epoch - b.epoch);
          set({
            trainLmEpochs: next,
            trainLmLast: {
              loss: ev.loss, lr: ev.lr ?? prev?.lr ?? 0,
              gradNorm: ev.gradNorm ?? prev?.gradNorm ?? 0, etaMs: ev.etaMs ?? prev?.etaMs ?? 0,
            },
          });
        } else if (ev.metric === 'epoch' || ev.metric === 'step') {
          if (typeof ev.loss === 'number' || typeof ev.lr === 'number' || typeof ev.gradNorm === 'number') {
            set({
              trainLmLast: {
                loss: ev.loss ?? prev?.loss ?? 0, lr: ev.lr ?? prev?.lr ?? 0,
                gradNorm: ev.gradNorm ?? prev?.gradNorm ?? 0, etaMs: ev.etaMs ?? prev?.etaMs ?? 0,
              },
            });
          }
        } else if (ev.metric === 'data' && typeof ev.skippedLong === 'number') {
          set({ trainLmSkippedLong: ev.skippedLong });
        } else if (ev.metric === 'vram') {
          // 4B plan §1.3: the per-size VRAM hint is driven by this one event.
          // Fires once, before the first step. `mode`/`baseMb`/`ckptMb`/
          // `segPeakMb` only exist for low-VRAM runs, so they default to 0/''.
          set({
            trainLmVram: {
              mode: typeof ev.mode === 'string' ? ev.mode : '',
              maxLen: ev.maxLen ?? 0, estMb: ev.estMb ?? 0, freeMb: ev.freeMb ?? 0,
              baseMb: ev.baseMb ?? 0, ckptMb: ev.ckptMb ?? 0, segPeakMb: ev.segPeakMb ?? 0,
            },
          });
        }
        // 'milestone' carries its own log line — nothing to store.
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
  // A finished train-lm job wrote (or failed to write) the adapter dir — the
  // done-state card reads that from disk, so re-read it here.
  if (get().activeJob?.kind === 'train-lm') void get().loadTrainLmStatus();
  // Same for the DiT adapter dir — the per-epoch export means even a cancelled
  // run usually leaves one on disk, so this refresh matters on every terminal
  // status, not just 'done'.
  if (get().activeJob?.kind === 'train-dit') void get().loadTrainDitStatus();
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
