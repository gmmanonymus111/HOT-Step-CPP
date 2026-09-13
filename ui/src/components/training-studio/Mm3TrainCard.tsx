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
  AlertTriangle, ChevronDown, ChevronRight, Cpu, Loader2, Play, ShieldCheck, Volume2, XCircle,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { estimateMm3PeakMb, estimateMm3PrefixMb, mm3FlashVramCalibrated, type Mm3PresetName } from '../../services/trainingApi';
import type { Mm3TrainLmRequest } from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { JobProgress } from './JobProgress';
import { Mm3PreviewStrip } from './Mm3PreviewStrip';
import { Mm3RunsPanel } from './Mm3RunsPanel';
import { useMm3Status } from './useMm3Status';
import { TrainingChart } from './TrainingChart';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';
const INPUT = 'w-full px-2.5 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 '
            + 'dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 outline-none '
            + 'focus:border-amber-500/50';

const BTN_SM = 'shrink-0 px-2.5 py-1.5 rounded-lg text-[11px] font-medium border border-zinc-300 '
             + 'dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 '
             + 'dark:hover:bg-zinc-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors';

interface FormState {
  steps: number;
  stopMode: 'steps' | 'loss';
  targetLoss: number;
  targetLossMetric: 'train' | 'eval';
  targetLossEpochs: number;
  saveEvery: number;
  keepResumeState: boolean;
  longTracks: 'exclude' | 'crop';
  rank: number;
  alpha: number;
  lr: number;
  maxFrames: number;
  cropMode: 'random' | 'beginning' | 'structured';
  cropStartFrac: number;
  cropEndFrac: number;
  cropStartTiles: number;
  depthLossWeight: number;
  depthLossFrames: number;
  optimizer: 'muon' | 'adamw' | 'prodigy';
  muonLrScale: number;
  adapterType: 'lora' | 'lokr';
  lokrFactor: number;
  // ── Flag-contract parity fields (2026-09-05) ───────────────────────────
  attnBackend: 'exact' | 'flash';
  rslora: boolean;
  dora: boolean;
  hira: boolean;
  loha: boolean;
  pissa: boolean;
  hotPizza: boolean;
  hra: boolean;
  loraPlusRatio: number;
  /** Soft prompt (a token + a TRAINABLE prefix, both new to MM3). Distinct
   *  from prefixFrames below, which is the FROZEN real-audio history prefix —
   *  a different engine mechanism (train/mm3-lm-kvprefix.h vs lm-prefix.h). */
  artistTokenOn: boolean;
  artistToken: string;
  artistTokenK: number;
  artistTokenLr: number;
  prefixN: number;
  /** '' = auto-pick (server-side ladder — see mm3Preview.ts). */
  previewSongId: string;
  gradAccum: number;
  seed: number;
  trigger: string;
  triggerPrepend: boolean;
  basePrecision: string;
  holdout: number;
  evalEvery: number;
  cropAnchor: 'song' | 'zero';
  prefixFrames: number;
  previewEverySteps: number;
  previewEveryMinutes: number;
  previewSeconds: number;
  previewSeed: number;
  previewCaption: string;
  previewControl: boolean;
  previewBaseline: boolean;
  previewScaleMlp: number;
  regDatasetId: string;
  regEvery: number;
  regTopK: number;
}

/** Same shape as TrainDitForm's DitMethod / TrainLmForm's LmMethod (2026-09-05
 *  flag-contract parity). LoKr is its own type; DoRA/HiRA/LoHa/HRA are the
 *  LoRA type with one flag set. */
type Mm3Method = 'lokr' | 'lora' | 'dora' | 'hira' | 'loha' | 'hra';
const methodOf = (s: FormState): Mm3Method =>
  s.adapterType === 'lokr' ? 'lokr' : s.dora ? 'dora' : s.hira ? 'hira' : s.loha ? 'loha' : s.hra ? 'hra' : 'lora';
const MM3_METHOD_KEYS: Record<Mm3Method, { label: string; info: string }> = {
  lokr: { label: 'adapterLokr', info: 'adapterTypeHint' },
  lora: { label: 'adapterLora', info: 'adapterTypeHint' },
  dora: { label: 'dora', info: 'doraInfo' },
  hira: { label: 'hira', info: 'hiraInfo' },
  loha: { label: 'loha', info: 'lohaInfo' },
  hra:  { label: 'hra',  info: 'hraInfo' },
};

const NumField: React.FC<{
  label: string; value: number; onChange: (v: number) => void; step?: number; hint?: string;
  disabled?: boolean;
}> = ({ label, value, onChange, step = 1, hint, disabled }) => (
  <label className={`flex flex-col gap-1${disabled ? ' opacity-50' : ''}`}>
    <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">{label}</span>
    <input
      type="number" className={INPUT} value={value} step={step} disabled={disabled}
      onChange={e => onChange(Number(e.target.value))}
    />
    {hint && <span className="text-[10px] text-zinc-500 leading-snug">{hint}</span>}
  </label>
);

export const Mm3TrainCard: React.FC<{ datasetId: string; trigger?: string }> = ({ datasetId, trigger }) => {
  const { t } = useTranslation();
  const activeJob = useTrainingStore(s => s.activeJob);
  const startMm3TrainLm = useTrainingStore(s => s.startMm3TrainLm);
  // Which codes cache this run trains on. Off = the standard cache, exactly as
  // every run before the launder gate existed.
  const [trainLaunder, setTrainLaunder] = useState(false);
  const mm3Live = useTrainingStore(s => s.mm3Live);
  const storeError = useTrainingStore(s => s.error);
  const setPhase = useTrainingStore(s => s.setPhase);
  const trainStepSeries = useTrainingStore(s => s.trainStepSeries);
  const trainMilestones = useTrainingStore(s => s.trainMilestones);
  const trainLmEpochs = useTrainingStore(s => s.trainLmEpochs);
  const trainEvalSeries = useTrainingStore(s => s.trainEvalSeries);
  const trainMaxEpochs = useTrainingStore(s => s.trainMaxEpochs);
  const trainTargetLoss = useTrainingStore(s => s.trainTargetLoss);

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

  // Muon normalises its update, so the schedule LR alone understates what the
  // optimizer applied by the scale factor.
  const lrScale = status?.defaults.optimizer === 'muon' ? (status.defaults.muonLrScale ?? 1) : 1;

  const form: FormState | null = status ? {
    steps: status.defaults.steps ?? 800,
    stopMode: status.defaults.stopMode ?? 'steps',
    targetLoss: status.defaults.targetLoss ?? 1.0,
    targetLossMetric: status.defaults.targetLossMetric ?? 'train',
    targetLossEpochs: status.defaults.targetLossEpochs ?? 5,
    saveEvery: status.defaults.saveEvery ?? 100,
    keepResumeState: status.defaults.keepResumeState ?? false,
    longTracks: status.defaults.longTracks === 'crop' ? 'crop' : 'exclude',
    // Rank follows the recommendation for the same reason as the base: at the
    // default 256 nothing fits below ~24 GB, so a 16 GB card would open on a
    // red 'will not fit' form with the fix two fields away and unstated.
    rank: status.recommended?.rank ?? status.defaults.rank ?? 256,
    alpha: status.defaults.alpha ?? 256,
    lr: status.defaults.lr ?? 8e-5,
    maxFrames: status.defaults.maxFrames ?? 1500,
    cropMode: (status.defaults.cropMode as 'random' | 'beginning' | 'structured') ?? 'structured',
    cropStartFrac: status.defaults.cropStartFrac ?? 0.55,
    cropEndFrac: status.defaults.cropEndFrac ?? 0.15,
    cropStartTiles: status.defaults.cropStartTiles ?? 3,
    depthLossWeight: status.defaults.depthLossWeight ?? 1.0,
    depthLossFrames: status.defaults.depthLossFrames ?? 128,
    optimizer: status.defaults.optimizer ?? 'prodigy',
    muonLrScale: status.defaults.muonLrScale ?? 64,
    adapterType: status.defaults.adapterType ?? 'lora',
    lokrFactor: status.defaults.lokrFactor ?? 6,
    // Flag-contract parity (2026-09-05). All off/exact/1 by default — none of
    // these has an MM3-specific measurement yet (see the hover text), and
    // --attn/--dora/--hira/--loha/--pissa/--hra/--prefix-n are landing in the
    // mm3-lm-train engine parser concurrently with this UI, not before it.
    attnBackend: status.defaults.attnBackend ?? 'exact',
    rslora: status.defaults.rslora ?? false,
    dora: status.defaults.dora ?? false,
    hira: status.defaults.hira ?? false,
    loha: status.defaults.loha ?? false,
    pissa: status.defaults.pissa ?? false,
    hotPizza: status.defaults.hotPizza ?? false,
    hra: status.defaults.hra ?? false,
    loraPlusRatio: status.defaults.loraPlusRatio ?? 1,
    artistTokenOn: !!status.defaults.artistToken,
    artistToken: status.defaults.artistToken ?? '',
    artistTokenK: status.defaults.artistTokenK ?? 32,
    artistTokenLr: status.defaults.artistTokenLr ?? 0.005,
    prefixN: status.defaults.prefixN ?? 0,
    previewSongId: '',
    gradAccum: status.defaults.gradAccum ?? 1,
    seed: status.defaults.seed ?? 42,
    trigger: trigger ?? '',
    triggerPrepend: status.defaults.triggerPrepend !== false,
    // The RECOMMENDED base, not the global default: the default was chosen on a
    // 32 GB card and on a 12 GB one it is simply wrong. The server picks the
    // highest-fidelity base that fits THIS GPU, and falls back to the default
    // when it cannot read the card.
    // The configured default WINS over the VRAM recommender's pick. The
    // recommender walks bases by fidelity and falls through to a smaller one
    // whenever the best does not fit its budget, which silently downgraded
    // f16 to q8_0. It still sets `rank` and still raises overBudget, so the
    // user is warned rather than quietly given a different base.
    basePrecision: status.defaults.basePrecision || status.recommended?.base || 'f16',
    holdout: status.defaults.holdout ?? 0.15,
    evalEvery: status.defaults.evalEvery ?? 50,
    cropAnchor: (status.defaults.cropAnchor as 'song' | 'zero') ?? 'song',
    // OFF by default, because nothing trained with a prefix has been heard yet.
    prefixFrames: status.defaults.prefixFrames ?? 0,
    // Previews default OFF. They are the fastest way to learn whether a run is
    // worth finishing, but each one costs about a minute, so opting in is the
    // user's call rather than a surprise on the clock.
    // Cadence follows checkpoints (Rob, 2026-08-25): a preview per checkpoint,
    // no minutes clock. `||` on purpose — a server default of 0 ("off") falls
    // back to the checkpoint cadence rather than disabling previews.
    previewEverySteps: status.defaults.previewEverySteps || (status.defaults.saveEvery ?? 50),
    previewEveryMinutes: 0,
    previewSeconds: 40,
    previewSeed: 424242,
    previewCaption: '',
    // Off by default (Rob, 2026-08-25): with a preview at every checkpoint the
    // control and baseline renders would double the pause cost for takes that
    // rarely get listened to. Both remain a checkbox away.
    previewControl: false,
    previewBaseline: false,
    previewScaleMlp: 1.0,
    // Prior preservation is off until a corpus is chosen: it needs a second
    // dataset the user has to nominate, and defaulting it on would silently
    // train a different objective than the form otherwise describes.
    regDatasetId: '',
    regEvery: status.defaults.regularisation?.every ?? 3,
    regTopK: status.defaults.regularisation?.topK ?? 64,
    ...edits,
  } : null;

  // Peak VRAM for the CURRENT form, not for the defaults — rank is the biggest
  // single term after the base itself, so an estimate pinned to the defaults
  // would be wrong for exactly the users who need it most.
  const chosen = status?.bases?.find(b => b.id === form?.basePrecision);
  const flashCalibrated = status ? (status.flashVramCalibrated ?? mm3FlashVramCalibrated(status.vramModel)) : false;
  // Whether this engine build can run --attn flash AT ALL. The fused
  // attention-training op is CUDA + CPU only, so on a Vulkan (AMD/Intel) or
  // Metal engine ace-train refuses to start — and since flash is the default,
  // that made the DEFAULT recipe unlaunchable with no way to see why (#149).
  // An older server does not report it; treat that as supported, as before.
  const flashSupported = status?.flashSupported !== false;
  // What will ACTUALLY run, which is what the estimate and the checkbox both
  // have to show — the route coerces to exact on an unsupported build.
  const attnEffective: 'exact' | 'flash' =
    form && flashSupported && form.attnBackend === 'flash' ? 'flash' : 'exact';
  const peak = (() => {
    if (!form || !status?.vramModel || !chosen) return null;
    const mb    = estimateMm3PeakMb(chosen.bytes, form.rank, form.maxFrames, status.vramModel,
                                    form.optimizer,
                                    form.cropAnchor === 'song' ? form.prefixFrames : 0,
                                    256, attnEffective)
                + (chosen.extraMb || 0);
    // Flash mode's own coefficients are not measured yet (MM3_VRAM_MODEL.flash
    // is null server-side), so the number above is silently standing in for
    // exact mode's — say so rather than presenting it as a proven saving.
    const flashCaveat = attnEffective === 'flash' && !flashCalibrated
      ? ` (${t('trainingStudio.mm3.flashVramPending', 'flash: estimate pending measurement')})` : '';
    const total = status.gpuTotalMb || 0;
    const gb    = (mb / 1024).toFixed(1);
    // 0 means the engine could not be read, NOT a card with no memory. Show the
    // estimate without a verdict rather than inventing a scary one.
    if (total <= 0) {
      return { text: t('trainingStudio.mm3.peakUnknown', { gb }) + flashCaveat, tone: 'text-zinc-500' };
    }
    const totalGb = (total / 1024).toFixed(1);
    if (mb + 1536 <= total) {
      return { text: t('trainingStudio.mm3.peakFits', { gb, totalGb }) + flashCaveat, tone: 'text-emerald-500' };
    }
    if (mb <= total) {
      // Fits on paper, with nothing left for the desktop. This is the state that
      // produced 12-14 s/step instead of 3.7 in the f16 A/B, so it is a warning
      // rather than an error.
      return { text: t('trainingStudio.mm3.peakTight', { gb, totalGb }) + flashCaveat, tone: 'text-amber-500' };
    }
    // "Pick a smaller base" is bad advice when the server already established
    // that nothing in the catalogue fits at any rank on the ladder.
    if (status.recommended?.overBudget) {
      return { text: t('trainingStudio.mm3.peakNoFit', { gb, totalGb }) + flashCaveat, tone: 'text-rose-500' };
    }
    return { text: t('trainingStudio.mm3.peakOver', { gb, totalGb }) + flashCaveat, tone: 'text-rose-500' };
  })();

  // The card has no per-type default BUNDLE the way TrainDitForm's pickType
  // does — lokrFactor/rank/alpha already coexist in one flat state, only
  // their visibility changes — so switching method here is just the flag
  // set. PiSSA drops when hopping off plain LoRA; rsLoRA drops for HRA.
  const method: Mm3Method = form ? methodOf(form) : 'lora';
  const pickMethod = (m: Mm3Method) => {
    if (!form || m === method) return;
    if (m === 'lokr') { set('adapterType', 'lokr'); return; }
    setEdits(prev => ({
      ...prev,
      adapterType: 'lora',
      dora: m === 'dora', hira: m === 'hira', loha: m === 'loha', hra: m === 'hra',
      pissa: m === 'lora' ? form.pissa : false,
      hotPizza: m === 'lora' ? form.hotPizza : false,
      rslora: m === 'hra' ? false : form.rslora,
    }));
  };

  const startTrain = async () => {
    if (!form) return;
    setBusy(true);
    try {
      const body: Mm3TrainLmRequest = {
        steps: form.steps, saveEvery: form.saveEvery, keepResumeState: form.keepResumeState,
        longTracks: form.longTracks,
        rank: form.rank, alpha: form.alpha,
        lr: form.lr, maxFrames: form.maxFrames, cropMode: form.cropMode,
        // Informational: the fields above already carry the recipe. The
        // route lays a named preset UNDER them, so this changes nothing here
        // and only tells a log reader which recipe the user started from.
        preset: activePreset === 'custom' ? undefined : activePreset,
        cropStartFrac: form.cropStartFrac, cropEndFrac: form.cropEndFrac,
        cropStartTiles: form.cropStartTiles,
        depthLossWeight: form.depthLossWeight, depthLossFrames: form.depthLossFrames,
        optimizer: form.optimizer, muonLrScale: form.muonLrScale,
        adapterType: form.adapterType, lokrFactor: form.lokrFactor,
        gradAccum: form.gradAccum, seed: form.seed,
        basePrecision: form.basePrecision, holdout: form.holdout, evalEvery: form.evalEvery,
        cropAnchor: form.cropAnchor,
        // `steps` above is the cap in BOTH modes, so nothing about it changes
        // here — only whether a target is allowed to end the run sooner.
        stopMode: form.stopMode,
        ...(form.stopMode === 'loss' ? {
          targetLoss: form.targetLoss,
          targetLossMetric: form.targetLossMetric,
          targetLossEpochs: form.targetLossEpochs,
        } : {}),
        // ALWAYS sent, 0 included, and 0 under `zero` anchoring where the
        // engine refuses a prefix outright. The route resolves a MISSING key to
        // its default, which was 4096 until the whole-song recipe landed, so
        // every omission here was a run trained with 164 s of history the form
        // said was off (#142).
        prefixFrames: form.cropAnchor === 'song' ? Math.max(0, form.prefixFrames) : 0,
        ...(form.trigger.trim()
          ? { trigger: form.trigger.trim(), triggerPrepend: form.triggerPrepend }
          : {}),
        ...(form.regDatasetId ? {
          regularisation: {
            datasetId: form.regDatasetId,
            every: form.regEvery,
            topK: form.regTopK,
          },
        } : {}),
        ...(trainLaunder ? { launder: true } : {}),
        ...(form.previewEverySteps > 0 || form.previewEveryMinutes > 0 ? {
          preview: {
            everySteps: form.previewEverySteps,
            everyMinutes: form.previewEveryMinutes,
            seconds: form.previewSeconds,
            seed: form.previewSeed,
            control: form.previewControl,
            baseline: form.previewBaseline,
            scaleMlp: form.previewScaleMlp,
            ...(form.previewCaption.trim() ? { caption: form.previewCaption.trim() } : {}),
            // Explicit pick wins over the caption box above only because the
            // form never lets both be non-empty at once (picking a song clears
            // the caption box's relevance) — the server resolves it into
            // caption/lyrics through the same override path either way.
            ...(!form.previewCaption.trim() && form.previewSongId
              ? { previewSongId: form.previewSongId } : {}),
          },
        } : {}),
        // Flag-contract parity fields (2026-09-05). EVERY ONE IS SENT, ALWAYS,
        // including the falses and the zeros.
        //
        // They used to be spread in only when switched on, on an "an older
        // engine never sees it" rule that does not apply here: this body goes
        // to our own route, and it is the ARG BUILDER, not the request, that
        // decides which flags an older ace-train sees. What omission actually
        // bought was a class of dead checkboxes — the route reads a missing key
        // as "use the default", so unticking anything whose default is ON left
        // the default standing. attnBackend (default flash since 2026-09-06)
        // was refused by the trainer on an AMD card with the box unticked
        // (#149); PiSSA/HOT-PiZZA (default on) was the same bug one line down.
        //
        // NOT gated on adapterType: --attn is orthogonal to the adapter
        // parameterization and mm3-lm-train accepts it under LoKr too, where the
        // VRAM saving is identical.
        // Forced to 'exact' when the engine build has no fused-attention kernel
        // (Vulkan/Metal): the trainer refuses to start there, and the checkbox
        // is disabled with that reason shown.
        attnBackend: flashSupported ? form.attnBackend : 'exact',
        // The LoRA-family group is meaningless under LoKr, so it is sent as all
        // off there rather than omitted — the route's defaults would otherwise
        // reinstate HOT-PiZZA under a LoKr request.
        dora:   form.adapterType === 'lora' && form.dora,
        hira:   form.adapterType === 'lora' && form.hira,
        loha:   form.adapterType === 'lora' && form.loha,
        rslora: form.adapterType === 'lora' && form.rslora,
        pissa:    form.adapterType === 'lora' && form.pissa
                  && !form.dora && !form.hira && !form.loha,
        hotPizza: form.adapterType === 'lora' && form.pissa && form.hotPizza
                  && !form.dora && !form.hira && !form.loha,
        hra: form.adapterType === 'lora' && form.hra
             && !form.dora && !form.hira && !form.loha && !form.pissa,
        loraPlusRatio: form.adapterType === 'lora' ? form.loraPlusRatio : 1,
        artistToken: form.adapterType === 'lora' && form.artistTokenOn ? form.artistToken : '',
        artistTokenK: form.artistTokenK,
        artistTokenLr: form.artistTokenLr,
        // 0 alongside a regularisation corpus — the engine refuses the pair and
        // the route 400s on it. The control is disabled there too.
        prefixN: form.adapterType === 'lora' && !form.regDatasetId
          ? Math.max(0, form.prefixN) : 0,
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

  // Presets are DERIVED from the fields, never stored: the row highlights
  // whichever preset the four governing fields currently equal, and shows
  // Custom otherwise. Editing a field therefore leaves the preset row honest
  // without a second piece of state that could disagree with the form.
  const PRESET_ORDER: Mm3PresetName[] = ['balanced', 'thorough', 'fast'];   // default first; Fast is experimental
  const presets = status?.presets;
  const activePreset: Mm3PresetName | 'custom' = (() => {
    if (!presets || !form) return 'custom';
    const hit = PRESET_ORDER.find(p => {
      const q = presets[p];
      return q && form.steps === q.steps && form.lr === q.lr
          && form.maxFrames === q.maxFrames && form.prefixFrames === q.prefixFrames;
    });
    return hit ?? 'custom';
  })();
  const applyPreset = (p: Mm3PresetName) => {
    const q = presets?.[p];
    if (!q) return;
    setEdits(e => ({ ...e, steps: q.steps, lr: q.lr, maxFrames: q.maxFrames, prefixFrames: q.prefixFrames,
                     stopMode: 'steps' }));
  };

  const hasCodes = (status?.codes ?? 0) > 0;
  const hasLaundered = (status?.codesLaundered ?? 0) > 0;
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
        ) : !hasCodes && !hasLaundered ? (
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
            {/* -- Codes cache (the launder gate) --------------------------
                Only rendered when a laundered cache exists; a dataset without
                one trains on the standard cache with no extra UI at all. */}
            {hasLaundered && (
              <label className="flex items-start gap-2 mb-3 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={trainLaunder}
                  onChange={e => setTrainLaunder(e.target.checked)}
                  disabled={!hasLaundered}
                  className="mt-0.5 accent-amber-500"
                />
                <span className="text-[11px] text-zinc-600 dark:text-zinc-400 leading-relaxed">
                  <span className="font-semibold text-zinc-800 dark:text-zinc-200">
                    {t('trainingStudio.mm3.trainLaunderLabel', 'Train on cover-laundered codes')}
                  </span>
                  {' — '}
                  {t('trainingStudio.mm3.trainLaunderBlurb',
                    'the vocal-forward code targets for dense mixes ({{n}} tracks laundered). Off = the '
                    + 'standard cache, exactly as before.',
                    { n: status?.codesLaundered ?? 0 })}
                </span>
              </label>
            )}
            {/* -- Presets (2026-09-07) -----------------------------------
                Three recipes that tied blind; they trade minutes, not
                audible quality. The row reflects the fields, so a hand edit
                shows as Custom rather than misreporting a preset. */}
            {presets && (
              <div className="mb-3">
                <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider mb-1">
                  {t('trainingStudio.mm3.preset', 'Recipe')}
                </div>
                <div className="flex flex-wrap gap-2">
                  {PRESET_ORDER.map(p => (
                    <button key={p} type="button" onClick={() => applyPreset(p)} disabled={busy}
                      className={BTN_SM + (activePreset === p
                        ? ' !border-amber-500 !text-amber-600 dark:!text-amber-400 !bg-amber-500/10' : '')}>
                      {t(`trainingStudio.mm3.preset.${p}`, p)}
                    </button>
                  ))}
                  {activePreset === 'custom' && (
                    <span className={BTN_SM + ' !border-amber-500 !text-amber-600 dark:!text-amber-400 !bg-amber-500/10 cursor-default'}>
                      {t('trainingStudio.mm3.preset.custom', 'Custom')}
                    </span>
                  )}
                </div>
                <p className="text-[11px] text-zinc-500 mt-1">
                  {activePreset === 'custom'
                    ? t('trainingStudio.mm3.preset.customInfo', 'Steps, learning rate, crop or history differ from every preset.')
                    : t(`trainingStudio.mm3.preset.${activePreset}Info`, '')}
                </p>
              </div>
            )}
            {/* -- Stopping strategy ---------------------------------------
                Two ways to answer "when is this run done": a step count, or a
                loss to reach. The step field never goes away, because in loss
                mode it is the CAP - a target that is never met still has to end
                the run somewhere. */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
              <label className="flex flex-col gap-1">
                <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                  {t('trainingStudio.mm3.stopMode', 'Train until')}
                </span>
                <select className={INPUT} value={form.stopMode}
                  onChange={e => set('stopMode', e.target.value as 'steps' | 'loss')}>
                  <option value="steps">{t('trainingStudio.mm3.stopSteps', 'Step count')}</option>
                  <option value="loss">{t('trainingStudio.mm3.stopLoss', 'Target loss')}</option>
                </select>
              </label>
              <NumField
                label={form.stopMode === 'loss'
                  ? t('trainingStudio.mm3.maxSteps', 'Max steps')
                  : t('trainingStudio.mm3.steps', 'Steps')}
                value={form.steps}
                onChange={v => set('steps', v)}
                hint={form.stopMode === 'loss'
                  ? t('trainingStudio.mm3.maxStepsHint',
                      'The cap. If the target never arrives, the run ends here.') as string
                  : undefined} />
              {form.stopMode === 'loss' && (
                <>
                  <NumField label={t('trainingStudio.mm3.targetLoss', 'Target loss')}
                    value={form.targetLoss} onChange={v => set('targetLoss', v)} step={0.05}
                    hint={t('trainingStudio.mm3.targetLossHint',
                      'Stops as soon as the loss reaches this. Down is NOT automatically '
                      + 'better: under ~0.05 training loss the runs that got there had '
                      + 'memorised the songs.') as string} />
                  <label className="flex flex-col gap-1">
                    <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                      {t('trainingStudio.mm3.targetLossMetric', 'Measured on')}
                    </span>
                    <select className={INPUT} value={form.targetLossMetric}
                      onChange={e => set('targetLossMetric', e.target.value as 'train' | 'eval')}>
                      <option value="train">
                        {t('trainingStudio.mm3.metricTrain', 'Training loss (trailing mean)')}
                      </option>
                      <option value="eval">
                        {t('trainingStudio.mm3.metricEval', 'Held-out loss')}
                      </option>
                    </select>
                    <span className="text-[10px] text-zinc-500 leading-snug">
                      {form.targetLossMetric === 'eval'
                        ? t('trainingStudio.mm3.metricEvalHint',
                            'The number that says the adapter GENERALISES rather than memorised. '
                            + 'It only lands on evaluation steps, so with Evaluate every set at '
                            + '{{n}} it can only fire every {{n}} steps — lower that first, and '
                            + 'keep a hold-out fraction above 0.',
                            { n: form.evalEvery })
                        : t('trainingStudio.mm3.metricTrainHint',
                            'The mean of the last {{n}} full passes over the dataset, because one '
                            + 'step is one crop of one song and swings more than the whole run '
                            + 'does. Whole passes, not a step count: a window that is not a '
                            + 'multiple of your {{songs}} songs weighs some tracks more than '
                            + 'others. Available on every run, but it cannot tell learning from '
                            + 'memorising.',
                            { n: form.targetLossEpochs, songs: status?.codes ?? 0 })}
                    </span>
                  </label>
                </>
              )}
            </div>
            {form.stopMode === 'loss' && form.targetLossMetric === 'eval'
              && (form.holdout <= 0 || form.evalEvery <= 0) && (
              <div className="flex items-start gap-2 text-[11px] text-amber-600 dark:text-amber-400 mb-3">
                <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
                <span>
                  {t('trainingStudio.mm3.targetNeedsEval',
                    'Targeting the held-out loss needs a hold-out fraction above 0 and Evaluate '
                    + 'every above 0 — both are in Advanced. The run will be refused otherwise.')}
                </span>
              </div>
            )}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <NumField label={t('trainingStudio.mm3.saveEvery', 'Checkpoint every')} value={form.saveEvery}
                onChange={v => set('saveEvery', v)} />
              <NumField label={t('trainingStudio.mm3.rank', 'Rank')} value={form.rank}
                onChange={v => set('rank', v)} />
              <NumField label={t('trainingStudio.mm3.maxFrames', 'Crop (frames)')} value={form.maxFrames}
                onChange={v => set('maxFrames', v)} step={50} />
              <NumField label={t('trainingStudio.mm3.cropStartFrac', 'Crops at song start')}
                value={form.cropStartFrac} onChange={v => set('cropStartFrac', v)} step={0.05}
                hint={t('trainingStudio.mm3.cropStartFracHint',
                  'Share anchored at frame 0 — what teaches songs to OPEN like songs. '
                  + 'Too low and renders jump in mid-flow.') as string} />
              <NumField label={t('trainingStudio.mm3.cropStartTiles', 'Start tiles')}
                value={form.cropStartTiles} onChange={v => set('cropStartTiles', v)} step={1}
                hint={t('trainingStudio.mm3.cropStartTilesHint',
                  'Half the start share stays at frame 0; the rest lands on aligned tiles '
                  + 'after it, teaching the intro→build→verse arc under short crops. '
                  + '1 = frame 0 only.') as string} />
              <NumField label={t('trainingStudio.mm3.cropEndFrac', 'Crops at song end')}
                value={form.cropEndFrac} onChange={v => set('cropEndFrac', v)} step={0.05}
                hint={t('trainingStudio.mm3.cropEndFracHint',
                  'Share flush to the track end — the only place EOS is taught. The '
                  + 'REMAINDER of these two is the random share (currently '
                  + `${Math.max(0, Math.round((1 - form.cropStartFrac - form.cropEndFrac) * 100))}% mid-song crops).`) as string} />
              <NumField label={t('trainingStudio.mm3.depthLossWeight', 'Acoustic loss weight')}
                value={form.depthLossWeight} onChange={v => set('depthLossWeight', v)} step={0.1}
                hint={t('trainingStudio.mm3.depthLossWeightHint',
                  'Trains the adapter to keep vocal timbre intact: acoustic codebooks are '
                  + 'supervised through the frozen depth decoder. 0 disables — renders then '
                  + 'drift into chipmunk/goblin voices. Leave at 1.') as string} />
              <label className="flex items-start gap-2 text-[11px] text-zinc-600 dark:text-zinc-300 col-span-2">
                <input type="checkbox" className="mt-0.5" checked={form.keepResumeState}
                  onChange={e => set('keepResumeState', e.target.checked)} />
                <span>
                  {t('trainingStudio.mm3.keepResumeState', 'Keep resume state after completion')}
                  <span className="block text-[10px] text-zinc-500">
                    {t('trainingStudio.mm3.keepResumeStateHint',
                      'The optimizer state (about 4 GB) lets a finished run be continued past its step '
                      + 'count. Off: it is deleted once the run ends. A run that stops early keeps it either way.')}
                  </span>
                </span>
              </label>
              <label className="flex items-start gap-2 text-[11px] text-zinc-600 dark:text-zinc-300 col-span-2">
                <input type="checkbox" className="mt-0.5" checked={form.longTracks === 'exclude'}
                  onChange={e => set('longTracks', e.target.checked ? 'exclude' : 'crop')} />
                <span>
                  {t('trainingStudio.mm3.longTracksExclude', 'Leave out tracks longer than the window')}
                  <span className="block text-[10px] text-zinc-500">
                    {t('trainingStudio.mm3.longTracksHint',
                      'The recipe trains each track as one whole sequence. A track longer than the window '
                      + '(360 s at 9000 frames) cannot be, so it is left out and named in the log. Unticked: it is '
                      + 'trained in crops of the window instead, which is the pre-2026-09-11 behaviour.')}
                  </span>
                </span>
              </label>
              <NumField label={t('trainingStudio.mm3.depthLossFrames', 'Acoustic frames/step')}
                value={form.depthLossFrames} onChange={v => set('depthLossFrames', v)} step={16}
                hint={t('trainingStudio.mm3.depthLossFramesHint',
                  'Frames sampled per step for the acoustic loss.') as string} />
            </div>
            <label className="flex flex-col gap-1 mt-3">
              <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                {t('trainingStudio.mm3.trigger', 'Trigger word')}
              </span>
              <input className={INPUT} value={form.trigger} onChange={e => set('trigger', e.target.value)} />
            </label>
            {/* A sibling, not a child: a label nested inside a label is invalid
                and clicking the checkbox would focus the text input instead. */}
            <label className="flex items-start gap-2 mt-2 text-[11px] text-zinc-600 dark:text-zinc-300">
              <input type="checkbox" className="mt-0.5" checked={form.triggerPrepend}
                onChange={e => set('triggerPrepend', e.target.checked)}
                disabled={!form.trigger.trim()} />
              <span>
                {t('trainingStudio.mm3.triggerPrepend', 'Train the trigger')}
                <span className="block text-[10px] text-zinc-500">
                  {t('trainingStudio.mm3.triggerPrependHint',
                    'Puts "trigger, " at the front of every training caption, in memory — your files '
                    + 'are not touched. Leave this ON. With it off the word is only recorded in the '
                    + 'adapter sidecar and never learned, so typing it at render time bolts an unseen '
                    + 'token sequence onto your prompt and makes the result WORSE, not better.')}
                </span>
              </span>
            </label>

            {/* -- Previews ------------------------------------------------
                Above Advanced on purpose: this is the control that decides
                whether the user finds out at step 200 or at step 800 that the
                run is going the wrong way. */}
            <div className="mt-3 rounded-lg border border-zinc-200 dark:border-white/10 p-3">
              <div className="flex items-center gap-2">
                <Volume2 size={13} className="text-amber-500" />
                <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">
                  {t('trainingStudio.mm3.previewTitle', 'Audio previews during training')}
                </span>
              </div>
              <p className="text-[10px] text-zinc-500 leading-snug mt-1">
                {t('trainingStudio.mm3.previewBlurb',
                  'Renders a sample from the checkpoint while the run is still going, so you can hear '
                  + 'whether it is heading the right way and abort if not. Each preview point pauses '
                  + 'training for about a minute: the trainer has to hand the whole card to the render, '
                  + 'so it saves its optimizer state, exits, renders, and resumes exactly where it was. '
                  + 'Zero in both fields turns previews off.')}
              </p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-2">
                <NumField label={t('trainingStudio.mm3.previewEverySteps', 'Every N steps')}
                  value={form.previewEverySteps} onChange={v => set('previewEverySteps', v)}
                  step={50} hint={t('trainingStudio.mm3.previewStepsHint', '0 = off') as string} />
                <NumField label={t('trainingStudio.mm3.previewEveryMinutes', 'Or every N minutes')}
                  value={form.previewEveryMinutes} onChange={v => set('previewEveryMinutes', v)}
                  step={5} hint={t('trainingStudio.mm3.previewMinutesHint',
                    'Whichever comes first') as string} />
                <NumField label={t('trainingStudio.mm3.previewSeconds', 'Length (s)')}
                  value={form.previewSeconds} onChange={v => set('previewSeconds', v)}
                  step={4} hint={t('trainingStudio.mm3.previewSecondsHint',
                    '24 s costs about 16 s of GPU') as string} />
                <NumField label={t('trainingStudio.mm3.previewSeed', 'Preview seed')}
                  value={form.previewSeed} onChange={v => set('previewSeed', v)}
                  hint={t('trainingStudio.mm3.previewSeedHint',
                    'Fixed across the run') as string} />
                <NumField label={t('trainingStudio.mm3.previewScaleMlp', 'Preview MLP scale')}
                  value={form.previewScaleMlp} onChange={v => set('previewScaleMlp', v)}
                  step={0.05} hint={t('trainingStudio.mm3.previewScaleMlpHint',
                    'How hard the adapter’s MLP delta is applied in previews only. '
                    + '1 = full, 0 = attention only.') as string} />
              </div>
              {/* Explicit song pick (2026-09-05): a select over the dataset's
                  usable rows, default = auto. Fixes the auto-pick landing on
                  a noise interlude (oasis_morningglory's "instrumental_2",
                  40s of "[Instrumental]") by letting a user route around
                  whatever the ladder in mm3Preview.ts picks — the ladder
                  itself was also fixed the same day, this is the manual
                  override for when it still isn't the song you want. Wired
                  through the SAME override the caption box below uses:
                  picking a song here only takes effect while that box is
                  blank. */}
              <label className="flex flex-col gap-1 mt-3">
                <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                  {t('trainingStudio.mm3.previewSong', 'Preview song')}
                </span>
                <select className={INPUT} value={form.previewSongId}
                  disabled={!!form.previewCaption.trim()}
                  onChange={e => set('previewSongId', e.target.value)}>
                  <option value="">{t('trainingStudio.mm3.previewSongAuto', 'Auto (held-out, real lyrics preferred)')}</option>
                  {(status?.previewSongs ?? []).map(s => (
                    <option key={s.id} value={s.id}>
                      {s.filename}{s.held ? '' : ` (${t('trainingStudio.mm3.previewSongTraining', 'training')})`}
                      {!s.usableLyrics ? ` — ${t('trainingStudio.mm3.previewSongInstrumental', 'likely instrumental')}` : ''}
                      {s.durationS > 0 ? ` · ${Math.round(s.durationS)}s` : ''}
                    </option>
                  ))}
                </select>
                <span className="text-[10px] text-zinc-500 leading-snug">
                  {form.previewCaption.trim()
                    ? t('trainingStudio.mm3.previewSongDisabled', 'Ignored while the caption box below is filled in.')
                    : t('trainingStudio.mm3.previewSongHint',
                        'Rows marked "likely instrumental" are exactly what the auto-pick now avoids — '
                        + 'pick one anyway if that is deliberately what you want to hear.')}
                </span>
              </label>
              <label className="flex flex-col gap-1 mt-3">
                <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                  {t('trainingStudio.mm3.previewCaption', 'Preview caption')}
                </span>
                <textarea
                  className={`${INPUT} font-mono text-[11px] leading-snug`} rows={3}
                  placeholder={t('trainingStudio.mm3.previewCaptionPlaceholder',
                    'Blank = the preview-song pick above, or auto (first held-out song with real '
                    + 'lyrics), with the trigger prepended') as string}
                  value={form.previewCaption}
                  onChange={e => set('previewCaption', e.target.value)}
                />
                <span className="text-[10px] text-zinc-500 leading-snug">
                  {t('trainingStudio.mm3.previewCaptionHint',
                    'Leave blank unless you have a reason not to: a held-out caption carries the '
                    + 'artist’s true BPM and tuning, and a wrong one becomes audible as soon as '
                    + 'identity bakes in. The trigger is prepended as "trigger, " on the caption’s '
                    + 'first line, which is the exact shape the training rows use.')}
                </span>
              </label>
              <div className="flex flex-col gap-1.5 mt-3">
                <label className="flex items-start gap-2 text-[11px] text-zinc-600 dark:text-zinc-300">
                  <input type="checkbox" className="mt-0.5" checked={form.previewControl}
                    onChange={e => set('previewControl', e.target.checked)} />
                  <span>
                    {t('trainingStudio.mm3.previewControl', 'Also render a neutral control caption')}
                    <span className="block text-[10px] text-zinc-500">
                      {t('trainingStudio.mm3.previewControlHint',
                        'An off-genre prompt rendered WITH the adapter. This is the one that catches '
                        + 'the adapter damaging the base planner - on the artist caption a damaged '
                        + 'model and a good one both sound roughly like the artist.')}
                    </span>
                  </span>
                </label>
                <label className="flex items-start gap-2 text-[11px] text-zinc-600 dark:text-zinc-300">
                  <input type="checkbox" className="mt-0.5" checked={form.previewBaseline}
                    onChange={e => set('previewBaseline', e.target.checked)} />
                  <span>
                    {t('trainingStudio.mm3.previewBaseline', 'Render a no-adapter reference first')}
                    <span className="block text-[10px] text-zinc-500">
                      {t('trainingStudio.mm3.previewBaselineHint',
                        'Free - the engine is still up before the run starts. Without it there is '
                        + 'nothing to judge "worse than base" against.')}
                    </span>
                  </span>
                </label>
              </div>
            </div>

            {/* -- Prior preservation -------------------------------------- */}
            <div className="mt-3 rounded-lg border border-zinc-200 dark:border-white/10 p-3">
              <div className="flex items-center gap-2">
                <ShieldCheck size={13} className="text-amber-500" />
                <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">
                  {t('trainingStudio.mm3.regTitle', 'Prior preservation')}
                </span>
              </div>
              <p className="text-[10px] text-zinc-500 leading-snug mt-1">
                {t('trainingStudio.mm3.regBlurb',
                  'Spends some steps on an UNRELATED dataset, scored against what the base model '
                  + 'itself predicted there rather than against that music. The adapter is then '
                  + 'penalised for changing its mind about material that has nothing to do with the '
                  + 'artist, which is the only thing in the objective that separates "learned the '
                  + 'voice" from "rewrote the planner". Leave the corpus unset to turn it off.')}
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-2">
                <label className="flex flex-col gap-1 md:col-span-1">
                  <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                    {t('trainingStudio.mm3.regDataset', 'Corpus')}
                  </span>
                  <select className={INPUT} value={form.regDatasetId}
                    onChange={e => set('regDatasetId', e.target.value)}>
                    <option value="">{t('trainingStudio.mm3.regOff', 'Off')}</option>
                    {(status?.regCandidates ?? []).map(d => (
                      <option key={d.id} value={d.id}>{d.name} ({d.songs})</option>
                    ))}
                  </select>
                  <span className="text-[10px] text-zinc-500 leading-snug">
                    {(status?.regCandidates?.length ?? 0) === 0
                      ? t('trainingStudio.mm3.regNone',
                          'No other dataset has RVQ codes yet — a regularisation corpus needs exactly '
                          + 'what a training corpus needs. Run the codes export on one.')
                      : t('trainingStudio.mm3.regDatasetHint',
                          'Pick something with nothing in common with this artist. Its lyrics may be '
                          + 'empty; only its captions and codes are used.')}
                  </span>
                </label>
                <NumField label={t('trainingStudio.mm3.regEvery', 'Every N steps')}
                  value={form.regEvery} onChange={v => set('regEvery', v)}
                  hint={t('trainingStudio.mm3.regEveryHint',
                    '3 = one prior step per two style steps') as string} />
                <NumField label={t('trainingStudio.mm3.regTopK', 'Classes kept')}
                  value={form.regTopK} onChange={v => set('regTopK', v)} step={64}
                  hint={t('trainingStudio.mm3.regTopKHint',
                    'Coverage: 64 = 90%, 128 = 94%, 256 = 97%') as string} />
              </div>
              {form.regDatasetId && (
                <div className="mt-2 text-[10px] text-amber-600/90 dark:text-amber-400/90 leading-snug">
                  {t('trainingStudio.mm3.regDilution',
                    'This dilutes style exposure: at every {{n}} steps only {{s}} of your {{total}} '
                    + 'steps train on the artist. Raise Steps to about {{want}} to keep the same '
                    + 'exposure as running without it.',
                    {
                      n: form.regEvery,
                      s: form.steps - Math.floor(form.steps / Math.max(2, form.regEvery)),
                      total: form.steps,
                      want: Math.round(form.steps / (1 - 1 / Math.max(2, form.regEvery))),
                    })}
                </div>
              )}
            </div>

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
                {/* The dataset-wide caption box lived here until 2026-09-09. One caption for
                    every track replaced the per-track .mm3.txt captions the renders use, and
                    adapters trained that way did not end songs (album B 0/6 vs 4/6). Removed. */}
                {/* ── Method row (2026-09-05) ───────────────────────────────
                    Same shape as TrainDitForm's DitMethod / TrainLmForm's
                    LmMethod: LoKr is its own type, DoRA/HiRA/LoHa/HRA are the
                    LoRA type with one flag set, PiSSA/rsLoRA/LoRA+ sit
                    underneath as checkboxes. None of the five new methods has
                    an MM3-specific measurement — see each button's hover
                    text — and --dora/--hira/--loha/--pissa/--hra are landing
                    in the mm3-lm-train engine parser concurrently with this
                    card, not before it: a run that picks one is refused by an
                    older ace-train, loudly, rather than silently training
                    plain LoRA. */}
                <div className="flex flex-col gap-1.5">
                  <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                    {t('trainingStudio.mm3.adapterType', 'Adapter type')}
                  </span>
                  <div className="flex flex-wrap items-center gap-1.5">
                    {(['lokr', 'lora', 'dora', 'hira', 'loha', 'hra'] as Mm3Method[]).map(m => {
                      const active = method === m;
                      return (
                        <button
                          key={m}
                          type="button"
                          onClick={() => pickMethod(m)}
                          title={t(`trainingStudio.mm3.${MM3_METHOD_KEYS[m].info}`)}
                          className={`px-4 py-1.5 rounded-lg text-xs font-bold border transition-colors ${
                            active
                              ? 'text-amber-500 bg-amber-500/10 border-amber-500/30'
                              : 'text-zinc-500 border-zinc-300 dark:border-white/10 hover:text-zinc-700 dark:hover:text-zinc-300'
                          }`}
                        >
                          {t(`trainingStudio.mm3.${MM3_METHOD_KEYS[m].label}`)}
                        </button>
                      );
                    })}
                  </div>
                  <span className="text-[10px] text-zinc-500 leading-snug">
                    {t('trainingStudio.mm3.adapterTypeHint',
                      'LoRA is the default. LoKr trains a Kronecker pair per slot for a smaller file '
                      + '(about 528 MB at factor 6 against a rank-128 LoRA\'s 1.4 GB) — NOT YET '
                      + 'VALIDATED BY EAR. DoRA/HiRA/LoHa/HRA are new here 2026-09-05 (see each '
                      + 'button\'s hover text); none has an MM3 measurement of its own yet.')}
                  </span>
                  {method !== 'lokr' && (
                    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 rounded-lg border border-zinc-200 dark:border-white/5 px-3 py-2 mt-1">
                      <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300"
                        title={t('trainingStudio.mm3.hotPizzaInfo')}>
                        <input type="checkbox" checked={form.pissa && form.hotPizza} disabled={method !== 'lora'} className="accent-amber-500"
                          onChange={e => setEdits(prev => ({ ...prev, hotPizza: e.target.checked, ...(e.target.checked ? { pissa: true } : {}) }))} />
                        {t('trainingStudio.mm3.hotPizza', 'HOT-PiZZA')}
                      </label>
                      <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300"
                        title={t('trainingStudio.mm3.pissaInfo')}>
                        <input type="checkbox" checked={form.pissa} disabled={method !== 'lora'} className="accent-amber-500"
                          onChange={e => setEdits(prev => ({ ...prev, pissa: e.target.checked, ...(e.target.checked ? {} : { hotPizza: false }) }))} />
                        {t('trainingStudio.mm3.pissa', 'PiSSA init')}
                      </label>
                      <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                        <input type="checkbox" checked={form.rslora} disabled={method === 'hra'} className="accent-amber-500"
                          onChange={e => set('rslora', e.target.checked)} />
                        {t('trainingStudio.mm3.rslora', 'rsLoRA')}
                      </label>
                      <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                        {t('trainingStudio.mm3.loraPlusRatio', 'LoRA+ ratio')}
                        <input type="number" min={1} max={64} step={1} value={form.loraPlusRatio}
                          onChange={e => set('loraPlusRatio', Math.max(1, Number(e.target.value) || 1))}
                          className={`${INPUT} w-20`} />
                      </label>
                    </div>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <label className="flex flex-col gap-1">
                    <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                      {t('trainingStudio.mm3.optimizer', 'Optimizer')}
                    </span>
                    <select className={INPUT} value={form.optimizer}
                      onChange={e => set('optimizer', e.target.value as 'muon' | 'adamw' | 'prodigy')}>
                      <option value="prodigy">Prodigy (sets its own LR)</option>
                      <option value="adamw">AdamW</option>
                      <option value="muon">Muon</option>
                    </select>
                    <span className="text-[10px] text-zinc-500 leading-snug">
                      {t('trainingStudio.mm3.optimizerHint',
                        'Prodigy estimates its own step size, so the learning rate below becomes a '
                        + 'schedule multiplier only. On album B it converged to 8.19e-5 against the '
                        + '8e-5 tuned by hand. It costs two extra state buffers (about +2.7 GB at rank '
                        + '128) and CANNOT resume, so it is unavailable when mid-training previews are '
                        + 'on. AdamW matches the published SimpleTuner recipe. Muon is NOT recommended: '
                        + 'at the default LR scale of 64 it produced an adapter that rendered digital '
                        + 'silence, because that value was tuned on a different model.')}
                    </span>
                  </label>
                  <label className={'flex items-start gap-2 text-[11px] text-zinc-600 dark:text-zinc-300 '
                                  + `self-end pb-1${flashSupported ? '' : ' opacity-60'}`}>
                    <input type="checkbox" className="mt-0.5" checked={attnEffective === 'flash'}
                      disabled={!flashSupported}
                      onChange={e => set('attnBackend', e.target.checked ? 'flash' : 'exact')} />
                    <span>
                      {t('trainingStudio.mm3.attnBackend', 'Flash attention')}
                      <span className="block text-[10px] text-zinc-500">
                        {!flashSupported
                          ? t('trainingStudio.mm3.attnBackendUnsupported',
                              'Unavailable on this build: the fused attention kernels exist for CUDA '
                              + 'only, and the trainer refuses to start on a {{backend}} engine rather '
                              + 'than run them on the CPU behind your back. Training runs in exact mode.',
                              { backend: status?.engineBackend ?? 'non-CUDA' })
                          : flashCalibrated
                            ? t('trainingStudio.mm3.attnBackendHelp',
                                'Fused attention kernels — see the VRAM estimate above for the measured saving.')
                            : t('trainingStudio.mm3.attnBackendHelpPending',
                                'Fused attention kernels, once mm3-lm-train gains --attn (landing '
                                + 'concurrently). VRAM saving not yet measured — the estimate above stands '
                                + 'in with exact mode\'s number until it is.')}
                      </span>
                    </span>
                  </label>
                  {form.adapterType === 'lokr' && (
                    <NumField label={t('trainingStudio.mm3.lokrFactor', 'LoKr factor')}
                      value={form.lokrFactor} onChange={v => set('lokrFactor', v)}
                      hint={t('trainingStudio.mm3.lokrFactorHint',
                        '6 gives ~528 MB and 264M parameters. Higher is smaller and less capable: '
                        + '8 -> 274 MB, 16 -> 109 MB.') as string} />
                  )}
                  {form.optimizer === 'muon' && (
                    <NumField label={t('trainingStudio.mm3.muonLrScale', 'Muon LR scale')}
                      value={form.muonLrScale} onChange={v => set('muonLrScale', v)}
                      hint={t('trainingStudio.mm3.muonLrScaleHint',
                        '64 is the best of the values measured so far, not a tuned optimum.') as string} />
                  )}
                </div>

                {/* ── Soft prompt (2026-09-05) ──────────────────────────────
                    A trainable token + a trainable K/V prefix, both new to
                    MM3 (the mm3-lm-train ENGINE parser already accepts
                    --artist-token/-k/-lr; --prefix-n is landing concurrently
                    with this card). Kept visually separate from "History
                    before crop" below, which is a DIFFERENT, FROZEN prefix
                    baked from real audio (train/mm3-lm-kvprefix.h) — the two
                    are not the same knob and must not be confused. */}
                <div className="rounded-lg border border-zinc-200 dark:border-white/10 p-3 flex flex-col gap-2">
                  <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">
                    {t('trainingStudio.mm3.softPromptGroup', 'Soft prompt (trainable, this run)')}
                  </span>
                  <span className="text-[10px] text-zinc-500 leading-snug">
                    {t('trainingStudio.mm3.softPromptGroupHelp',
                      'A named token adds k learned vectors behind a placeholder in every caption; the '
                      + 'prefix adds n trainable key/value columns per layer. Both train WITH the LoRA '
                      + 'and ship in the same adapter file. Different mechanism from "History before '
                      + 'crop" below, which is FROZEN real-audio context, not a trained parameter.')}
                  </span>
                  <label className="flex items-start gap-2 text-[11px] text-zinc-600 dark:text-zinc-300">
                    <input type="checkbox" className="mt-0.5" checked={form.artistTokenOn}
                      onChange={e => {
                        const on = e.target.checked;
                        setEdits(prev => ({
                          ...prev, artistTokenOn: on,
                          artistToken: on && !form.artistToken.trim()
                            ? (form.trigger.trim() || 'artist') : form.artistToken,
                        }));
                      }} />
                    {t('trainingStudio.mm3.artistTokenOn', 'Train an artist token')}
                  </label>
                  {form.artistTokenOn && (
                    <div className="grid grid-cols-3 gap-2">
                      <label className="flex flex-col gap-1">
                        <span className="text-[10px] text-zinc-500">
                          {t('trainingStudio.mm3.artistToken', 'Token name')}
                        </span>
                        <input className={INPUT} value={form.artistToken}
                          onChange={e => set('artistToken', e.target.value.replace(/[^A-Za-z0-9_-]/g, ''))} />
                      </label>
                      <NumField label={t('trainingStudio.mm3.artistTokenK', 'Token vectors (k)')}
                        value={form.artistTokenK} onChange={v => set('artistTokenK', v)} />
                      <NumField label={t('trainingStudio.mm3.artistTokenLr', 'Token LR')}
                        value={form.artistTokenLr} onChange={v => set('artistTokenLr', v)} step={0.0005} />
                    </div>
                  )}
                  <NumField label={t('trainingStudio.mm3.prefixN', 'Prefix columns (trainable)')}
                    value={form.regDatasetId ? 0 : form.prefixN}
                    onChange={v => set('prefixN', Math.max(0, Math.min(64, v)))}
                    disabled={!!form.regDatasetId}
                    hint={(form.regDatasetId
                      ? t('trainingStudio.mm3.prefixNVsReg',
                          'Unavailable with a regularisation dataset: the prior capture needs an inert '
                          + 'model, and a prefix is non-zero from initialisation. The engine refuses '
                          + 'the pair.')
                      : t('trainingStudio.mm3.prefixNInfo',
                          'n trainable key/value columns per layer that every position attends to. 0 = off. '
                          + 'Not the frozen history prefix below — this one trains.')) as string} />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <label className="flex flex-col gap-1">
                    <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                      {t('trainingStudio.mm3.base', 'Base precision')}
                    </span>
                    <select className={INPUT} value={form.basePrecision}
                      onChange={e => set('basePrecision', e.target.value)}>
                      {(status?.bases ?? []).map(b => (
                        <option key={b.id} value={b.id}>
                          {b.id} — {(b.bytes / 1073741824).toFixed(1)} GB
                          {b.quality === 'poor' ? ' (not recommended)' : ''}
                          {b.id === status?.recommended?.base ? ' \u2713' : ''}
                        </option>
                      ))}
                    </select>
                    {peak && (
                      <span className={`text-[10px] leading-snug ${peak.tone}`}>{peak.text}</span>
                    )}
                    <span className="text-[10px] text-zinc-500 leading-snug">
                      {chosen?.quality === 'poor'
                        ? t('trainingStudio.mm3.basePoor',
                            'This base is too lossy to train against: measured +14% on the first-step loss '
                            + 'and roughly double the gradient norm, i.e. the quantizer injects more error '
                            + 'than the adapter is being asked to learn. Prefer the smallest base marked '
                            + 'good or better that fits.')
                        : t('trainingStudio.mm3.baseHint',
                            'Every base trains the same way — the frozen weights are dequantized in-graph, '
                            + 'so only VRAM and fidelity change, and step time barely moves (within ~5% '
                            + 'across the whole range). Pick the smallest one that comfortably fits your '
                            + 'card, then spend what is left on LoRA rank.')}
                    </span>
                  </label>
                  <NumField label={t('trainingStudio.mm3.holdout', 'Hold-out fraction')}
                    value={form.holdout} onChange={v => set('holdout', v)} step={0.05}
                    hint={t('trainingStudio.mm3.holdoutHint',
                      '0 disables evaluation — the training loss then cannot tell learning from '
                      + 'memorising. Ignored below 6 songs.') as string} />
                  <NumField label={t('trainingStudio.mm3.evalEvery', 'Evaluate every')}
                    value={form.evalEvery} onChange={v => set('evalEvery', v)} step={25}
                    hint={t('trainingStudio.mm3.evalEveryHint',
                      'Steps between held-out evaluations. 0 = off. This is also the finest '
                      + 'grain a held-out target can stop on.') as string} />
                </div>
                <label className="flex flex-col gap-1">
                  <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                    {t('trainingStudio.mm3.cropMode', 'Crop mode')}
                  </span>
                  <select className={INPUT} value={form.cropMode}
                    onChange={e => set('cropMode', e.target.value as 'random' | 'beginning' | 'structured')}>
                    <option value="structured">structured</option>
                    <option value="random">random</option>
                    <option value="beginning">beginning</option>
                  </select>
                  <span className={`text-[10px] leading-snug ${
                    form.cropMode !== 'structured'
                      ? 'text-amber-600/80 dark:text-amber-400/80' : 'text-zinc-500'
                  }`}>
                    {form.cropMode === 'random'
                      ? t('trainingStudio.mm3.cropModeRandom',
                          '`random` hands the model a prompt followed straight by mid-song audio with '
                          + 'no history in front of it, and supervises it — so it learns that a song may '
                          + 'begin at any position. It also leaves the ending to chance: EOS is only '
                          + 'supervised by a crop that reaches the track end, which on a 3-minute song '
                          + 'is under 3% of steps. Renders begin mid-flow and never resolve.')
                      : form.cropMode === 'beginning'
                        ? t('trainingStudio.mm3.cropModeIntros',
                            '`beginning` takes every track from frame 0, so every supervised position '
                            + 'has the song’s real history in front of it. What it can never teach is an '
                            + 'ending: EOS is only supervised by a crop that reaches the track end, and '
                            + 'it never does unless the whole song fits in the window.')
                        : t('trainingStudio.mm3.cropModeStructured',
                            '`structured` anchors most crops at frame 0, so every supervised position '
                            + 'carries the song’s real history exactly as it will at generation time, '
                            + 'and pins the rest flush to the track end — the only place EOS is '
                            + 'supervised. This is the default and the two others are each broken at '
                            + 'one end.')}
                  </span>
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                    {t('trainingStudio.mm3.cropAnchor', 'Crop position')}
                  </span>
                  <select className={INPUT} value={form.cropAnchor}
                    onChange={e => set('cropAnchor', e.target.value as 'song' | 'zero')}>
                    <option value="song">song (true position)</option>
                    <option value="zero">zero (legacy)</option>
                  </select>
                  <span className="text-[10px] text-zinc-500 leading-snug">
                    {t('trainingStudio.mm3.cropAnchorHint',
                      'Under "zero" every crop was presented to the model as if it were the opening of '
                      + 'the song, whatever part of the track it came from - while generation always '
                      + 'starts at frame 0. That mismatch teaches the model that a song can begin '
                      + 'anywhere, and shows up as sound arriving instantly at 0:00 and as tempo '
                      + 'drifting mid-track. "song" labels each crop with where it actually is. Runs '
                      + 'trained under the two are not comparable.')}
                  </span>
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                    {t('trainingStudio.mm3.prefixFrames', 'History before crop (frames)')}
                  </span>
                  <div className="flex gap-2">
                    <input type="number" className={INPUT} step={50} min={0}
                      value={form.prefixFrames}
                      disabled={form.cropAnchor === 'zero'}
                      onChange={e => set('prefixFrames', Math.max(0, Number(e.target.value)))} />
                    <button type="button" className={BTN_SM}
                      disabled={form.cropAnchor === 'zero'}
                      onClick={() => set('prefixFrames',
                        form.prefixFrames > 0 ? 0 : form.maxFrames)}>
                      {/* An ACTION, not a state. It read "Off" while the field
                          still said 4096, which is how at least one user
                          reported having switched the prefix off and trained
                          with it anyway (#142). */}
                      {form.prefixFrames > 0
                        ? t('trainingStudio.mm3.prefixTurnOff', 'Turn off')
                        : t('trainingStudio.mm3.prefixMatch', 'Match crop')}
                    </button>
                  </div>
                  <span className="text-[10px] text-zinc-500 leading-snug">
                    {form.cropAnchor === 'zero'
                      ? t('trainingStudio.mm3.prefixNeedsSong',
                          'Needs crop position "song". History placed at positions the crop then '
                          + 're-uses is not a history.')
                      : form.prefixFrames > 0
                        ? t('trainingStudio.mm3.prefixOn',
                            '{{sec}} s of the track before each crop is run through the model first, '
                            + 'with no gradient, so the crop attends back over real history instead of '
                            + 'starting from nothing. Adds about {{gb}} GB and roughly 60% to step '
                            + 'time. UNHEARD: no adapter trained this way has been auditioned.',
                            { sec: (form.prefixFrames / 25).toFixed(1),
                              gb: (estimateMm3PrefixMb(form.prefixFrames, form.maxFrames,
                                                       status!.vramModel) / 1024).toFixed(1) })
                        : t('trainingStudio.mm3.prefixOff2',
                            'Off. Each crop is presented at its true position in the track with an '
                            + 'EMPTY context, so the middle third of the planner learns to produce '
                            + 'late-song behaviour from a 30-second view. That is the band that renders '
                            + 'better with the Middle Third dial at 0. History is LINEAR in VRAM where '
                            + 'crop length is quadratic, so it buys context far more cheaply than a '
                            + 'longer crop does.')}
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

      {/* ── Previous runs, and continuing one ── */}
      <Mm3RunsPanel datasetId={datasetId} />

      {/* ── Live run: the shared job machinery, unchanged ── */}
      {mine && activeJob && (
        <div className={CARD}>
          <JobProgress />
          {jobKind === 'mm3-train-lm' && mm3Live && (
            // The run-stats row. MM3 has no epochs, so none of the ACE tiles
            // apply — these are the numbers that actually say what the run is
            // doing, and STEP TIME is the one that exposes a VRAM spill (about
            // 4 s when it fits, ~10x that when it pages to host memory).
            <div className="mt-3 grid grid-cols-2 md:grid-cols-5 gap-2">
              {[
                { k: 'loss', v: mm3Live.loss ? mm3Live.loss.toFixed(4) : '—' },
                { k: 'gradNorm', v: mm3Live.gradNorm ? mm3Live.gradNorm.toFixed(3) : '—' },
                // The SCHEDULE value is not what Muon applies: its update is
                // normalised, so the effective rate is lr x muon-lr-scale.
                // Showing 3.4e-5 while the optimizer used 2.2e-3 was a quietly
                // misleading tile.
                {
                  k: 'lr',
                  v: mm3Live.lr
                    ? (lrScale > 1
                      ? `${(mm3Live.lr * lrScale).toExponential(2)}`
                      : mm3Live.lr.toExponential(2))
                    : '—',
                },
                { k: 'stepTime', v: mm3Live.stepMs ? `${(mm3Live.stepMs / 1000).toFixed(1)}s` : '—' },
                {
                  k: 'vram',
                  v: mm3Live.totalMb
                    ? `${Math.round(mm3Live.usedMb / 1024)}/${Math.round(mm3Live.totalMb / 1024)} GB`
                    : '—',
                  warn: mm3Live.totalMb > 0 && mm3Live.usedMb > mm3Live.totalMb - 512,
                },
              ].map(tile => (
                <div key={tile.k}
                  className="rounded-lg border border-zinc-200 dark:border-white/5 px-2.5 py-1.5">
                  <div className="text-[10px] uppercase tracking-wider text-zinc-500">
                    {t(`trainingStudio.mm3.stat.${tile.k}`, tile.k)}
                  </div>
                  <div className={`text-sm font-semibold tabular-nums ${
                    (tile as { warn?: boolean }).warn ? 'text-amber-500' : 'text-zinc-800 dark:text-zinc-200'
                  }`}>{tile.v}</div>
                </div>
              ))}
            </div>
          )}
          {jobKind === 'mm3-train-lm' && <Mm3PreviewStrip />}
          {jobKind === 'mm3-train-lm' && trainStepSeries.length > 1 && (
            <div className="mt-3">
              {/* Four series on one fractional-epoch axis: per-step noise, the
                  epoch mean, its 5-epoch average, and — the one that matters —
                  held-out loss. The target line is drawn only when the run is
                  actually stopping on one; the store seeds it from the engine's
                  own announcement, so a run adopted after a reload still has
                  it. */}
              <TrainingChart
                epochs={trainLmEpochs}
                steps={trainStepSeries}
                milestones={trainMilestones}
                evals={trainEvalSeries}
                target={trainTargetLoss}
                maxEpochs={trainMaxEpochs}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Mm3TrainCard;
