// TrainDitForm.tsx — the DiT LoRA training controls
//
// Structural sibling of TrainLmForm: four visible controls (name, quality dial,
// target loss, epoch cap) and everything else behind the Advanced drawer, same
// pattern as PreprocessOptionsForm.
//
// The base model is a READ-ONLY chip, not a picker. The cached latents, context
// latents and encoder states in a preprocess variant are that DiT's own outputs,
// so training against a different base is meaningless — the server resolves
// --dit from preprocess_meta.json and ignores anything the client sends.

import React from 'react';
import { Loader2, Lock, Waves } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type {
  DitAdapterType,
  TrainDitStage,
} from '../../services/trainingApi';

export type DitQuality = 'fast' | 'balanced' | 'thorough';

/** How the crop box is being *displayed*. Form-local only — the wire field stays
 *  `crop`, in raw latent frames, exactly as the engine and the API expect. */
type CropMode = 'auto' | 'seconds' | 'frames';

/** Latent frame rate of the preprocess cache (preprocess_meta.json latent_fps). */
const LATENT_FPS = 25;

/** The DiT patch size is 2, so an odd crop is a frame the trainer cannot use. */
const secondsToFrames = (seconds: number): number => {
  const f = Math.round(seconds * LATENT_FPS);
  return f - (f % 2);
};

/** Everything the panel needs to POST, with every default already resolved. */
export interface TrainDitFormState {
  adapterName: string;
  quality: DitQuality;
  targetLoss: number;
  epochs: number;
  adapterType: DitAdapterType;
  rank: number;
  alpha: number;
  /** LyCORIS LoKR factors (K2). Only meaningful when adapterType==='lokr' —
   *  always present so a type toggle never has to invent them. */
  lokrDim: number;
  lokrAlpha: number;
  lokrFactor: number;
  lokrDecomposeBoth: boolean;
  targetMlp: boolean;
  layers: number;            // 0 = auto (top-K depth)
  crop: number;              // 0 = auto-fit
  cropMin: number;
  cropMax: number;
  learningRate: number;
  gradAccum: number;
  gradClip: number;
  warmupRatio: number;
  weightDecay: number;
  lossWeighting: 'none' | 'flow_snr';
  snrGamma: number;
  tBias: number;
  channelBalance: boolean;
  timestepMu: number;
  timestepSigma: number;
  tMin: number;
  tMax: number;
  cfgRatio: number;
  genreRatio: number;
  seed: number;
  order: 'shuffle' | 'fixed';
  milestoneStep: number;
  milestoneKeep: number;
  vramReserveMb: number;
  mirror: 'f32' | 'bf16';
  /** Crops per micro-batch, from that many DIFFERENT songs (design §2.2). */
  batch: number;
  /** Gradient-checkpointing segments: 0 = off, 1 = auto, 2-32 = fixed. */
  ckptSegments: number;
  stages: TrainDitStage[];
  overwrite: boolean;
  stopEngine: boolean;
}

/** §2.6 defaults, with the DiT retune applied: 400 epochs, r128/a256, 30 %
 *  genre conditioning and the MLP projections trained. Kept in lockstep with
 *  the train-dit route's own numOpt fallbacks (server/src/routes/training.ts)
 *  and DitTrainArgs (engine/src/train/dit-train-run.h). */
export const TRAIN_DIT_DEFAULTS: TrainDitFormState = {
  adapterName: '',
  quality: 'balanced',
  targetLoss: 0.4,
  epochs: 400,
  adapterType: 'lora',
  rank: 128,
  alpha: 256,
  // Inert while adapterType==='lora'; kept at the K2 values so a toggle to
  // 'lokr' via TRAIN_DIT_LOKR_DEFAULTS below is the only place they change.
  lokrDim: 512,
  lokrAlpha: 512,
  lokrFactor: 6,
  lokrDecomposeBoth: true,
  targetMlp: true,
  layers: 0,
  crop: 0,
  cropMin: 375,
  cropMax: 1250,
  learningRate: 0.0005,
  gradAccum: 4,
  gradClip: 1.0,
  warmupRatio: 0.05,
  weightDecay: 0.01,
  lossWeighting: 'flow_snr',
  snrGamma: 5.0,
  tBias: 0.5,
  channelBalance: true,
  timestepMu: -0.4,
  timestepSigma: 1.0,
  tMin: 0,
  tMax: 1,
  cfgRatio: 0.15,
  genreRatio: 30,
  seed: 42,
  order: 'shuffle',
  milestoneStep: 0.1,
  milestoneKeep: 6,
  vramReserveMb: 2048,
  // Default 'bf16' for both adapter types (2026-07-29); TRAIN_DIT_LOKR_DEFAULTS
  // spreads this object and doesn't override mirror, so LoKR inherits it too.
  // The engine itself falls back to 'f32' with a warning on a non-CUDA backend.
  mirror: 'bf16',
  // Micro-batching + checkpointing defaults (design §2.2 / C3/C5) — the engine's
  // own train-dit defaults. batch 1 = OFF (2026-07-29, measured): batching is
  // ~2.5x SLOWER at full depth on a 32 GB card and ~2.4x faster on shallow /
  // partial-depth runs, so it is opt-in for the latter. ckptSegments 1 = auto,
  // which resolves to a single (i.e. unsegmented) run whenever full depth fits.
  batch: 1,
  ckptSegments: 1,
  stages: ['train', 'export'],
  overwrite: false,
  stopEngine: true,
};

/** K1/K2 (lokr-dit-training plan §0): LoKR is the UI's DEFAULT adapter type —
 *  Rob's validated preference (Uber-LoKR-4). Same base as TRAIN_DIT_DEFAULTS,
 *  patched with the preset's four adapter fields plus the loss-shape knobs the
 *  preset agrees with (lr/loss-weighting/target-loss/weight-decay). Everything
 *  else (epochs, gradAccum, targetMlp, cfgRatio, channelBalance, seed,
 *  milestoneStep…) stays at the shared default.
 *
 *  crop stays 0 (auto-fit) like the shared default. It shipped as a fixed 60 s
 *  window to match Side-Step's chunk_duration, but the auto-fit reduces CROP
 *  before it reduces DEPTH: pinning the window is what forces the depth ladder
 *  down instead, so a 32 GB card would train a partial-depth adapter to keep the
 *  60 s. Auto keeps all 32 layers and takes the longest window that genuinely
 *  fits. DIT_LOKR_CROP_FRAMES below is still where the Seconds/Frames control
 *  lands when the user leaves auto. */
export const DIT_LOKR_CROP_FRAMES = 1500;   // 60 s at 25 latent fps

export const TRAIN_DIT_LOKR_DEFAULTS: TrainDitFormState = {
  ...TRAIN_DIT_DEFAULTS,
  adapterType: 'lokr',
  lokrDim: 512,
  lokrAlpha: 512,
  lokrFactor: 6,
  lokrDecomposeBoth: true,
  learningRate: 0.01,
  lossWeighting: 'none',
  targetLoss: 0.6,
  weightDecay: 0.001,
  crop: 0,
};

/** Which unit to show `crop` in. State, not a pure derivation: once the box holds
 *  a number, seconds and frames are the same value and only the user's last
 *  choice distinguishes them. This is the seed for that state. */
const deriveCropMode = (state: TrainDitFormState): CropMode => {
  if (state.crop === 0) return 'auto';
  // The LoKR window is authored in seconds, so show it in seconds.
  if (state.adapterType === 'lokr' && state.crop === DIT_LOKR_CROP_FRAMES) return 'seconds';
  return 'frames';
};

/** §5.2 quality dial. Each preset patches the SAME key set so flipping back and
 *  forth is reversible — the plan names targetLoss only under Thorough, which on
 *  its own would leave 0.3 stuck after switching away from it. */
// Balanced IS the default state, so its epoch count has to be the new default
// (400) — leaving it at 100 would mean the dial silently undid the retune the
// moment anyone clicked the button that was already highlighted. Fast and
// Thorough are re-spaced around it to keep the dial monotone.
const DIT_QUALITY_PRESETS: Record<DitQuality, Partial<TrainDitFormState>> = {
  fast:     { epochs: 100, cropMax: 750,  milestoneStep: 0.2, targetLoss: 0.4 },
  balanced: { epochs: 400, cropMax: 1250, milestoneStep: 0.1, targetLoss: 0.4 },
  thorough: { epochs: 800, cropMax: 1250, milestoneStep: 0.1, targetLoss: 0.3 },
};

/** …but the loss thresholds above are LoRA's. LoKR's validated auto-stop is 0.6
 *  (K2, frozen contract), so applying the LoRA numbers on a quality click would
 *  silently undo the default the LoKR form ships with. Only targetLoss is
 *  type-aware — every other preset field means the same thing for both. */
const DIT_LOKR_TARGET_LOSS: Record<DitQuality, number> = {
  fast: 0.6, balanced: 0.6, thorough: 0.5,
};

const ALL_STAGES: TrainDitStage[] = ['train', 'export'];
const QUALITIES: DitQuality[] = ['fast', 'balanced', 'thorough'];

const FIELD =
  'rounded-lg px-3 py-2 text-sm bg-zinc-100 dark:bg-black/20 border border-zinc-300 dark:border-white/10 text-zinc-800 dark:text-zinc-200 focus:outline-none focus:border-amber-500 disabled:opacity-50 disabled:cursor-not-allowed';
const LABEL = 'text-xs font-semibold text-zinc-600 dark:text-zinc-400';

interface Props {
  value: TrainDitFormState;
  onChange: (patch: Partial<TrainDitFormState>) => void;
  /** The variant's own DiT, from trainDitStatus.ditModel. Read-only. */
  ditModel: string;
  /** Locks every control (a job is running, or a start is in flight). */
  disabled?: boolean;
  starting?: boolean;
  onStart: () => void;
}

export const TrainDitForm: React.FC<Props> = ({
  value, onChange, ditModel, disabled, starting, onStart,
}) => {
  const { t } = useTranslation();

  const lock = !!disabled;

  // Display unit for `crop`. Seeded from the incoming state, then re-seeded
  // whenever something outside the component (a preset load, an adapter-type
  // swap) moves `crop` across the auto boundary — which is the only move the
  // current mode can no longer represent.
  const [cropMode, setCropMode] = React.useState<CropMode>(() => deriveCropMode(value));
  React.useEffect(() => {
    if ((cropMode === 'auto') !== (value.crop === 0)) {
      setCropMode(deriveCropMode(value));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value.crop, value.adapterType, cropMode]);

  // Mirrors the server's ADAPTER_NAME_RE (§4.5 step 6). The GET path sanitises
  // the same input instead of rejecting it, so without this the panel can render
  // a green "Adapter ready" card for a directory the POST will refuse.
  const trimmed = value.adapterName.trim();
  const adapterNameOk = /^[A-Za-z0-9._-]{1,64}$/.test(trimmed)
    && !/^\.+$/.test(trimmed)
    && !trimmed.startsWith('.');

  const num = (raw: string, fallback: number): number => {
    // '' -> Number('') === 0, which is finite. targetLoss 0 legitimately means
    // "no auto-stop" and crop/layers 0 mean "auto", so an emptied box must fall
    // back to the default rather than silently selecting one of those modes.
    if (raw.trim() === '') return fallback;
    const n = Number(raw);
    return Number.isFinite(n) ? n : fallback;
  };

  const toggleStage = (stage: TrainDitStage, on: boolean) => {
    const next = ALL_STAGES.filter(s => (s === stage ? on : value.stages.includes(s)));
    onChange({ stages: next });
  };

  const pickQuality = (q: DitQuality) => onChange({
    quality: q,
    ...DIT_QUALITY_PRESETS[q],
    ...(value.adapterType === 'lokr' ? { targetLoss: DIT_LOKR_TARGET_LOSS[q] } : {}),
  });

  // §4: switching adapter type swaps the WHOLE form state to that type's
  // default set (simplest option the plan sanctions over per-field "was it
  // touched" tracking) — except the adapter name, which is the one field
  // losing on a toggle would actually cost the user something.
  const pickType = (ty: DitAdapterType) => {
    if (ty === value.adapterType) return;
    const defaults = ty === 'lokr' ? TRAIN_DIT_LOKR_DEFAULTS : TRAIN_DIT_DEFAULTS;
    setCropMode(deriveCropMode(defaults));
    onChange({ ...defaults, adapterName: value.adapterName });
  };

  const pickCropMode = (mode: CropMode) => {
    setCropMode(mode);
    if (mode === 'auto') {
      onChange({ crop: 0 });
    } else if (value.crop === 0) {
      // seconds <-> frames is lossless (same field, same number), so only the
      // hop out of auto has to invent one. Land on the documented 60 s window.
      onChange({ crop: DIT_LOKR_CROP_FRAMES });
    }
  };

  const setCropValue = (raw: string) => {
    if (cropMode === 'seconds') {
      onChange({ crop: secondsToFrames(num(raw, DIT_LOKR_CROP_FRAMES / LATENT_FPS)) });
    } else {
      onChange({ crop: num(raw, 0) });
    }
  };

  const cropShown = cropMode === 'seconds'
    ? Math.round((value.crop / LATENT_FPS) * 10) / 10
    : value.crop;

  // cropMax below cropMin is a 400 from the server (§4.5 step 7); flag it here.
  const cropRangeOk = value.cropMax >= value.cropMin;
  const tWindowOk = value.tMin < value.tMax;
  const isLokr = value.adapterType === 'lokr';
  // Server range: -1 or [2,64] (§2.1). 0/1/negative-not-(-1) values are refused.
  const lokrFactorOk = value.lokrFactor === -1 || (value.lokrFactor >= 2 && value.lokrFactor <= 64);
  // The server accepts crop 0 (auto) or 128..8192 — the 2..126 band the stepper
  // walks through is a guaranteed 400. Same for lr (strictly > 0) and seed.
  const cropOk = value.crop === 0 || (value.crop >= 128 && value.crop <= 8192);
  const lrOk = value.learningRate > 0 && value.learningRate <= 1;
  const seedOk = value.seed >= 0 && value.seed <= 2147483647;

  return (
    <div className="flex flex-col gap-4">
      {/* ── Adapter type (K1: LoKR is the default) ──────────────────────── */}
      <div className="flex flex-col gap-1.5">
        <span className={LABEL}>{t('trainingStudio.train.dit.adapterType')}</span>
        <div className="flex items-center gap-1.5">
          {(['lokr', 'lora'] as DitAdapterType[]).map(ty => {
            const active = value.adapterType === ty;
            return (
              <button
                key={ty}
                type="button"
                disabled={lock}
                onClick={() => pickType(ty)}
                className={`px-4 py-1.5 rounded-lg text-xs font-bold border transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                  active
                    ? 'text-amber-500 bg-amber-500/10 border-amber-500/30'
                    : 'text-zinc-500 border-zinc-300 dark:border-white/10 hover:text-zinc-700 dark:hover:text-zinc-300'
                }`}
              >
                {ty === 'lokr' ? t('trainingStudio.train.dit.adapterTypeLokr') : t('trainingStudio.train.dit.adapterTypeLora')}
              </button>
            );
          })}
        </div>
        <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.dit.adapterTypeHelp')}</span>
      </div>

      {/* ── Base model (read-only) ────────────────────────────────────── */}
      <div className="flex flex-col gap-1.5">
        <span className={LABEL}>{t('trainingStudio.train.dit.baseModel')}</span>
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-zinc-300 dark:border-white/10 bg-zinc-200/50 dark:bg-white/5">
          <Lock size={12} className="text-zinc-500 flex-shrink-0" />
          <code className="min-w-0 truncate font-mono text-[11px] text-zinc-700 dark:text-zinc-300" title={ditModel}>
            {ditModel || '—'}
          </code>
        </div>
        <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.dit.baseModelHelp')}</span>
      </div>

      {/* ── Quality dial ──────────────────────────────────────────────── */}
      <div className="flex flex-col gap-1.5">
        <span className={LABEL}>{t('trainingStudio.train.dit.quality')}</span>
        <div className="flex items-center gap-1.5 flex-wrap">
          {QUALITIES.map(q => {
            const active = value.quality === q;
            const label = q === 'fast'
              ? t('trainingStudio.train.dit.qualityFast')
              : q === 'balanced'
                ? t('trainingStudio.train.dit.qualityBalanced')
                : t('trainingStudio.train.dit.qualityThorough');
            return (
              <button
                key={q}
                type="button"
                disabled={lock}
                onClick={() => pickQuality(q)}
                className={`px-4 py-1.5 rounded-lg text-xs font-bold border transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                  active
                    ? 'text-amber-500 bg-amber-500/10 border-amber-500/30'
                    : 'text-zinc-500 border-zinc-300 dark:border-white/10 hover:text-zinc-700 dark:hover:text-zinc-300'
                }`}
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {/* ── Adapter name ────────────────────────────────────────────── */}
        <label className="flex flex-col gap-1.5">
          <span className={LABEL}>{t('trainingStudio.train.dit.adapterName')}</span>
          <input
            type="text"
            value={value.adapterName}
            disabled={lock}
            spellCheck={false}
            onChange={(e) => onChange({ adapterName: e.target.value })}
            className={`${FIELD}${trimmed && !adapterNameOk ? ' border-red-500/50' : ''}`}
          />
          {trimmed && !adapterNameOk && (
            <span className="text-[11px] text-red-500 dark:text-red-400">
              {t('trainingStudio.train.adapterNameInvalid')}
            </span>
          )}
        </label>

        {/* ── Target loss ─────────────────────────────────────────────── */}
        <label className="flex flex-col gap-1.5">
          <span className={LABEL}>{t('trainingStudio.train.dit.targetLoss')}</span>
          <input
            type="number"
            min={0}
            max={20}
            step={0.05}
            value={value.targetLoss}
            disabled={lock}
            onChange={(e) => onChange({ targetLoss: num(e.target.value, isLokr ? 0.6 : 0.4) })}
            className={FIELD}
          />
        </label>
      </div>

      <span className="text-[11px] text-zinc-500 -mt-2">{t('trainingStudio.train.dit.targetLossHelp')}</span>

      {/* ── Max epochs ────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <label className="flex flex-col gap-1.5">
          <span className={LABEL}>{t('trainingStudio.train.dit.maxEpochs')}</span>
          <input
            type="number"
            min={1}
            max={2000}
            step={1}
            value={value.epochs}
            disabled={lock}
            onChange={(e) => onChange({ epochs: num(e.target.value, 400) })}
            className={FIELD}
          />
          <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.maxEpochsHelp')}</span>
        </label>
      </div>

      {/* ── Advanced ──────────────────────────────────────────────────── */}
      <details className="rounded-lg border border-zinc-200 dark:border-white/5 px-3 py-2">
        <summary className="cursor-pointer text-xs font-semibold text-zinc-600 dark:text-zinc-400 select-none">
          {t('trainingStudio.train.dit.advanced')}
        </summary>

        <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
          {isLokr ? (
            <>
              <label className="flex flex-col gap-1.5">
                <span className={LABEL}>{t('trainingStudio.train.dit.lokrDim')}</span>
                <input
                  type="number" min={4} max={4096} step={1}
                  value={value.lokrDim} disabled={lock}
                  onChange={(e) => onChange({ lokrDim: num(e.target.value, 512) })}
                  className={FIELD}
                />
              </label>

              <label className="flex flex-col gap-1.5">
                <span className={LABEL}>{t('trainingStudio.train.dit.lokrAlpha')}</span>
                <input
                  type="number" min={0} max={8192} step={1}
                  value={value.lokrAlpha} disabled={lock}
                  onChange={(e) => onChange({ lokrAlpha: num(e.target.value, 512) })}
                  className={FIELD}
                />
              </label>

              <label className="flex flex-col gap-1.5">
                <span className={LABEL}>{t('trainingStudio.train.dit.lokrFactor')}</span>
                <input
                  type="number" min={-1} max={64} step={1}
                  value={value.lokrFactor} disabled={lock}
                  onChange={(e) => onChange({ lokrFactor: num(e.target.value, 6) })}
                  className={`${FIELD}${lokrFactorOk ? '' : ' border-red-500/60'}`}
                />
                <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.dit.lokrFactorHelp')}</span>
                {!lokrFactorOk && (
                  <span className="text-[11px] text-red-500">
                    {t('trainingStudio.train.dit.lokrFactorInvalid')}
                  </span>
                )}
              </label>

              <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300 sm:col-span-2">
                <input
                  type="checkbox"
                  checked={value.lokrDecomposeBoth}
                  disabled={lock}
                  onChange={(e) => onChange({ lokrDecomposeBoth: e.target.checked })}
                  className="accent-amber-500"
                />
                {t('trainingStudio.train.dit.lokrDecomposeBoth')}
              </label>
            </>
          ) : (
            <>
              <label className="flex flex-col gap-1.5">
                <span className={LABEL}>{t('trainingStudio.train.dit.rank')}</span>
                <input
                  type="number" min={1} max={256} step={1}
                  value={value.rank} disabled={lock}
                  onChange={(e) => onChange({ rank: num(e.target.value, 128) })}
                  className={FIELD}
                />
              </label>

              <label className="flex flex-col gap-1.5">
                <span className={LABEL}>{t('trainingStudio.train.dit.alpha')}</span>
                <input
                  type="number" min={1} max={1024} step={1}
                  value={value.alpha} disabled={lock}
                  onChange={(e) => onChange({ alpha: num(e.target.value, 256) })}
                  className={FIELD}
                />
              </label>
            </>
          )}

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.layers')}</span>
            <input
              type="number" min={0} max={64} step={1}
              value={value.layers} disabled={lock}
              onChange={(e) => onChange({ layers: num(e.target.value, 0) })}
              className={FIELD}
            />
            <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.dit.layersAuto')}</span>
            <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.dit.layersHelp')}</span>
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.crop')}</span>
            <div className="flex items-center gap-1.5">
              <select
                value={cropMode} disabled={lock}
                onChange={(e) => pickCropMode(e.target.value as CropMode)}
                className={`${FIELD} shrink-0`}
              >
                <option value="auto">{t('trainingStudio.train.dit.cropModeAuto')}</option>
                <option value="seconds">{t('trainingStudio.train.dit.cropModeSeconds')}</option>
                <option value="frames">{t('trainingStudio.train.dit.cropModeFrames')}</option>
              </select>
              {cropMode !== 'auto' && (
                <input
                  type="number"
                  min={cropMode === 'seconds' ? 5 : 0}
                  max={cropMode === 'seconds' ? 320 : 8192}
                  step={cropMode === 'seconds' ? 1 : 2}
                  value={cropShown} disabled={lock}
                  onChange={(e) => setCropValue(e.target.value)}
                  className={`${FIELD} w-full min-w-0${cropOk ? '' : ' border-red-500/60'}`}
                />
              )}
            </div>
            <span className="text-[11px] text-zinc-500">
              {cropMode === 'auto'
                ? t('trainingStudio.train.dit.cropAuto')
                : cropMode === 'seconds'
                  ? t('trainingStudio.train.dit.cropSecondsHelp')
                  : t('trainingStudio.train.dit.cropFramesHelp')}
            </span>
            {!cropOk && (
              <span className="text-[11px] text-red-500">
                {t('trainingStudio.train.dit.cropHelp')}
              </span>
            )}
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.cropMin')}</span>
            <input
              type="number" min={128} max={8192} step={1}
              value={value.cropMin} disabled={lock}
              onChange={(e) => onChange({ cropMin: num(e.target.value, 375) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.cropMax')}</span>
            <input
              type="number" min={128} max={8192} step={1}
              value={value.cropMax} disabled={lock}
              onChange={(e) => onChange({ cropMax: num(e.target.value, 1250) })}
              className={`${FIELD}${cropRangeOk ? '' : ' border-red-500/50'}`}
            />
          </label>

          <span className="text-[11px] text-zinc-500 sm:col-span-2 -mt-1">
            {t('trainingStudio.train.dit.cropHelp')}
          </span>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.learningRate')}</span>
            <input
              type="number" min={0.00001} max={1} step={0.00001}
              value={value.learningRate} disabled={lock}
              onChange={(e) => onChange({ learningRate: num(e.target.value, isLokr ? 0.01 : 0.0005) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.gradAccum')}</span>
            <input
              type="number" min={1} max={64} step={1}
              value={value.gradAccum} disabled={lock}
              onChange={(e) => onChange({ gradAccum: num(e.target.value, 4) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.gradClip')}</span>
            <input
              type="number" min={0} max={100} step={0.1}
              value={value.gradClip} disabled={lock}
              onChange={(e) => onChange({ gradClip: num(e.target.value, 1) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.warmupRatio')}</span>
            <input
              type="number" min={0} max={0.5} step={0.01}
              value={value.warmupRatio} disabled={lock}
              onChange={(e) => onChange({ warmupRatio: num(e.target.value, 0.05) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.weightDecay')}</span>
            <input
              type="number" min={0} max={1} step={0.005}
              value={value.weightDecay} disabled={lock}
              onChange={(e) => onChange({ weightDecay: num(e.target.value, isLokr ? 0.001 : 0.01) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.lossWeighting')}</span>
            <select
              value={value.lossWeighting}
              disabled={lock}
              onChange={(e) => onChange({ lossWeighting: e.target.value === 'none' ? 'none' : 'flow_snr' })}
              className={FIELD}
            >
              <option value="flow_snr">flow_snr</option>
              <option value="none">none</option>
            </select>
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.snrGamma')}</span>
            <input
              type="number" min={1} max={100} step={0.5}
              value={value.snrGamma} disabled={lock}
              onChange={(e) => onChange({ snrGamma: num(e.target.value, 5) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.tBias')}</span>
            <input
              type="number" min={0} max={4} step={0.05}
              value={value.tBias} disabled={lock}
              onChange={(e) => onChange({ tBias: num(e.target.value, 0.5) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.timestepMu')}</span>
            <input
              type="number" min={-4} max={4} step={0.05}
              value={value.timestepMu} disabled={lock}
              onChange={(e) => onChange({ timestepMu: num(e.target.value, -0.4) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.timestepSigma')}</span>
            <input
              type="number" min={0.01} max={4} step={0.05}
              value={value.timestepSigma} disabled={lock}
              onChange={(e) => onChange({ timestepSigma: num(e.target.value, 1) })}
              className={FIELD}
            />
          </label>

          <div className="flex flex-col gap-1.5 sm:col-span-2">
            <span className={LABEL}>{t('trainingStudio.train.dit.tWindow')}</span>
            <div className="flex items-center gap-2">
              <input
                type="number" min={0} max={1} step={0.01}
                value={value.tMin} disabled={lock}
                onChange={(e) => onChange({ tMin: num(e.target.value, 0) })}
                className={`${FIELD} flex-1 min-w-0${tWindowOk ? '' : ' border-red-500/50'}`}
              />
              <span className="text-xs text-zinc-500">→</span>
              <input
                type="number" min={0} max={1} step={0.01}
                value={value.tMax} disabled={lock}
                onChange={(e) => onChange({ tMax: num(e.target.value, 1) })}
                className={`${FIELD} flex-1 min-w-0${tWindowOk ? '' : ' border-red-500/50'}`}
              />
            </div>
          </div>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.cfgRatio')}</span>
            <input
              type="number" min={0} max={1} step={0.05}
              value={value.cfgRatio} disabled={lock}
              onChange={(e) => onChange({ cfgRatio: num(e.target.value, 0.15) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.genreRatio')}</span>
            <input
              type="number" min={0} max={100} step={1}
              value={value.genreRatio} disabled={lock}
              onChange={(e) => onChange({ genreRatio: num(e.target.value, 30) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.seed')}</span>
            <input
              type="number" min={0} max={2147483647} step={1}
              value={value.seed} disabled={lock}
              onChange={(e) => onChange({ seed: num(e.target.value, 42) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.order')}</span>
            <select
              value={value.order}
              disabled={lock}
              onChange={(e) => onChange({ order: e.target.value === 'fixed' ? 'fixed' : 'shuffle' })}
              className={FIELD}
            >
              <option value="shuffle">shuffle</option>
              <option value="fixed">fixed</option>
            </select>
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.milestoneStep')}</span>
            <input
              type="number" min={0} max={5} step={0.05}
              value={value.milestoneStep} disabled={lock}
              onChange={(e) => onChange({ milestoneStep: num(e.target.value, 0.1) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.milestoneKeep')}</span>
            <input
              type="number" min={0} max={64} step={1}
              value={value.milestoneKeep} disabled={lock}
              onChange={(e) => onChange({ milestoneKeep: num(e.target.value, 6) })}
              className={FIELD}
            />
          </label>

          <span className="text-[11px] text-zinc-500 sm:col-span-2 -mt-1">
            {t('trainingStudio.train.dit.milestoneDiskHelp')}
          </span>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.vramReserve')}</span>
            <input
              type="number" min={0} max={16384} step={128}
              value={value.vramReserveMb} disabled={lock}
              onChange={(e) => onChange({ vramReserveMb: num(e.target.value, 2048) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.mirror')}</span>
            <select
              value={value.mirror}
              disabled={lock}
              onChange={(e) => onChange({ mirror: e.target.value === 'bf16' ? 'bf16' : 'f32' })}
              className={FIELD}
            >
              <option value="f32">{t('trainingStudio.train.dit.mirrorF32')}</option>
              <option value="bf16">{t('trainingStudio.train.dit.mirrorBf16')}</option>
            </select>
          </label>

          <span className="text-[11px] text-zinc-500 sm:col-span-2 -mt-1">
            {t('trainingStudio.train.dit.mirrorHelp')}
          </span>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.batch')}</span>
            <input
              type="number" min={1} max={16} step={1}
              value={value.batch} disabled={lock}
              onChange={(e) => onChange({ batch: num(e.target.value, 1) })}
              className={FIELD}
            />
            <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.dit.batchHelp')}</span>
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.dit.ckptSegments')}</span>
            <select
              value={value.ckptSegments}
              disabled={lock}
              onChange={(e) => onChange({ ckptSegments: Number(e.target.value) })}
              className={FIELD}
            >
              <option value={1}>{t('trainingStudio.train.dit.ckptAuto')}</option>
              <option value={0}>{t('trainingStudio.train.dit.ckptOff')}</option>
              <option value={2}>2</option>
              <option value={4}>4</option>
              <option value={8}>8</option>
              <option value={16}>16</option>
            </select>
            <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.dit.ckptSegmentsHelp')}</span>
          </label>
        </div>

        <div className="mt-3 flex flex-col gap-2">
          <span className={LABEL}>{t('trainingStudio.train.dit.stages')}</span>
          <div className="flex items-center gap-4 flex-wrap">
            {ALL_STAGES.map(stage => (
              <label key={stage} className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                <input
                  type="checkbox"
                  checked={value.stages.includes(stage)}
                  disabled={lock}
                  onChange={(e) => toggleStage(stage, e.target.checked)}
                  className="accent-amber-500"
                />
                {stage}
              </label>
            ))}
          </div>

          <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
            <input
              type="checkbox"
              checked={value.targetMlp}
              disabled={lock}
              onChange={(e) => onChange({ targetMlp: e.target.checked })}
              className="accent-amber-500"
            />
            {t('trainingStudio.train.dit.targetMlp')}
          </label>
          <span className="text-[11px] text-zinc-500 pl-6">{t('trainingStudio.train.dit.targetMlpHelp')}</span>

          <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
            <input
              type="checkbox"
              checked={value.channelBalance}
              disabled={lock}
              onChange={(e) => onChange({ channelBalance: e.target.checked })}
              className="accent-amber-500"
            />
            {t('trainingStudio.train.dit.channelBalance')}
          </label>

          <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
            <input
              type="checkbox"
              checked={value.stopEngine}
              disabled={lock}
              onChange={(e) => onChange({ stopEngine: e.target.checked })}
              className="accent-amber-500"
            />
            {t('trainingStudio.train.dit.stopEngine')}
          </label>
          <span className="text-[11px] text-zinc-500 pl-6">{t('trainingStudio.preprocess.stopEngineHelp')}</span>
        </div>
      </details>

      <button
        onClick={onStart}
        disabled={
          lock || !adapterNameOk || !cropRangeOk || !tWindowOk || !cropOk || !lrOk || !seedOk ||
          !lokrFactorOk || value.stages.length === 0
        }
        className="self-start flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        {starting ? <Loader2 size={15} className="animate-spin" /> : <Waves size={15} />}
        {t('trainingStudio.train.dit.start')}
      </button>
    </div>
  );
};

export default TrainDitForm;
