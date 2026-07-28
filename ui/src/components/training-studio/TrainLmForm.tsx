// TrainLmForm.tsx — the LM LoRA training controls
//
// Four controls are always visible because they are the only ones a normal run
// needs: which base size, what the adapter is called, when to stop, and a
// backstop epoch cap. Everything else is Side-Step's own defaults and lives
// behind the Advanced drawer, same pattern as PreprocessOptionsForm.

import React from 'react';
import { Cpu, Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type {
  LmSize,
  TrainLmStage,
  TrainingCapabilities,
} from '../../services/trainingApi';

/** Everything the panel needs to POST, with every default already resolved. */
export interface TrainLmFormState {
  lmSize: LmSize;
  lmModel: string;          // '' = let the server pick the BF16 base for lmSize
  adapterName: string;
  targetLoss: number;
  epochs: number;
  rank: number;
  alpha: number;
  learningRate: number;
  gradAccum: number;
  gradClip: number;
  warmupRatio: number;
  weightDecay: number;
  maxLen: number;           // 0 = auto-fit from free VRAM
  seed: number;
  order: 'shuffle' | 'fixed';
  lossOnCot: boolean;
  milestoneStep: number;
  milestoneKeep: number;
  stages: TrainLmStage[];
  overwrite: boolean;
  stopEngine: boolean;
}

export const TRAIN_LM_DEFAULTS: TrainLmFormState = {
  lmSize: '0.6B',
  lmModel: '',
  adapterName: '',
  targetLoss: 0.4,
  epochs: 16,
  rank: 16,
  alpha: 32,
  learningRate: 0.0001,
  gradAccum: 4,
  gradClip: 1.0,
  warmupRatio: 0.05,
  weightDecay: 0.01,
  maxLen: 0,
  seed: 42,
  order: 'shuffle',
  lossOnCot: true,
  milestoneStep: 0.1,
  milestoneKeep: 6,
  stages: ['extract', 'train', 'export'],
  overwrite: false,
  stopEngine: true,
};

const ALL_STAGES: TrainLmStage[] = ['extract', 'train', 'export'];
const LM_SIZES: LmSize[] = ['0.6B', '1.7B'];

const FIELD =
  'rounded-lg px-3 py-2 text-sm bg-zinc-100 dark:bg-black/20 border border-zinc-300 dark:border-white/10 text-zinc-800 dark:text-zinc-200 focus:outline-none focus:border-amber-500 disabled:opacity-50 disabled:cursor-not-allowed';
const LABEL = 'text-xs font-semibold text-zinc-600 dark:text-zinc-400';

interface Props {
  capabilities: TrainingCapabilities | null;
  value: TrainLmFormState;
  onChange: (patch: Partial<TrainLmFormState>) => void;
  /** Locks every control (a job is running, or a start is in flight). */
  disabled?: boolean;
  starting?: boolean;
  onStart: () => void;
}

export const TrainLmForm: React.FC<Props> = ({
  capabilities, value, onChange, disabled, starting, onStart,
}) => {
  const { t } = useTranslation();

  const tl = capabilities?.trainLm;
  const sizes = tl?.sizes?.length ? tl.sizes : LM_SIZES;
  // Only bases whose name carries the selected size token — the same `-<size>-`
  // rule pickLmFor() uses. The raw registry also holds the 4B files, and the
  // server's 4B guard only inspects `body.lmSize`: picking a 4B base at size
  // 0.6B is accepted, the runner then STOPS ace-server, and only then does
  // ace-train infer 4B from the weights and exit 1. §4.5 step 4 is explicit that
  // a 4B request must not reach the engine.
  const sizeToken = new RegExp(`-${value.lmSize.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}-`, 'i');
  const lmModels = (tl?.lmModels ?? []).filter(n => sizeToken.test(n));
  // '' means "no installed base matches this size" — the server would 400.
  const defaultForSize = tl?.defaultLmBySize?.[value.lmSize] ?? '';
  // A cold boot with ace-server unreachable leaves the CACHED model snapshot
  // empty, so defaultLmBySize is '' for every size even though the file is
  // installed. Saying "download one in the Model Manager" then is simply wrong,
  // and the POST would succeed (the route re-probes). Distinguish the two, as
  // PreprocessOptionsForm does for its own model list.
  const modelsUnknown = capabilities?.engine?.up === false && (tl?.lmModels?.length ?? 0) === 0;
  const noLmForSize = !value.lmModel && defaultForSize === '' && !modelsUnknown;
  const lock = !!disabled;

  // Mirrors the server's ADAPTER_NAME_RE. The GET path sanitises the same input
  // instead of rejecting it, so without this the panel can render a green
  // "Adapter ready" card for a directory the POST will refuse outright.
  const adapterNameOk = /^[A-Za-z0-9._-]{1,64}$/.test(value.adapterName.trim())
    && !/^\.+$/.test(value.adapterName.trim())
    && !value.adapterName.trim().startsWith('.');

  const num = (raw: string, fallback: number): number => {
    // '' -> Number('') === 0, which is finite. For every other field 0 is
    // rejected by the server, but targetLoss 0 legitimately means "no auto-stop",
    // so an emptied box would silently disable the panel's headline feature.
    if (raw.trim() === '') return fallback;
    const n = Number(raw);
    return Number.isFinite(n) ? n : fallback;
  };

  const toggleStage = (stage: TrainLmStage, on: boolean) => {
    const next = ALL_STAGES.filter(s => (s === stage ? on : value.stages.includes(s)));
    onChange({ stages: next });
  };

  return (
    <div className="flex flex-col gap-4">
      {/* ── Base size ─────────────────────────────────────────────────── */}
      <div className="flex flex-col gap-1.5">
        <span className={LABEL}>{t('trainingStudio.train.baseSize')}</span>
        <div className="flex items-center gap-1.5">
          {sizes.map(size => {
            const active = value.lmSize === size;
            return (
              <button
                key={size}
                type="button"
                disabled={lock}
                onClick={() => onChange({ lmSize: size, lmModel: '' })}
                className={`px-4 py-1.5 rounded-lg text-xs font-bold border transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                  active
                    ? 'text-amber-500 bg-amber-500/10 border-amber-500/30'
                    : 'text-zinc-500 border-zinc-300 dark:border-white/10 hover:text-zinc-700 dark:hover:text-zinc-300'
                }`}
              >
                {size}
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {/* ── Adapter name ────────────────────────────────────────────── */}
        <label className="flex flex-col gap-1.5">
          <span className={LABEL}>{t('trainingStudio.train.adapterName')}</span>
          <div className="flex items-stretch">
            <input
              type="text"
              value={value.adapterName}
              disabled={lock}
              spellCheck={false}
              onChange={(e) => onChange({ adapterName: e.target.value })}
              className={`${FIELD} flex-1 min-w-0 rounded-r-none border-r-0${
                value.adapterName.trim() && !adapterNameOk ? ' border-red-500/50' : ''}`}
            />
            <span className="flex items-center px-2.5 rounded-r-lg border border-l-0 border-zinc-300 dark:border-white/10 bg-zinc-200/60 dark:bg-white/5 text-[11px] font-mono font-semibold text-zinc-500 whitespace-nowrap">
              -{value.lmSize}
            </span>
          </div>
          {value.adapterName.trim() && !adapterNameOk && (
            <span className="text-[11px] text-red-500 dark:text-red-400">
              {t('trainingStudio.train.adapterNameInvalid')}
            </span>
          )}
        </label>

        {/* ── Target loss ─────────────────────────────────────────────── */}
        <label className="flex flex-col gap-1.5">
          <span className={LABEL}>{t('trainingStudio.train.targetLoss')}</span>
          <input
            type="number"
            min={0}
            max={20}
            step={0.05}
            value={value.targetLoss}
            disabled={lock}
            onChange={(e) => onChange({ targetLoss: num(e.target.value, 0.4) })}
            className={FIELD}
          />
        </label>
      </div>

      <span className="text-[11px] text-zinc-500 -mt-2">{t('trainingStudio.train.targetLossHelp')}</span>

      {/* ── Max epochs ────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <label className="flex flex-col gap-1.5">
          <span className={LABEL}>{t('trainingStudio.train.maxEpochs')}</span>
          <input
            type="number"
            min={1}
            max={200}
            step={1}
            value={value.epochs}
            disabled={lock}
            onChange={(e) => onChange({ epochs: num(e.target.value, 16) })}
            className={FIELD}
          />
          <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.maxEpochsHelp')}</span>
        </label>
      </div>

      {/* ── Advanced ──────────────────────────────────────────────────── */}
      <details className="rounded-lg border border-zinc-200 dark:border-white/5 px-3 py-2">
        <summary className="cursor-pointer text-xs font-semibold text-zinc-600 dark:text-zinc-400 select-none">
          {t('trainingStudio.train.advanced')}
        </summary>

        <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
          <label className="flex flex-col gap-1.5 sm:col-span-2">
            <span className={LABEL}>{t('trainingStudio.train.lmModel')}</span>
            <select
              value={value.lmModel}
              disabled={lock}
              onChange={(e) => onChange({ lmModel: e.target.value })}
              className={FIELD}
            >
              <option value="">{defaultForSize || '—'}</option>
              {lmModels.map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.rank')}</span>
            <input
              type="number" min={1} max={256} step={1}
              value={value.rank} disabled={lock}
              onChange={(e) => onChange({ rank: num(e.target.value, 16) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.alpha')}</span>
            <input
              type="number" min={1} max={1024} step={1}
              value={value.alpha} disabled={lock}
              onChange={(e) => onChange({ alpha: num(e.target.value, 32) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.learningRate')}</span>
            <input
              type="number" min={0} max={1} step={0.00001}
              value={value.learningRate} disabled={lock}
              onChange={(e) => onChange({ learningRate: num(e.target.value, 0.0001) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.gradAccum')}</span>
            <input
              type="number" min={1} max={64} step={1}
              value={value.gradAccum} disabled={lock}
              onChange={(e) => onChange({ gradAccum: num(e.target.value, 4) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.gradClip')}</span>
            <input
              type="number" min={0} max={100} step={0.1}
              value={value.gradClip} disabled={lock}
              onChange={(e) => onChange({ gradClip: num(e.target.value, 1) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.warmupRatio')}</span>
            <input
              type="number" min={0} max={0.5} step={0.01}
              value={value.warmupRatio} disabled={lock}
              onChange={(e) => onChange({ warmupRatio: num(e.target.value, 0.05) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.weightDecay')}</span>
            <input
              type="number" min={0} max={1} step={0.005}
              value={value.weightDecay} disabled={lock}
              onChange={(e) => onChange({ weightDecay: num(e.target.value, 0.01) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.maxLen')}</span>
            <input
              type="number" min={0} max={16384} step={64}
              value={value.maxLen} disabled={lock}
              onChange={(e) => onChange({ maxLen: num(e.target.value, 0) })}
              className={FIELD}
            />
            <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.maxLenAuto')}</span>
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.seed')}</span>
            <input
              type="number" min={0} step={1}
              value={value.seed} disabled={lock}
              onChange={(e) => onChange({ seed: num(e.target.value, 42) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.order')}</span>
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
            <span className={LABEL}>{t('trainingStudio.train.milestoneStep')}</span>
            <input
              type="number" min={0} max={5} step={0.05}
              value={value.milestoneStep} disabled={lock}
              onChange={(e) => onChange({ milestoneStep: num(e.target.value, 0.1) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.train.milestoneKeep')}</span>
            <input
              type="number" min={0} max={64} step={1}
              value={value.milestoneKeep} disabled={lock}
              onChange={(e) => onChange({ milestoneKeep: num(e.target.value, 6) })}
              className={FIELD}
            />
          </label>
        </div>

        <div className="mt-3 flex flex-col gap-2">
          <span className={LABEL}>{t('trainingStudio.train.stages')}</span>
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
              checked={value.lossOnCot}
              disabled={lock}
              onChange={(e) => onChange({ lossOnCot: e.target.checked })}
              className="accent-amber-500"
            />
            {t('trainingStudio.train.lossOnCot')}
          </label>

          <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
            <input
              type="checkbox"
              checked={value.overwrite}
              disabled={lock}
              onChange={(e) => onChange({ overwrite: e.target.checked })}
              className="accent-amber-500"
            />
            {t('trainingStudio.train.reextract')}
          </label>

          <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
            <input
              type="checkbox"
              checked={value.stopEngine}
              disabled={lock}
              onChange={(e) => onChange({ stopEngine: e.target.checked })}
              className="accent-amber-500"
            />
            {t('trainingStudio.train.stopEngine')}
          </label>
          <span className="text-[11px] text-zinc-500 pl-6">{t('trainingStudio.preprocess.stopEngineHelp')}</span>
        </div>
      </details>

      {noLmForSize && (
        <div className="px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
          {t('trainingStudio.train.noLmForSize', { size: value.lmSize })}
        </div>
      )}

      {modelsUnknown && (
        <div className="px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
          {t('trainingStudio.preprocess.engineDownForModels')}
        </div>
      )}

      <button
        onClick={onStart}
        disabled={lock || noLmForSize || !adapterNameOk || value.stages.length === 0}
        className="self-start flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        {starting ? <Loader2 size={15} className="animate-spin" /> : <Cpu size={15} />}
        {t('trainingStudio.train.start')}
      </button>
    </div>
  );
};

export default TrainLmForm;
