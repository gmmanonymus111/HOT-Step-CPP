// BackendModelsDropdown.tsx — model picker for backends whose model choice is
// ENGINE STATE rather than a per-request parameter.
//
// ACE passes model file names on every generate call, so its picker
// (ModelsDropdown.tsx) writes to globalParams and nothing is posted until you
// hit generate. MiniMax-Music3 is the opposite: the engine loads one quant of
// each role and holds it resident, so choosing a model is a POST that evicts
// the old weights. That difference — not the backend's name — is what this
// component keys off: it renders whenever the catalogue reports
// `selectable: true`, and renders whatever buckets it is handed.
//
// Buckets are backend-shaped by design (plan §2 principle 2): MM3 reports
// { lm, synth }, a future backend may report something else entirely. Nothing
// here hardcodes a bucket name beyond a label lookup with a sane fallback.

import React, { useEffect, useState } from 'react';
import { Download } from 'lucide-react';
import { useBackendStore } from '../../stores/backendStore';
import { ModelManagerModal } from '../model-manager/ModelManagerModal';
import { ModelSelect } from './ModelSelect';

/** Friendly names for the buckets we know about. Unknown buckets fall back to
 *  their raw key rather than being hidden — a backend that grows a bucket
 *  should show up in the UI without a matching edit here. */
const BUCKET_LABELS: Record<string, string> = {
  lm: 'Language Model',
  synth: 'Synth / Flow Model',
  dit: 'Flow DiT',
  depth: 'Depth Decoder',
  cond: 'Condition Encoder',
  voc: 'Vocoder',
  vae: 'VAE Decoder',
  embedding: 'Text Encoder',
};

function formatBytes(bytes?: number): string {
  if (!bytes || bytes <= 0) return '';
  return `${(bytes / 1e9).toFixed(2)} GB`;
}

export const BackendModelsDropdown: React.FC = () => {
  const activeBackendId = useBackendStore(s => s.activeBackendId);
  const catalogue = useBackendStore(s => s.models[s.activeBackendId] ?? null);
  const fetchModels = useBackendStore(s => s.fetchModels);
  const selectModels = useBackendStore(s => s.selectModels);

  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [showModelManager, setShowModelManager] = useState(false);

  useEffect(() => {
    if (!activeBackendId) return;
    void fetchModels(activeBackendId);
  }, [activeBackendId, fetchModels]);

  const buckets = Object.entries(catalogue?.buckets ?? {}).filter(([, opts]) => opts.length > 0);

  const onPick = async (bucket: string, value: string) => {
    if (!catalogue) return;
    setBusy(bucket);
    setNote(null);
    // Post the WHOLE selection, not just the changed bucket: the engine's
    // select-model contract is idempotent and takes both roles, so sending
    // the full picture avoids a partial state if two changes race.
    const selection: Record<string, string> = {};
    for (const [key] of Object.entries(catalogue.buckets)) {
      selection[key] = String(catalogue.defaults?.[key] ?? '');
    }
    selection[bucket] = value;

    const ok = await selectModels(selection, activeBackendId);
    setBusy(null);
    setNote(ok
      ? 'Switched — the next generation reloads weights (adds a warm-up).'
      : 'Could not switch model; the previous one is still loaded.');
  };

  if (!catalogue) {
    return <p className="text-[11px] text-zinc-500">Loading models…</p>;
  }

  if (buckets.length === 0) {
    return (
      <div className="space-y-3">
        <p className="text-[11px] text-zinc-500 leading-relaxed">
          No model files found for this backend. Download them from the Model Manager.
        </p>
        <button
          onClick={() => setShowModelManager(true)}
          className="w-full px-3 py-2 rounded-xl bg-pink-500/10 border border-pink-500/20
                     text-sm text-pink-400 hover:bg-pink-500/20 hover:text-pink-300
                     transition-colors flex items-center justify-center gap-2"
        >
          <Download size={14} />
          Get Models
        </button>
        {showModelManager && <ModelManagerModal onClose={() => setShowModelManager(false)} />}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {buckets.map(([bucket, options]) => {
        const current = String(catalogue.defaults?.[bucket] ?? '');
        const meta = catalogue.meta?.[bucket] ?? {};
        return (
          <div key={bucket}>
            <label
              htmlFor={`backend-model-${bucket}`}
              className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5"
            >
              {BUCKET_LABELS[bucket] ?? bucket}
            </label>
            <ModelSelect
              id={`backend-model-${bucket}`}
              value={current}
              onChange={(v) => void onPick(bucket, v)}
              options={options}
              // Every MM3 weight file is a GGUF; the default sniffer reads the
              // option string as a file name and these are quant tokens.
              formatOf={() => 'gguf'}
              // Short, fixed list — a filter box would be furniture.
              filterable={false}
              disabled={busy !== null || !catalogue.selectable}
              formatLabel={(opt) =>
                `${opt}${meta[opt]?.bytes ? ` — ${formatBytes(meta[opt].bytes)}` : ''}`}
              placeholder={busy === bucket ? 'Switching…' : 'Select model…'}
            />
            {meta[current]?.label && (
              <p className="text-[10px] text-zinc-500 mt-1.5 font-mono truncate">{meta[current].label}</p>
            )}
          </div>
        );
      })}

      {note && <p className="text-[10px] text-zinc-500 leading-relaxed">{note}</p>}

      <div className="border-t border-zinc-200 dark:border-white/5 pt-3 mt-1">
        <button
          onClick={() => setShowModelManager(true)}
          className="w-full px-3 py-2 rounded-xl bg-pink-500/10 border border-pink-500/20
                     text-sm text-pink-400 hover:bg-pink-500/20 hover:text-pink-300
                     transition-colors flex items-center justify-center gap-2"
        >
          <Download size={14} />
          Get More Models
        </button>
      </div>

      {showModelManager && <ModelManagerModal onClose={() => setShowModelManager(false)} />}
    </div>
  );
};

/** Summary badge — the quant of each bucket currently in force. */
export const BackendModelsBadge: React.FC = () => {
  const catalogue = useBackendStore(s => s.models[s.activeBackendId] ?? null);
  const parts = Object.keys(catalogue?.buckets ?? {})
    .map(k => String(catalogue?.defaults?.[k] ?? ''))
    .filter(Boolean);
  return (
    <span className="text-[10px] text-zinc-500 font-mono truncate">
      {parts.length ? parts.join(' · ') : '—'}
    </span>
  );
};
