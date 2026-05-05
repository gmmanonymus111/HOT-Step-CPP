// ModelSelector.tsx — Model selection dropdowns (DiT, LM, VAE)
//
// Adapter controls have been moved to AdaptersAccordion.tsx.

import React, { useEffect, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { modelApi } from '../../services/api';
import type { AceModels } from '../../types';

interface ModelSelectorProps {
  ditModel: string;
  onDitModelChange: (v: string) => void;
  lmModel: string;
  onLmModelChange: (v: string) => void;
  vaeModel: string;
  onVaeModelChange: (v: string) => void;
}

const selectClasses = "w-full px-3 py-2 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors cursor-pointer";

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  ditModel, onDitModelChange, lmModel, onLmModelChange,
  vaeModel, onVaeModelChange,
}) => {
  const [models, setModels] = useState<AceModels | null>(null);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    modelApi.list()
      .then(setModels)
      .catch(() => {});
  }, []);

  const ditModels = models?.models?.dit || [];
  const lmModels = models?.models?.lm || [];
  const vaeModels = models?.models?.vae || [];

  return (
    <div className="space-y-1 pt-3 border-t border-zinc-200 dark:border-white/5">
      {/* Accordion header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3 py-2.5 rounded-xl hover:bg-white/5 transition-colors"
      >
        <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400 uppercase tracking-wider">Models</span>
        <ChevronDown size={14} className={`text-zinc-500 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="px-3 pb-3 space-y-3">
          {/* DiT Model */}
          <div>
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">DiT Model</label>
            <select className={selectClasses} value={ditModel}
              onChange={e => onDitModelChange(e.target.value)}>
              {ditModels.length === 0 && <option value="">Loading...</option>}
              {ditModels.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          {/* LM Model */}
          <div>
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">LM Model</label>
            <select className={selectClasses} value={lmModel}
              onChange={e => onLmModelChange(e.target.value)}>
              {lmModels.length === 0 && <option value="">Loading...</option>}
              {lmModels.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          {/* VAE Model — only show when multiple VAEs are available */}
          {vaeModels.length > 1 && (
            <div>
              <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">VAE Decoder</label>
              <select className={selectClasses} value={vaeModel}
                onChange={e => onVaeModelChange(e.target.value)}>
                {vaeModels.map(m => (
                  <option key={m} value={m}>{m.replace(/-BF16\.gguf$/, '')}</option>
                ))}
              </select>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
