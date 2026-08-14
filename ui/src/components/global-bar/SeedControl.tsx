// SeedControl.tsx — the shared "Generation Seed" block
//
// Extracted from GenerationDropdown so the ACE cluster and the generic
// backend cluster render the SAME control rather than two that drift apart.
// It writes to the same globalParams fields (seed / randomSeed), so a seed set
// under one backend is the seed used by the other — which is what "the seed"
// means to a user.

import React, { useEffect, useState } from 'react';
import { Save } from 'lucide-react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { ToggleSwitch } from './BarSection';
import { SeedManagerDrawer } from './SeedManagerDrawer';

/** Seed input with local string buffer — prevents parseInt("-") snap-back. */
const SeedInput: React.FC<{ value: number; onChange: (v: number) => void; className: string }> = ({
  value, onChange, className,
}) => {
  const [local, setLocal] = useState(String(value));
  useEffect(() => { setLocal(String(value)); }, [value]);
  const commit = () => { onChange(parseInt(local) || 42); };
  return (
    <input type="number" className={className} value={local}
      onChange={e => setLocal(e.target.value)}
      onBlur={commit}
      onKeyDown={e => { if (e.key === 'Enter') commit(); }}
    />
  );
};

export const SeedControl: React.FC<{ inputClasses: string; hint?: string }> = ({ inputClasses, hint }) => {
  const gp = useGlobalParams();
  const [seedDrawerOpen, setSeedDrawerOpen] = useState(false);

  return (
    <div className="relative">
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-1.5">
          <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Generation Seed</label>
          <button onClick={() => setSeedDrawerOpen(true)} title="Seed Manager"
            className="text-zinc-500 hover:text-amber-400 transition-colors">
            <Save size={12} />
          </button>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-zinc-500">Random</span>
          <ToggleSwitch checked={gp.randomSeed} onChange={gp.setRandomSeed} accentColor="sky" />
        </div>
      </div>
      {!gp.randomSeed && (
        <SeedInput value={gp.seed} onChange={gp.setSeed} className={inputClasses} />
      )}
      <p className="text-[10px] text-zinc-500 mt-1">
        {hint ?? 'Drives audio synthesis (DiT). Varies per track during batch generation. See LM Seed for caption/lyrics/code sampling.'}
      </p>
      <SeedManagerDrawer
        isOpen={seedDrawerOpen}
        onClose={() => setSeedDrawerOpen(false)}
        currentSeed={gp.seed}
        onLoad={(seed) => { gp.setSeed(seed); gp.setRandomSeed(false); setSeedDrawerOpen(false); }}
        onLoadRandom={(seed) => { gp.setSeed(seed); gp.setRandomSeed(false); }}
      />
    </div>
  );
};
