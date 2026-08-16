// BackendGenerationDropdown.tsx — Generation cluster for backends without the
// ACE plugin system.
//
// ACE's GenerationDropdown is built around the Lua plugin registry (solvers,
// schedulers, guidance) plus a pile of ACE-specific sampler knobs, none of
// which exist for another backend. But "Generation" is not only plugins: the
// SEED lives here, and seed is backend-agnostic — hiding the whole cluster left
// MiniMax-Music3 with no way to change it, so every render of a given prompt
// came back identical (the persisted hs-randomSeed / hs-seed pair was being
// used with no control to reach it).
//
// This renders the parts that are genuinely backend-agnostic:
//   - the shared SeedControl (same component ACE uses)
//   - whatever knobs the backend DECLARES via capabilities().extensions,
//     rendered generically from the schema
//
// Nothing here is MM3-specific; a future backend that declares extensions gets
// them rendered without touching this file.

import React from 'react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { useCapabilities } from '../../hooks/useCapabilities';
import { SeedControl } from './SeedControl';
import { SamplerPluginControls } from './SamplerPluginControls';
import { Slider } from '../shared/Slider';

const inputClasses =
  'w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 ' +
  'text-sm text-zinc-800 dark:text-zinc-200 outline-none focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20';

export const BackendGenerationDropdown: React.FC = () => {
  const gp = useGlobalParams() as any;
  const { capabilities } = useCapabilities();

  const extensions = capabilities?.extensions ?? [];
  const supportsSeed = capabilities?.core?.seed !== false;

  return (
    <div className="space-y-3">
      {supportsSeed && (
        <SeedControl
          inputClasses={inputClasses}
          hint="Drives the whole render — the token plan and the flow sampling. Turn Random off to reproduce a take exactly."
        />
      )}

      {extensions.map((p) => {
        const value = gp.backendParams?.[p.key] ?? p.default;
        if (p.type === 'slider') {
          return (
            <div key={p.key}>
              <Slider
                label={p.label}
                value={typeof value === 'number' ? value : Number(p.default ?? 0)}
                onChange={(v: number) => gp.setBackendParam?.(p.key,v)}
                min={p.min ?? 0}
                max={p.max ?? 1}
                step={p.step ?? 0.1}
              />
              {p.hint && <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed">{p.hint}</p>}
            </div>
          );
        }
        if (p.type === 'toggle') {
          return (
            <label key={p.key} className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider">{p.label}</span>
              <input
                type="checkbox"
                checked={!!value}
                onChange={(e) => gp.setBackendParam?.(p.key,e.target.checked)}
              />
            </label>
          );
        }
        if (p.type === 'select') {
          return (
            <div key={p.key}>
              <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
                {p.label}
              </label>
              <select
                className={inputClasses}
                value={String(value ?? '')}
                onChange={(e) => gp.setBackendParam?.(p.key,e.target.value)}
              >
                {(p.options ?? []).map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
              {p.hint && <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed">{p.hint}</p>}
            </div>
          );
        }
        return (
          <div key={p.key}>
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
              {p.label}
            </label>
            <input
              className={inputClasses}
              value={String(value ?? '')}
              onChange={(e) => gp.setBackendParam?.(p.key,e.target.value)}
            />
            {p.hint && <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed">{p.hint}</p>}
          </div>
        );
      })}

      {/* Shared Lua sampler plugins, for a backend that runs them but does not
          render ACE's GenerationDropdown. Two conditions, both necessary: the
          backend must claim the capability, AND the user must have switched on
          the declared `samplerPluginsEnabled` knob above — the picks are shared
          global state, so showing live pickers that are not being sent would be
          a lie in the other direction. */}
      {capabilities?.features?.samplerPlugins && !!gp.backendParams?.samplerPluginsEnabled && (
        <div className="pt-2 border-t border-zinc-300 dark:border-white/10">
          <SamplerPluginControls />
        </div>
      )}

      {extensions.length === 0 && !supportsSeed && (
        <p className="text-[11px] text-zinc-500 leading-relaxed">
          The active backend exposes no generation controls.
        </p>
      )}
    </div>
  );
};

/** Summary badge — seed mode plus any declared knobs. */
export const BackendGenerationBadge: React.FC = () => {
  const gp = useGlobalParams() as any;
  const { capabilities } = useCapabilities();
  const parts: string[] = [gp.randomSeed ? 'Rnd' : `Seed ${gp.seed}`];
  for (const p of capabilities?.extensions ?? []) {
    const v = gp.backendParams?.[p.key] ?? p.default;
    if (v !== undefined && v !== null) parts.push(`${p.label}: ${v}`);
  }
  return (
    <span className="text-[10px] text-zinc-500 font-mono truncate">{parts.join(' · ')}</span>
  );
};
