// Sa3SamplerControls.tsx — solver / scheduler / guidance pickers for the
// StableStep (SA3) refine.
//
// Same registry, same PluginControls renderer and same shared pluginParams map
// as the generation-side pickers — but bound to its OWN store fields
// (stableStepSolver/Scheduler/GuidanceMode). StableStep is a post-processing
// refine, not the generation sampler; sharing gp.inferMethod would mean picking
// a solver here silently changed the DiT's solver too.
//
// Engine side: engine/src/sa3-refine.h (Sa3PluginParams), reached through the
// /sa3-refine query string. Leaving every picker on Native keeps the original
// pingpong path, bit-identical.
//
// DIFFERENCES FROM SamplerPluginControls (the MM3 one) — both are real:
//   • owns_loop solvers ARE offered here. MM3 refuses them because a full-loop
//     solver bypasses its per-step window-overlap blend and breaks the seams;
//     SA3 refines the clip in a single pass, so there is no seam to break.
//   • Guidance is close to decorative. SA3 was trained at cfg=1 and has no
//     unconditional branch, so the engine passes the cond velocity as both
//     predictions. APG-family modes see diff = 0 and pass through unchanged;
//     only plugins that do cond-side work have any effect. The warning below
//     says so rather than letting the dropdown imply more than it delivers.

import React from 'react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { usePluginRegistry } from '../../hooks/usePluginRegistry';
import { PluginControls } from './PluginControls';
import { EditableSlider } from '../shared/EditableSlider';

const selectClasses =
  'w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 ' +
  'text-sm text-zinc-800 dark:text-zinc-200 focus:border-sky-500/50 focus:ring-1 focus:ring-sky-500/20 ' +
  'outline-none transition-colors cursor-pointer';

const NATIVE = '';

export const Sa3SamplerControls: React.FC = () => {
  const gp = useGlobalParams() as any;
  const { registry, findSolver, findScheduler, findGuidance } = usePluginRegistry();

  const solvers = registry.solvers ?? [];
  const schedulers = registry.schedulers ?? [];
  const guidance = registry.guidance ?? [];

  const solverMeta = findSolver(gp.stableStepSolver);
  const schedMeta = findScheduler(gp.stableStepScheduler);
  const guideMeta = findGuidance(gp.stableStepGuidanceMode);

  const singleNfe = solvers.filter((s: any) => (s.nfe ?? 1) === 1);
  const multiNfe = solvers.filter((s: any) => (s.nfe ?? 1) > 1);

  return (
    <div className="space-y-3 pt-1">
      <EditableSlider
        label="Refine steps"
        value={gp.stableStepSteps}
        min={1} max={64} step={1}
        onChange={gp.setStableStepSteps}
        tooltip="Sampler steps for the refine. 8 is the tuned default; more steps cost time roughly linearly."
      />

      <p className="text-[10px] text-zinc-500 leading-relaxed">
        The same solver, scheduler and guidance plugins the generation sampler uses, applied to
        the StableStep refine. Leaving a picker on <em>Native</em> keeps the original tested path.
        These are separate from the Generation dropdown's picks.
      </p>

      {/* Solver */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
          Refine solver
        </label>
        <select
          className={selectClasses}
          value={gp.stableStepSolver ?? NATIVE}
          onChange={e => gp.setStableStepSolver(e.target.value)}
        >
          <option value={NATIVE}>Native (ping-pong)</option>
          {singleNfe.length > 0 && (
            <optgroup label="── Single Evaluation (1 NFE) ──">
              {singleNfe.map((s: any) => (
                <option key={s.name} value={s.name}>{s.display}</option>
              ))}
            </optgroup>
          )}
          {multiNfe.length > 0 && (
            <optgroup label="── Multi Evaluation ──">
              {multiNfe.map((s: any) => (
                <option key={s.name} value={s.name}>{s.display} ({s.nfe} NFE)</option>
              ))}
            </optgroup>
          )}
        </select>
        {solverMeta?.description && (
          <p className="text-[10px] text-zinc-500 mt-1.5 leading-relaxed">{solverMeta.description}</p>
        )}
        {gp.stableStepSolver && (
          <p className="text-[10px] text-amber-500/90 mt-1 leading-relaxed">
            A solver replaces the ping-pong re-noise rather than stacking with it — the refine
            becomes deterministic in that stage.
          </p>
        )}
        {(solverMeta?.nfe ?? 1) > 1 && (
          <p className="text-[10px] text-amber-500/90 mt-1 leading-relaxed">
            Multi-evaluation solvers run extra forward passes per step; the refine will scale
            roughly with NFE.
          </p>
        )}
      </div>

      {solverMeta && solverMeta.params?.length > 0 && (
        <PluginControls
          pluginName={solverMeta.name}
          displayName={solverMeta.display}
          accent={solverMeta.accent}
          params={solverMeta.params}
          values={gp.pluginParams}
          onChange={gp.setPluginParam}
          onReset={() => gp.resetPluginParams(solverMeta.name)}
        />
      )}

      {/* Scheduler */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
          Refine schedule
        </label>
        <select
          className={selectClasses}
          value={gp.stableStepScheduler ?? NATIVE}
          onChange={e => gp.setStableStepScheduler(e.target.value)}
        >
          <option value={NATIVE}>Native (SA3 LogSNR)</option>
          {schedulers.map((s: any) => (
            <option key={s.name} value={s.name}>{s.display}</option>
          ))}
        </select>
        {schedMeta?.description && (
          <p className="text-[10px] text-zinc-500 mt-1.5 leading-relaxed">{schedMeta.description}</p>
        )}
        {gp.stableStepScheduler && (
          <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed">
            Schedulers are written for full denoising (starting at 1.0). The refine only partly
            re-noises, so the curve is rescaled onto the refine strength — its shape is kept, its
            starting point is not.
          </p>
        )}
      </div>

      {schedMeta && schedMeta.params?.length > 0 && (
        <PluginControls
          pluginName={schedMeta.name}
          displayName={schedMeta.display}
          accent={schedMeta.accent}
          params={schedMeta.params}
          values={gp.pluginParams}
          onChange={gp.setPluginParam}
          onReset={() => gp.resetPluginParams(schedMeta.name)}
        />
      )}

      {/* Guidance */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
          Refine guidance
        </label>
        <select
          className={selectClasses}
          value={gp.stableStepGuidanceMode ?? NATIVE}
          onChange={e => gp.setStableStepGuidanceMode(e.target.value)}
        >
          <option value={NATIVE}>None</option>
          {guidance.map((g: any) => (
            <option key={g.name} value={g.name}>{g.display}</option>
          ))}
        </select>
        {guideMeta?.description && (
          <p className="text-[10px] text-zinc-500 mt-1.5 leading-relaxed">{guideMeta.description}</p>
        )}
        {gp.stableStepGuidanceMode && (
          <p className="text-[10px] text-amber-500/90 mt-1.5 leading-relaxed">
            SA3 has no unconditional branch (it was trained without CFG), so guidance runs with the
            conditional prediction as both inputs. APG-style modes pass straight through unchanged;
            only plugins that enhance the conditional prediction on its own will do anything here.
          </p>
        )}
      </div>

      {gp.stableStepGuidanceMode && (
        <EditableSlider
          label="Refine guidance scale"
          value={gp.stableStepGuidanceScale}
          min={1.0} max={10.0} step={0.1}
          onChange={gp.setStableStepGuidanceScale}
          tooltip="Passed to the guidance plugin. 1.0 disables guidance entirely."
        />
      )}

      {guideMeta && guideMeta.params?.length > 0 && (
        <PluginControls
          pluginName={guideMeta.name}
          displayName={guideMeta.display}
          accent={guideMeta.accent}
          params={guideMeta.params}
          values={gp.pluginParams}
          onChange={gp.setPluginParam}
          onReset={() => gp.resetPluginParams(guideMeta.name)}
        />
      )}
    </div>
  );
};
