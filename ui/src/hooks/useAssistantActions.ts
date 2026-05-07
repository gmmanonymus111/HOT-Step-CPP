// useAssistantActions.ts — Hook for applying LLM-suggested settings changes
//
// Maps action field names to GlobalParamsContext setters, provides
// preview diffs and one-click apply functionality.

import { useGlobalParams } from '../context/GlobalParamsContext';
import type { AssistantAction } from '../services/assistantApi';

export interface ActionDiff {
  key: string;
  label: string;
  from: any;
  to: any;
}

/** Human-readable labels for parameter field names */
const LABEL_MAP: Record<string, string> = {
  // Models
  ditModel: 'DiT Model',
  lmModel: 'LM Model',
  vaeModel: 'VAE Model',
  // Adapter
  adapter: 'Adapter',
  adapterScale: 'Adapter Scale',
  adapterMode: 'Adapter Mode',
  // Generation
  inferMethod: 'Solver',
  inferenceSteps: 'Steps',
  guidanceScale: 'Guidance Scale',
  shift: 'Shift',
  scheduler: 'Scheduler',
  guidanceMode: 'Guidance Mode',
  seed: 'Seed',
  randomSeed: 'Random Seed',
  batchSize: 'Batch Size',
  // Solver sub-params
  storkSubsteps: 'STORK Substeps',
  beatStability: 'Beat Stability',
  frequencyDamping: 'Frequency Damping',
  temporalSmoothing: 'Temporal Smoothing',
  // Guidance sub-params
  apgMomentum: 'APG Momentum',
  apgNormThreshold: 'APG Norm Threshold',
  // LM
  skipLm: 'Skip LM',
  useCotCaption: 'CoT Caption',
  lmTemperature: 'LM Temperature',
  lmCfgScale: 'LM CFG Scale',
  lmTopK: 'LM Top-K',
  lmTopP: 'LM Top-P',
  lmNegativePrompt: 'LM Negative Prompt',
  // Post-processing
  postProcessingEnabled: 'Post-Processing',
  spectralLifterEnabled: 'Spectral Lifter',
  slDenoiseStrength: 'SL Denoise',
  slNoiseFloor: 'SL Noise Floor',
  slHfMix: 'SL HF Mix',
  slTransientBoost: 'SL Transient Boost',
  slShimmerReduction: 'SL Shimmer Reduction',
  masteringEnabled: 'Mastering',
  masteringReference: 'Mastering Reference',
  // Denoiser
  denoiseStrength: 'Denoise Strength',
  denoiseSmoothing: 'Denoise Smoothing',
  denoiseMix: 'Denoise Mix',
  // PP-VAE
  ppVaeReencode: 'PP-VAE Re-encode',
  ppVaeBlend: 'PP-VAE Blend',
  // DCW
  dcwEnabled: 'DCW Enabled',
  dcwMode: 'DCW Mode',
  dcwScaler: 'DCW Scaler',
  dcwHighScaler: 'DCW High Scaler',
  // Latent
  latentShift: 'Latent Shift',
  latentRescale: 'Latent Rescale',
  customTimesteps: 'Custom Timesteps',
  // Duration
  autoTrimEnabled: 'Auto-Trim',
  durationBuffer: 'Duration Buffer',
  autoTrimFadeMs: 'Trim Fade',
};

export function useAssistantActions() {
  const gp = useGlobalParams();

  // Map field names → setter functions
  const setterMap: Record<string, (v: any) => void> = {
    // Models
    ditModel: gp.setDitModel,
    lmModel: gp.setLmModel,
    vaeModel: gp.setVaeModel,
    // Adapter
    adapter: gp.setAdapter,
    adapterScale: gp.setAdapterScale,
    adapterMode: gp.setAdapterMode,
    adapterGroupScales: gp.setAdapterGroupScales,
    // Generation
    inferMethod: gp.setInferMethod,
    inferenceSteps: gp.setInferenceSteps,
    guidanceScale: gp.setGuidanceScale,
    shift: gp.setShift,
    scheduler: gp.setScheduler,
    guidanceMode: gp.setGuidanceMode,
    seed: gp.setSeed,
    randomSeed: gp.setRandomSeed,
    batchSize: gp.setBatchSize,
    // Solver sub-params
    storkSubsteps: gp.setStorkSubsteps,
    beatStability: gp.setBeatStability,
    frequencyDamping: gp.setFrequencyDamping,
    temporalSmoothing: gp.setTemporalSmoothing,
    // Guidance sub-params
    apgMomentum: gp.setApgMomentum,
    apgNormThreshold: gp.setApgNormThreshold,
    // LM
    skipLm: gp.setSkipLm,
    useCotCaption: gp.setUseCotCaption,
    lmTemperature: gp.setLmTemperature,
    lmCfgScale: gp.setLmCfgScale,
    lmTopK: gp.setLmTopK,
    lmTopP: gp.setLmTopP,
    lmNegativePrompt: gp.setLmNegativePrompt,
    // Post-processing
    postProcessingEnabled: gp.setPostProcessingEnabled,
    spectralLifterEnabled: gp.setSpectralLifterEnabled,
    slDenoiseStrength: gp.setSlDenoiseStrength,
    slNoiseFloor: gp.setSlNoiseFloor,
    slHfMix: gp.setSlHfMix,
    slTransientBoost: gp.setSlTransientBoost,
    slShimmerReduction: gp.setSlShimmerReduction,
    masteringEnabled: gp.setMasteringEnabled,
    masteringReference: gp.setMasteringReference,
    // Denoiser
    denoiseStrength: gp.setDenoiseStrength,
    denoiseSmoothing: gp.setDenoiseSmoothing,
    denoiseMix: gp.setDenoiseMix,
    // PP-VAE
    ppVaeReencode: gp.setPpVaeReencode,
    ppVaeBlend: gp.setPpVaeBlend,
    // DCW
    dcwEnabled: gp.setDcwEnabled,
    dcwMode: gp.setDcwMode,
    dcwScaler: gp.setDcwScaler,
    dcwHighScaler: gp.setDcwHighScaler,
    // Latent
    latentShift: gp.setLatentShift,
    latentRescale: gp.setLatentRescale,
    customTimesteps: gp.setCustomTimesteps,
    // Duration
    autoTrimEnabled: gp.setAutoTrimEnabled,
    durationBuffer: gp.setDurationBuffer,
    autoTrimFadeMs: gp.setAutoTrimFadeMs,
  };

  /** Apply a list of actions to the global params */
  function applyActions(actions: AssistantAction[]): number {
    let applied = 0;
    for (const action of actions) {
      const setter = setterMap[action.set];
      if (setter) {
        setter(action.value);
        applied++;
      } else {
        console.warn(`[Assistant] Unknown action field: ${action.set}`);
      }
    }
    return applied;
  }

  /** Preview actions as a diff array (before applying) */
  function previewActions(actions: AssistantAction[]): ActionDiff[] {
    const current = gp.getGlobalParams();
    return actions
      .filter(a => a.set in setterMap)
      .map(a => ({
        key: a.set,
        label: LABEL_MAP[a.set] || a.set,
        from: (current as any)[a.set],
        to: a.value,
      }));
  }

  return { applyActions, previewActions };
}
