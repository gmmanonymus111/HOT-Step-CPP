// vocalNaturalizer.ts — Vocal Naturalizer DSP engine
//
// Ports the 5-stage vocal naturalisation pipeline from jeankassio/ComfyUI_MusicTools
// (src/vocal_enhance.py → apply_vocal_naturalizer) to TypeScript.
//
// Attribution: Original algorithm by Jean Kassio (MIT License)
// https://github.com/jeankassio/ComfyUI_MusicTools
//
// All DSP is pure math — no native dependencies. Butterworth IIR filters are
// computed from coefficient formulas, applied via direct-form II transposed.
//
// Architecture note: DSP is applied directly to the full mix. The 5 stages
// are all frequency-band-targeted and primarily affect vocal content without
// needing stem separation. This avoids the destructive separate/remix cycle
// that degrades signal quality and raises the noise floor.

import fs from 'fs';

// ── Types ──────────────────────────────────────────────────────────────────

export interface NaturalizerParams {
  amount: number;           // 0.0–1.0 master intensity
  vibratoRate: number;      // 3.0–7.0 Hz
  vibratoDepth: number;     // 0.0–1.0 (relative to master)
  formantStrength: number;  // 0.0–1.0
  metallicReduction: number; // 0.0–1.0
  quantizationMask: number; // 0.0–1.0 — CAUTION: injects shaped noise, off by default
  transitionSmooth: number; // 0.0–1.0
}

export const DEFAULT_NATURALIZER: NaturalizerParams = {
  amount: 0.5,
  vibratoRate: 4.5,
  vibratoDepth: 1.0,
  formantStrength: 1.0,
  metallicReduction: 1.0,
  quantizationMask: 0.0,   // Off by default — injects 1–4kHz noise (audible hiss)
  transitionSmooth: 1.0,
};

type LogFn = (level: 'INFO' | 'DEBUG' | 'WARNING' | 'ERROR', msg: string) => void;
type StageFn = (stage: string) => void;

// ── WAV Parsing ────────────────────────────────────────────────────────────

interface WavData {
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
  /** Interleaved float samples — always normalised to [-1, 1] */
  samples: Float32Array;
}

/** Parse a WAV buffer into float samples. Supports 16-bit, 24-bit, and 32-bit float PCM. */
function parseWav(buf: Buffer): WavData {
  // RIFF header
  if (buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('Not a valid WAV file');
  }

  let offset = 12;
  let fmt: { audioFormat: number; channels: number; sampleRate: number; bitsPerSample: number } | null = null;
  let dataStart = 0;
  let dataSize = 0;

  while (offset < buf.length - 8) {
    const chunkId = buf.toString('ascii', offset, offset + 4);
    const chunkSize = buf.readUInt32LE(offset + 4);
    if (chunkId === 'fmt ') {
      fmt = {
        audioFormat: buf.readUInt16LE(offset + 8),
        channels: buf.readUInt16LE(offset + 10),
        sampleRate: buf.readUInt32LE(offset + 12),
        bitsPerSample: buf.readUInt16LE(offset + 22),
      };
    } else if (chunkId === 'data') {
      dataStart = offset + 8;
      dataSize = chunkSize;
      break;
    }
    offset += 8 + chunkSize + (chunkSize % 2); // pad to even
  }

  if (!fmt || dataStart === 0) throw new Error('Malformed WAV: missing fmt/data chunks');

  const { channels, sampleRate, bitsPerSample, audioFormat } = fmt;
  const bytesPerSample = bitsPerSample / 8;
  const numSamples = Math.floor(dataSize / bytesPerSample);
  const samples = new Float32Array(numSamples);

  if (audioFormat === 3 && bitsPerSample === 32) {
    // IEEE float
    for (let i = 0; i < numSamples; i++) {
      samples[i] = buf.readFloatLE(dataStart + i * 4);
    }
  } else if (audioFormat === 1 && bitsPerSample === 16) {
    for (let i = 0; i < numSamples; i++) {
      samples[i] = buf.readInt16LE(dataStart + i * 2) / 32768;
    }
  } else if (audioFormat === 1 && bitsPerSample === 24) {
    for (let i = 0; i < numSamples; i++) {
      const off = dataStart + i * 3;
      const val = buf[off] | (buf[off + 1] << 8) | (buf[off + 2] << 16);
      samples[i] = ((val & 0x800000) ? val - 0x1000000 : val) / 8388608;
    }
  } else {
    throw new Error(`Unsupported WAV format: audioFormat=${audioFormat}, bits=${bitsPerSample}`);
  }

  return { sampleRate, channels, bitsPerSample, samples };
}

/** Write float samples back to a WAV buffer (32-bit float PCM). */
function writeWav(data: WavData): Buffer {
  const { sampleRate, channels, samples } = data;
  const bitsPerSample = 32;
  const bytesPerSample = 4;
  const dataSize = samples.length * bytesPerSample;
  const buf = Buffer.alloc(44 + dataSize);

  // RIFF header
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + dataSize, 4);
  buf.write('WAVE', 8);

  // fmt chunk (IEEE float)
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(3, 20);              // audioFormat = IEEE float
  buf.writeUInt16LE(channels, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * channels * bytesPerSample, 28); // byteRate
  buf.writeUInt16LE(channels * bytesPerSample, 32);              // blockAlign
  buf.writeUInt16LE(bitsPerSample, 34);

  // data chunk
  buf.write('data', 36);
  buf.writeUInt32LE(dataSize, 40);
  for (let i = 0; i < samples.length; i++) {
    buf.writeFloatLE(samples[i], 44 + i * 4);
  }

  return buf;
}

// ── IIR Filter Design ──────────────────────────────────────────────────────
// Butterworth filter coefficient computation — replaces scipy.signal.butter

interface SosSection { b0: number; b1: number; b2: number; a0: number; a1: number; a2: number; }

/** Design a 2nd-order Butterworth lowpass filter (single SOS section). */
function butterLowpass2(cutoff: number, fs: number): SosSection {
  const wc = Math.tan(Math.PI * cutoff / fs);
  const wc2 = wc * wc;
  const sqrt2 = Math.SQRT2;
  const norm = 1 / (1 + sqrt2 * wc + wc2);
  return {
    b0: wc2 * norm, b1: 2 * wc2 * norm, b2: wc2 * norm,
    a0: 1, a1: 2 * (wc2 - 1) * norm, a2: (1 - sqrt2 * wc + wc2) * norm,
  };
}

/** Design a 2nd-order Butterworth bandpass filter (single SOS section). */
function butterBandpass2(low: number, high: number, fs: number): SosSection {
  const wl = Math.tan(Math.PI * low / fs);
  const wh = Math.tan(Math.PI * high / fs);
  const bw = wh - wl;
  const w0 = Math.sqrt(wl * wh);
  const w02 = w0 * w0;
  const Q = w0 / bw;
  const alpha = Math.sin(2 * Math.atan(w0)) / (2 * Q);
  // Bilinear-transformed bandpass
  const cosW0 = (1 - w02) / (1 + w02);
  const norm = 1 / (1 + alpha);
  return {
    b0: alpha * norm, b1: 0, b2: -alpha * norm,
    a0: 1, a1: -2 * cosW0 * norm, a2: (1 - alpha) * norm,
  };
}

/** Apply a single SOS section to an array (direct-form II transposed). */
function applySos(sos: SosSection, input: Float32Array): Float32Array {
  const out = new Float32Array(input.length);
  let z1 = 0, z2 = 0;
  for (let i = 0; i < input.length; i++) {
    const x = input[i];
    const y = sos.b0 * x + z1;
    z1 = sos.b1 * x - sos.a1 * y + z2;
    z2 = sos.b2 * x - sos.a2 * y;
    out[i] = y;
  }
  return out;
}

/** Cascade multiple SOS sections for higher-order filters. */
function applySosCascade(sections: SosSection[], input: Float32Array): Float32Array {
  let signal = input;
  for (const sos of sections) {
    signal = applySos(sos, signal);
  }
  return signal;
}

// ── Seeded PRNG ────────────────────────────────────────────────────────────
// Simple xoshiro128** for deterministic noise generation

function xoshiro128ss(seed: number) {
  let s0 = seed | 0 || 1;
  let s1 = (seed * 1664525 + 1013904223) | 0;
  let s2 = (s1 * 1664525 + 1013904223) | 0;
  let s3 = (s2 * 1664525 + 1013904223) | 0;
  return (): number => {
    const t = s1 << 9;
    let r = (s1 * 5) | 0;
    r = ((r << 7) | (r >>> 25)) * 9;
    s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3;
    s2 ^= t; s3 = (s3 << 11) | (s3 >>> 21);
    return (r >>> 0) / 4294967296; // [0, 1)
  };
}

/** Generate Gaussian noise using Box-Muller transform with seeded PRNG. */
function gaussianNoise(length: number, seed: number): Float32Array {
  const rng = xoshiro128ss(seed);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i += 2) {
    const u1 = rng() || 1e-10;
    const u2 = rng();
    const mag = Math.sqrt(-2 * Math.log(u1));
    out[i] = mag * Math.cos(2 * Math.PI * u2);
    if (i + 1 < length) out[i + 1] = mag * Math.sin(2 * Math.PI * u2);
  }
  return out;
}

// ── 5-Stage Naturalizer Pipeline ───────────────────────────────────────────

/** Apply the 5-stage vocal naturalisation to a mono signal. */
function naturalizeChannel(
  audio: Float32Array,
  sampleRate: number,
  params: NaturalizerParams,
  seed: number,
): Float32Array {
  const { amount } = params;
  if (amount < 0.01) return audio;

  const result = new Float32Array(audio);
  const N = audio.length;

  // Stage 1: Pitch Variation (vibrato-like AM modulation)
  const vDepth = params.vibratoDepth;
  if (vDepth > 0.01) {
    const vibRate = params.vibratoRate;
    const depth = 0.002 * amount * vDepth;
    for (let i = 0; i < N; i++) {
      const t = i / sampleRate;
      const pitchVar = Math.sin(2 * Math.PI * vibRate * t) * depth;
      // Phase modulation approximation via AM
      const phaseMod = pitchVar * 2 * Math.PI;
      const modulated = audio[i] * (1 + Math.sin(phaseMod) * 0.01 * amount * vDepth);
      result[i] = result[i] * 0.7 + modulated * 0.3;
    }
  }

  // Stage 2: Formant Variation
  const fStr = params.formantStrength;
  if (fStr > 0.01) {
    const noise = gaussianNoise(N, seed);
    const formantVariation = new Float32Array(N);
    for (let i = 0; i < N; i++) formantVariation[i] = noise[i] * 0.005 * amount * fStr;

    const formantSos = butterBandpass2(200, 3000, sampleRate);
    const formantSignal = applySos(formantSos, audio);
    for (let i = 0; i < N; i++) {
      result[i] += formantSignal[i] * (1 + formantVariation[i]) * 0.15 * amount * fStr
                 - formantSignal[i] * 0.15 * amount * fStr; // net: add modulated delta only
    }
  }

  // Stage 3: Metallic Artifact Removal (6–10 kHz)
  const mRed = params.metallicReduction;
  if (mRed > 0.01 && sampleRate > 12000) {
    const metallicSos = butterBandpass2(6000, Math.min(10000, sampleRate * 0.45), sampleRate);
    const metallic = applySos(metallicSos, audio);
    for (let i = 0; i < N; i++) {
      result[i] -= metallic[i] * 0.3 * amount * mRed;
    }
  }

  // Stage 4: Quantization Masking (shaped noise 1–4 kHz)
  // WARNING: This stage injects noise. Off by default (quantizationMask = 0).
  // Only enable if you specifically want dither-like masking of quantization
  // artifacts and accept the trade-off of a slightly raised noise floor.
  const qMask = params.quantizationMask;
  if (qMask > 0.01) {
    const rawNoise = gaussianNoise(N, seed + 42);
    for (let i = 0; i < N; i++) rawNoise[i] *= 0.002 * amount * qMask;
    const noiseSos = butterBandpass2(1000, 4000, sampleRate);
    const shapedNoise = applySos(noiseSos, rawNoise);
    for (let i = 0; i < N; i++) result[i] += shapedNoise[i];
  }

  // Stage 5: Transition Smoothing (low-pass filtered differential)
  const tSmooth = params.transitionSmooth;
  if (tSmooth > 0.01) {
    // Compute differential
    const diff = new Float32Array(N);
    diff[0] = 0;
    for (let i = 1; i < N; i++) diff[i] = result[i] - result[i - 1];

    // Low-pass at 80 Hz (slightly less aggressive than original's 50 Hz)
    const smoothSos = butterLowpass2(80, sampleRate);
    const smoothedDiff = applySos(smoothSos, diff);
    const blend = 0.4 * amount * tSmooth;
    for (let i = 0; i < N; i++) {
      result[i] = result[i] - diff[i] * blend + smoothedDiff[i] * blend;
    }
  }

  // Normalise to prevent clipping
  let maxVal = 0;
  for (let i = 0; i < N; i++) {
    const abs = Math.abs(result[i]);
    if (abs > maxVal) maxVal = abs;
  }
  if (maxVal > 0.95) {
    const scale = 0.95 / maxVal;
    for (let i = 0; i < N; i++) result[i] *= scale;
  }

  return result;
}

/** Apply naturaliser to all channels of a WAV. */
function naturalizeWav(wav: WavData, params: NaturalizerParams, seed: number): WavData {
  const { channels, sampleRate, samples } = wav;
  const framesPerChannel = Math.floor(samples.length / channels);
  const result = new Float32Array(samples.length);

  for (let ch = 0; ch < channels; ch++) {
    // De-interleave
    const mono = new Float32Array(framesPerChannel);
    for (let i = 0; i < framesPerChannel; i++) mono[i] = samples[i * channels + ch];

    // Process
    const processed = naturalizeChannel(mono, sampleRate, params, seed + ch);

    // Re-interleave
    for (let i = 0; i < framesPerChannel; i++) result[i * channels + ch] = processed[i];
  }

  return { ...wav, samples: result, bitsPerSample: 32 };
}

// ── Public API ─────────────────────────────────────────────────────────────

/**
 * Run the vocal naturaliser pipeline directly on the full mix.
 *
 * DSP stages are frequency-band-targeted and primarily affect vocal content
 * without needing stem separation. This preserves signal integrity, avoids
 * phase smearing, and keeps the full dynamic range intact for downstream
 * processing (VST chains, mastering).
 */
export async function runVocalNaturalizer(
  processedPath: string,
  params: NaturalizerParams,
  log: LogFn,
  setStage: StageFn,
  trackIndex?: number,
  totalTracks?: number,
): Promise<boolean> {
  const suffix = (totalTracks && totalTracks > 1 && trackIndex !== undefined)
    ? ` (${trackIndex + 1}/${totalTracks})`
    : '';

  setStage(`Vocal Naturalizer: processing${suffix}...`);

  try {
    // Read the WAV directly — no separation step
    const wavBuf = fs.readFileSync(processedPath);
    const wav = parseWav(wavBuf);

    const seed = Date.now();
    const processed = naturalizeWav(wav, params, seed);

    // Write back — same sample rate, same channels, no resampling needed
    fs.writeFileSync(processedPath, writeWav(processed));

    log('INFO', `[Vocal Naturalizer] Applied to full mix (${wav.sampleRate}Hz, ${wav.channels}ch)`);
    return true;
  } catch (err: any) {
    log('WARNING', `[Vocal Naturalizer] Failed (non-fatal): ${err.message}`);
    return false;
  }
}
