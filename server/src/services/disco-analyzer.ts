// disco-analyzer.ts — Server-side WAV analysis for disco beat visualization.
//
// Reads WAV stem files, computes RMS energy per ~16ms window,
// normalises to [0–1], and saves as a compact JSON file.
//
// The browser loads just this JSON (~30-60 KB) instead of three WAV
// files (~15 MB total). Energy lookup is a pure array index — zero
// sync issues with the main player.

import fs from 'fs';
import path from 'path';

// ── Types ────────────────────────────────────────────────────────────────────

export interface DiscoData {
  version: number;          // Schema version
  fps: number;              // Analysis windows per second
  duration: number;         // Total duration in seconds
  kick: number[];           // Energy per window [0–1], 2 decimal places
  snare: number[];
  hihat: number[];
}

// ── WAV Parsing ──────────────────────────────────────────────────────────────
//
// Handles 16-bit PCM, 24-bit PCM, and 32-bit IEEE float.
// Mono or stereo — stereo is mixed to mono.

interface WavData {
  sampleRate: number;
  channels: number;
  samples: Float32Array;    // Mono mix, normalised to [-1, 1]
}

function parseWav(filePath: string): WavData {
  const buf = fs.readFileSync(filePath);

  // RIFF header
  const riff = buf.toString('ascii', 0, 4);
  const wave = buf.toString('ascii', 8, 12);
  if (riff !== 'RIFF' || wave !== 'WAVE') {
    throw new Error(`Not a WAV file: ${filePath}`);
  }

  // Find fmt and data chunks
  let fmtOffset = -1;
  let dataOffset = -1;
  let dataSize = 0;
  let pos = 12;

  while (pos < buf.length - 8) {
    const chunkId = buf.toString('ascii', pos, pos + 4);
    const chunkSize = buf.readUInt32LE(pos + 4);

    if (chunkId === 'fmt ') {
      fmtOffset = pos + 8;
    } else if (chunkId === 'data') {
      dataOffset = pos + 8;
      dataSize = chunkSize;
    }

    pos += 8 + chunkSize;
    // Chunks must be word-aligned
    if (chunkSize % 2 !== 0) pos++;
  }

  if (fmtOffset < 0) throw new Error(`No fmt chunk in: ${filePath}`);
  if (dataOffset < 0) throw new Error(`No data chunk in: ${filePath}`);

  // Parse fmt chunk
  const audioFormat = buf.readUInt16LE(fmtOffset);      // 1=PCM, 3=IEEE float
  const channels = buf.readUInt16LE(fmtOffset + 2);
  const sampleRate = buf.readUInt32LE(fmtOffset + 4);
  const bitsPerSample = buf.readUInt16LE(fmtOffset + 14);

  // Calculate samples
  const bytesPerSample = bitsPerSample / 8;
  const totalFrames = Math.floor(dataSize / (bytesPerSample * channels));
  const samples = new Float32Array(totalFrames);

  // Read PCM data and mix to mono
  for (let i = 0; i < totalFrames; i++) {
    let monoSum = 0;
    for (let ch = 0; ch < channels; ch++) {
      const offset = dataOffset + (i * channels + ch) * bytesPerSample;

      let sample: number;
      if (audioFormat === 3 && bitsPerSample === 32) {
        // 32-bit IEEE float
        sample = buf.readFloatLE(offset);
      } else if (audioFormat === 1 && bitsPerSample === 16) {
        // 16-bit signed PCM
        sample = buf.readInt16LE(offset) / 32768;
      } else if (audioFormat === 1 && bitsPerSample === 24) {
        // 24-bit signed PCM (manual 3-byte read)
        const b0 = buf[offset];
        const b1 = buf[offset + 1];
        const b2 = buf[offset + 2];
        const val = (b2 << 16) | (b1 << 8) | b0;
        sample = (val >= 0x800000 ? val - 0x1000000 : val) / 8388608;
      } else if (audioFormat === 1 && bitsPerSample === 32) {
        // 32-bit signed PCM
        sample = buf.readInt32LE(offset) / 2147483648;
      } else {
        throw new Error(`Unsupported WAV format: ${audioFormat}/${bitsPerSample}bit in ${filePath}`);
      }

      monoSum += sample;
    }
    samples[i] = monoSum / channels;
  }

  return { sampleRate, channels, samples };
}

// ── Energy Analysis ──────────────────────────────────────────────────────────

const ANALYSIS_FPS = 60;  // ~16.7ms windows

/**
 * Compute normalised RMS energy per window for a WAV file.
 * Returns array of values [0–1], rounded to 2 decimal places.
 */
function analyzeWav(filePath: string): { energy: number[]; duration: number; sampleRate: number } {
  const wav = parseWav(filePath);
  const windowSamples = Math.floor(wav.sampleRate / ANALYSIS_FPS);
  const totalWindows = Math.ceil(wav.samples.length / windowSamples);
  const energy = new Float32Array(totalWindows);

  // Compute RMS per window
  let maxRms = 0;
  for (let w = 0; w < totalWindows; w++) {
    const start = w * windowSamples;
    const end = Math.min(start + windowSamples, wav.samples.length);
    let sumSq = 0;
    for (let i = start; i < end; i++) {
      sumSq += wav.samples[i] * wav.samples[i];
    }
    const rms = Math.sqrt(sumSq / (end - start));
    energy[w] = rms;
    if (rms > maxRms) maxRms = rms;
  }

  // Normalise to [0–1]
  const result: number[] = new Array(totalWindows);
  if (maxRms > 1e-8) {
    for (let w = 0; w < totalWindows; w++) {
      result[w] = Math.round((energy[w] / maxRms) * 100) / 100;
    }
  } else {
    result.fill(0);
  }

  const duration = wav.samples.length / wav.sampleRate;
  return { energy: result, duration, sampleRate: wav.sampleRate };
}

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * Analyze drum stem WAV files and save a compact disco data JSON file.
 *
 * @param songId - Song ID (used for filenames)
 * @param audioDir - Directory containing stem WAV files
 * @param stemUrls - Object with kick/snare/hihat URL paths (e.g., "/audio/abc_kick.wav")
 * @returns URL path to the saved JSON file, or '' if no stems available
 */
export function analyzeAndSaveDiscoData(
  songId: string,
  audioDir: string,
  stemUrls: { kick?: string; snare?: string; hihat?: string },
): string {
  const stemCount = [stemUrls.kick, stemUrls.snare, stemUrls.hihat].filter(Boolean).length;
  if (stemCount === 0) {
    console.log(`[DiscoAnalyzer] Song ${songId}: no stems to analyze`);
    return '';
  }

  console.log(`[DiscoAnalyzer] Song ${songId}: analyzing ${stemCount} stem(s)...`);
  const t0 = Date.now();

  let duration = 0;

  // Analyze each available stem
  function analyzeStem(url: string | undefined, label: string): number[] {
    if (!url) return [];
    const filename = path.basename(url);
    const filePath = path.join(audioDir, filename);
    if (!fs.existsSync(filePath)) {
      console.warn(`[DiscoAnalyzer] ${label} stem file not found: ${filePath}`);
      return [];
    }
    try {
      const result = analyzeWav(filePath);
      if (result.duration > duration) duration = result.duration;
      console.log(`[DiscoAnalyzer]   ${label}: ${result.energy.length} windows, ${result.duration.toFixed(1)}s`);
      return result.energy;
    } catch (err: any) {
      console.error(`[DiscoAnalyzer]   ${label}: analysis failed: ${err.message}`);
      return [];
    }
  }

  const kick = analyzeStem(stemUrls.kick, 'kick');
  const snare = analyzeStem(stemUrls.snare, 'snare');
  const hihat = analyzeStem(stemUrls.hihat, 'hihat');

  // Build disco data
  const data: DiscoData = {
    version: 1,
    fps: ANALYSIS_FPS,
    duration,
    kick,
    snare,
    hihat,
  };

  // Save to JSON
  const filename = `${songId}_disco.json`;
  const filePath = path.join(audioDir, filename);
  fs.writeFileSync(filePath, JSON.stringify(data));

  const fileSize = fs.statSync(filePath).size;
  const elapsed = Date.now() - t0;
  console.log(`[DiscoAnalyzer] Song ${songId}: saved ${filename} (${(fileSize / 1024).toFixed(1)} KB) in ${elapsed}ms`);

  return `/audio/${filename}`;
}
