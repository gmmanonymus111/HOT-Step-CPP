/**
 * whisperTranscribe.ts — Whisper CLI transcription service
 *
 * Wraps whisper.cpp's whisper-cli to transcribe audio files to word-level
 * timestamped text. Used for lyrics synchronisation in the player.
 *
 * Workflow:
 *   1. Locate whisper-cli.exe via config.whisper.exe
 *   2. Find the best available GGML model in config.whisper.modelsDir
 *   3. Run whisper-cli with -oj (JSON output) and --max-len 1 (word-level)
 *   4. Parse the sidecar JSON file whisper writes, clean up, return result
 */
import { execFile } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';
import { config } from '../config.js';

const execFileAsync = promisify(execFile);

// ── Types ───────────────────────────────────────────────────────────

export interface WhisperWord {
  word: string;
  start: number;
  end: number;
  probability: number;
}

export interface WhisperSegment {
  start: number;
  end: number;
  text: string;
  words?: WhisperWord[];
}

export interface WhisperResult {
  segments: WhisperSegment[];
}

export interface WhisperOptions {
  /** Whisper model name override (e.g. 'ggml-large-v3-turbo.bin') */
  model?: string;
  /** Language code (e.g. 'en', 'ja') or 'auto' for auto-detect. Default: 'auto' */
  language?: string;
  /** Beam size for decoding. Default: 5 */
  beamSize?: number;
}

// ── Model priority for fallback selection ───────────────────────────

const MODEL_PRIORITY = [
  'ggml-large-v3-turbo.bin',
  'ggml-large-v3.bin',
  'ggml-medium.bin',
  'ggml-base.bin',
];

// ── Helpers ─────────────────────────────────────────────────────────

/**
 * Strip section markers like [Verse 1], [Chorus], etc. from lyrics text.
 * Collapses multiple newlines, joins into a single line, and trims to 800 chars.
 * Used to build a vocabulary-priming prompt for whisper.
 */
export function stripSectionMarkers(lyrics: string): string {
  return lyrics
    .replace(/\[.*?\]/g, '')           // remove [Verse 1], [Chorus], etc.
    .replace(/\r\n/g, '\n')            // normalise line endings
    .replace(/\n{2,}/g, '\n')          // collapse multiple newlines
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0)
    .join(' ')
    .slice(0, 800);
}

/**
 * Find the best available whisper GGML model file.
 *
 * If `preferredModel` is given and exists in the models directory, use it.
 * Otherwise falls back through MODEL_PRIORITY in order.
 *
 * @returns Absolute path to the model file, or null if none found.
 */
export function findWhisperModel(preferredModel?: string): string | null {
  const modelsDir = config.whisper.modelsDir;

  if (!fs.existsSync(modelsDir)) {
    console.warn(`[Whisper] Models directory does not exist: ${modelsDir}`);
    return null;
  }

  // Try preferred model first
  if (preferredModel) {
    const preferredPath = path.join(modelsDir, preferredModel);
    if (fs.existsSync(preferredPath)) {
      return preferredPath;
    }
    console.warn(`[Whisper] Preferred model not found: ${preferredModel}`);
  }

  // Fall through priority list
  for (const modelName of MODEL_PRIORITY) {
    const modelPath = path.join(modelsDir, modelName);
    if (fs.existsSync(modelPath)) {
      console.log(`[Whisper] Using model: ${modelName}`);
      return modelPath;
    }
  }

  // Last resort: any .bin file in the directory
  try {
    const files = fs.readdirSync(modelsDir).filter(f => f.endsWith('.bin'));
    if (files.length > 0) {
      const fallback = path.join(modelsDir, files[0]);
      console.log(`[Whisper] Fallback model: ${files[0]}`);
      return fallback;
    }
  } catch {
    // Can't read directory
  }

  console.warn('[Whisper] No model files found in models directory');
  return null;
}

/**
 * Check whether whisper-cli is available at the configured path.
 */
export function isWhisperAvailable(): boolean {
  try {
    return fs.existsSync(config.whisper.exe);
  } catch {
    return false;
  }
}

/**
 * Transcribe an audio file using whisper-cli.
 *
 * Runs whisper.cpp with JSON output (-oj) and word-level timestamps (--max-len 1).
 * Source lyrics are passed as a vocabulary-priming --prompt to improve accuracy.
 *
 * @param audioPath    Absolute path to the audio file (WAV/MP3)
 * @param sourceLyrics Original lyrics text for vocabulary priming
 * @param options      Optional overrides for model, language, beam size
 * @returns            Parsed WhisperResult, or null on failure
 */
export async function transcribeWithWhisper(
  audioPath: string,
  sourceLyrics: string,
  options: WhisperOptions = {},
): Promise<WhisperResult | null> {
  const whisperExe = config.whisper.exe;

  // Validate exe exists
  if (!fs.existsSync(whisperExe)) {
    console.error(`[Whisper] whisper-cli not found at: ${whisperExe}`);
    return null;
  }

  // Find model
  const modelPath = findWhisperModel(options.model);
  if (!modelPath) {
    console.error('[Whisper] No model available — cannot transcribe');
    return null;
  }

  // Validate audio file
  if (!fs.existsSync(audioPath)) {
    console.error(`[Whisper] Audio file not found: ${audioPath}`);
    return null;
  }

  const beamSize = options.beamSize ?? 5;
  const language = options.language ?? 'auto';

  // Build CLI args
  const args: string[] = [
    '-m', modelPath,
    '-f', audioPath,
    '-oj',                  // output JSON (writes <input>.json sidecar)
    '--max-len', '1',       // word-level segmentation
    '--beam-size', String(beamSize),
    '--no-prints',          // suppress progress to stderr
  ];

  // Language (skip if auto-detect)
  if (language !== 'auto') {
    args.push('--language', language);
  }

  // Vocabulary priming prompt from source lyrics
  const prompt = stripSectionMarkers(sourceLyrics);
  if (prompt.length > 0) {
    args.push('--prompt', prompt);
  }

  // whisper.cpp -oj writes JSON to <audioPath>.json
  const jsonOutputPath = audioPath + '.json';

  console.log(`[Whisper] Transcribing: ${path.basename(audioPath)}`);
  console.log(`[Whisper] Model: ${path.basename(modelPath)}, language: ${language}, beam: ${beamSize}`);
  const t0 = Date.now();

  try {
    await execFileAsync(whisperExe, args, {
      timeout: 120_000,       // 2 minute timeout
      maxBuffer: 10 * 1024 * 1024,  // 10 MB stdout/stderr buffer
    });
  } catch (err: any) {
    const stderr = err.stderr?.toString()?.slice(-500) || '';
    const code = err.code || 'unknown';
    console.error(`[Whisper] Process failed (code: ${code}): ${stderr || err.message}`);
    // Clean up any partial JSON output
    try { fs.unlinkSync(jsonOutputPath); } catch {}
    return null;
  }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

  // Read and parse the JSON output
  if (!fs.existsSync(jsonOutputPath)) {
    console.error('[Whisper] Expected JSON output file not found — whisper may have failed silently');
    return null;
  }

  let result: WhisperResult;
  try {
    const raw = fs.readFileSync(jsonOutputPath, 'utf-8');
    const parsed = JSON.parse(raw);

    // whisper.cpp JSON format: { transcription: [{ timestamps: {from, to}, text, ... }] }
    // Normalise into our WhisperResult format
    result = normaliseWhisperJson(parsed);
  } catch (err: any) {
    console.error(`[Whisper] Failed to parse JSON output: ${err.message}`);
    try { fs.unlinkSync(jsonOutputPath); } catch {}
    return null;
  }

  // Clean up the sidecar JSON file
  try { fs.unlinkSync(jsonOutputPath); } catch {}

  const segCount = result.segments.length;
  const wordCount = result.segments.reduce((n, s) => n + (s.words?.length ?? 0), 0);
  console.log(`[Whisper] Done in ${elapsed}s — ${segCount} segments, ${wordCount} words`);

  return result;
}

// ── Internal: normalise whisper.cpp JSON ────────────────────────────

/**
 * whisper.cpp -oj produces JSON in this shape:
 * {
 *   "transcription": [
 *     {
 *       "timestamps": { "from": "00:00:00,000", "to": "00:00:02,500" },
 *       "offsets": { "from": 0, "to": 2500 },
 *       "text": " Hello world",
 *       "tokens": [
 *         { "text": " Hello", "timestamps": { "from": "...", "to": "..." },
 *           "offsets": { "from": 0, "to": 1200 }, "p": 0.95 },
 *         ...
 *       ]
 *     }
 *   ]
 * }
 *
 * We normalise this into our simpler WhisperResult format.
 */
function normaliseWhisperJson(raw: any): WhisperResult {
  const segments: WhisperSegment[] = [];

  const transcription = raw?.transcription;
  if (!Array.isArray(transcription)) {
    console.warn('[Whisper] Unexpected JSON structure — no transcription array');
    return { segments };
  }

  for (const seg of transcription) {
    const startMs = seg?.offsets?.from ?? 0;
    const endMs = seg?.offsets?.to ?? 0;
    const text = (seg?.text ?? '').trim();

    const words: WhisperWord[] = [];
    if (Array.isArray(seg?.tokens)) {
      for (const tok of seg.tokens) {
        const tokText = (tok?.text ?? '').trim();
        if (tokText.length === 0) continue;

        words.push({
          word: tokText,
          start: (tok?.offsets?.from ?? 0) / 1000,
          end: (tok?.offsets?.to ?? 0) / 1000,
          probability: tok?.p ?? 0,
        });
      }
    }

    if (text.length > 0) {
      segments.push({
        start: startMs / 1000,
        end: endMs / 1000,
        text,
        words: words.length > 0 ? words : undefined,
      });
    }
  }

  return { segments };
}
