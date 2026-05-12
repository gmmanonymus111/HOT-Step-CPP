// download.ts — Audio download route with format conversion + metadata embedding
//
// GET /api/songs/:id/download?format=wav|flac|opus|mp3&bitrate=192&version=original|mastered
//
// Converts the source WAV to the requested format, embeds metadata tags
// and cover art (when available), and streams it back with a
// Content-Disposition header for browser download.

import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { config, getFFmpegPath } from '../config.js';
import { getDb } from '../db/database.js';
import {
  gatherSongMetadata, buildMetadataArgs, buildCoverArtArgs,
  type AudioMetadata,
} from '../services/audioMetadata.js';

const execFileAsync = promisify(execFile);
const router = Router();

/** Get mp3-codec binary path (platform-aware: no .exe on macOS/Linux) */
function getMp3CodecPath(): string {
  const ext = process.platform === 'win32' ? '.exe' : '';
  const aceExe = config.aceServer.exe;
  if (aceExe) return path.join(path.dirname(aceExe), `mp3-codec${ext}`);
  return path.resolve(process.cwd(), '..', 'engine', 'build', 'Release', `mp3-codec${ext}`);
}

/** Convert WAV to target format with optional metadata embedding */
async function convertAudio(
  sourcePath: string,
  format: string,
  bitrate: number,
  outputPath: string,
  metadata?: AudioMetadata,
): Promise<void> {
  const hasMeta = !!metadata;

  // WAV without metadata — fast copy, no conversion
  if (format === 'wav' && !hasMeta) {
    fs.copyFileSync(sourcePath, outputPath);
    return;
  }

  // MP3 without metadata — use mp3-codec.exe if available (faster, no ffmpeg dep)
  // When metadata IS present, we must use ffmpeg so ID3 tags get written.
  if (format === 'mp3' && !hasMeta) {
    const codec = getMp3CodecPath();
    if (fs.existsSync(codec)) {
      await execFileAsync(codec, [
        '-i', sourcePath, '-o', outputPath, '-b', String(bitrate),
      ], { timeout: 120_000 });
      return;
    }
    // Fallback to ffmpeg
  }

  // All other cases use ffmpeg (conversion + metadata + cover art)
  const ffmpegPath = getFFmpegPath();
  if (!ffmpegPath) {
    // No ffmpeg — for WAV, fall back to raw copy (no metadata)
    if (format === 'wav') {
      fs.copyFileSync(sourcePath, outputPath);
      return;
    }
    throw new Error(`Cannot convert to ${format} — ffmpeg not available`);
  }

  // ── Build ffmpeg command ──
  const args = ['-y', '-i', sourcePath];

  // Cover art: add as second input (PNG → JPEG transcoded on-the-fly)
  let coverArtArgs: { inputArgs: string[]; outputArgs: string[] } = { inputArgs: [], outputArgs: [] };
  if (metadata?.coverArtPath) {
    coverArtArgs = buildCoverArtArgs(metadata.coverArtPath, format);
    args.push(...coverArtArgs.inputArgs);
  }

  // Audio codec selection
  switch (format) {
    case 'wav':
      // Re-encode through ffmpeg so INFO chunks get written
      args.push('-c:a', 'pcm_s16le');
      break;
    case 'flac':
      args.push('-c:a', 'flac', '-sample_fmt', 's32', '-compression_level', '8');
      break;
    case 'opus':
      args.push('-c:a', 'libopus', '-b:a', `${bitrate}k`);
      break;
    case 'mp3':
      args.push('-c:a', 'libmp3lame', '-b:a', `${bitrate}k`);
      break;
    default:
      throw new Error(`Unsupported format: ${format}`);
  }

  // Cover art output args (stream mapping, codec, disposition)
  if (coverArtArgs.outputArgs.length > 0) {
    args.push(...coverArtArgs.outputArgs);
  }

  // Metadata tags
  if (metadata) {
    args.push(...buildMetadataArgs(metadata, format));
  }

  args.push(outputPath);

  try {
    await execFileAsync(ffmpegPath, args, { timeout: 120_000 });
  } catch (err: any) {
    throw new Error(`ffmpeg conversion failed: ${err.message}`);
  }
}

/** MIME types for audio formats */
const mimeTypes: Record<string, string> = {
  wav: 'audio/wav',
  mp3: 'audio/mpeg',
  flac: 'audio/flac',
  opus: 'audio/ogg',
};

// GET /api/download/:id?format=wav&bitrate=192&version=original&artist=Name&prepend=Prefix
router.get('/:id', async (req, res) => {
  const { id } = req.params;
  const format = (req.query.format as string || 'wav').toLowerCase();
  const bitrate = parseInt(req.query.bitrate as string) || 192;
  const version = (req.query.version as string || 'original').toLowerCase();
  const artistName = (req.query.artist as string || '').trim();
  const prepend = (req.query.prepend as string || '').trim();

  // Validate format
  if (!['wav', 'mp3', 'flac', 'opus'].includes(format)) {
    res.status(400).json({ error: `Invalid format: ${format}. Use wav, mp3, flac, or opus.` });
    return;
  }

  // Get song from DB — try by ID first, then by audio_url
  let song = getDb().prepare('SELECT * FROM songs WHERE id = ?').get(id) as any;
  if (!song) {
    // Fallback: try looking up by audio_url (for Lyric Studio queue items)
    const audioUrlParam = req.query.audioUrl as string;
    if (audioUrlParam) {
      song = getDb().prepare('SELECT * FROM songs WHERE audio_url = ?').get(audioUrlParam) as any;
    }
  }
  if (!song) {
    // Last resort: serve the audio file directly without DB metadata
    const audioUrlParam = req.query.audioUrl as string;
    if (audioUrlParam) {
      const filename = path.basename(audioUrlParam);
      const sourcePath = path.join(config.data.audioDir, filename);
      if (fs.existsSync(sourcePath)) {
        // Clean up parsed parameters just in case DB doesn't have standard naming
        const badPrefixes = /^_?(XL|STD)(\s*\(CPP\))?_?\s*-?\s*/i;
        const cleanPrepend = prepend.trim();
        const cleanArtist = artistName.replace(badPrefixes, '').trim();
        const titleSuffix = version === 'original' ? ' - Unmastered' : '';
        const titleParts = [cleanPrepend, cleanArtist, 'Untitled'].filter(Boolean);
        const downloadFilename = `${titleParts.join(' - ')}${titleSuffix}.${format}`;
        if (format === 'wav' && sourcePath.endsWith('.wav')) {
          res.setHeader('Content-Type', mimeTypes.wav);
          res.setHeader('Content-Disposition', `attachment; filename="${downloadFilename}"`);
          res.setHeader('Content-Length', fs.statSync(sourcePath).size);
          fs.createReadStream(sourcePath).pipe(res);
          return;
        }
        // Convert
        const tempDir = path.join(config.data.dir, 'download_temp');
        fs.mkdirSync(tempDir, { recursive: true });
        const tempFile = path.join(tempDir, `dl_${Date.now().toString(36)}.${format}`);
        await convertAudio(sourcePath, format, bitrate, tempFile);
        const stat = fs.statSync(tempFile);
        res.setHeader('Content-Type', mimeTypes[format] || 'application/octet-stream');
        res.setHeader('Content-Disposition', `attachment; filename="${downloadFilename}"`);
        res.setHeader('Content-Length', stat.size);
        const stream = fs.createReadStream(tempFile);
        stream.pipe(res);
        stream.on('end', () => { try { fs.unlinkSync(tempFile); } catch {} });
        return;
      }
    }
    res.status(404).json({ error: 'Song not found' });
    return;
  }

  // Latent download — raw HSLAT binary, no format conversion
  if (version === 'latent') {
    const latentUrl = song?.latent_url;
    if (!latentUrl) {
      res.status(404).json({ error: 'No latent file available for this track' });
      return;
    }
    const latentFilename = path.basename(latentUrl);
    const latentPath = path.join(config.data.audioDir, latentFilename);
    if (!fs.existsSync(latentPath)) {
      res.status(404).json({ error: 'Latent file not found on disk' });
      return;
    }
    const rawTitle = song.title || 'track';
    const downloadName = `${rawTitle.replace(/[^a-zA-Z0-9 _()-]/g, '').trim() || 'track'}.latent`;
    res.setHeader('Content-Type', 'application/octet-stream');
    res.setHeader('Content-Disposition', `attachment; filename="${downloadName}"`);
    res.setHeader('Content-Length', fs.statSync(latentPath).size);
    fs.createReadStream(latentPath).pipe(res);
    return;
  }

  // Determine which audio URL to use
  let audioUrl: string;
  if (version === 'mastered' && song.mastered_audio_url) {
    audioUrl = song.mastered_audio_url;
  } else {
    audioUrl = song.audio_url;
  }

  if (!audioUrl) {
    res.status(404).json({ error: 'No audio file available' });
    return;
  }

  // Resolve to filesystem path
  const audioFilename = path.basename(audioUrl);
  const sourcePath = path.join(config.data.audioDir, audioFilename);

  if (!fs.existsSync(sourcePath)) {
    res.status(404).json({ error: `Audio file not found on disk: ${audioFilename}` });
    return;
  }

  // Gather metadata for embedding into the output file
  let metadata: AudioMetadata | undefined;
  try {
    metadata = gatherSongMetadata(song);
  } catch (metaErr: any) {
    // Non-fatal — proceed without metadata if gathering fails
    console.warn(`[Download] Metadata gathering failed (non-fatal): ${metaErr.message}`);
  }

  // Build download filename: Prepend - Artist - Title_suffix.format
  // Strip leading/trailing underscores and the old backend-injected prefix patterns
  const badPrefixes = /^_?(XL|STD)(\s*\(CPP\))?_?\s*-?\s*/i;
  
  let rawTitle = song.title || 'Untitled';
  // Strip backend-generated prefix strings if they accidentally got committed to the DB
  rawTitle = rawTitle.replace(badPrefixes, '');
  rawTitle = rawTitle.replace(/_mastered/g, ''); // User wants mastered as default, so explicitly strip it out just in case
  const songTitle = rawTitle.replace(/[^a-zA-Z0-9 _()-]/g, '').trim();

  const suffix = version === 'original' ? ' - Unmastered' : '';
  const resolvedArtist = artistName || (song.artist || '').replace(badPrefixes, '').replace(/[^a-zA-Z0-9 _()-]/g, '').trim();

  // Strip leading "Artist - " prefix from title if it duplicates the resolved artist.
  // generate.ts stores titles as "Artist - Song Title", so without this the artist
  // would appear twice in the download filename.
  let finalTitle = songTitle;
  if (resolvedArtist) {
    const artistPrefix = new RegExp(`^${resolvedArtist.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*-\\s*`, 'i');
    finalTitle = songTitle.replace(artistPrefix, '').trim() || songTitle;
  }

  // Clean prepend: the user typed this, just trim whitespace
  const cleanPrepend = prepend.trim();
  const parts = [cleanPrepend, resolvedArtist, `${finalTitle}${suffix}`].filter(Boolean);
  const downloadFilename = `${parts.join(' - ')}.${format}`;

  try {
    if (format === 'wav' && sourcePath.endsWith('.wav') && !metadata) {
      // Source is already WAV with no metadata to embed — stream directly
      res.setHeader('Content-Type', mimeTypes.wav);
      res.setHeader('Content-Disposition', `attachment; filename="${downloadFilename}"`);
      res.setHeader('Content-Length', fs.statSync(sourcePath).size);
      fs.createReadStream(sourcePath).pipe(res);
      return;
    }

    // Convert to temp file, then stream
    const tempDir = path.join(config.data.dir, 'download_temp');
    fs.mkdirSync(tempDir, { recursive: true });
    const tempFile = path.join(tempDir, `dl_${Date.now().toString(36)}.${format}`);

    await convertAudio(sourcePath, format, bitrate, tempFile, metadata);

    const stat = fs.statSync(tempFile);
    res.setHeader('Content-Type', mimeTypes[format] || 'application/octet-stream');
    res.setHeader('Content-Disposition', `attachment; filename="${downloadFilename}"`);
    res.setHeader('Content-Length', stat.size);

    const stream = fs.createReadStream(tempFile);
    stream.pipe(res);

    // Clean up temp file after stream completes
    stream.on('end', () => {
      try { fs.unlinkSync(tempFile); } catch {}
      try { fs.rmdirSync(tempDir); } catch {}
    });
    stream.on('error', () => {
      try { fs.unlinkSync(tempFile); } catch {}
    });
  } catch (err: any) {
    console.error(`[Download] Conversion failed:`, err.message);
    res.status(500).json({ error: err.message });
  }
});

export default router;
