// training/audioMeta.ts — duration + embedded-tag reader (music-metadata)
//
// Duration is measured here and ONLY here — never taken from /understand, whose
// `duration` is the LM's chain-of-thought guess rather than a measurement (§4.8).
// Tag strings arrive with Latin-1 mojibake and NULs often enough that every one
// is sanitised before use (§7.6).

import { parseFile } from 'music-metadata';

export interface AudioMetaResult {
  duration: number;   // seconds, 0 when unknown
  artist: string;
  title: string;
  album: string;
  bpm: number | null; // embedded BPM tag, when the container carries one
}

const EMPTY: AudioMetaResult = { duration: 0, artist: '', title: '', album: '', bpm: null };

/** Control chars to drop from tag text — everything below 0x20 except \t and \n,
 *  plus DEL. Built from escapes so the source file stays plain ASCII. */
const CONTROL_CHARS = new RegExp('[\\u0000-\\u0008\\u000B\\u000C\\u000E-\\u001F\\u007F]', 'g');

/** Strip NULs and control chars (keeping \n and \t), trim, cap at 300 (§7.6). */
export function sanitizeTag(value: unknown): string {
  if (typeof value !== 'string') return '';
  return value.replace(CONTROL_CHARS, '').trim().slice(0, 300);
}

/** Read duration + artist/title/album. Never throws — returns zeros on failure. */
export async function read(audioPath: string): Promise<AudioMetaResult> {
  try {
    const md = await parseFile(audioPath, { duration: true, skipCovers: true });
    const rawBpm = (md.common as { bpm?: unknown }).bpm;
    const bpm = typeof rawBpm === 'number' && Number.isFinite(rawBpm) && rawBpm > 0
      ? Math.trunc(rawBpm)
      : null;
    return {
      duration: md.format.duration && Number.isFinite(md.format.duration) ? md.format.duration : 0,
      artist: sanitizeTag(md.common.artist),
      title: sanitizeTag(md.common.title),
      album: sanitizeTag(md.common.album),
      bpm,
    };
  } catch (err: any) {
    console.warn(`[Training] Tag read failed for ${audioPath}: ${err?.message || err}`);
    return { ...EMPTY };
  }
}

/** Duration only, in seconds. 0 when it cannot be determined. */
export async function readDuration(audioPath: string): Promise<number> {
  const meta = await read(audioPath);
  return meta.duration;
}
