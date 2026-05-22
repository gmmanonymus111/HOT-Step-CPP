// wordLrcUtils.ts — Parse .lyrics.json and find active word/line by time

export interface LyricsWord {
  word: string;
  start: number;
  end: number;
  confidence: number;
  source: 'matched' | 'whisper' | 'ad-lib';
}

export interface LyricsLine {
  start: number;
  end: number;
  text: string;
  words: LyricsWord[];
  section?: string;
}

export interface LyricsJson {
  version: number;
  method: string;
  whisperModel: string;
  vocalsIsolated: boolean;
  lines: LyricsLine[];
}

/** Fetch .lyrics.json for the given audio URL. Returns null if not found. */
export async function fetchLyricsJson(audioUrl: string): Promise<LyricsJson | null> {
  if (!audioUrl) return null;
  try {
    const jsonUrl = audioUrl.replace(/\.\w+$/, '.lyrics.json');
    const res = await fetch(jsonUrl);
    if (!res.ok) return null;
    const data = await res.json();
    return data?.version && data?.lines ? data : null;
  } catch {
    return null;
  }
}

/** Find the index of the current line at the given time. */
export function findCurrentLineIndex(lines: LyricsLine[], time: number): number {
  if (lines.length === 0) return -1;
  let result = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].start <= time) result = i;
    else break;
  }
  if (result >= 0 && time <= lines[result].end + 2.0) return result;
  return result;
}

/** Find the index of the active word within a line at the given time. */
export function findActiveWordIndex(words: LyricsWord[], time: number): number {
  if (words.length === 0) return -1;
  for (let i = words.length - 1; i >= 0; i--) {
    if (time >= words[i].start) return i;
  }
  return -1;
}
