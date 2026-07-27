// training/lyricsSanitizer.ts — Genius section-header normaliser
//
// `[Verse 1: Flume]` → `[Verse 1]`, `[Chorus - KUCKA]` → `[Chorus]`,
// `[Bridge (feat. X)]` → `[Bridge]`. `[Pre-Chorus]` is left alone, and any
// bracket line the pattern does not recognise passes through untouched.
//
// Applied to Genius lyrics only — never to user-typed lyrics (§6.7).

const HEADER_RE = /^\[([A-Za-z][A-Za-z0-9 -]*?)(?:\s*[:—–,]\s*.+|\s+-\s+.+|\s*\([^)]*\))?\]$/;

export function sanitizeHeaders(lyrics: string): string {
  if (!lyrics) return lyrics;
  return lyrics.split(/\r\n|\r|\n/).map(line => {
    const s = line.trim();
    if (s.startsWith('[') && s.endsWith(']')) {
      const m = HEADER_RE.exec(s);
      if (m) return `[${m[1].trim()}]`;
    }
    return line;
  }).join('\n');
}
