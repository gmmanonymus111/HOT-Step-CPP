// promptBuilder.ts — Build text-to-image prompts from song metadata
//
// Ported from HOT-Step 9000's acestep/core/cover_art.py
//
// When `subject` is provided (from Lireek metadata), it's used as the
// primary prompt for more evocative imagery. Otherwise falls back to
// keyword extraction from lyrics.

const STOP_WORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
  'of', 'with', 'by', 'from', 'is', 'it', 'its', 'are', 'was', 'were',
  'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
  'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
  'not', 'no', 'so', 'if', 'up', 'out', 'just', 'like', 'my', 'me',
  'we', 'you', 'your', 'they', 'them', 'he', 'she', 'her', 'his',
  'i', 'im', 'ive', 'dont', 'that', 'this', 'all', 'got', 'get',
  'when', 'what', 'where', 'how', 'why', 'oh', 'yeah', 'ya', 'na',
  'la', 'da', 'uh', 'ah', 'ooh', 'hey', 'go', 'know', 'come', 'take',
  'make', 'see', 'let', 'say', 'one', 'way', 'back', 'now',
  'more', 'than', 'into', 'over', 'down', 'been',
]);

/** Extract the most common meaningful words from lyrics. */
export function extractThemeKeywords(lyrics: string, maxKeywords = 5): string[] {
  if (!lyrics?.trim()) return [];

  // Strip section headers like [Verse 1]
  let cleaned = lyrics.replace(/\[.*?\]/g, '');
  // Remove punctuation, lowercase
  cleaned = cleaned.replace(/[^\w\s]/g, '').toLowerCase();

  const words = cleaned.split(/\s+/)
    .filter(w => w.length > 3 && !STOP_WORDS.has(w));

  if (words.length === 0) return [];

  // Count frequencies
  const freq: Record<string, number> = {};
  for (const w of words) freq[w] = (freq[w] || 0) + 1;

  // Sort by frequency, take top N
  return Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxKeywords)
    .map(([word]) => word);
}

/**
 * Map music genre keywords to visual moods/palettes for image generation.
 * Avoids any text-triggering words like "album", "cover", "title", etc.
 */
const GENRE_VISUALS: Record<string, string> = {
  rock: 'dramatic lighting, electric atmosphere, high contrast',
  metal: 'dark dramatic scene, intense fire and shadows, heavy atmosphere',
  punk: 'gritty urban scene, raw energy, bold colors, rebellion',
  pop: 'vibrant colors, clean aesthetic, bright lighting, contemporary',
  electronic: 'neon lights, futuristic environment, glowing particles, cyberpunk',
  jazz: 'warm golden tones, smoky atmosphere, elegant mood, sophisticated',
  blues: 'moody blue tones, deep shadows, soulful atmosphere',
  folk: 'natural landscapes, warm earth tones, rustic beauty, pastoral',
  classical: 'elegant composition, renaissance lighting, grand architecture',
  hip: 'urban cityscape, bold colors, street culture, dynamic perspective',
  rap: 'urban environment, dramatic angles, street aesthetic',
  country: 'wide open landscapes, golden hour, rural beauty, americana',
  indie: 'dreamy atmosphere, soft pastel colors, artistic composition',
  r: 'warm intimate lighting, smooth gradients, elegant silhouettes',
  ambient: 'ethereal landscapes, soft focus, atmospheric mist, dreamlike',
  bossa: 'tropical sunset, warm golden light, coastal paradise',
  reggae: 'tropical colors, island vibes, sunset hues, laid-back mood',
  soul: 'warm rich tones, intimate atmosphere, emotional depth',
  funk: 'bold psychedelic colors, retro vibes, dynamic energy',
  alternative: 'moody atmosphere, artistic composition, unconventional beauty',
};

/** Get visual mood keywords based on genre/style string */
function getGenreVisuals(style: string): string {
  if (!style) return '';
  const lower = style.toLowerCase();
  for (const [genre, visuals] of Object.entries(GENRE_VISUALS)) {
    if (lower.includes(genre)) return visuals;
  }
  return '';
}

export interface CoverArtPromptOpts {
  title?: string;
  style?: string;
  lyrics?: string;
  subject?: string;
  /**
   * Fully user-authored prompt. When present (non-empty), it is used VERBATIM
   * as the positive prompt — all auto-assembly (subject/genre/art-direction) is
   * skipped. Set by the per-track "Generate Cover Art" prompt modal (#67).
   */
  prompt?: string;
}

/**
 * Build a text-to-image prompt from song metadata.
 *
 * IMPORTANT: Avoids ALL text-triggering language. FLUX models at cfg_scale=1
 * ignore negative prompts, so the only way to prevent text in the output is
 * to never mention text-related concepts (album, cover, title, etc.) in the
 * positive prompt. We describe only visual scenes and moods.
 */
export function buildCoverArtPrompt(opts: CoverArtPromptOpts): string {
  // User-authored prompt wins outright — used exactly as typed (#67).
  if (opts.prompt?.trim()) {
    return opts.prompt.trim();
  }

  const parts: string[] = [];

  if (opts.subject?.trim()) {
    // Rich subject path: use the curated description as a visual scene
    parts.push(opts.subject.trim());
  } else {
    // Fallback: extract visual themes from lyrics
    const keywords = extractThemeKeywords(opts.lyrics || '', 5);
    if (keywords.length > 0) {
      parts.push(`a scene evoking ${keywords.join(', ')}`);
    } else {
      // Absolute fallback — generic but text-free
      parts.push('a striking visual composition with dramatic lighting');
    }
  }

  // Genre-aware visual mood
  const genreVisuals = getGenreVisuals(opts.style || '');
  if (genreVisuals) {
    parts.push(genreVisuals);
  } else if (opts.style) {
    // Use raw style words as mood descriptors (but skip any that look like names)
    const styleWords = opts.style.split(',')
      .map(w => w.trim().toLowerCase())
      .filter(w => w.length > 2 && !w.includes('_'))
      .slice(0, 2);
    if (styleWords.length > 0) {
      parts.push(`${styleWords.join(' ')} aesthetic`);
    }
  }

  // Art direction — purely visual, zero text-triggering words
  parts.push('digital painting, cinematic composition, highly detailed, beautiful lighting, 8k');

  return parts.join(', ');
}
