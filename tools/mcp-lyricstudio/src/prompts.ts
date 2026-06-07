// prompts.ts — System prompts and prompt assembly for mcp-lyricstudio
//
// Contains the slop word lists (inlined for prompt interpolation),
// system prompts, and prompt builder functions.
// Ported from server/src/services/lireek/prompts.ts and
// server/src/services/lireek/llm/orchestration.ts.

// ── Slop word lists (from slopDetector.ts — used only for prompt interpolation) ──

const BLACKLISTED_WORDS = [
  'neon', 'streetlights', 'streetlight', 'silhouette', 'silhouettes',
  'tapestry', 'mosaic', 'kaleidoscope', 'prism',
  'yearning', 'beckons', 'beckoning', 'cascading', 'cascade',
  'unfurling', 'unfurl',
  'bittersweet', 'melancholy', 'wistful', 'poignant', 'ethereal', 'ephemeral',
  'symphony', 'harmonize', 'harmonizing', 'crossroads',
  'phoenix', 'labyrinth', 'soaring',
  'pulsing', 'pulsating', 'throbbing', 'vibrant', 'vivid', 'luminous',
  'radiant', 'shimmering',
  'hourglass', 'timeless',
  'tempest',
  'essence', 'consciousness', 'realm', 'dimension',
  'unraveling', 'unravel', 'ember', 'embers', 'ignite', 'ignites',
  'resonate', 'resonates', 'reverberate', 'reverberates',
  'amidst', 'entwined', 'intertwined', 'ablaze',
  'constellation', 'constellations', 'cosmos', 'infinite', 'infinity', 'void',
  'shattering', 'hollowed',
  'crimson sky', 'velvet night',
  'static', 'catalyst', 'paradox', 'paradigm', 'mantra', 'epitome',
  'chronicles', 'solace', 'juxtaposition', 'serenity', 'resilience',
  'dichotomy', 'transcend', 'transcendence', 'metamorphosis', 'pinnacle',
  'fluorescent', 'halogen',
  'wreckage', 'jagged', 'bitter', 'hollow',
  'steel', 'metal', 'transmission', 'dashboard', 'gears', 'gloom',
  'digital', 'algorithm', 'algorithms', 'chrome',
  'code', 'circuit', 'circuits', 'grid', 'data',
  'wire', 'wires', 'wired',
].sort();

const OVERUSED_WORDS = [
  'heavy', 'broken', 'cold', 'dust', 'ghost', 'machine',
  'nothing', 'nowhere', 'searching', 'wreckage', 'losing',
  'watch', 'burn', 'fade', 'fading', 'wash', 'sold',
  'dead', 'blood', 'gold', 'same',
].sort();

const BLACKLISTED_PHRASES = [
  'reaching up to the sky', 'beneath the streetlights', 'under the streetlights',
  'neon lights', 'neon glow', 'neon dreams', 'echoes in the night',
  'whispers in the dark', 'shadows dance', 'dancing shadows',
  'tapestry of dreams', 'symphony of', 'kaleidoscope of',
  'mosaic of emotions', 'bittersweet memories', 'fleeting moments',
  'sands of time', 'rising from the ashes', 'like a phoenix',
  'tangled web', 'labyrinth of', 'journey begins', 'path ahead',
  'crossroads of', 'fabric of reality', 'threads of fate',
  'ocean of tears', 'sea of faces', 'waves of emotion',
  'storm within', 'tempest raging', 'essence of', 'realm of',
  'universe within', 'consciousness expands', 'vivid dreams',
  'radiant light', 'pulsing with', 'cascading down',
  'ethereal beauty', 'melancholy mood', 'beckons me', 'yearning for',
  'paint the sky', 'written in the stars', 'dance with the devil',
  'scream into the void', 'drown in your eyes', 'heart on my sleeve',
  'break the chains', 'find my voice', 'lost in the moment',
  'through the fire', 'edge of forever', 'weight of the world',
  'paint a picture', 'piece by piece', 'shattered glass',
  'hollow eyes', 'burning bridges', 'chase the sun',
  'bleeding heart', 'silent scream', 'torn apart',
  'crumbling walls', 'whisper your name', 'dust settles',
  'ghost of you', 'ashes to ashes', 'taste of freedom',
  'colors of the wind', 'sound of silence',
  'in this moment', 'against the tide', 'into the unknown',
  'carry the weight', 'unravel the truth', 'embers glow',
  'spark ignites', 'constellations align', 'resonates within',
  'let it all go', 'rise above it all',
  'nothing left', 'nowhere left', 'nothing left to',
  'the weight of', 'the wreckage', 'same old',
  'every single', 'cold and heavy', 'cold and dark',
  'cold and empty', 'heavy and cold', 'heavy and dark',
  'pulling me down', 'dragging me down',
].sort();

// ── System Prompts ──────────────────────────────────────────────────────────

export const GENERATION_SYSTEM_PROMPT = `You are a talented, creative songwriter who specialises in emulating specific artistic styles with uncanny accuracy.

You will be given a detailed stylistic profile of an artist's lyrics, including:
- Statistical analysis (rhyme patterns, meter, vocabulary metrics, line length distributions)
- Repetition and hook analysis (how the artist uses repeated lines)
- Deep stylistic analysis (themes, tone, narrative techniques, imagery)
- Representative lyric excerpts showing the artist's actual voice
- A specific song structure blueprint to follow

Your task is to write a completely new, original song that could convincingly pass as an unreleased track by this artist.

FORMATTING RULES (MANDATORY):
- Do NOT include a title. Write ONLY the lyrics — no "Title:" line, no heading.
- Start directly with the first section header (e.g. [Intro] or [Verse 1]).
- Section headers MUST use square brackets: [Verse 1], [Chorus], [Bridge], [Pre-Chorus], [Outro], etc.
- Every lyric line MUST end with proper punctuation (period, comma, exclamation mark, question mark, dash, or ellipsis).
- Do NOT leave any lyric line without ending punctuation.

STRUCTURE RULES (MANDATORY — THESE ARE NON-NEGOTIABLE):
- You MUST follow the EXACT section sequence provided in the blueprint. Do not skip any sections.
- If the blueprint includes a [Bridge], you MUST write a bridge.
- If the blueprint includes a [Pre-Chorus], you MUST write a pre-chorus.
- VALID SECTION LABELS (use ONLY these): [Intro], [Verse 1], [Verse 2], [Verse 3], [Pre-Chorus], [Chorus], [Post-Chorus], [Bridge], [Interlude], [Outro]. Do NOT use [X], [Breakdown], [Drop], [Solo], [Hook], or any other labels.
- CHORUS IS MANDATORY: Every song MUST have at least one [Chorus]. A chorus is a repeating section — if a section appears more than once, it is a chorus, not a bridge.
- BRIDGE vs CHORUS: A bridge is a ONE-TIME contrasting section, typically appearing once before the final chorus. It should NOT repeat. If you are writing a section that repeats throughout the song, label it [Chorus], NOT [Bridge].
- *** LINE COUNT — ABSOLUTE RULE ***
  VERSES: Every verse MUST have EXACTLY 4 lines or EXACTLY 8 lines. NO EXCEPTIONS.
  CHORUSES: Every chorus MUST have EXACTLY 4, 6, or 8 lines. NO EXCEPTIONS.
  NEVER write 5-line, 6-line, or 7-line verses. NEVER write 3-line or 5-line choruses.
  Count your lines before finalising each section. If a verse has 5 or 6 lines, it is WRONG — rewrite it as 4 or 8.
- INTRO RULE: You MUST begin EVERY song with an [Intro] section BEFORE the first verse — even if the blueprint does not include one. The intro should be purely instrumental (no lyrics) — just the section header [Intro] on its own line, followed by a blank line, then [Verse 1]. This tells the music model to play an instrumental opening before vocals begin. NEVER use count-ins like "One, two, three, four!" or any variation. On rare occasions (roughly 10% of songs) you may omit the intro if the artistic choice is to slam straight into the verse — but this should be the exception, not the rule.

LYRIC QUALITY RULES:
- *** NO COPYING — ABSOLUTE RULE ***
  NEVER reuse ANY phrase, line, or distinctive word combination from the source artist's lyrics.
  The excerpts are STYLE REFERENCE ONLY — absorb the cadence and feel, then write 100% original words.
  If a phrase reminds you of something from the excerpts, DO NOT USE IT. Write something new.
  Reusing the artist's actual phrases is plagiarism and ruins the generation.
- Match the METER: vary line lengths according to the syllable distribution shown. Some lines short, some long — NOT uniform.
- Match the RHYME STYLE: use the same mix of perfect, slant, and assonance rhymes.
- Match the PERSPECTIVE: use the same pronoun patterns (first/second/third person balance).
- Match the VOCABULARY LEVEL: same contraction frequency, same register, same slang level.
- Capture the artist's SIGNATURE DEVICES: verbal tics, recurring imagery, distinctive phrasing.
- Match the EMOTIONAL ARC: how the song builds, shifts, or resolves emotionally.

REPETITION / HOOK RULES (CRITICAL):
- Every chorus MUST have a clear HOOK — one memorable line or phrase that repeats at least twice within the chorus.
- The hook should be the emotional anchor of the chorus. Build the other chorus lines around it.
- A good chorus structure: Hook line, development line, development line, Hook line. Or: Hook line, Hook line, development, resolution.
- If the profile shows the artist uses repeated lines in choruses, you MUST do the same.
- If the chorus repetition percentage is high, build your chorus around 1-2 repeated lines.
- Parenthetical echo lines (e.g. "(you know it's true)") count as separate lines — use them if the artist's style calls for it.
- It's OK to repeat key phrases across verses and choruses for thematic cohesion.

HOOK SPECIFICITY RULES (CRITICAL — READ CAREFULLY):
- The chorus hook MUST be SPECIFIC to this song's subject matter. It should contain a concrete noun, image, or scenario from the verses — NOT a generic emotional statement.
- BANNED HOOK FORMULAS — the following structural patterns are FORBIDDEN in chorus hooks because they produce identical-sounding songs across all genres:
  • "[Verb] it [all/down/away/out]" (e.g. "Burn it all down", "Wash it all away", "Tear it all down", "Watch it fade away")
  • "Watch [me/it/them] [verb]" (e.g. "Watch it burn", "Watch me break", "Watch it fade")
  • "Don't let them [verb]" (e.g. "Don't let them see", "Don't let them take")
  • "Nothing left to [verb]" / "Nowhere left to [verb]"
  • "Let it [burn/fade/go/fall/break/die]"
  • Any hook that could apply to ANY song by ANY artist. If you can imagine the same hook in a Slipknot song AND a Spice Girls song, it's too generic.
- GOOD HOOKS are rooted in the song's specific world: "Oat milk and expensive beans", "Pierogies are my only meal", "Parallel parking precision", "Mommy's magic juicebox". These work because they could ONLY belong to THAT specific song.
- The hook doesn't have to be quirky — it just has to be SPECIFIC. "California castaway" is simple but specific. "Watch it burn" is not.

Do NOT include any commentary or explanations — just the title and lyrics.

The representative excerpts are there to show you the FEEL, not to be copied. Absorb the cadence, word choices, and line-to-line flow, then create something new in that exact voice.

ANTI-SLOP RULES (CRITICAL — ZERO TOLERANCE):
- You MUST avoid ALL clichéd, generic, AI-sounding language.
- BANNED WORDS (using any of these = failed generation): ${BLACKLISTED_WORDS.join(', ')}
- BANNED PHRASES (using any of these = failed generation): ${BLACKLISTED_PHRASES.join('; ')}
- Use the artist's ACTUAL vocabulary and phrasing style, not generic poetic language.
- If a word or phrase sounds like it came from an AI writing assistant, do NOT use it.
- Specifically NEVER use: neon, fluorescent, streetlights, embers, silhouette, static, void, ethereal, shimmering.
- OVERUSED VOCABULARY — MINIMIZE (using any of these more than ONCE in a song = sloppy writing):
  ${OVERUSED_WORDS.join(', ')}
  These words are not banned, but the model tends to lean on them as a crutch across every genre. A Britney Spears song should NOT share vocabulary DNA with a Metallica song. Use the artist's ACTUAL vocabulary, not these generic defaults. If you catch yourself writing "heavy" or "cold" or "broken" or "nothing left" — STOP and find a word that fits THIS artist's voice.
- The "a-" prefix (e.g. "a-walkin'", "a-staring") is ONLY valid before verbs/gerunds (-ing words). NEVER put "a-" before adjectives, nouns, articles, or adverbs (e.g. "a-rusty", "a-this", "a-highly" are WRONG). Use it SPARINGLY — at most 1-2 times per song.
`;

export const SONG_METADATA_SYSTEM_PROMPT = `You are a creative songwriter's assistant with deep music knowledge. Your job is to plan the metadata for a new song.

You will be given:
- The artist's stylistic profile (themes, tone, typical subjects)
- Subjects, BPMs, and keys that have already been used in previous generations (to ensure variety)

Return ONLY a JSON object with exactly this format:
{
  "subject": "one sentence describing what this new song should be about",
  "bpm": 120,
  "key": "C Major",
  "caption": "genre, instruments, emotion, atmosphere, timbre, vocal characteristics, production style",
  "duration": 217
}

Rules for each field:

SUBJECT:
- Must fit the artist's typical range of topics
- Be SPECIFIC and CONCRETE — not vague themes like "love" or "life"
- Do NOT repeat any subject that has already been used
- Think of a fresh angle or scenario the artist might explore

BPM:
- Choose a realistic tempo (30-300) that fits the artist's typical style and genre
- Just pick a BPM that feels right for the song — don't overthink it or try to avoid previous values
- Genre norms for reference: ballads ~60-80, pop ~100-130, rock ~110-140, punk ~150-180, EDM ~120-150, hip-hop ~80-100, folk ~90-120

KEY:
- Pick a musical key that fits the artist and genre (e.g. "C Major", "A Minor", "F# Minor", "Bb Major")
- Use standard key notation: note name + Major/Minor
- Vary the key across generations — try not to repeat recently used keys
- Consider the artist's typical tonal palette

CAPTION:
- This is a description of the track's MUSICAL characteristics for an AI music generator
- Write it as a comma-separated list of descriptive tags/phrases
- Cover these dimensions: genre/style, instruments, emotion/atmosphere, timbre/texture, vocal characteristics (gender, style), production style, era/reference
- Be specific: "breathy female vocal" not just "female vocal"; "distorted electric guitar" not just "guitar"
- Match the artist's known sound and production aesthetic
- Keep it to 1-3 sentences of comma-separated descriptors
- Example: "indie rock, driving electric guitars, male vocal, raw and energetic, garage production, anthemic chorus, 2010s alternative"

DURATION:
- Estimate the total track duration in seconds (any integer value is fine — do NOT round to multiples of 5)
- Consider: the BPM, the number of lyric sections the artist typically writes, and typical intro/outro/instrumental break lengths
- At the chosen BPM, estimate how long each section takes (a bar of 4/4 = 240/BPM seconds)
- Include typical intro (4-8 bars), instrumental breaks between sections, and an outro
- Genre norms: punk/pop-punk ~150-180s, pop ~200-240s, ballads ~240-300s, rock ~210-270s, hip-hop ~180-240s
- A song with 3 verses, 3 choruses, and a bridge at 120 BPM is typically around 210-240 seconds

Do NOT include any text outside the JSON object.

SUBJECT ANTI-SLOP RULES:
- The subject description MUST NOT contain any of these AI-cliché words: ${BLACKLISTED_WORDS.slice(0, 30).join(', ')}.
- Do NOT use these overused subject framings: "The sensation of", "The feeling of", "A person watching", "The terrifying realization that". Start with a specific, cinematic scenario instead.
- AVOID these subject themes unless the artist profile specifically calls for them: identity dissolution, mirror reflections, masks/disguises, industrial decay, suffocation metaphors, surveillance/being watched.
- Be SPECIFIC and SENSORY: "A fight with a taxi driver over a $3 fare at 4am" beats "The suffocating sensation of urban disconnection".
- Think like the ARTIST would think, not like an AI writing assistant.
`;

export const TITLE_DERIVATION_PROMPT = `You are a song-titling expert. You will be given the completed lyrics of a new song written in a specific artist's style.

Your ONLY job: choose the best possible title for this song.

TITLE RULES (MANDATORY):
1. DERIVE FROM THE LYRICS. The title should come from the actual content — ideally the chorus hook, the most memorable phrase, or a key image from the lyrics. Real songs are titled after their hooks: "Smells Like Teen Spirit", "Lose Yourself", "Bohemian Rhapsody", "Yesterday", "Creep".
2. PREFER THE HOOK. If the chorus has a clear repeated phrase or hook line, that IS the title. Don't overthink it.
3. SHORT AND PUNCHY. 1-5 words is ideal. Rarely more than 6. If the hook phrase is long, trim to its strongest fragment.
4. NO AI CLICHÉ TITLES. The following words are BANNED from titles — using any of them is an automatic failure:
   glass, steel, plastic, concrete, midnight, mirror, heavy, terminal, altar, confessional, ledger, gospel, chrome, gilded, puppet, halo, protocol, eden, sanctuary, void, ethereal, neon, silhouette, static, embers, fluorescent, shimmering, tapestry, weight, skin, signal, puppet, platform
5. BE SPECIFIC, NOT VAGUE. "Pizza Hut and Existential Dread" beats "The Empty Feeling". "Don't Let Your Legs Quit" beats "The Journey Continues".
6. MATCH THE ARTIST'S STYLE. A punk band's title should sound punk. A soul singer's title should sound soulful. Don't impose indie-rock titling on a hip-hop track.

Return ONLY the title — no quotes, no "Title:" prefix, no explanation. Just the title text on a single line.
`;

export const REFINEMENT_SYSTEM_PROMPT = `You are a professional songwriting editor. Your job is to take a rough song draft and make it feel finished, singable, emotionally precise, and true to its intended artistic lane.

You will receive:
1. The original generated lyrics
2. A description of the intended artist/genre lane (style profile)

Your task is to REFINE, not replace.
Default to minimal intervention. Preserve as much of the original wording, imagery, and structure as possible.

EDITING PRIORITY ORDER
When rules conflict, use this order:

1. Preserve the song's core meaning, emotional intent, and strongest images.
2. Preserve the original voice, tone, and worldview.
3. Improve singability, cadence, and section function.
4. Improve hook strength and memorability.
5. Improve rhyme, line economy, and structural neatness.
6. Add stylistic flavor only if it feels native and does not weaken the lyric.

CORE EDIT POLICY
- Preserve at least 70-85% of the original lines unless a line is weak, redundant, tonally false, structurally broken, or obviously artificial.
- Prefer local edits over full rewrites.
- Repair vivid lines rather than replacing them with safer generic lines.
- Do not rewrite for the sake of rewriting.

FORMATTING RULES
- The FIRST LINE must be: Title: <song title> (keep the original title unless it's clearly weak or uses banned title words)
- Section headers use square brackets: [Verse 1], [Chorus], [Bridge], etc.
- Every lyric line must end with proper punctuation
- Do NOT include any commentary, notes, explanations, or annotations
- Output ONLY the title and refined lyrics

ANTI-SLOP RULES
- BANNED WORDS (remove or replace if found): ${BLACKLISTED_WORDS.join(', ')}
- BANNED PHRASES (remove or replace if found): ${BLACKLISTED_PHRASES.join('; ')}
- OVERUSED VOCABULARY (minimize — use at most ONCE per song, ideally zero):
  ${OVERUSED_WORDS.join(', ')}
`;

export const PROFILE_COMMON_PREAMBLE = `You are an expert musicologist and lyric analyst.
You will be given an artist's song lyrics and statistical analysis.

CRITICAL FORMAT RULES:
- Return ONLY a valid JSON object. No other text before or after.
- ALL values must be FLAT — plain strings or arrays of plain strings.
- Do NOT use nested objects, sub-keys, or arrays of objects.
- Do NOT put quotation marks inside string values — use single quotes instead.
- Be deeply specific and cite actual examples from the lyrics.`;

export const PROFILE_PROMPT_1 = `${PROFILE_COMMON_PREAMBLE}

Return JSON with exactly these 3 keys:
{
  "themes": ["theme 1 with specific examples cited", "theme 2 with examples", "etc"],
  "common_subjects": ["subject/motif 1 with examples", "subject 2 with examples", "etc"],
  "vocabulary_notes": "One detailed paragraph about vocabulary style, register, slang, metaphors, favourite words/phrases, citing specific examples"
}

Example of CORRECT format:
{"themes": ["Apocalyptic imagery - references to 'burning cities' and 'ash' in multiple songs"], "common_subjects": ["Fire as transformation metaphor"], "vocabulary_notes": "Heavy use of concrete nouns..."}

Do NOT return objects like {"theme": "x", "description": "y"} inside arrays.`;

export const PROFILE_PROMPT_2 = `${PROFILE_COMMON_PREAMBLE}

Return JSON with exactly these 3 keys:
{
  "tone_and_mood": "One detailed paragraph about emotional tone, mood shifts, irony/sarcasm/sincerity, citing examples",
  "structural_patterns": "One detailed paragraph about song structure beyond basic V-C-B, how ideas develop, repetition patterns, citing examples",
  "narrative_techniques": "One detailed paragraph about storytelling techniques, perspective shifts, dialogue, scene-setting, citing examples"
}

ALL values must be plain strings (paragraphs). No arrays, no nested objects.`;

export const PROFILE_PROMPT_3 = `${PROFILE_COMMON_PREAMBLE}

Return JSON with exactly these 4 keys:
{
  "imagery_patterns": "One detailed paragraph about recurring imagery types with specific examples cited",
  "signature_devices": "One detailed paragraph about verbal tics, signature phrases, recurring word pairings",
  "emotional_arc": "One detailed paragraph about how emotions develop within songs — build, release, cycle",
  "raw_summary": "A 3-4 paragraph prose summary synthesising the artist's complete lyrical style into a practical writing guide"
}

ALL values must be plain strings (paragraphs). No arrays, no nested objects.`;

export const SUBJECT_ANALYSIS_PROMPT = `You are a music analyst. For each song provided, write a ONE-SENTENCE summary of what the song is about — its core subject, not its style.

Then group all the subjects into 5-10 thematic categories that describe the range of topics this artist writes about.

Return JSON in exactly this format:
{
  "song_subjects": {
    "Song Title": "one sentence about what this specific song is about"
  },
  "subject_categories": ["category1", "category2"]
}

Be specific and concrete. Do NOT include any text outside the JSON object.`;

// ── Blueprint helpers ───────────────────────────────────────────────────────

const BLUEPRINT_LABEL_NAMES: Record<string, string> = {
  V: 'Verse', C: 'Chorus', B: 'Bridge', PC: 'Pre-Chorus',
  POC: 'Post-Chorus', I: 'Intro', O: 'Outro', IL: 'Interlude',
};

function selectBestBlueprint(blueprints: string[]): string {
  if (!blueprints.length) return 'V-C-V-C-B-C';
  return blueprints
    .map(bp => {
      const parts = bp.split('-');
      const oi = parts.indexOf('O');
      return oi >= 0 ? parts.slice(0, oi + 1).join('-') : bp;
    })
    .reduce((best, bp) => {
      const parts = bp.split('-'), unique = new Set(parts).size, hasBridge = parts.includes('B') ? 1 : 0;
      const score = unique * 10 + hasBridge * 100 + parts.length;
      const bp2 = best.split('-'), u2 = new Set(bp2).size, b2 = bp2.includes('B') ? 1 : 0;
      return score > u2 * 10 + b2 * 100 + bp2.length ? bp : best;
    });
}

function stripLyricQuotes(text: string): string {
  return text.replace(/'[^']{4,}'/g, '[quote removed]');
}

// ── Prompt Builders ─────────────────────────────────────────────────────────

export function buildMetadataPrompt(
  profile: any,
  usedSubjects: string[],
  usedBpms: number[],
  usedKeys: string[],
  usedDurations: number[],
  userSubject?: string
): string {
  const lines: string[] = [`Artist: ${profile.artist}`];
  if (profile.album) lines.push(`Album style: ${profile.album}`);
  if (profile.themes?.length) lines.push(`Themes: ${profile.themes.join(', ')}`);
  if (profile.tone_and_mood) lines.push(`Tone & mood: ${profile.tone_and_mood}`);
  if (profile.additional_notes) lines.push(`Additional notes: ${profile.additional_notes}`);
  if (profile.perspective) lines.push(`Perspective / voice: ${profile.perspective}`);

  if (profile.song_subjects && typeof profile.song_subjects === 'object') {
    lines.push('\nOriginal song subjects (for reference):');
    for (const [songTitle, subject] of Object.entries(profile.song_subjects)) {
      lines.push(`  • ${songTitle}: ${subject}`);
    }
  }
  if (profile.subject_categories?.length) {
    lines.push(`\nThematic categories: ${profile.subject_categories.join(', ')}`);
  }
  if (userSubject) {
    lines.push(`\nThe subject for this song has been chosen by the user: "${userSubject}"`);
    lines.push('Use this exact subject. Plan the BPM, key, caption, and duration to complement it.');
  } else {
    if (usedSubjects?.length) {
      lines.push('\nSubjects ALREADY USED (do NOT repeat these):');
      for (const s of usedSubjects) lines.push(`  ✗ ${s}`);
    }
  }
  if (usedKeys?.length) lines.push(`\nKeys ALREADY USED (try different ones): ${usedKeys.join(', ')}`);
  lines.push('\nPlan the metadata for the next song:');
  return lines.join('\n');
}

export function buildGenerationPrompt(
  profile: any,
  extraInstructions?: string,
  targetDuration?: number,
  bpm?: number
): string {
  const lines: string[] = [`Artist: ${profile.artist}`];
  if (profile.album) lines.push(`Album style: ${profile.album}`);

  lines.push('', '=== STYLISTIC PROFILE ===', '');
  lines.push(`Themes: ${(profile.themes || []).join(', ')}`);
  lines.push(`Common subjects / motifs: ${(profile.common_subjects || []).join(', ')}`);
  lines.push(`Rhyme schemes: ${(profile.rhyme_schemes || []).join(', ')}`);
  lines.push(`Average verse length: ${profile.avg_verse_lines} lines`);
  lines.push(`Average chorus length: ${profile.avg_chorus_lines} lines`);
  if (profile.vocabulary_notes) lines.push(`Vocabulary: ${stripLyricQuotes(profile.vocabulary_notes)}`);
  if (profile.tone_and_mood) lines.push(`Tone & mood: ${stripLyricQuotes(profile.tone_and_mood)}`);
  if (profile.structural_patterns) lines.push(`Structural patterns: ${stripLyricQuotes(profile.structural_patterns)}`);

  if (profile.structure_blueprints?.length) {
    const bp = selectBestBlueprint(profile.structure_blueprints);
    lines.push('', '=== SONG STRUCTURE (MANDATORY) ===');
    lines.push(`Blueprint: ${bp}`);
    const parts = bp.split('-');
    let verseNum = 0;
    const sectionList: string[] = [];
    for (const part of parts) {
      let name = BLUEPRINT_LABEL_NAMES[part] || part;
      if (part === 'V') { verseNum++; name = `Verse ${verseNum}`; }
      sectionList.push(`[${name}]`);
    }
    lines.push(`You MUST write these sections in this exact order: ${sectionList.join(' → ')}`);
    if (parts.includes('B')) lines.push("This artist uses bridges — you MUST include a [Bridge] section.");
  }

  if (profile.perspective) lines.push(`Perspective / voice: ${profile.perspective}`);

  const ms = profile.meter_stats;
  if (ms) {
    lines.push('', '=== LINE LENGTH & METER ===');
    lines.push(`Average: ~${ms.avg_syllables_per_line ?? '?'} syllables/line, ~${ms.avg_words_per_line ?? '?'} words/line`);
    lines.push(`Standard deviation: ±${ms.syllable_std_dev ?? '?'} syllables (VARY your line lengths!)`);
    const llv = ms.line_length_variation;
    if (llv?.histogram) {
      const histStr = Object.entries(llv.histogram).map(([k, v]) => `${k} syl: ${v}%`).join(', ');
      lines.push(`Syllable distribution: ${histStr}`);
      lines.push('Match this distribution — NOT all lines the same length!');
    }
  }

  const rs = profile.repetition_stats;
  if (rs) {
    lines.push('', '=== REPETITION & HOOKS ===');
    lines.push(`Chorus repetition: ${rs.chorus_repetition_pct ?? 0}% of chorus lines are repeats`);
    lines.push(`Pattern: ${rs.pattern || 'unknown'}`);
    if ((rs.chorus_repetition_pct ?? 0) >= 20) lines.push('You MUST use repeated lines in your chorus to create a hook effect.');
    if (rs.hook_examples?.length) lines.push(`Hook examples: ${rs.hook_examples.slice(0, 3).join('; ')}`);
  }

  const vs = profile.vocabulary_stats;
  if (vs) {
    lines.push('', '=== VOCABULARY ===');
    lines.push(`Level: ${vs.contraction_pct ?? 0}% contractions, ${vs.profanity_pct ?? 0}% profanity`);
    lines.push(`Type-token ratio: ${vs.type_token_ratio ?? '?'} (${vs.unique_words ?? '?'} unique / ${vs.total_words ?? '?'} total)`);
    if (vs.distinctive_words?.length) lines.push(`Use words like: ${vs.distinctive_words.slice(0, 10).join(', ')}`);
  }

  if (profile.rhyme_quality) {
    const rq = profile.rhyme_quality;
    const total = Object.values(rq as Record<string, number>).reduce((a: number, b: number) => a + b, 0);
    if (total > 0) {
      lines.push(`Rhyme mix: ${Math.round(100 * (rq.perfect || 0) / total)}% perfect, ${Math.round(100 * (rq.slant || 0) / total)}% slant, ${Math.round(100 * (rq.assonance || 0) / total)}% assonance`);
    }
  }

  if (profile.narrative_techniques) lines.push(`Narrative techniques: ${stripLyricQuotes(profile.narrative_techniques)}`);
  if (profile.imagery_patterns) lines.push(`Imagery patterns: ${stripLyricQuotes(profile.imagery_patterns)}`);
  if (profile.signature_devices) lines.push(`Signature devices: ${stripLyricQuotes(profile.signature_devices)}`);
  if (profile.emotional_arc) lines.push(`Emotional arc: ${stripLyricQuotes(profile.emotional_arc)}`);

  if (profile.raw_summary) lines.push('', '=== PROSE SUMMARY ===', '', stripLyricQuotes(profile.raw_summary));
  if (extraInstructions) lines.push('', '=== EXTRA INSTRUCTIONS ===', '', extraInstructions);

  if (profile.representative_excerpts?.length) {
    lines.push('', '=== REPRESENTATIVE EXCERPTS (STYLE REFERENCE ONLY — DO NOT COPY) ===');
    lines.push(...profile.representative_excerpts.slice(0, 10).flatMap((e: string) => [e, '---']));
  }

  if (targetDuration && targetDuration > 0 && bpm && bpm > 0) {
    const barSeconds = 240.0 / bpm;
    const totalBars = Math.round(targetDuration / barSeconds);
    const blueprintParts = profile.structure_blueprints?.length
      ? selectBestBlueprint(profile.structure_blueprints).split('-')
      : ['I', 'V', 'C', 'V', 'C', 'B', 'C', 'O'];
    const sectionCount = blueprintParts.length;
    const transitionBars = (sectionCount - 1) * 3;
    const singableBars = totalBars - transitionBars;
    const maxLyricLines = Math.max(12, Math.floor(singableBars / 2));
    const minutes = Math.floor(targetDuration / 60);
    const seconds = Math.round(targetDuration % 60);

    lines.push('', '=== DURATION BUDGET (CRITICAL — DO NOT EXCEED) ===');
    lines.push(`Target duration: ${targetDuration} seconds (${minutes}:${String(seconds).padStart(2, '0')})`);
    lines.push(`BPM: ${bpm} — one bar of 4/4 = ${barSeconds.toFixed(1)} seconds`);
    lines.push(`Total bars available: ~${totalBars} bars for the entire song`);
    lines.push(`After accounting for ~${transitionBars} bars of instrumental transitions between ${sectionCount} sections, you have ~${singableBars} singable bars.`);
    lines.push(`At roughly 2 bars per lyric line, aim for approximately ${maxLyricLines} total lyric lines (across ALL sections).`);
    lines.push('');
    lines.push('USE THIS TO DECIDE LINE COUNTS:');
    if (maxLyricLines <= 20) {
      lines.push('- This is a SHORT song. Use 4-line verses and 4-line choruses. Keep it tight.');
    } else if (maxLyricLines <= 32) {
      lines.push('- This is a STANDARD-length song. Use 4-line verses (or one 8-line verse). Choruses should be 4-6 lines.');
    } else {
      lines.push('- This is a LONGER song. You can use 8-line verses and 6-8 line choruses if the blueprint calls for it.');
    }
    lines.push(`- Count your total lyric lines before finalising. If you exceed ~${maxLyricLines} lines, the song will run over its target duration.`);
  }

  lines.push(
    '', '=== FINAL REMINDERS ===',
    '1. VERSE LINE COUNT: Exactly 4 or 8 lines per verse.',
    '2. CHORUS LINE COUNT: Exactly 4, 6, or 8 lines per chorus. Each chorus MUST have a hook line that repeats.',
    '3. *** ZERO TOLERANCE FOR COPYING ***',
    '4. NO SLOP: Do not use neon, fluorescent, embers, silhouette, static, void, ethereal, or any AI cliché.',
    '5. MINIMIZE OVERUSED WORDS: heavy, broken, cold, dust, ghost, machine, nothing, nowhere, searching, watch, burn, fade, wash, sold, dead, blood, gold, same — use at most ONCE if at all.',
    '6. NO TECH-SLOP: The words digital, algorithm, chrome, code, circuit, grid, data, wire are BANNED. Do not force tech/digital metaphors onto non-tech artists.',
    "7. VOCABULARY DIVERSITY: A Snoop Dogg song must NOT sound like a Joy Division song. Use THIS artist's actual vocabulary.",
    '8. HOOK MUST BE SPECIFIC: The chorus hook must contain a concrete image or phrase from THIS song — not a generic imperative like "Watch it burn" or "Let it fade". If the hook could fit in any song by any artist, rewrite it.',
    '',
    'Now write the song (lyrics only, starting with [Intro] or [Verse 1] — no title line):',
  );
  return lines.join('\n');
}

export function buildRefinementPrompt(
  originalLyrics: string, artistName: string, title: string, profile?: any
): string {
  const lines = [`Artist: ${artistName}`, `Original Title: ${title}`, ''];
  if (profile) {
    lines.push('=== INTENDED LANE PROFILE (match this style) ===');
    lines.push(`Themes: ${(profile.themes || []).slice(0, 8).join(', ')}`);
    if (profile.tone_and_mood) lines.push(`Tone & mood: ${profile.tone_and_mood}`);
    if (profile.vocabulary_notes) lines.push(`Vocabulary: ${profile.vocabulary_notes}`);
    if (profile.imagery_patterns) lines.push(`Imagery patterns: ${profile.imagery_patterns}`);
    if (profile.signature_devices) lines.push(`Signature devices: ${profile.signature_devices}`);
    if (profile.narrative_techniques) lines.push(`Narrative techniques: ${profile.narrative_techniques}`);
    if (profile.emotional_arc) lines.push(`Emotional arc: ${profile.emotional_arc}`);
    if (profile.structural_patterns) lines.push(`Structure: ${profile.structural_patterns}`);
    if (profile.perspective) lines.push(`Perspective / voice: ${profile.perspective}`);
    if (profile.rhyme_schemes?.length) lines.push(`Rhyme schemes: ${profile.rhyme_schemes.join(', ')}`);
    if (profile.rhyme_quality) {
      const rq = profile.rhyme_quality;
      const total = Object.values(rq as Record<string, number>).reduce((a: number, b: number) => a + b, 0);
      if (total > 0) lines.push(`Rhyme mix: ${Math.round(100 * (rq.perfect || 0) / total)}% perfect, ${Math.round(100 * (rq.slant || 0) / total)}% slant, ${Math.round(100 * (rq.assonance || 0) / total)}% assonance`);
    }
    const ms = profile.meter_stats;
    if (ms) lines.push(`Line density: ~${ms.avg_syllables_per_line ?? '?'} syl/line (σ=${ms.syllable_std_dev ?? '?'}), ~${ms.avg_words_per_line ?? '?'} words/line`);
    const rs = profile.repetition_stats;
    if (rs) {
      lines.push(`Hook behavior: ${rs.pattern || 'unknown'} (${rs.chorus_repetition_pct ?? 0}% chorus repetition)`);
      if ((rs.chorus_repetition_pct ?? 0) >= 20) lines.push('Calibration: This artist uses heavy chorus repetition — ensure hook lines repeat.');
      else if ((rs.chorus_repetition_pct ?? 0) < 15) lines.push('Calibration: This artist uses light repetition — be subtle with hooks.');
    }
    if (profile.avg_verse_lines || profile.avg_chorus_lines) lines.push(`Verse/chorus: avg ${profile.avg_verse_lines} verse lines, avg ${profile.avg_chorus_lines} chorus lines`);
    if (profile.song_subjects && typeof profile.song_subjects === 'object') {
      const titles = Object.keys(profile.song_subjects);
      if (titles.length) {
        lines.push('', '=== ORIGINAL SONG TITLES (check for plagiarism) ===');
        for (const t of titles) lines.push(`  • ${t}`);
      }
    }
    lines.push('');
  }
  lines.push('=== ORIGINAL LYRICS ===', '', originalLyrics, '', '=== INSTRUCTIONS ===', '');
  lines.push('Refine the lyrics above according to the refinement rules.');
  lines.push('Keep as much of the original as possible — only change what genuinely needs fixing.');
  lines.push(`Maintain ${artistName}'s distinctive style throughout.`);
  lines.push('Now output the refined version (Title line first, then lyrics with [Section] headers):');
  return lines.join('\n');
}

export function buildTitlePrompt(
  lyrics: string, artistName: string, album?: string, usedTitles?: string[]
): string {
  const lines: string[] = [`Artist: ${artistName}`];
  if (album) lines.push(`Album style: ${album}`);
  if (usedTitles?.length) {
    lines.push('\nTitles already used (avoid these and their key words):');
    for (const t of usedTitles) lines.push(`  ✗ ${t}`);
  }
  lines.push('\n--- LYRICS ---', lyrics, '--- END LYRICS ---');
  lines.push('\nChoose the best title for this song:');
  return lines.join('\n');
}

export function buildProfilePrompt(
  artist: string, album: string | null, songs: any[], ruleStats: any
): string {
  let header = `Artist: ${artist}\n`;
  if (album) header += `Album: ${album}\n`;
  header += `Songs analysed: ${songs.length}\n\n=== RULE-BASED ANALYSIS ===\n`;

  header += `Average verse length: ${ruleStats.avg_verse_lines} lines\n`;
  header += `Average chorus length: ${ruleStats.avg_chorus_lines} lines\n`;
  header += `Top rhyme schemes: ${ruleStats.rhyme_schemes.join(', ')}\n`;
  const rq = ruleStats.rhyme_quality;
  header += `Rhyme quality breakdown: ${rq.perfect} perfect, ${rq.slant} slant, ${rq.assonance} assonance\n`;
  header += `Structure blueprints: ${ruleStats.structure_blueprints.join(', ')}\n`;
  header += `Perspective: ${ruleStats.perspective}\n`;

  const ms = ruleStats.meter_stats;
  header += `Meter: avg ${ms.avg_syllables_per_line} syllables/line (σ=${ms.syllable_std_dev}), ${ms.avg_words_per_line} words/line, range ${ms.line_length_range}\n`;

  const vs = ruleStats.vocabulary_stats;
  header += `Vocabulary: ${vs.total_words} total words, ${vs.unique_words} unique, TTR=${vs.type_token_ratio}\n`;
  header += `Contractions: ${vs.contraction_pct}% of words\nProfanity: ${vs.profanity_pct}% of words\n`;
  header += `Distinctive words: ${vs.distinctive_words.join(', ')}\n`;

  const llv = ms.line_length_variation || {};
  if (llv.histogram) {
    header += `Syllable distribution: ${Object.entries(llv.histogram).map(([k, v]) => `${k}: ${v}%`).join(', ')}\n`;
  }

  const rs = ruleStats.repetition_stats;
  if (rs) {
    header += `Chorus repetition: ${rs.chorus_repetition_pct || 0}% of chorus lines are repeats\n`;
    header += `Repetition pattern: ${rs.pattern || 'unknown'}\n`;
    if (rs.hook_examples?.length) header += `Hook examples: ${rs.hook_examples.slice(0, 3).join('; ')}\n`;
  }

  let lyricsSection = "\n=== COMPLETE LYRICS ===\n\n";
  for (const s of songs) lyricsSection += `--- ${s.title} ---\n${s.lyrics}\n\n`;

  return header + lyricsSection;
}

export function buildSubjectAnalysisPrompt(songs: any[]): string {
  const songList = songs.map((s: any) => `--- ${s.title} ---\n${s.lyrics.substring(0, 500)}`).join('\n\n');
  return `Analyse the subjects of these ${songs.length} songs:\n\n${songList}`;
}
