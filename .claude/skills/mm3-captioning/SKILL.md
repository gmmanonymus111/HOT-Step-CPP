---
name: mm3-captioning
description: Explains MiniMax-Music3's Structured Caption format (Global Metadata / Vocal Details / Arrangement) and the vendored upstream music-caption-rewriter reference library. Use when formatting a caption/prompt for the MiniMax-Music3 backend (mm3-* engine, /api/generate with backend=minimax), building the prompt-assembly / request-translator increment for MM3, wiring Lyric Studio output toward an MM3 caption, or debugging genre drift / low adherence in MM3 generations.
---

# MiniMax-Music3 Captioning

MiniMax-Music3 is the second local generation backend (alongside the native ACE-Step
LM→DiT→VAE pipeline) — see `engine/src/minimax/mm3-*.h`, `engine/tools/convert-mm3.py`,
`server/src/services/backends/types.ts`, `ui/src/stores/backendStore.ts`. Unlike
ACE-Step's caption field, MM3 was trained on a specific three-section **Structured
Caption** format, and adherence to that format is the main lever for controlling the
output (see "Empirical context" below).

This skill vendors MiniMax's own official caption-authoring skill
(`upstream/`, fetched verbatim from `MiniMax-AI/MiniMax-Music3`) and summarizes it for
our agents. **Read `upstream/SKILL.md` in full before writing or reviewing any
MM3 caption-assembly code** — this file is a map and a set of house notes, not a
replacement.

## When to use this skill

- Formatting a caption/prompt string for an MM3 generation request (engine `mm3-*`
  backend, or the `instructions` field the reference HTTP API expects).
- Building or reviewing the MM3 prompt-assembly / request-translator increment —
  the code that turns Lyric Studio / UI form fields into the text MM3's tokenizer
  wraps in `<|caption_start|>...<|caption_end|>`.
- Wiring Lyric Studio (`server/src/services/lireek/`) output toward an MM3-targeted
  caption instead of (or alongside) the ACE-Step caption it already produces.
- Debugging MM3 output that drifts genre/mood across seeds, ignores section-local
  instructions, or otherwise seems to be "not listening" to the caption.

## The Structured Caption contract (from `upstream/SKILL.md`)

MM3 captions have exactly three top-level sections, in this order, as **plain text
labels** (not markdown headings — see "Skill vs. pipeline-template discrepancy"
below):

1. **Global Metadata** — genre/subgenre, tempo (exact BPM only if explicit/strongly
   justified, else a range or qualitative tempo), key/scale (only if explicit or
   musically useful), global emotional progression, sonics/production profile.
2. **Vocal Details** — for vocal music: lead configuration, timbre, register,
   delivery, harmony/backing vocals, restrained vocal FX. For instrumental music:
   state it's instrumental and name the instrument/texture carrying the lead line.
   Never invent lyrical subject matter or reproduce lyrics here.
3. **Arrangement** — section-by-section timeline: what enters/exits/changes/
   intensifies per section, instrument lifecycle (primary/secondary), groove
   development, transitions, embellishments/spatial FX. ~250–450 English words by
   default.

Other hard rules worth internalizing:

- **5,000-token limit** on the tokenized text prompt (README limitation) — the
  caption competes with lyrics for that budget; don't let Arrangement balloon.
- **Lyrics stay out of the caption.** Bracketed section tags (`[Verse]`, `[Chorus]`,
  `[Bridge]`, `[Instrumental]`, …) in the lyric text act as *musical directives* for
  that section's local arrangement — they must be honored in the Arrangement section
  — but the lyric words themselves are never quoted, paraphrased, or summarized into
  the caption.
- **Explicit instrumental requests must stay instrumental** — never add vocals to
  fill in an unspecified case; be conservative instead.
- **Never fabricate precision**: no invented exact BPM/key/production technique
  when the user's brief only supports a broader description.
- Section-local directives (tag-level) can change *that section's* local
  arrangement but must never silently override a global exclusion or an explicit
  top-level constraint (vocal gender, instrumental requirement, tempo limit,
  required/prohibited instrument).

## Navigating `upstream/references/` (genre router → family index → template)

This is a **progressive-disclosure** retrieval system — do not scan all 1,000
templates. Three layers:

1. **`upstream/references/genre-router.md`** — entry point. Maps genre/mood/cultural
   cues (with a CN/EN alias table) to one of 18 style families, e.g.
   `east-asian-modern`, `hip-hop-rap`, `metal-heavy-rock`, `cinematic-orchestral-epic`,
   `contemporary-folk-acoustic`, `general-pop-ballad` (fallback when only mood/imagery
   is given). Read this first, pick at most one primary + one secondary family.
2. **`upstream/references/index-<family>.md`** (18 files) — compact style cards per
   family. Read only the 1–2 files the router pointed at.
3. **`upstream/templates/<slug>_NNNN.txt`** (1,000 files) — full example captions in
   the exact target format. Select up to three by distinct role — **Foundation**
   (overall identity/groove), **Modifier** (one specific requested dimension:
   secondary genre, vocal character, cultural color, production texture),
   **Arrangement** (section timeline/energy-contour logic only) — then synthesize a
   *new* caption; never copy a template's sentences, exact key/BPM, or full section
   order verbatim.

Example templates worth opening as calibration references (plain-text label format,
not markdown):
- `upstream/templates/acoustic-blues-folk_0001.txt` — sparse solo-instrument
  arrangement, good minimal-instrumentation example.
- `upstream/templates/index-east-asian-modern.md` cards → templates therein, if
  MM3 output needs to match HOT-Step's existing East-Asian-heavy adapter/dataset mix.
- Any `index-hip-hop-rap.md` or `index-metal-heavy-rock.md` card, for genres where
  our current ACE-Step captions already lean on strong groove/production language —
  good starting point for a side-by-side format comparison.

## Skill vs. pipeline-template discrepancy (the request-translator seam)

Cross-checked `upstream/SKILL.md`'s Output Contract against
`D:\Ace-Step-Latest\mm3-weights\fixtures\tok_prompt_template.txt`, the literal
prompt-assembly template the reference pipeline builds:

```
<|im_start|><|caption_start|>Energetic synthwave with driving bass, retro drums, and soaring lead synths. 120 BPM, A minor.<|caption_end|><|lyrics_start|>[start]
[verse]
Neon lights across the bay
...
<|lyrics_end|><|im_end|><|audio_start|>
```

Findings:

- `upstream/SKILL.md` presents the three required sections as markdown `###`
  headings in its *own instructions* (`### Global Metadata`, etc.), which could be
  misread as "the output caption text should literally contain `###` markdown".
  The actual reference templates (e.g. `templates/acoustic-blues-folk_0001.txt`)
  use **plain text section-name lines** (`Global Metadata` / `Vocal Details` /
  `Arrangement`, no `#`, no blank-line separation requirement) — confirmed by
  reading a template directly. **The request-translator should emit plain-text
  labels, not markdown**, to match what MM3 was actually trained/templated on.
- `tok_prompt_template.txt`'s `<|caption_start|>...<|caption_end|>` slot is a fully
  **opaque string** — the tokenizer doesn't parse or require internal structure, it
  just wraps whatever text is handed to it. The fixture's own example caption is a
  minimal one-liner ("Energetic synthwave... 120 BPM, A minor."), not a
  three-section Structured Caption — i.e. the reference pipeline was smoke-tested
  with minimal captions, not the skill's prescribed structured form. This is
  consistent with the skill's own README calling the plain description "used
  directly" as one valid mode, with the Structured Caption as the *richer,
  optional* upgrade path — both are legal inputs to the same
  `<|caption_start|>` slot.
- Net implication for the request-translator increment: build the caption as
  plain-text section labels (matching template convention) when emitting a
  Structured Caption, keep it under the shared 5,000-token budget alongside lyrics,
  and don't assume the pipeline needs or wants markdown syntax anywhere in the
  string it hands to `<|caption_start|>`.

## Empirical context (measured 2026-08-13)

Minimal one-line captions (the fixture's own smoke-test style) produce **high
take-variance across seeds — genre drift** for MM3: the same short prompt lands in
noticeably different genre/mood territory seed to seed. Detailed Structured
Captions (the three-section form this skill describes) are the adherence lever —
more explicit Global Metadata / Vocal Details / Arrangement content reduces that
drift. This directly motivates building the request-translator to *always* emit a
full Structured Caption rather than passing a short user-typed description straight
through to `<|caption_start|>`.

## The format is MANDATORY, not preferred (ear-verified 2026-08-14)

A controlled A/B settled this for training data. One track (Alkaline Trio,
`alk3_crimson`), 30 s, identical lyrics, 5 seeds per arm, f16/f16, no adapter:

- **Arm A** — a good ACE-style caption verbatim: 219 words of Gemini descriptive
  prose already covering groove, per-instrument detail, timbre, mix AND
  arrangement-over-time. Not tag prose; genuinely rich.
- **Arm B** — the SAME content restructured into the three sections, 503 words,
  `Basic Attributes` synthesised from the sidecar's bpm/key/signature.

Rob's verdict by ear: **B better "by a massive margin"; most A takes were not even
the right genre (1 of 5 was), while B was on-genre and sounded far better.**

Consequences, both load-bearing:

1. **Rich descriptive prose is NOT a substitute for the format.** Arm A had most of
   the CONTENT and still produced wrong-genre output. So restructuring an existing
   caption corpus is required work before it can condition MM3 — for generation and
   for building a training conditioning cache alike.
2. **Lead the Arrangement with whatever OPENS the track.** B lost a piano intro that
   A reproduced in 4 of 5 takes. The piano was present in B, but as a subordinate
   clause two-thirds into a 503-word caption ("Primary: Distorted electric guitars
   carry the harmonic weight… A clean, bright piano opens the piece alone"). Put the
   opening instrument first in Instrument Lifecycle, and name it in Global Metadata.

**Do not judge this class of change by spectral proxies.** In the same run,
flatness (0.080 → 0.200) and centroid (1847 → 3187 Hz) both rose sharply, which
reads as "noisier/harsher" — and by ear that was simply the genre arriving (crash
cymbals and wall-of-sound distortion ARE spectrally flat and bright). An
`intro_ratio` (opening RMS ÷ body RMS) also *favoured* B, while B was the arm with
no piano at all: it measures a dynamic envelope, not instrumentation. Cross-seed
spread rose for B rather than falling, contradicting the "informative caption →
tighter clustering" hypothesis. Every proxy either misled or measured something
adjacent. Ears decided it; keep it that way.

Artifacts: `M:\HOT-Step-CPP\_experiments\caption-ab-2026-08-14\` (10 WAVs, both
captions, `results.json`, and the runner).

## Restructuring has a ceiling, and it is below hand-written (2026-08-14)

Follow-up to the above: `engine/tools/mm3-caption-restructure.py` converts the
existing Gemini caption corpus into the format mechanically. Five ear-judged
rounds on `alk3_crimson`, 5 seeds each, same track/lyrics/model:

| caption | on-genre |
|---|---|
| raw ACE caption | 1/5 |
| **hand-written Structured Caption** | **4–5/5** |
| scripted restructure v1 | 1–2/5 |
| scripted v2 (genre statement moved to lead Global Emotional Progression) | worse — country, "big band" |
| scripted, on a *typical* track (no piano) | heavy rock / old-school punk, still not pop-punk |

**Only hand-written prose reached the target.** Restructuring preserves the
source's emphasis faithfully — which is the problem, because the source was
written for a different model and weights things differently. It is a bridge for
an existing corpus, not a solution. **The real fix is to caption in MM3 format in
the first place**, from the audio, via the Training Studio Gemini prompt; that
also turns the ~22 % boilerplate (Application Scenarios & Imagery, Vocal Style,
Harmony/Backing Vocals, Vocal FX — identical across every track a script emits)
into real per-track observation.

Three traps, each of which cost a wrong conclusion:

1. **SPECTRAL PROXIES DO NOT MEASURE GENRE. Four consecutive wrong calls.**
   Flatness/centroid rose → read as "harsh", heard as the genre arriving. An
   `intro_ratio` favoured the arm with no piano (it measures a dynamic envelope,
   not instrumentation). Scripted v2 was *statistically identical* to the
   hand-written winner (0.192 ± 0.070 vs 0.189 ± 0.075) and sounded like big
   band. Cross-seed spread rose where the hypothesis said it should fall. Judge
   caption changes BY EAR; use the numbers only to spot a gross regression.
2. **A source caption can be off-genre in its own content, and no restructuring
   fixes that.** Track 01 was the only one of 13 mentioning piano (×4) and the
   only one saying "classic rock-influenced". Piano + classic rock + major key +
   male vocal lands in country/southern rock, and it did, repeatedly. Grep a
   corpus for off-genre content words before blaming the tool.
3. **Do not benchmark on an atypical track.** Track 01 was chosen *because* its
   piano intro gave a checkable adherence claim — which made it a good adherence
   probe and a bad genre benchmark. Several rounds were spent tuning against the
   hardest track in the set and generalising from it.

**Scope note.** All of the above judges the BASE model's genre fidelity from a
caption. For *training* conditioning that is a proxy, not the objective: the
conditioning rollout is hallucinated and content-misaligned with the target audio
by construction, so what the adapter can learn is the target's timbre/production
marginal, and the trigger word is what binds style to invocation. Do not spend
unbounded effort perfecting base-model genre before a training run has ever been
judged.

## Directory contents

```
mm3-captioning/
├── SKILL.md                 This file — house notes for HOT-Step agents
└── upstream/                 Verbatim vendor copy, MiniMax-AI/MiniMax-Music3 @ main, 2026-08-13
    ├── PROVENANCE.md          Source, fetch method, license note (read before redistributing further)
    ├── SKILL.md               MiniMax's own skill instructions — the authoritative workflow
    ├── README.md               MiniMax's skill-level README (usage, output contract, layout)
    ├── agents/openai.yaml      Agent metadata (display name, default prompt)
    ├── references/
    │   ├── genre-router.md     Entry point: 18-family routing table + CN/EN aliases + fusion rules
    │   └── index-*.md          18 family indexes (compact style cards), e.g. index-hip-hop-rap.md
    └── templates/               1,000 full example Structured Captions, `<slug>_NNNN.txt`
```

## Reusable-for-Lyric-Studio notes

- The genre-router's alias table (Mandopop/C-pop/Cantopop CN↔EN normalization,
  `华语流行`/`国风流行`/`氛围 R&B` etc.) is directly reusable if Lyric Studio ever
  needs to normalize CN genre input the way it already handles artist-profile
  vocabulary — see `project-vocal-pacing` / `project-section-tag-vocabulary` memory
  entries for our existing tag-vocabulary discipline.
- The Foundation/Modifier/Arrangement three-reference selection pattern (pick up to
  three templates with *distinct roles*, never inherit their exact key/BPM/section
  order) is a clean template-fusion pattern that parallels how Lyric Studio already
  blends multiple sampled blueprints — worth reusing verbatim as a prompt-assembly
  strategy rather than reinventing one for MM3.
- `upstream/references/genre-router.md`'s "modifier vs. genre" discipline (treat
  `ballad`, `emotional`, `epic`, `modern`, `dark`, `cinematic` as modifiers, never
  primary-genre evidence) is a good sanity check to borrow for any future MM3-side
  genre-tag validation, mirroring the OOD-tag lesson in the
  `project-section-tag-vocabulary` memory entry (don't infer out-of-distribution
  tags from our own datasets — check the authoritative doc first).
