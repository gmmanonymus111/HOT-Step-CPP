#pragma once
// minimax/mm3-request.h — MM3 synth request parsing + prompt assembly.
//
// HOT-Step file (does not exist upstream). The seam between "what a caller
// asked for" (caption, lyrics, duration, seed) and "the exact byte string the
// checkpoint was trained on". Everything here is text work — no GGML, no VRAM,
// no GPU — so it is cheap and safe to call on a cold server. That is what makes
// POST /mm3/tokenize-check a useful pre-flight for the UI.
//
// ── The template (checkpoint contract) ──────────────────────────────────────
//
// diffusers @ dafe3733, modular_pipelines/minimax_music3/encoders.py:218-221:
//
//     text = (
//         f"{_IM_START}{_CAPTION_START}{_clean_caption(prompt)}{_CAPTION_END}"
//         f"{_LYRICS_START}{_normalize_lyrics(lyrics)}{_LYRICS_END}{_IM_END}{_AUDIO_START}"
//     )
//
// which expands to, verbatim (verified byte-for-byte against the fixture
// D:\Ace-Step-Latest\mm3-weights\fixtures\tok_prompt_template.txt):
//
//     <|im_start|><|caption_start|>CAPTION<|caption_end|><|lyrics_start|>[start]
//     LYRICS<|lyrics_end|><|im_end|><|audio_start|>
//
// No spaces, no newlines anywhere except the ones _normalize_lyrics itself
// emits. `[start]\n` is produced by _normalize_lyrics, NOT by the template, and
// the lyric body carries no trailing newline before <|lyrics_end|>.
// encoders.py:34-35 says it outright: "even whitespace-level changes to the
// assembled prompt change the generated audio."
//
// ── Text hygiene, replicated ────────────────────────────────────────────────
//
// _clean_caption (encoders.py:56-79) — the caption goes in as an opaque string,
// but the reference first strips the markdown an LLM-written caption tends to
// carry:
//   1. `<|anything|>` -> "first is rest" (or the bare inner text if it is one
//      word). This is a defusing rule: a caption containing a literal special
//      token would otherwise tokenise as a control token and corrupt the frame
//      structure.
//   2. per line: strip a leading ATX heading (`^\s{0,3}#{1,6}\s+`), a leading
//      bullet (`^\s*[*+-]\s+`, then `^\s*\*\s+` again), unwrap `**bold**`
//      repeatedly until it stops changing, unwrap `*italic*` (with a
//      not-preceded-by-`*` / not-followed-by-`*` guard), then rstrip.
//   3. blank out horizontal rules (`^\s*[-*_]{3,}\s*$`, multiline).
//   4. delete every "• " and every run of exactly four spaces.
//   5. collapse runs of 2+ newlines to one.
//
// _normalize_lyrics (encoders.py:82-93) — the lyric side is a real reformat:
//   1. per line: if the line STARTS with one or more `[tag]` (optionally
//      separated by spaces/tabs), the line is replaced by just those tags —
//      any text sharing the line with a leading tag is DROPPED. Lines with no
//      leading tag pass through untouched.
//   2. "] " -> "]\n" and " [" -> "\n[" — every tag ends up alone on its line.
//   3. " ^ " -> "\n" — the checkpoint's inline line-break marker.
//   4. lowercase the inside of every `[...]`.
//   5. prepend "[start]\n".
//
// The two `re.sub` calls that use lookbehind/lookahead (`*italic*`) and the
// MULTILINE horizontal rule are hand-rolled here: std::regex's ECMAScript
// grammar has no lookbehind, and Python's `\s` spanning newlines under
// re.MULTILINE makes the rule pattern subtly non-line-local. Both hand-rolled
// forms are documented at their definition with the equivalence argument.
//
// ── Where we deliberately differ ────────────────────────────────────────────
//
// The reference REQUIRES non-empty lyrics (encoders.py:210-211,
// `check_inputs`); it has no instrumental path at all. HOT-Step accepts an
// empty `lyrics` and substitutes the literal `[instrumental]` as the lyric
// body, giving `[start]\n[instrumental]`. `[Instrumental]` is a documented
// section tag in MiniMax's own skill README, so this stays inside the trained
// vocabulary rather than inventing one — but it IS a HOT-Step decision, not a
// reference behaviour. The caption is expected to say the piece is instrumental
// too (see .claude/skills/mm3-captioning/SKILL.md).

#include "mm3-model.h"
#include "mm3-pipeline.h"
#include "mm3-tokenizer.h"

#include "yyjson.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// encoders.py:44-45. Both are checkpoint limits, not our policy.
#define MM3_MAX_PROMPT_TOKENS 5000
#define MM3_MAX_AUDIO_FRAMES  9000

// The five template literals, encoders.py:36-39.
#define MM3_TPL_IM_START      "<|im_start|>"
#define MM3_TPL_IM_END        "<|im_end|>"
#define MM3_TPL_CAPTION_START "<|caption_start|>"
#define MM3_TPL_CAPTION_END   "<|caption_end|>"
#define MM3_TPL_LYRICS_START  "<|lyrics_start|>"
#define MM3_TPL_LYRICS_END    "<|lyrics_end|>"
#define MM3_TPL_AUDIO_START   "<|audio_start|>"

// What we put in the lyric slot when the caller sends none. See the header
// note — this is HOT-Step policy, the reference simply rejects empty lyrics.
#define MM3_INSTRUMENTAL_LYRIC "[instrumental]"

// ── Small text primitives ───────────────────────────────────────────────────

// Python's `\s` inside a single line. `\n` is excluded because every caller
// below has already split on newlines; keeping it out is what makes the
// hand-rolled rules line-local.
static inline bool mm3_is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f';
}

static std::string mm3_rstrip(const std::string & s) {
    size_t e = s.size();
    while (e > 0 && (mm3_is_space(s[e - 1]) || s[e - 1] == '\n')) {
        e--;
    }
    return s.substr(0, e);
}

static std::string mm3_strip(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && (mm3_is_space(s[b]) || s[b] == '\n')) {
        b++;
    }
    while (e > b && (mm3_is_space(s[e - 1]) || s[e - 1] == '\n')) {
        e--;
    }
    return s.substr(b, e - b);
}

static bool mm3_str_blank(const std::string & s) {
    for (char c : s) {
        if (!mm3_is_space(c) && c != '\n') {
            return false;
        }
    }
    return true;
}

// Python `str.split("\n")` — a trailing newline DOES yield an empty last field.
// The reference uses this form for the lyrics (encoders.py:85).
static std::vector<std::string> mm3_split_lines(const std::string & s) {
    std::vector<std::string> out;
    size_t                   start = 0;
    for (size_t i = 0; i <= s.size(); i++) {
        if (i == s.size() || s[i] == '\n') {
            out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

// Python `str.splitlines()` — the reference uses THIS form for the caption
// (encoders.py:65), and the difference from split("\n") is load-bearing: a
// caption ending in a newline must not gain a trailing empty line, because the
// join below would then leave a trailing '\n' that survives the \n{2,} collapse
// and lands inside <|caption_start|>...<|caption_end|>. Verified against the
// reference: modelling this as split("\n") produced exactly that one-byte
// divergence on every caption with a trailing newline.
//
// Boundaries handled: "\n", "\r\n", "\r". Python also splits on \v, \f, \x1c-\x1e,
// \x85 and U+2028/9; those never appear in a caption and are left alone.
static std::vector<std::string> mm3_splitlines(const std::string & s) {
    std::vector<std::string> out;
    size_t                   start = 0;
    size_t                   i     = 0;
    while (i < s.size()) {
        if (s[i] == '\n' || s[i] == '\r') {
            out.push_back(s.substr(start, i - start));
            if (s[i] == '\r' && i + 1 < s.size() && s[i + 1] == '\n') {
                i++;
            }
            start = i + 1;
        }
        i++;
    }
    if (start < s.size()) {
        out.push_back(s.substr(start));
    }
    return out;
}

static std::string mm3_join_lines(const std::vector<std::string> & v) {
    std::string out;
    for (size_t i = 0; i < v.size(); i++) {
        if (i) {
            out += '\n';
        }
        out += v[i];
    }
    return out;
}

static void mm3_replace_all(std::string & s, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s.compare(i, from.size(), from) == 0) {
            out += to;
            i += from.size();
        } else {
            out += s[i];
            i++;
        }
    }
    s.swap(out);
}

// ── _clean_caption ──────────────────────────────────────────────────────────

// `_SPECIAL_TAG_RE.sub(_rewrite_special_tag, caption)` (encoders.py:52,57-62).
// Pattern `<\|([^|]*)\|>`: the inner run may not contain '|', so the scan is a
// literal "<|" then everything up to the first '|', which must be followed by
// '>'. Replacement: inner.strip(); split on the first whitespace RUN; two
// fields -> "a is b", otherwise the stripped inner verbatim.
static std::string mm3_rewrite_special_tags(const std::string & in) {
    std::string out;
    out.reserve(in.size());
    size_t i = 0;
    while (i < in.size()) {
        if (in[i] == '<' && i + 1 < in.size() && in[i + 1] == '|') {
            size_t j = i + 2;
            while (j < in.size() && in[j] != '|') {
                j++;
            }
            if (j + 1 < in.size() && in[j] == '|' && in[j + 1] == '>') {
                const std::string inner = mm3_strip(in.substr(i + 2, j - (i + 2)));
                size_t            k     = 0;
                while (k < inner.size() && !mm3_is_space(inner[k]) && inner[k] != '\n') {
                    k++;
                }
                if (k < inner.size()) {
                    size_t r = k;
                    while (r < inner.size() && (mm3_is_space(inner[r]) || inner[r] == '\n')) {
                        r++;
                    }
                    if (r < inner.size()) {
                        out += inner.substr(0, k) + " is " + inner.substr(r);
                    } else {
                        // Trailing whitespace only: str.split(None, 1) yields ONE
                        // field, so the "a is b" branch does not fire.
                        out += inner.substr(0, k);
                    }
                } else {
                    out += inner;
                }
                i = j + 2;
                continue;
            }
        }
        out += in[i];
        i++;
    }
    return out;
}

// `^\s{0,3}#{1,6}\s+` -> "" (encoders.py:66).
static void mm3_strip_heading(std::string & line) {
    size_t i = 0;
    while (i < line.size() && i < 3 && mm3_is_space(line[i])) {
        i++;
    }
    size_t h = 0;
    while (i + h < line.size() && h < 6 && line[i + h] == '#') {
        h++;
    }
    if (h == 0) {
        return;
    }
    size_t j = i + h;
    if (j >= line.size() || !mm3_is_space(line[j])) {
        return;  // `\s+` needs at least one
    }
    while (j < line.size() && mm3_is_space(line[j])) {
        j++;
    }
    line.erase(0, j);
}

// `^\s*[*+-]\s+` -> "" (encoders.py:67), and the redundant `^\s*\*\s+` at
// encoders.py:68 (a strict subset — replicated anyway so a future divergence
// upstream is a one-line diff here).
static void mm3_strip_bullet(std::string & line, const char * marks) {
    size_t i = 0;
    while (i < line.size() && mm3_is_space(line[i])) {
        i++;
    }
    if (i >= line.size() || !strchr(marks, line[i])) {
        return;
    }
    size_t j = i + 1;
    if (j >= line.size() || !mm3_is_space(line[j])) {
        return;
    }
    while (j < line.size() && mm3_is_space(line[j])) {
        j++;
    }
    line.erase(0, j);
}

// One pass of `re.sub(r"\*\*([^*]+)\*\*", r"\1", line)` — leftmost,
// non-overlapping, inner run non-empty and '*'-free.
static std::string mm3_unwrap_bold_once(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == '*' && i + 1 < s.size() && s[i + 1] == '*') {
            size_t j = i + 2;
            while (j < s.size() && s[j] != '*') {
                j++;
            }
            if (j > i + 2 && j + 1 < s.size() && s[j] == '*' && s[j + 1] == '*') {
                out += s.substr(i + 2, j - (i + 2));
                i = j + 2;
                continue;
            }
        }
        out += s[i];
        i++;
    }
    return out;
}

// `re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)` (encoders.py:74),
// hand-rolled because ECMAScript std::regex has no lookbehind.
//
// Equivalence argument: `[^*\n]+` can never cross a '*', so once the greedy run
// stops the only candidate closer is the character it stopped on. Shrinking the
// run therefore cannot produce a different match, and neither guard can be
// satisfied by backtracking — a failed attempt at position i is a hard failure
// at position i, exactly as this loop treats it. The lookbehind reads the
// ORIGINAL string (Python's does too, even across a previous substitution), so
// the check is against `s`, not against the output built so far.
static std::string mm3_unwrap_italic(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == '*' && (i == 0 || s[i - 1] != '*')) {
            size_t j = i + 1;
            while (j < s.size() && s[j] != '*' && s[j] != '\n') {
                j++;
            }
            if (j > i + 1 && j < s.size() && s[j] == '*' && (j + 1 >= s.size() || s[j + 1] != '*')) {
                out += s.substr(i + 1, j - (i + 1));
                i = j + 1;
                continue;
            }
        }
        out += s[i];
        i++;
    }
    return out;
}

// `^\s*[-*_]{3,}\s*$` (encoders.py:77).
static bool mm3_is_hrule(const std::string & line) {
    size_t i = 0;
    while (i < line.size() && mm3_is_space(line[i])) {
        i++;
    }
    size_t n = 0;
    while (i + n < line.size() && (line[i + n] == '-' || line[i + n] == '*' || line[i + n] == '_')) {
        n++;
    }
    if (n < 3) {
        return false;
    }
    size_t j = i + n;
    while (j < line.size() && mm3_is_space(line[j])) {
        j++;
    }
    return j == line.size();
}

// encoders.py:56-79, in order.
static std::string mm3_clean_caption(const std::string & caption) {
    std::string text = mm3_rewrite_special_tags(caption);

    std::vector<std::string> lines_out;
    for (std::string line : mm3_splitlines(text)) {
        mm3_strip_heading(line);
        mm3_strip_bullet(line, "*+-");
        mm3_strip_bullet(line, "*");
        while (line.find("**") != std::string::npos) {
            const std::string updated = mm3_unwrap_bold_once(line);
            if (updated == line) {
                break;
            }
            line = updated;
        }
        line = mm3_unwrap_italic(line);
        lines_out.push_back(mm3_rstrip(line));
    }

    // The horizontal-rule sub is MULTILINE with a `\s` that also matches '\n',
    // so in Python it can swallow the newline(s) BEFORE a rule line. It can
    // never swallow the one after (`$` must land at a line end), and any extra
    // blank line it would have eaten is removed by the `\n{2,}` collapse three
    // lines below regardless. Blanking the line in place is therefore
    // equivalent on the collapsed output.
    for (auto & line : lines_out) {
        if (mm3_is_hrule(line)) {
            line.clear();
        }
    }
    text = mm3_join_lines(lines_out);

    mm3_replace_all(text, "\xE2\x80\xA2 ", "");  // "• " (U+2022 + space)
    mm3_replace_all(text, "    ", "");           // exactly four spaces

    // `re.sub(r"\n{2,}", "\n", text)`
    std::string collapsed;
    collapsed.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        if (text[i] == '\n') {
            size_t j = i;
            while (j < text.size() && text[j] == '\n') {
                j++;
            }
            collapsed += '\n';
            i = j;
        } else {
            collapsed += text[i];
            i++;
        }
    }
    return collapsed;
}

// ── _normalize_lyrics ───────────────────────────────────────────────────────

// `_LEADING_TAGS_RE = r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)"` (encoders.py:53).
// Returns true and writes group(1).strip() when the line starts with one or
// more bracketed tags; false leaves the line alone.
static bool mm3_leading_tags(const std::string & line, std::string * out) {
    size_t i = 0;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
        i++;
    }
    const size_t first = i;
    size_t       last  = i;  // end of the last consumed `[tag]` + trailing [ \t]*
    int          n     = 0;
    while (i < line.size() && line[i] == '[') {
        size_t j = i + 1;
        while (j < line.size() && line[j] != ']') {
            j++;
        }
        if (j >= line.size() || j == i + 1) {
            break;  // unterminated, or `[]` (the `[^\]]+` is one-or-more)
        }
        i = j + 1;
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
            i++;
        }
        last = i;
        n++;
    }
    if (n == 0) {
        return false;
    }
    *out = mm3_strip(line.substr(first, last - first));
    return true;
}

// `re.sub(r"\[([^\]]+)\]", lambda m: f"[{m.group(1).lower()}]", text)`
// (encoders.py:92). ASCII-only lowering: every tag in MiniMax's documented
// vocabulary is ASCII, and Python's Unicode casefold on a non-ASCII tag would
// be a behaviour we cannot replicate without ICU. Non-ASCII bytes pass through.
static std::string mm3_lower_tags(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == '[') {
            size_t j = i + 1;
            while (j < s.size() && s[j] != ']') {
                j++;
            }
            if (j < s.size() && j > i + 1) {
                out += '[';
                for (size_t k = i + 1; k < j; k++) {
                    out += (char) std::tolower((unsigned char) s[k]);
                }
                out += ']';
                i = j + 1;
                continue;
            }
        }
        out += s[i];
        i++;
    }
    return out;
}

// encoders.py:82-93, in order. Always returns a string starting "[start]\n".
static std::string mm3_normalize_lyrics(const std::string & lyrics) {
    std::vector<std::string> out;
    for (const std::string & line : mm3_split_lines(lyrics)) {
        std::string tags;
        out.push_back(mm3_leading_tags(line, &tags) ? tags : line);
    }
    std::string text = mm3_join_lines(out);
    mm3_replace_all(text, "] ", "]\n");
    mm3_replace_all(text, " [", "\n[");
    mm3_replace_all(text, " ^ ", "\n");
    text = mm3_lower_tags(text);
    return "[start]\n" + text;
}

// ── Assembly ────────────────────────────────────────────────────────────────

// encoders.py:218-221. `lyrics` empty/whitespace-only -> the instrumental
// substitution documented at the top of this file.
static std::string mm3_assemble_prompt(const std::string & caption, const std::string & lyrics,
                                       bool * out_instrumental = nullptr) {
    const bool instrumental = mm3_str_blank(lyrics);
    if (out_instrumental) {
        *out_instrumental = instrumental;
    }
    return std::string(MM3_TPL_IM_START) + MM3_TPL_CAPTION_START + mm3_clean_caption(caption) + MM3_TPL_CAPTION_END +
           MM3_TPL_LYRICS_START + mm3_normalize_lyrics(instrumental ? std::string(MM3_INSTRUMENTAL_LYRIC) : lyrics) +
           MM3_TPL_LYRICS_END + MM3_TPL_IM_END + MM3_TPL_AUDIO_START;
}

// ── The wire request ────────────────────────────────────────────────────────

// Everything POST /mm3/synth accepts, resolved against the checkpoint's own
// defaults. `gen` is ready to hand to mm3_generate() once a cancel hook is
// attached.
struct MM3SynthRequest {
    std::string caption;
    std::string lyrics;
    bool        instrumental = false;

    double  duration   = 0.0;  // as asked, seconds
    int64_t max_frames = 0;    // = min(round(duration * frame_rate), 9000)
    int64_t seed_in    = -1;   // as asked (-1 = random)
    int     wav_bits   = 16;

    std::string prompt;  // the assembled template
    int64_t     n_tokens = 0;

    MM3GenRequest gen;
};

// Read one JSON field with a type check that fails LOUDLY: a caller that sends
// `"seed": "42"` must be told, not silently given the default.
static bool mm3_req_num(yyjson_val * root, const char * key, double * out, bool * present, std::string * err) {
    *present       = false;
    yyjson_val * v = root ? yyjson_obj_get(root, key) : nullptr;
    if (!v || yyjson_is_null(v)) {
        return true;
    }
    if (!yyjson_is_num(v)) {
        if (err) {
            *err = std::string("\"") + key + "\" must be a number";
        }
        return false;
    }
    // yyjson_get_num, NOT yyjson_get_real: a JSON integer literal is stored as
    // uint/sint and yyjson_get_real returns 0.0 for it, so `"duration": 10`
    // would read as zero. Caught by the first smoke test.
    *out     = yyjson_get_num(v);
    *present = true;
    return true;
}

static bool mm3_req_str(yyjson_val * root, const char * key, std::string * out, bool * present, std::string * err) {
    *present       = false;
    yyjson_val * v = root ? yyjson_obj_get(root, key) : nullptr;
    if (!v || yyjson_is_null(v)) {
        return true;
    }
    if (!yyjson_is_str(v)) {
        if (err) {
            *err = std::string("\"") + key + "\" must be a string";
        }
        return false;
    }
    out->assign(yyjson_get_str(v), yyjson_get_len(v));
    *present = true;
    return true;
}

// Parse + assemble. Does NOT tokenise (that needs the LM GGUF header) — call
// mm3_request_tokenize() next.
//
// Contract, with types and defaults:
//   caption    string, REQUIRED, non-blank
//   lyrics     string, optional, "" -> instrumental
//   duration   number (seconds), REQUIRED unless max_frames is given
//   max_frames integer, optional escape hatch; wins over duration
//   seed       integer, default -1 (= draw one from std::random_device)
//   cfg_flow   number, default = the checkpoint's flow.cfg_scale (1.7)
//   steps      integer, default = the checkpoint's flow.steps (30)
//   get_wav_bits integer in {16, 24, 32}, default 16
static bool mm3_parse_synth_request(const MM3Model & m, yyjson_val * root, MM3SynthRequest * out, std::string * err) {
    *out = MM3SynthRequest{};

    bool present = false;
    if (!mm3_req_str(root, "caption", &out->caption, &present, err)) {
        return false;
    }
    if (!present || mm3_str_blank(out->caption)) {
        if (err) {
            *err = "\"caption\" is required and must be a non-empty string";
        }
        return false;
    }
    if (!mm3_req_str(root, "lyrics", &out->lyrics, &present, err)) {
        return false;
    }

    const int64_t frame_rate = m.lm_cfg.frame_rate ? (int64_t) m.lm_cfg.frame_rate : 25;
    int64_t       cap        = (int64_t) m.lm_cfg.max_audio_frames;
    if (cap <= 0 || cap > MM3_MAX_AUDIO_FRAMES) {
        cap = MM3_MAX_AUDIO_FRAMES;
    }

    double  dur = 0.0;
    bool    have_dur = false;
    if (!mm3_req_num(root, "duration", &dur, &have_dur, err)) {
        return false;
    }
    double  mf       = 0.0;
    bool    have_mf  = false;
    if (!mm3_req_num(root, "max_frames", &mf, &have_mf, err)) {
        return false;
    }
    if (have_mf) {
        out->max_frames = (int64_t) llround(mf);
        out->duration   = (double) out->max_frames / (double) frame_rate;
    } else if (have_dur) {
        if (!(dur > 0.0)) {
            if (err) {
                *err = "\"duration\" must be greater than 0 seconds";
            }
            return false;
        }
        out->duration   = dur;
        out->max_frames = (int64_t) llround(dur * (double) frame_rate);
    } else {
        if (err) {
            *err = "\"duration\" (seconds) is required";
        }
        return false;
    }
    if (out->max_frames < 1) {
        out->max_frames = 1;
    }
    if (out->max_frames > cap) {
        out->max_frames = cap;  // min(duration * 25, 9000)
    }

    double  seed_d  = -1.0;
    bool    have_seed = false;
    if (!mm3_req_num(root, "seed", &seed_d, &have_seed, err)) {
        return false;
    }
    out->seed_in = have_seed ? (int64_t) llround(seed_d) : -1;
    uint64_t seed;
    if (out->seed_in < 0) {
        std::random_device rd;
        seed = ((uint64_t) rd() << 32) ^ (uint64_t) rd();
    } else {
        seed = (uint64_t) out->seed_in;
    }

    double cfg = m.synth_cfg.flow.cfg_scale > 0.0f ? (double) m.synth_cfg.flow.cfg_scale : 1.7;
    if (!mm3_req_num(root, "cfg_flow", &cfg, &present, err)) {
        return false;
    }
    if (!(cfg > 0.0) || cfg > 100.0) {
        if (err) {
            *err = "\"cfg_flow\" must be in (0, 100]";
        }
        return false;
    }

    double steps = m.synth_cfg.flow.steps ? (double) m.synth_cfg.flow.steps : 30.0;
    if (!mm3_req_num(root, "steps", &steps, &present, err)) {
        return false;
    }
    const int64_t nsteps = (int64_t) llround(steps);
    if (nsteps < 1 || nsteps > 1000) {
        if (err) {
            *err = "\"steps\" must be in 1..1000";
        }
        return false;
    }

    double bits = 16.0;
    if (!mm3_req_num(root, "get_wav_bits", &bits, &present, err)) {
        return false;
    }
    out->wav_bits = (int) llround(bits);
    if (out->wav_bits != 16 && out->wav_bits != 24 && out->wav_bits != 32) {
        if (err) {
            *err = "\"get_wav_bits\" must be 16, 24 or 32";
        }
        return false;
    }

    out->prompt = mm3_assemble_prompt(out->caption, out->lyrics, &out->instrumental);

    out->gen            = MM3GenRequest{};
    out->gen.prompt     = out->prompt;
    out->gen.max_frames = out->max_frames;
    out->gen.seed       = seed;
    out->gen.steps      = (int) nsteps;
    out->gen.cfg_flow   = (float) cfg;
    return true;
}

// Tokenise the assembled prompt and enforce the 5,000-token checkpoint limit
// (encoders.py:224-227). Header-only GGUF read — works on a cold server.
// The error names the ACTUAL count so a UI can tell the user how much to cut.
static bool mm3_request_tokenize(const MM3Model & m, MM3Tokenizer * tok, MM3SynthRequest * req, std::string * err) {
    if (!mm3_tokenizer_load(m, tok, err)) {
        return false;
    }
    mm3_tokenizer_encode(*tok, req->prompt, &req->gen.ids_cond);
    req->n_tokens = (int64_t) req->gen.ids_cond.size();
    if (req->n_tokens < 3) {
        if (err) {
            *err = "the assembled prompt tokenises to fewer than 3 tokens";
        }
        return false;
    }
    if (req->n_tokens > MM3_MAX_PROMPT_TOKENS) {
        if (err) {
            char buf[192];
            snprintf(buf, sizeof(buf),
                     "the assembled prompt is %lld tokens; the checkpoint limit is %d "
                     "(shorten the caption or the lyrics by about %lld tokens)",
                     (long long) req->n_tokens, MM3_MAX_PROMPT_TOKENS,
                     (long long) (req->n_tokens - MM3_MAX_PROMPT_TOKENS));
            *err = buf;
        }
        return false;
    }
    mm3_tokenizer_uncond(m.lm_cfg, req->gen.ids_cond, &req->gen.ids_uncond);
    return true;
}
