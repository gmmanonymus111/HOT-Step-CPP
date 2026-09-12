#pragma once
// yue2/yue2-tokenizer.h — BPE wrapper + protocol assembly for YuE2.
//
// HOT-Step file (no acestep.cpp analog). Milestone M0,
// docs/plans/yue2/06-engine-port-plan.md §2. Wraps engine/src/bpe.h's
// arch-agnostic byte-level BPE primitives with the two things YuE2 needs
// that MM3/Qwen3 tokenization does not, plus protocol.py's prompt assembly
// (docs/plans/yue2/03-reference-numerics.md §1).
//
// bpe.h itself is UNTOUCHED by this file — build_byte_encoder/encode_chunk/
// bpe_merge/load_bpe_from_gguf/load_bpe_from_files are called, not copied,
// and are never modified, so MM3's own tokenizer parity corpus is not at
// risk (docs/plans/yue2/00-oracle-pin.md §6 / mm3-backend skill trap #2).
//
// What's deliberately NOT here: NFC normalization. Decision (per the M1
// task brief, overriding 06-engine-port-plan.md §2 item 1's "vendor
// utf8proc" recommendation): NFC happens in Node before a request reaches
// the engine. Every yue2_bpe_encode()/yue2_token_prefixes()/
// yue2_negative_prefix() call below assumes its string arguments are
// ALREADY NFC-normalized. A caller that skips that step will silently
// diverge from the reference on any input containing a decomposed (NFD)
// sequence — see the oracle-pin's "café résumé naïve" worked example.
//
// What IS here, corrected vs. bpe.h's shared bpe_encode() (00-oracle-pin.md
// §6, three of its four items — NFC is the fourth, handled by Node instead):
//   1. Widened whitespace class in the pre-tokenizer: U+3000 (ideographic
//      space, the corpus-confirmed CJK-lyrics failure) plus the
//      untested-but-flagged siblings U+2003/U+2028/U+2029/U+FEFF. Done via
//      a YuE2-local copy of gpt2_pre_tokenize() (yue2_gpt2_pre_tokenize),
//      never by widening bpe.h's shared is_whitespace() in place — MM3's
//      own tokenizer already validates against that function as it stands.
//   2. No "<|endoftext|>" literal-substring scan. The reference's
//      encode_ordinary() never special-cases any substring found inside
//      text; every special/structural token is spliced by protocol.py (and
//      by the functions below) as a raw int around the tokenizer's output.
//      yue2_bpe_encode() below simply never looks for the substring, so
//      there's nothing to gate with a flag.
//   3. No trailing EOS append. protocol.py's token_prefixes()/
//      negative_prefix() place EOD (same id as EOS, different role — see
//      below) and every other structural token explicitly; the encoder
//      itself never appends anything.
//
// Known, unfixed, latent divergence (documented, not manifested in the
// 64-string parity corpus): bpe.h's Rule 6 (reused unchanged in
// yue2_gpt2_pre_tokenize below) folds ANY whitespace run before a
// punctuation run into one chunk, where the reference regex only pulls in
// a single literal space. No corpus case currently exercises this; flagged
// per 00-oracle-pin.md §6 item 5 / 06-engine-port-plan.md §2.

#include "../bpe.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------
// Protocol constants (03-reference-numerics.md §1.1). Fixed values baked
// into this checkpoint's tokenizer/vocab layout, not config-derived --
// yue2-model.h separately reads the same numbers back out of the LM
// GGUF's `yue2.token.*` KV (Yue2LmConfig::tok_*) as a cross-check; a
// mismatch there means the GGUF disagrees with this header about which
// checkpoint revision is in play, not that one of the two is "more right".
// ---------------------------------------------------------------------
static constexpr int YUE2_EOD          = 151643;  // also BOS and PAD
static constexpr int YUE2_ABC_START    = 151847;
static constexpr int YUE2_ABC_END      = 151848;  // ABC-stage EOS
static constexpr int YUE2_MUSIC_START  = 151851;
static constexpr int YUE2_MUSIC_END    = 151852;  // semantic-stage EOS
static constexpr int YUE2_CODEC_OFFSET = 151853;
static constexpr int YUE2_CODEC_SIZE   = 32768;
static constexpr int YUE2_LATENT_START = 184621;  // training-time only, never in the live token stream
static constexpr int YUE2_LATENT_END   = 184622;
static constexpr int YUE2_LATENT_PAD   = 184623;
static constexpr int YUE2_VOCAB_SIZE   = 184704;

enum Yue2Cot {
    YUE2_COT_OFF    = 0,
    YUE2_COT_MELODY = 1,
    YUE2_COT_FULL   = 2,
};

static inline const char * yue2_cot_name(Yue2Cot cot) {
    switch (cot) {
        case YUE2_COT_OFF:
            return "off";
        case YUE2_COT_MELODY:
            return "melody";
        case YUE2_COT_FULL:
            return "full";
    }
    return "?";
}

static inline bool yue2_cot_from_name(const std::string & s, Yue2Cot * out) {
    if (s == "off") {
        *out = YUE2_COT_OFF;
        return true;
    }
    if (s == "melody") {
        *out = YUE2_COT_MELODY;
        return true;
    }
    if (s == "full") {
        *out = YUE2_COT_FULL;
        return true;
    }
    return false;
}

// protocol.py's INSTRUCTIONS dict (03-reference-numerics.md §1.2), verbatim.
static inline const char * yue2_instruction(Yue2Cot cot) {
    switch (cot) {
        case YUE2_COT_OFF:
            return "Generate music with codec tokens from the given conditions.";
        case YUE2_COT_MELODY:
            return "Generate a melody-only ABC transcription without chord symbols, then generate music with "
                   "codec tokens from the given conditions.";
        case YUE2_COT_FULL:
            return "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the "
                   "given conditions.";
    }
    return "";
}

// ---------------------------------------------------------------------
// Widened whitespace (00-oracle-pin.md §6 item 2) + a YuE2-local copy of
// bpe.h's gpt2_pre_tokenize(). is_letter/is_digit/is_newline/
// utf8_codepoint are reused from bpe.h unchanged (arch-agnostic already);
// only the whitespace predicate differs, so only the one function whose
// behavior depends on it needs its own copy.
// ---------------------------------------------------------------------
static bool yue2_is_whitespace(int cp) {
    return is_whitespace(cp) ||
           cp == 0x3000 ||  // ideographic space (CJK full-width) -- confirmed corpus failure without this
           cp == 0x2003 ||  // em space
           cp == 0x2028 ||  // line separator
           cp == 0x2029 ||  // paragraph separator
           cp == 0xFEFF;    // BOM / zero-width no-break space
}

// Verbatim copy of bpe.h's gpt2_pre_tokenize(), every is_whitespace( call
// replaced with yue2_is_whitespace(. Keep in sync by inspection if bpe.h's
// version changes -- see the file-header rationale for why this is a copy
// rather than a shared parametrized function.
static std::vector<std::string> yue2_gpt2_pre_tokenize(const std::string & text) {
    std::vector<std::string> chunks;
    const char *             s   = text.c_str();
    int                      len = (int) text.size();
    int                      i   = 0;

    while (i < len) {
        int adv;
        int cp = utf8_codepoint(s + i, &adv);

        // Rule 1: Contractions 's 't 're 've 'm 'll 'd
        if ((cp == '\'' || cp == 0x2019) && i + adv < len) {
            const char * rest      = s + i + adv;
            int          rlen      = len - i - adv;
            auto         try_match = [&](const char * suffix, int slen) -> bool {
                if (rlen >= slen) {
                    for (int k = 0; k < slen; k++) {
                        char c1 = rest[k], c2 = suffix[k];
                        if (c1 >= 'A' && c1 <= 'Z') {
                            c1 = (char) (c1 + 32);
                        }
                        if (c1 != c2) {
                            return false;
                        }
                    }
                    if (rlen > slen) {
                        int a2;
                        int cp2 = utf8_codepoint(rest + slen, &a2);
                        if (is_letter(cp2)) {
                            return false;
                        }
                    }
                    chunks.push_back(std::string(s + i, adv + slen));
                    i += adv + slen;
                    return true;
                }
                return false;
            };
            if (try_match("ll", 2)) {
                continue;
            }
            if (try_match("re", 2)) {
                continue;
            }
            if (try_match("ve", 2)) {
                continue;
            }
            if (try_match("s", 1)) {
                continue;
            }
            if (try_match("t", 1)) {
                continue;
            }
            if (try_match("m", 1)) {
                continue;
            }
            if (try_match("d", 1)) {
                continue;
            }
        }

        // Rule 2: [^\r\n\p{L}\p{N}]?\p{L}+
        if (is_letter(cp)) {
            int start = i;
            i += adv;
            while (i < len) {
                int a2;
                int cp2 = utf8_codepoint(s + i, &a2);
                if (!is_letter(cp2)) {
                    break;
                }
                i += a2;
            }
            chunks.push_back(std::string(s + start, i - start));
            continue;
        }
        if (!is_newline(cp) && !is_letter(cp) && !is_digit(cp) && !yue2_is_whitespace(cp)) {
            int start = i;
            int after = i + adv;
            if (after < len) {
                int a2;
                int cp2 = utf8_codepoint(s + after, &a2);
                if (is_letter(cp2)) {
                    i = after + a2;
                    while (i < len) {
                        int a3;
                        int cp3 = utf8_codepoint(s + i, &a3);
                        if (!is_letter(cp3)) {
                            break;
                        }
                        i += a3;
                    }
                    chunks.push_back(std::string(s + start, i - start));
                    continue;
                }
            }
        }

        // Rule 3: \p{N}+ (digits, one chunk per digit)
        if (is_digit(cp)) {
            int start = i;
            while (i < len && is_digit((unsigned char) s[i])) {
                i++;
            }
            for (int j = start; j < i; j++) {
                chunks.push_back(std::string(s + j, 1));
            }
            continue;
        }

        // Rule 5: \s*[\r\n]+
        if (is_newline(cp)) {
            int start = i;
            while (i < len && is_newline((unsigned char) s[i])) {
                i++;
            }
            chunks.push_back(std::string(s + start, i - start));
            continue;
        }

        // Rule 6: whitespace handling.
        //
        // This is a codepoint-aware rewrite of bpe.h's version, not a
        // literal copy: bpe.h's own run-continuation scan reads
        // `(unsigned char) s[ws_end]` -- a single raw BYTE -- and its
        // run-length test is `ws_end - start > 1`, a BYTE-length check.
        // Both are silently wrong for any whitespace codepoint that isn't
        // 1 byte (U+3000 is 3 bytes; even NBSP U+00A0 is 2), because they
        // never decode past the run's first codepoint. For a single such
        // character followed by a word, the byte-length check reads
        // "length 2 or 3 > 1" and wrongly treats one whitespace character
        // as a multi-character run to peel apart -- confirmed via the real
        // corpus (`ws_ideographic_space`, `ws_nbsp` both diverged from the
        // reference before this rewrite). Decoding every step with
        // utf8_codepoint() and counting whitespace CODEPOINTS (ws_count),
        // not bytes, fixes it. See the file header for why this lives here
        // rather than as a fix to bpe.h's shared function.
        if (yue2_is_whitespace(cp)) {
            int start         = i;
            int last_ws_start = i;  // byte offset of the most recently consumed ws codepoint
            int ws_end        = i + adv;
            int ws_count      = 1;
            while (ws_end < len) {
                int a2;
                int cp2 = utf8_codepoint(s + ws_end, &a2);
                if (!yue2_is_whitespace(cp2) || is_newline(cp2)) {
                    break;
                }
                last_ws_start = ws_end;
                ws_end += a2;
                ws_count++;
            }
            bool followed_by_non_ws = false;
            if (ws_end < len) {
                int a2;
                int cp2            = utf8_codepoint(s + ws_end, &a2);
                followed_by_non_ws = !yue2_is_whitespace(cp2) && !is_newline(cp2);
            }

            if (followed_by_non_ws && ws_count > 1) {
                // \s+(?!\S): peel off every whitespace codepoint but the
                // last, which is left for the branch below to combine with
                // what follows (re-entering this same Rule on the next
                // main-loop iteration, now with ws_count==1).
                chunks.push_back(std::string(s + start, last_ws_start - start));
                i = last_ws_start;
                continue;
            }
            if (followed_by_non_ws) {
                // Exactly one whitespace codepoint before a word/digit/
                // symbol run -- combine, mirroring rule 2's own
                // "[^\r\n\p{L}\p{N}]?\p{L}+" one-leading-char allowance.
                // followed_by_non_ws guarantees cp2 here is neither
                // whitespace nor newline, so this trichotomy is exhaustive.
                int wstart = start;
                i          = start + adv;
                int a2;
                int cp2 = utf8_codepoint(s + i, &a2);
                if (is_letter(cp2)) {
                    i += a2;
                    while (i < len) {
                        int a3;
                        int cp3 = utf8_codepoint(s + i, &a3);
                        if (!is_letter(cp3)) {
                            break;
                        }
                        i += a3;
                    }
                } else if (is_digit(cp2)) {
                    // \p{N} alone has no leading-char allowance -- push the
                    // whitespace codepoint by itself; the digit becomes its
                    // own chunk via Rule 3 on the next iteration.
                    chunks.push_back(std::string(s + wstart, i - wstart));
                    continue;
                } else {
                    while (i < len) {
                        int a3;
                        int cp3 = utf8_codepoint(s + i, &a3);
                        if (yue2_is_whitespace(cp3) || is_letter(cp3) || is_digit(cp3) || is_newline(cp3)) {
                            break;
                        }
                        i += a3;
                    }
                    while (i < len && is_newline((unsigned char) s[i])) {
                        i++;
                    }
                }
                chunks.push_back(std::string(s + wstart, i - wstart));
                continue;
            }
            // Trailing whitespace (end of string, or the run is cut short
            // by a newline that Rule 5 picks up next iteration) -- one chunk.
            chunks.push_back(std::string(s + start, ws_end - start));
            i = ws_end;
            continue;
        }

        // Rule 4: [^\s\p{L}\p{N}]+[\r\n]*
        {
            int start = i;
            i += adv;
            while (i < len) {
                int a2;
                int cp2 = utf8_codepoint(s + i, &a2);
                if (yue2_is_whitespace(cp2) || is_letter(cp2) || is_digit(cp2) || is_newline(cp2)) {
                    break;
                }
                i += a2;
            }
            while (i < len && is_newline((unsigned char) s[i])) {
                i++;
            }
            chunks.push_back(std::string(s + start, i - start));
        }
    }
    return chunks;
}

// ---------------------------------------------------------------------
// Encoder: text -> ordinary-vocab token ids. Equivalent to
// tokenization_yue2.py's encode_ordinary() call (post-NFC, which is the
// caller's job -- see file header). No special-substring scan, no EOS.
// encode_chunk() itself is bpe.h's, reused unchanged: it only does
// byte-level-encode + bpe_merge + vocab lookup, nothing whitespace- or
// special-token-shaped.
// ---------------------------------------------------------------------
static std::vector<int> yue2_bpe_encode(const BPETokenizer * tok, const std::string & text) {
    std::vector<int> ids;
    for (const auto & chunk : yue2_gpt2_pre_tokenize(text)) {
        encode_chunk(tok, chunk, ids);
    }
    return ids;
}

// Reverse of yue2_bpe_encode: ordinary-vocab ids (< EOD) -> UTF-8 text.
// Milestone M7 addition (not part of M0's original scope) -- needed only to
// render the plan stage's sampled ABC span back to readable text for the
// `score.abc` result artifact (docs/plans/yue2/06-engine-port-plan.md §7).
// Display-only / best-effort: an id outside the vocab, or a byte-string
// fragment this process's own build_byte_encoder() table doesn't recognise,
// is silently dropped rather than aborting the whole decode -- nothing
// downstream is numeric here, unlike every other function in this file.
static inline std::string yue2_bpe_decode(const BPETokenizer * tok, const std::vector<int32_t> & ids) {
    static std::unordered_map<std::string, uint8_t> * byte_of_str = nullptr;
    if (!byte_of_str) {
        auto * m = new std::unordered_map<std::string, uint8_t>();
        std::string byte2str[256];
        build_byte_encoder(byte2str);
        for (int b = 0; b < 256; b++) {
            (*m)[byte2str[b]] = (uint8_t) b;
        }
        byte_of_str = m;
    }
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || (size_t) id >= tok->id_to_str.size()) {
            continue;
        }
        const std::string & tstr = tok->id_to_str[(size_t) id];
        size_t              i    = 0;
        while (i < tstr.size()) {
            int adv     = 1;
            utf8_codepoint(tstr.c_str() + i, &adv);
            const std::string cp_str = tstr.substr(i, (size_t) adv);
            auto               it    = byte_of_str->find(cp_str);
            if (it != byte_of_str->end()) {
                out.push_back((char) it->second);
            }
            i += (size_t) adv;
        }
    }
    return out;
}

// Load the tokenizer straight from the LM GGUF's tokenizer.ggml.tokens/
// tokenizer.ggml.merges KV (convert-yue2.py's add_tokenizer_kv() writes
// both; load_bpe_from_gguf() is bpe.h's existing, unmodified loader).
static inline bool yue2_tokenizer_load_from_gguf(BPETokenizer * tok, const std::string & lm_gguf_path) {
    return load_bpe_from_gguf(tok, lm_gguf_path.c_str());
}

// Fallback / bring-up path: load from the sidecar tokenizer/ directory
// (vocab.json + merges.txt) engine/tools/yue2-tokenizer-convert.py writes,
// same convention every other bpe.h caller uses for HF-style directories.
static inline bool yue2_tokenizer_load_from_dir(BPETokenizer * tok, const std::string & tokenizer_dir) {
    std::string vocab_path  = tokenizer_dir + "/vocab.json";
    std::string merges_path = tokenizer_dir + "/merges.txt";
    return load_bpe_from_files(tok, vocab_path.c_str(), merges_path.c_str());
}

// ---------------------------------------------------------------------
// Protocol assembly (03-reference-numerics.md §1.2-§1.4). Every structural
// token below is spliced as a raw int around BPE output, exactly as
// protocol.py does -- never encoded from a literal string.
// ---------------------------------------------------------------------

// protocol.py's SongRequest.text(): "{instr}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"
static inline std::string yue2_assemble_text(const std::string & style, const std::string & lyrics, Yue2Cot cot) {
    std::string out;
    out.reserve(style.size() + lyrics.size() + 64);
    out += yue2_instruction(cot);
    out += "\n[Tags]\n";
    out += style;
    out += "\n[Lyrics]\n";
    out += lyrics;
    out += "\n";
    return out;
}

static inline void yue2_validate_abc_ids(const std::vector<int> & abc_ids, const char * caller) {
    for (int id : abc_ids) {
        if (id < 0 || id >= YUE2_EOD) {
            throw std::invalid_argument(std::string(caller) +
                                         ": ABC ids must remain inside the ordinary text vocabulary [0, EOD)");
        }
    }
}

// token_prefixes() -- 03-reference-numerics.md §1.3.
//
// abc_ids == nullptr: cot=off's own base case needs no ABC content at all;
//   for melody/full it means "stage 1's own ABC-planning prompt" (base +
//   [ABC_START], the prompt the AR sampler free-runs against to produce
//   the ABC ids in the first place).
// abc_ids != nullptr: melody/full's stage-2 (semantic) prefix, with the
//   ABC span (sampled, or supplied via an external --abc encode) spliced
//   in. Every id must be < EOD (protocol.py's own guard); cot=off ignores
//   this argument entirely (its base+[ABC_START,ABC_END,MUSIC_START]
//   shape never has a real ABC span to splice).
static std::vector<int> yue2_token_prefixes(const BPETokenizer * tok, const std::string & style,
                                             const std::string & lyrics, Yue2Cot cot,
                                             const std::vector<int> * abc_ids = nullptr) {
    std::vector<int> out;
    out.push_back(YUE2_EOD);
    for (int id : yue2_bpe_encode(tok, yue2_assemble_text(style, lyrics, cot))) {
        out.push_back(id);
    }

    if (cot == YUE2_COT_OFF) {
        out.push_back(YUE2_ABC_START);
        out.push_back(YUE2_ABC_END);
        out.push_back(YUE2_MUSIC_START);
        return out;
    }
    if (abc_ids == nullptr) {
        out.push_back(YUE2_ABC_START);
        return out;
    }
    yue2_validate_abc_ids(*abc_ids, "yue2_token_prefixes");
    out.push_back(YUE2_ABC_START);
    for (int id : *abc_ids) {
        out.push_back(id);
    }
    out.push_back(YUE2_ABC_END);
    out.push_back(YUE2_MUSIC_START);
    return out;
}

// negative_prefix() -- 03-reference-numerics.md §1.4. The CFG negative
// branch: same leading EOD, but the base text is JUST the bare instruction
// sentence (no [Tags]/style/[Lyrics]/lyrics wrapper at all).
//
// For melody/full, abc_ids is REQUIRED (non-null) and MUST be the exact
// same ids as the positive branch used -- protocol.py's own docstring/
// raise: "Symbolic CFG must retain the exact positive-branch ABC IDs".
// For cot=off, abc_ids is ignored -- off's negative branch never had an
// ABC bracket to begin with (base + [MUSIC_START] only).
static std::vector<int> yue2_negative_prefix(const BPETokenizer * tok, Yue2Cot cot,
                                              const std::vector<int> * abc_ids = nullptr) {
    std::vector<int> out;
    out.push_back(YUE2_EOD);
    for (int id : yue2_bpe_encode(tok, yue2_instruction(cot))) {
        out.push_back(id);
    }

    if (cot == YUE2_COT_OFF) {
        out.push_back(YUE2_MUSIC_START);
        return out;
    }
    if (abc_ids == nullptr) {
        throw std::invalid_argument(
            "yue2_negative_prefix: melody/full requires the positive branch's exact abc_ids");
    }
    yue2_validate_abc_ids(*abc_ids, "yue2_negative_prefix");
    out.push_back(YUE2_ABC_START);
    for (int id : *abc_ids) {
        out.push_back(id);
    }
    out.push_back(YUE2_ABC_END);
    out.push_back(YUE2_MUSIC_START);
    return out;
}
