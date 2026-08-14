#pragma once
// minimax/mm3-tokenizer.h — prompt tokenisation for MiniMax-Music3.
//
// HOT-Step file (does not exist upstream). Included only by minimax/mm3-ar-loop.h
// (and transitively minimax/mm3-server.h), the single hook into upstream code.
//
// SCOPE (increment 5): turn the assembled prompt STRING into the two token id
// rows the AR stage prefills — conditional and unconditional. Assembling that
// string from a caption + lyrics (the reference's `_clean_caption` /
// `_normalize_lyrics` text hygiene) is deliberately NOT here; it is pure string
// munging with no model coupling and it belongs with the request translator in a
// later increment. What lives here is the part that must agree with the
// checkpoint bit for bit.
//
// ── Why the engine's own tokenizer is reused ─────────────────────────────────
//
// `engine/src/bpe.h` is a self-contained Qwen3/GPT-2 byte-level BPE with a
// GGUF loader that reads the llama.cpp-standard `tokenizer.ggml.tokens` and
// `tokenizer.ggml.merges` arrays — exactly what `convert-mm3.py` writes into
// mm3-lm-*.gguf (layout doc §6: 200000 tokens, 151387 merges, `model = gpt2`,
// `pre = qwen2`). It has NO global state — `model-store.cpp` already keeps
// several live instances keyed by path — so a second instance over the MM3 vocab
// is free. No fork, no vendored copy, no Python.
//
// Two gaps had to be closed on top of it, both here rather than in bpe.h so the
// ACE path cannot be perturbed:
//
// 1. ADDED-TOKEN MATCHING. `bpe_encode` only recognises the literal
//    `<|endoftext|>`; the ACE path pushes every other special token's id by hand
//    (`prompt.h`). The MM3 prompt carries EIGHT specials, so this file builds a
//    longest-match table from the vocab itself — ids [151643, semantic_vocab_offset)
//    are precisely `tokenizer.json`'s 32 added tokens (layout doc §6), and their
//    vocab strings are their literal spellings because every byte in `<|...|>` is
//    printable ASCII and maps to itself under the GPT-2 byte encoder.
//    Deriving the table from the file rather than hardcoding nine ids means a
//    checkpoint that renumbers a token cannot silently mis-tokenise.
//
// 2. `add_eos = false`. bpe_encode's default appends `<|endoftext|>`; the MM3
//    template ends at `<|audio_start|>` and an extra token would shift every
//    RoPE position.
//
// ── The pre-tokenizer question, settled empirically ──────────────────────────
//
// The GGUF says `tokenizer.ggml.pre = qwen2`, which in llama.cpp means the
// tiktoken-style pattern with `\p{N}{1,3}` (digits grouped in threes). bpe.h
// implements the CLASSIC GPT-2 pattern with `\p{N}` (one digit at a time). Those
// disagree — but the reference pipeline instantiates `Qwen2Tokenizer`, the SLOW
// transformers tokenizer, whose `PRETOKENIZE_REGEX` is the single-digit form.
// The fixture settles it: `tok_ids_cond.bin` tokenises " 120" as
// [220, 16, 17, 15] = ["Ġ", "1", "2", "0"] — one digit per token. bpe.h's
// pre-tokenizer is therefore the CORRECT one for this checkpoint, and the GGUF's
// `pre` KV (written for llama.cpp's benefit) is the misleading one. Do not
// "fix" this to match the KV.
//
// Validation: `POST /mm3/lm-plan {"prompt": <tok_prompt_template.txt>,
// "tokenize_only": true}` must return `tok_ids_cond.bin` and `tok_ids_uncond.bin`
// verbatim. That path works COLD (header-only GGUF read, no 23 GB warm).

#include "mm3-model.h"

#include "bpe.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct MM3Tokenizer {
    BPETokenizer bpe;
    bool         loaded = false;
    std::string  path;  // the GGUF the vocab came from; identity for the cache

    // Added tokens, longest first so a longest-match scan is a linear probe.
    std::vector<std::pair<std::string, int32_t>> specials;
};

// Load (once) from the LM GGUF. Header-only read — no weights, no VRAM, so this
// is safe to call on a cold server.
static bool mm3_tokenizer_load(const MM3Model & m, MM3Tokenizer * t, std::string * err) {
    if (t->loaded && t->path == m.lm_file.path) {
        return true;
    }
    if (!m.lm_file.found) {
        if (err) {
            *err = "the MiniMax-Music3 LM GGUF was not found; nothing to tokenise with";
        }
        return false;
    }

    *t = MM3Tokenizer{};
    if (!load_bpe_from_gguf(&t->bpe, m.lm_file.path.c_str())) {
        if (err) {
            *err = "mm3-lm GGUF carries no tokenizer.ggml.tokens / .merges arrays";
        }
        return false;
    }

    // Added tokens = the ids between the last ordinary BPE token and the first
    // audio code. Everything at or above semantic_vocab_offset is a synthetic
    // `<|audio_code_N|>` name the AR loop reaches arithmetically and must never
    // match in text.
    const int32_t hi = (int32_t) m.lm_cfg.semantic_vocab_offset;
    const int32_t lo = hi > 32 ? hi - 32 : 0;
    for (int32_t id = lo; id < hi && id < (int32_t) t->bpe.id_to_str.size(); id++) {
        const std::string & s = t->bpe.id_to_str[(size_t) id];
        if (s.size() >= 4 && s.compare(0, 2, "<|") == 0 && s.compare(s.size() - 2, 2, "|>") == 0) {
            t->specials.emplace_back(s, id);
        }
    }
    std::sort(t->specials.begin(), t->specials.end(),
              [](const std::pair<std::string, int32_t> & a, const std::pair<std::string, int32_t> & b) {
                  return a.first.size() > b.first.size();
              });

    t->loaded = true;
    t->path   = m.lm_file.path;
    fprintf(stderr, "[MM3-Tok] Loaded %d vocab entries, %zu added tokens from %s\n", t->bpe.n_vocab,
            t->specials.size(), m.lm_file.name.c_str());
    return true;
}

// Encode the assembled prompt template. Added tokens are matched longest-first at
// every position; everything between them goes through the byte-level BPE.
static void mm3_tokenizer_encode(const MM3Tokenizer & t, const std::string & text, std::vector<int32_t> * out) {
    out->clear();

    auto flush_text = [&](size_t from, size_t to) {
        if (to <= from) {
            return;
        }
        const std::string segment = text.substr(from, to - from);
        std::vector<int>  ids;
        for (const auto & chunk : gpt2_pre_tokenize(segment)) {
            encode_chunk(&t.bpe, chunk, ids);
        }
        for (int id : ids) {
            out->push_back((int32_t) id);
        }
    };

    size_t seg_start = 0;
    size_t i         = 0;
    while (i < text.size()) {
        if (text[i] != '<') {  // every added token starts "<|"
            i++;
            continue;
        }
        const std::pair<std::string, int32_t> * hit = nullptr;
        for (const auto & sp : t.specials) {
            if (text.compare(i, sp.first.size(), sp.first) == 0) {
                hit = &sp;  // specials are length-sorted, so the first hit is the longest
                break;
            }
        }
        if (!hit) {
            i++;
            continue;
        }
        flush_text(seg_start, i);
        out->push_back(hit->second);
        i += hit->first.size();
        seg_start = i;
    }
    flush_text(seg_start, text.size());
}

// The unconditional row. Reference (`encoders.py`):
//     unconditional_ids = input_ids.clone()
//     unconditional_ids[:, 1:-2] = _AUDIO_CFG_TOKEN_ID
// i.e. token 0 (`<|im_start|>`) and the final TWO (`<|im_end|>`, `<|audio_start|>`)
// survive; every id in between becomes 151654. Exactly three real tokens remain,
// which is why the unconditional branch carries no caption and no lyrics but still
// sits at the same sequence length — the CFG rows must stay position-aligned.
static void mm3_tokenizer_uncond(const MM3LmConfig & c, const std::vector<int32_t> & cond,
                                 std::vector<int32_t> * uncond) {
    *uncond = cond;
    if (cond.size() < 4) {
        return;  // nothing strictly between index 0 and the last two
    }
    for (size_t i = 1; i + 2 < cond.size(); i++) {
        (*uncond)[i] = (int32_t) c.tok_audio_cfg;
    }
}
