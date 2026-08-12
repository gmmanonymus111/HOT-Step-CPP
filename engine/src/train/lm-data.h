#pragma once
// lm-data.h — lm_codes.jsonl -> training sequences, and the sparse one-hot
// label helpers.
//
// The single most important function here is `lm_build_sequence()` (L6a):
//
//   S_tr        = S - n_masked          (NOT "- 1")
//   targets[i]  = tokens[n_masked + i]  for i in [0, s_tr)
//   logits come from hidden columns [n_masked-1, S-2]
//
// Side-Step trains position n_masked-1 (it must learn to emit the FIRST
// <think> token) — train_lm.py:88-95, 152-160: the chunked loss uses positions
// [i,j) to predict labels[i+1:j+1), and n_tok = (labels[:,1:] != -100).sum()
// = S - n_masked. The Phase-0 spike missed exactly that one token.
//
// Over-length samples are SKIPPED, never truncated (L5, train_lm.py:98-105).
//
// docs/plans/2026-07-27-lm-trainer-implementation.md §3.3

#include "bpe.h"
#include "prompt.h"
#include "train/lm-common.h"
#include "train/lm-extract.h"

#include <string>
#include <vector>

struct LmSample {
    std::string          file;
    std::vector<int32_t> tokens;    // full sequence
    std::vector<int32_t> targets;   // s_tr entries; targets[i] = tokens[n_masked + i]
    int                  n_masked = 0;
    int                  s_tr     = 0;  // == (int)tokens.size() - n_masked
};

// Load lm_codes.jsonl into rows (thin wrapper over lm-extract.h's reader).
static bool lm_load_codes(const char * path, std::vector<LmCodeRow> * out, std::string * err) {
    if (!pm_file_exists(path)) {
        *err = std::string("codes file not found: ") + path;
        return false;
    }
    std::string warn;
    if (!lm_codes_read_file(path, out, &warn)) {
        *err = std::string("cannot read ") + path;
        return false;
    }
    if (!warn.empty()) {
        lm_log("warn", warn);
    }
    if (out->empty()) {
        *err = std::string("no usable rows in ") + path;
        return false;
    }
    return true;
}

static inline AcePrompt lm_prompt_from_row(const LmCodeRow & r) {
    AcePrompt p;
    p.caption        = r.caption;
    p.lyrics         = r.lyrics;
    p.duration       = (float) r.duration;
    p.bpm            = r.bpm;
    p.keyscale       = r.keyscale;
    // Older lm_codes.jsonl rows still store '4/4'; normalize at prompt build
    // so retraining them conditions like inference without a re-extract.
    p.timesignature  = pm_timesig_numerator(r.timesignature);
    p.vocal_language = r.language;
    return p;
}

// Build one training sequence. Returns false (with *why_skipped set) when the
// sample must be skipped.
// `over_length` (optional) distinguishes the L5 over-length skip — the only one
// `skippedLong` is allowed to count (§2.2) — from the structural rejections
// (empty prompt / nothing trainable / n_masked < 1), which point at malformed
// data, not at --max-len.
static bool lm_build_sequence(BPETokenizer & bpe, const LmCodeRow & r, bool loss_on_cot, int max_len, LmSample * out,
                              std::string * why_skipped, bool * over_length = nullptr) {
    *out = LmSample{};
    out->file = r.file;
    if (over_length) {
        *over_length = false;
    }

    const AcePrompt        p   = lm_prompt_from_row(r);
    const std::string      cot = build_cot_yaml(p);
    const std::vector<int> pre = build_lm_prompt_with_cot(bpe, p, cot);

    if (pre.empty()) {
        *why_skipped = "empty prompt";
        return false;
    }

    // n_think = index of TOKEN_THINK inside `pre`; pre.size() when absent.
    int n_think = (int) pre.size();
    for (size_t i = 0; i < pre.size(); i++) {
        if (pre[i] == TOKEN_THINK) {
            n_think = (int) i;
            break;
        }
    }

    out->tokens.assign(pre.begin(), pre.end());
    for (size_t i = 0; i < r.codes.size(); i++) {
        out->tokens.push_back((int32_t) (AUDIO_CODE_BASE + r.codes[i]));
    }
    out->tokens.push_back((int32_t) TOKEN_IM_END);

    out->n_masked = loss_on_cot ? n_think : (int) pre.size();
    out->s_tr     = (int) out->tokens.size() - out->n_masked;  // L6(a): NO "- 1"

    // L5: SKIP, never truncate.
    if ((int) out->tokens.size() > max_len) {
        char b[128];
        snprintf(b, sizeof(b), "%d tokens > max_len %d", (int) out->tokens.size(), max_len);
        *why_skipped = b;
        if (over_length) {
            *over_length = true;
        }
        return false;
    }
    if (out->s_tr <= 0) {
        *why_skipped = "nothing trainable";
        return false;
    }
    // The prompt always begins with an im_start turn, so the first trained
    // position (n_masked-1) never underflows. Assert it anyway.
    if (out->n_masked < 1) {
        *why_skipped = "n_masked < 1";
        return false;
    }

    out->targets.resize((size_t) out->s_tr);
    for (int i = 0; i < out->s_tr; i++) {
        out->targets[(size_t) i] = out->tokens[(size_t) (out->n_masked + i)];
    }
    return true;
}

// ─── sparse-write dense labels (L3) ─────────────────────────────────────────
//
// t_lab is one persistent zeroed [V, S_TR_MAX] F32 buffer. Per micro-step we
// write exactly s_tr single floats (1.0f) and then write them back to 0.0f.
// ~2*s_tr 4-byte H2D transfers instead of a ~1 GB blanket upload.
//
// lm_labels_clear MUST run before the next sample's set, in the same order,
// including on the error path — a stale 1.0 poisons every later step.

static inline void lm_labels_write(ggml_tensor * t_lab, const int32_t * tgt, int s_tr, int V, float value) {
    for (int i = 0; i < s_tr; i++) {
        const int32_t c = tgt[i];
        if (c < 0 || c >= V) {
            continue;  // guarded at build time; never write out of range
        }
        const size_t off = ((size_t) i * (size_t) V + (size_t) c) * sizeof(float);
        ggml_backend_tensor_set(t_lab, &value, off, sizeof(float));
    }
}

static inline void lm_labels_set(ggml_tensor * t_lab, const int32_t * tgt, int s_tr, int V) {
    lm_labels_write(t_lab, tgt, s_tr, V, 1.0f);
}

static inline void lm_labels_clear(ggml_tensor * t_lab, const int32_t * tgt, int s_tr, int V) {
    lm_labels_write(t_lab, tgt, s_tr, V, 0.0f);
}

// RAII guard: the clear runs on every exit path, including exceptions/returns.
struct LmLabelGuard {
    ggml_tensor *   t   = nullptr;
    const int32_t * tgt = nullptr;
    int             s_tr = 0, V = 0;

    LmLabelGuard(ggml_tensor * t_lab, const int32_t * targets, int n, int vocab)
        : t(t_lab), tgt(targets), s_tr(n), V(vocab) {
        lm_labels_set(t, tgt, s_tr, V);
    }
    ~LmLabelGuard() { lm_labels_clear(t, tgt, s_tr, V); }
    LmLabelGuard(const LmLabelGuard &)             = delete;
    LmLabelGuard & operator=(const LmLabelGuard &) = delete;
};

// ─── causal mask ────────────────────────────────────────────────────────────

// [S, S] f32, 0 / -INF. Row i = query, col j = key.
static void lm_causal_mask(int S, std::vector<float> * out) {
    out->assign((size_t) S * (size_t) S, 0.0f);
    float * m = out->data();
    for (int i = 0; i < S; i++) {
        float * row = m + (size_t) i * (size_t) S;
        for (int j = 0; j < S; j++) {
            row[j] = (j <= i) ? 0.0f : -INFINITY;
        }
    }
}

// ─── epoch order ────────────────────────────────────────────────────────────
//
// V2 (§7): this is NOT bit-identical to Python's random.Random(seed).shuffle
// (Mersenne Twister) and cannot be. `--order fixed` disables shuffling so the
// §6.5 A/B run does not depend on RNG parity.
static void lm_epoch_order(std::vector<int> * order, int n, bool shuffle, uint64_t seed, int epoch) {
    order->resize((size_t) n);
    for (int i = 0; i < n; i++) {
        (*order)[(size_t) i] = i;
    }
    if (!shuffle) {
        return;
    }
    LmRng rng;
    lm_rng_seed(&rng, seed + (uint64_t) epoch);
    lm_rng_shuffle(&rng, *order);
}
