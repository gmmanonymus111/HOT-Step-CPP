#pragma once
// minimax/mm3-ar-loop.h — the MiniMax-Music3 autoregressive planning loop.
//
// HOT-Step file (does not exist upstream). Included only by minimax/mm3-server.h,
// which is itself the single hook include into tools/hot-step-server.cpp.
//
// SCOPE (increment 5b): the loop that turns a tokenised prompt into the
// [F, 8, 4096] block of per-frame hidden states the condition encoder consumes,
// plus the RVQ codes that produced them. This is stage 1 of the pipeline end to
// end: prefill -> {sample semantic, depth-decode 7 acoustic, feed back} x F.
// What is still missing for a full render is the orchestration above it — chunk
// windowing, condition encoding, flow sampling, vocoding — all of which already
// exist as validated modules.
//
// ── The loop, from the reference (diffusers `encoders.py` @ dafe3733) ─────────
//
//   text_embeds = embed_tokens(text_ids)                  # [2, T, H]
//   out         = lm(inputs_embeds=text_embeds, use_cache=True)
//   last_hidden = out.last_hidden_state[:, -1]            # [2, H]
//
//   for i in range(max_frames + 1):
//       logits  = lm_head(last_hidden).float()            # [2, V]
//       logits  = logits.masked_fill(vocab_mask, -inf)    # keep semantic + EOS
//       guided  = uncond + (cond - uncond) * 1.5
//       thresh  = topk(cond, 50).values[..., -1]
//       guided  = guided.masked_fill(cond < thresh, -inf)
//       guided  = guided.masked_fill(vocab_mask, -inf)    # CFG on two -inf -> NaN
//       sampled = sample_top_k(guided)                    # top-50 AGAIN, then multinomial
//       if sampled == 151670: break                       # EOS
//       codes, depth_hidden = depth_decode(last_hidden, sampled - 151675)
//       if i > 0:
//           frame_hiddens.append(cat(last_hidden[:1], depth_hidden))
//           if len(frame_hiddens) >= max_frames: break
//       feedback    = embed_audio_frame(codes)            # [2, 1, H]
//       out         = lm(inputs_embeds=feedback, past_key_values=..., use_cache=True)
//       last_hidden = out.last_hidden_state[:, -1]
//
// ── The five things that are easy to get wrong ───────────────────────────────
//
// 1. ITERATION 0 IS FED BACK BUT NOT EMITTED. Its codes are sampled, depth-decoded
//    and pushed into the LM's history, and then its hidden states are DISCARDED.
//    Emitted frame j is iteration j+1. The fixtures encode this twice over:
//    `codes_semantic_all` has 301 entries for 300 emitted frames, and
//    `codes_semantic_emitted == codes_semantic_all[1:]`. Get this off by one and
//    the audio is a frame out of step with its own conditioning — subtle, and it
//    would survive every per-module parity check.
//
// 2. THE MAX-FRAMES BREAK COMES BEFORE THE FEEDBACK. The final emitted frame never
//    runs a decode step, so the last iteration has no feedback embedding at all.
//    (The dump for that iteration carries zeros, not stale data.)
//
// 3. TOP-K IS A DOUBLE FILTER, AND THE TWO PASSES RANK BY DIFFERENT THINGS. The
//    first keeps the CONDITIONAL row's top 50 (ties survive, so 50..55 candidates
//    in practice); the second, inside the sampler, keeps the top 50 of what is
//    left ranked by the GUIDED value. Collapsing them into one filter changes the
//    candidate set. `mm3-sample.h` owns the second pass.
//
// 4. THE UNCONDITIONAL ROW IS NOT AN EMPTY PROMPT. It is the conditional token
//    sequence with everything between index 0 and the final two replaced by
//    token 151654, so it keeps the same length and the same RoPE positions.
//    `mm3-tokenizer.h` owns that; the loop just prefills both rows.
//
// 5. THE VOCAB MASK IS PART OF THE CONTRACT, NOT A SAFETY NET. Only the 16384
//    semantic codes and the EOS token may be sampled; the other 183615 vocab
//    entries are masked to -inf on BOTH rows before CFG. This loop never
//    materialises them: it gathers the 16385 candidate logits straight out of the
//    head output, which is arithmetically identical (every masked entry would be
//    -inf, hence below any threshold and outside every top-k) and turns a
//    400000-element sweep per frame into a 32770-element one.
//
// ── Sampling determinism ─────────────────────────────────────────────────────
//
// `torch.multinomial` with a torch Generator cannot be reproduced in C++, so a
// seeded run here does NOT reproduce the reference's code sequence. Parity is
// validated with the sampler bypassed: `forced_semantic` / `forced_acoustic` feed
// the fixture's own codes so the LM and the depth decoder see exactly the token
// sequence the reference saw, and the LOGITS are then comparable. The seeded path
// is validated separately for determinism (same seed twice -> identical codes),
// range, and absence of NaN. Same protocol the depth-decoder increment used.
//
// ── Validation, measured 2026-08-13 (RTX 5090, f16 GGUF) ─────────────────────
//
// THE OFF-BY-ONE IS PROVEN, NOT ASSERTED. A forced 301-iteration replay of the
// fixture's own codes produced a [300, 8, 4096] block; its first 200 frames
// against `cond_in_w0.bin` (the condition encoder's real input for window 0):
//
//     corr 0.9998494   rel RMSE 1.737e-2   (per layer 1.43e-2 .. 2.02e-2, flat)
//
// 1.74e-2 is, to three digits, the flow DiT's independently measured bf16-dump
// floor — so the whole stage-1 chain (prefill -> 301 x {LM step, depth decode,
// feedback}) reproduces the reference to within the capture's own precision.
// Re-running the SAME comparison one frame out of alignment gives 8.50e-1, a 49x
// worse fit: the emission indexing is right, and it is the kind of bug that would
// otherwise pass every per-module check.
//
// Sampled path (seed 1234, 30 frames): identical codes across two runs, different
// codes for a different seed, all semantic codes in [0, 16384), all acoustic in
// [0, 1024), zero non-finite logits. A 7500-frame request hit EOS naturally at
// frame 1200 and stopped there, exercising the stop token.
//
// Cost per emitted frame, steady state over 1200 frames:
//     LM decode step   15.26 ms
//     depth decode      9.20 ms   (7 codebook graphs; matches the standalone
//                                  increment's 9.2 ms exactly)
//     host (mask, CFG, double top-k, sampling)   0.22 ms
//     ------------------------------------------------
//     total            24.70 ms/frame
//
// At 25 fps that is 0.62 s of wall clock per second of audio — 1.6x faster than
// realtime for the planning stage, against the reference's 539 ms/frame on CPU.
// The depth decoder is 37 % of it despite being 7 % of the parameters, for the
// reason its own header gives: seven sweeps of 0.6 B weights over <=8 tokens is
// launch-bound, not bandwidth-bound. That is where the next speed lever is.

#include "mm3-depth-graph.h"
#include "mm3-lm-graph.h"
#include "mm3-model.h"
#include "mm3-sample.h"
#include "mm3-tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

// Per-iteration parity capture. Only the first `dump_iters` iterations are kept —
// the blocks are ~370 kB each and the loop runs up to 9000 times.
struct MM3ArDump {
    std::vector<float> last_hidden;   // [2, H]      rows [cond, uncond]
    std::vector<float> sem_logits;    // [2, SV]     RAW head logits over the semantic slice
    std::vector<float> guided;        // [SV]        post-CFG, post-double-filter (-inf where masked)
    std::vector<float> feedback;      // [2, H]      rows identical; zeros if no feedback ran
    std::vector<float> depth_hidden;  // [NC, H]     conditional row only
};

struct MM3ArResult {
    int64_t n_iterations = 0;  // AR steps actually run (emitted frames + 1, unless EOS)
    int64_t n_frames     = 0;  // emitted frames F
    int64_t hidden_dim   = 0;  // H
    int64_t n_codebooks  = 0;  // 8 = 1 semantic + 7 acoustic
    int64_t sem_vocab    = 0;  // SV
    bool    eos_hit      = false;

    std::vector<int32_t> semantic_all;    // [I]        includes the un-emitted iteration 0
    std::vector<int32_t> acoustic_all;    // [I, NC]
    std::vector<float>   frame_hiddens;   // [F, NC+1, H] layer-major: LM hidden then 7 depth hiddens
    std::vector<float>   prefill_hidden;  // [2, H]
    std::vector<MM3ArDump> dumps;

    // diagnostics
    int64_t nonfinite_logits = 0;  // candidate logits that were NaN/+inf across the whole run
    double  prefill_ms       = 0.0;
    double  lm_ms            = 0.0;  // decode steps only
    double  depth_ms         = 0.0;
    double  host_ms          = 0.0;  // masking, CFG, top-k, sampling
    double  total_ms         = 0.0;
    int64_t lm_steps         = 0;
};

struct MM3ArOptions {
    int64_t  max_frames = 300;
    uint64_t seed       = 42;

    // Forced-replay parity mode. Both arrays are indexed by ITERATION (so entry 0
    // is the un-emitted iteration 0), and `forced_len` caps the loop.
    const int32_t * forced_semantic = nullptr;  // [forced_len]
    const int32_t * forced_acoustic = nullptr;  // [forced_len, NC]
    int64_t         forced_len      = 0;

    int64_t dump_iters      = 0;
    bool    collect_hiddens = true;

    // Called after every emitted frame. Cheap; used for server-side progress.
    std::function<void(int64_t /*frames*/, int64_t /*max_frames*/)> on_frame;

    // Returns true to abort. Polled once per AR iteration — the finest grain
    // this loop has, and the only one that matters: at ~25 frames of real audio
    // per second of wall clock, a 60 s song is thousands of poll points. On
    // abort mm3_ar_plan() returns false with *err == MM3_ERR_CANCELLED so the
    // caller can tell a user cancel from a real failure.
    std::function<bool()> should_cancel;
};

// The sentinel a cancelled run reports. Compared by value, not by prefix, so a
// genuine error can never be mistaken for a cancel.
#define MM3_ERR_CANCELLED "cancelled"

static MM3LmGraph g_mm3_lm;

// Plan one song. `cond_ids` / `uncond_ids` are the two prefill rows.
//
// Not thread-safe: the caller serialises (mm3-server.h holds g_mm3_mutex).
static bool mm3_ar_plan(const MM3Model & m, const int32_t * cond_ids, const int32_t * uncond_ids, int64_t n_prompt,
                        const MM3ArOptions & opt, MM3ArResult * out, std::string * err) {
    const MM3LmConfig & c  = m.lm_cfg;
    const int64_t       H  = (int64_t) c.embedding_length;
    const int64_t       V  = (int64_t) c.vocab_size;
    const int64_t       SV = (int64_t) c.semantic_vocab_size;
    const int64_t       NC = (int64_t) c.num_codebooks - 1;
    const int64_t       AV = (int64_t) c.acoustic_vocab_size;
    const int64_t       OFF  = (int64_t) c.semantic_vocab_offset;
    const int64_t       EOS  = (int64_t) c.eos_audio;
    const float         CFG  = c.ar_cfg_scale > 0.0f ? c.ar_cfg_scale : 1.5f;
    const int           TOPK = c.ar_top_k > 0 ? (int) c.ar_top_k : 50;

    if (n_prompt <= 0) {
        if (err) {
            *err = "the prompt tokenised to zero tokens";
        }
        return false;
    }
    if (c.max_prompt_tokens > 0 && n_prompt > (int64_t) c.max_prompt_tokens) {
        if (err) {
            *err = "the prompt is " + std::to_string((long long) n_prompt) + " tokens; the checkpoint's limit is " +
                   std::to_string(c.max_prompt_tokens);
        }
        return false;
    }
    if (EOS < 0 || EOS >= V || OFF + SV > V) {
        if (err) {
            *err = "the LM vocabulary metadata does not cover the semantic range and EOS";
        }
        return false;
    }

    int64_t max_frames = opt.max_frames;
    if (max_frames <= 0) {
        if (err) {
            *err = "max_frames must be positive";
        }
        return false;
    }
    if (c.max_audio_frames > 0 && max_frames > (int64_t) c.max_audio_frames) {
        max_frames = (int64_t) c.max_audio_frames;
    }
    const bool forced = opt.forced_semantic != nullptr;
    if (forced) {
        if (!opt.forced_acoustic || opt.forced_len <= 0) {
            if (err) {
                *err = "forced replay needs both forced_semantic and forced_acoustic, and a positive length";
            }
            return false;
        }
        if (opt.forced_len - 1 < max_frames) {
            max_frames = opt.forced_len - 1;  // entry 0 is the un-emitted iteration
        }
        if (max_frames <= 0) {
            if (err) {
                *err = "forced replay needs at least 2 iterations (one un-emitted, one emitted)";
            }
            return false;
        }
    }

    // KV budget: prompt + one iteration per emitted frame + the un-emitted first
    // one, plus slack so the last decode step never sits exactly on the boundary.
    if (!mm3_lm_prepare(m, &g_mm3_lm, n_prompt + max_frames + 2, err)) {
        return false;
    }

    *out             = MM3ArResult{};
    out->hidden_dim  = H;
    out->n_codebooks = NC + 1;
    out->sem_vocab   = SV;

    std::vector<float> hidden((size_t) (H * MM3_LM_CFG_ROWS));
    std::vector<float> logits((size_t) (V * MM3_LM_CFG_ROWS));
    std::vector<float> feedback((size_t) H);

    const auto t_start = std::chrono::steady_clock::now();
    {
        const auto t0 = std::chrono::steady_clock::now();
        if (!mm3_lm_prefill(m, &g_mm3_lm, cond_ids, uncond_ids, n_prompt, hidden.data(), logits.data(), err)) {
            return false;
        }
        out->prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    out->prefill_hidden.assign(hidden.begin(), hidden.end());

    // Candidate layout: index 0 is EOS, indices 1..SV are semantic codes 0..SV-1.
    // Everything else in the 200000-entry vocabulary is masked and never gathered.
    const int64_t      NCAND = SV + 1;
    std::vector<float> cand_cond((size_t) NCAND);
    std::vector<float> cand_unc((size_t) NCAND);
    std::vector<float> cand_guided((size_t) NCAND);
    std::vector<float> sel_scratch;
    std::vector<float> samp_scratch;

    std::mt19937_64      rng(opt.seed);
    std::vector<int32_t> ac_rows((size_t) NC);
    MM3DepthFrame        frame;

    out->semantic_all.reserve((size_t) (max_frames + 1));
    out->acoustic_all.reserve((size_t) ((max_frames + 1) * NC));
    if (opt.collect_hiddens) {
        out->frame_hiddens.reserve((size_t) (max_frames * (NC + 1) * H));
    }

    for (int64_t it = 0; it <= max_frames; it++) {
        if (forced && it >= opt.forced_len) {
            break;
        }
        const auto t_host0 = std::chrono::steady_clock::now();

        // ── gather the candidate logits, both rows ──
        const float * lrow_c = logits.data();
        const float * lrow_u = logits.data() + V;
        auto          fix    = [&](float x) -> float {
            if (std::isnan(x) || (std::isinf(x) && x > 0.0f)) {
                out->nonfinite_logits++;
                return -INFINITY;
            }
            return x;
        };
        cand_cond[0] = fix(lrow_c[EOS]);
        cand_unc[0]  = fix(lrow_u[EOS]);
        for (int64_t j = 0; j < SV; j++) {
            cand_cond[(size_t) (j + 1)] = fix(lrow_c[OFF + j]);
            cand_unc[(size_t) (j + 1)]  = fix(lrow_u[OFF + j]);
        }

        // ── CFG, then the first top-k filter (ranked by the CONDITIONAL row) ──
        for (int64_t i = 0; i < NCAND; i++) {
            const float u          = cand_unc[(size_t) i];
            cand_guided[(size_t) i] = u + (cand_cond[(size_t) i] - u) * CFG;
        }
        {
            int64_t k = TOPK < NCAND ? (int64_t) TOPK : NCAND;
            if (k < 1) {
                k = 1;
            }
            float threshold = -INFINITY;
            if (k < NCAND) {
                sel_scratch = cand_cond;
                std::nth_element(sel_scratch.begin(), sel_scratch.begin() + (size_t) (k - 1), sel_scratch.end(),
                                 std::greater<float>());
                threshold = sel_scratch[(size_t) (k - 1)];
            }
            for (int64_t i = 0; i < NCAND; i++) {
                // Strictly less: ties at the threshold survive, which is why the
                // reference dumps carry 50..55 finite entries and not exactly 50.
                if (cand_cond[(size_t) i] < threshold) {
                    cand_guided[(size_t) i] = -INFINITY;
                }
            }
        }

        // ── sample (or replay) ──
        int32_t semantic;
        if (forced) {
            semantic = opt.forced_semantic[it];
            if (semantic < 0 || (int64_t) semantic >= SV) {
                if (err) {
                    *err = "forced semantic code " + std::to_string(semantic) + " at iteration " +
                           std::to_string((long long) it) + " is outside [0, " + std::to_string((long long) SV) + ")";
                }
                return false;
            }
        } else {
            const int64_t idx = mm3_sample_top_k(cand_guided.data(), NCAND, TOPK, rng, &samp_scratch);
            if (idx == 0) {
                out->eos_hit = true;
                out->host_ms +=
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_host0).count();
                break;
            }
            semantic = (int32_t) (idx - 1);
        }
        out->host_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_host0).count();

        // ── depth decoder: the seven acoustic codes and their hidden states ──
        const int32_t * forced_ac = forced ? opt.forced_acoustic + it * NC : nullptr;
        if (!mm3_depth_decode_frame(m, hidden.data(), hidden.data() + H, semantic, forced_ac, &frame, err,
                                    forced ? nullptr : &rng, TOPK)) {
            return false;
        }
        out->depth_ms += frame.ms;
        if (frame.n_codes != (int) NC) {
            if (err) {
                *err = "the depth decoder returned " + std::to_string(frame.n_codes) + " codes, expected " +
                       std::to_string((long long) NC);
            }
            return false;
        }

        out->semantic_all.push_back(semantic);
        for (int64_t i = 0; i < NC; i++) {
            out->acoustic_all.push_back(frame.codes[i]);
        }
        out->n_iterations++;

        const bool dumping = (int64_t) out->dumps.size() < opt.dump_iters;
        if (dumping) {
            MM3ArDump d;
            d.last_hidden.assign(hidden.begin(), hidden.end());
            d.sem_logits.resize((size_t) (2 * SV));
            memcpy(d.sem_logits.data(), lrow_c + OFF, (size_t) SV * sizeof(float));
            memcpy(d.sem_logits.data() + SV, lrow_u + OFF, (size_t) SV * sizeof(float));
            d.guided.assign(cand_guided.begin() + 1, cand_guided.end());
            d.feedback.assign((size_t) (2 * H), 0.0f);
            d.depth_hidden = frame.hiddens;
            out->dumps.push_back(std::move(d));
        }

        // ── emit (iteration 0 is fed back but never emitted) ──
        if (it > 0) {
            if (opt.collect_hiddens) {
                out->frame_hiddens.insert(out->frame_hiddens.end(), hidden.begin(), hidden.begin() + H);
                out->frame_hiddens.insert(out->frame_hiddens.end(), frame.hiddens.begin(), frame.hiddens.end());
            }
            out->n_frames++;
            if (opt.on_frame) {
                opt.on_frame(out->n_frames, max_frames);
            }
            if (opt.should_cancel && opt.should_cancel()) {
                if (err) {
                    *err = MM3_ERR_CANCELLED;
                }
                return false;
            }
            if (out->n_frames >= max_frames) {
                break;  // note 2: no feedback, no decode step, for the last frame
            }
        }

        // ── feed back and advance ──
        for (int64_t i = 0; i < NC; i++) {
            ac_rows[(size_t) i] = frame.codes[i] + (int32_t) (i * AV);
        }
        {
            const auto t0 = std::chrono::steady_clock::now();
            if (!mm3_lm_decode(m, &g_mm3_lm, semantic + (int32_t) OFF, ac_rows.data(), hidden.data(), logits.data(),
                               feedback.data(), err)) {
                return false;
            }
            out->lm_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            out->lm_steps++;
        }
        if (dumping) {
            MM3ArDump & d = out->dumps.back();
            // The reference dumps [2, H]; both rows are the same vector because
            // the sampled codes are shared across the CFG pair.
            memcpy(d.feedback.data(), feedback.data(), (size_t) H * sizeof(float));
            memcpy(d.feedback.data() + H, feedback.data(), (size_t) H * sizeof(float));
        }
    }

    out->total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();

    if (out->n_frames == 0) {
        if (err) {
            *err = out->eos_hit ? "the LM emitted EOS on the first iteration; zero audio frames were generated"
                                : "zero audio frames were generated";
        }
        return false;
    }

    fprintf(stderr,
            "[MM3-AR] %lld frames (%lld iterations%s) in %.0f ms — prefill %.0f, LM %.0f (%lld steps, %.1f ms/step), "
            "depth %.0f (%.1f ms/frame), host %.0f\n",
            (long long) out->n_frames, (long long) out->n_iterations, out->eos_hit ? ", EOS" : "", out->total_ms,
            out->prefill_ms, out->lm_ms, (long long) out->lm_steps,
            out->lm_steps ? out->lm_ms / (double) out->lm_steps : 0.0, out->depth_ms,
            out->n_iterations ? out->depth_ms / (double) out->n_iterations : 0.0, out->host_ms);
    if (out->nonfinite_logits) {
        fprintf(stderr, "[MM3-AR] WARNING: %lld non-finite candidate logits were clamped to -inf\n",
                (long long) out->nonfinite_logits);
    }
    return true;
}
