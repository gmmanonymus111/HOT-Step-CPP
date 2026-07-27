#pragma once
// spike-s3.h — Rung 3: full 28-layer Qwen3-0.6B LM LoRA micro-train on a real
// prompt + real FSQ codes, with cross-entropy over the 217204-entry vocab.
//
// Two loss paths, both measured:
//   "naive"   : one lm_head over ALL trained positions -> [V, S_tr] logits +
//               [V, S_tr] dense one-hot labels, ggml-opt CROSS_ENTROPY.
//   "chunked" : Side-Step's trick ported to GGML — trunk forward once, then
//               per-chunk lm_head + CE + MANUAL dL/dhidden written into a
//               persistent [H, S] gradient buffer G (chunks are disjoint
//               column ranges so no accumulation is needed), then the trunk
//               backward is driven by the surrogate loss  sum(hidden * G),
//               whose parameter gradients are identical to the real ones.

#include "bpe.h"
#include "prompt.h"
#include "spike-lm.h"
#include "yyjson.h"

#include <algorithm>
#include <cstdio>

// ─── BF16 -> F32 weight mirror ──────────────────────────────────────────────
//
// WHY: ggml's backward for MUL_MAT w.r.t. its ACTIVATION input is
//      ggml_out_prod(W, grad^T). CUDA's OUT_PROD is F32-only
//      (ggml-cuda.cu supports_op) and the CPU fallback GGML_ABORTs for
//      F16/BF16 (ops.cpp ggml_compute_forward_out_prod). So a BF16 frozen
//      base weight cannot pass a gradient through to earlier layers.

struct SpikeF32Mirror {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    size_t                bytes = 0;
};

static ggml_tensor * s3_mirror_add(SpikeF32Mirror * M, ggml_tensor * src, std::vector<ggml_tensor *> * srcs,
                                   std::vector<ggml_tensor **> * slots, ggml_tensor ** slot, ggml_type ty) {
    if (!src) return nullptr;
    ggml_tensor * d = ggml_new_tensor_4d(M->ctx, ty, src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    ggml_set_name(d, src->name);
    srcs->push_back(src);
    slots->push_back(slot);
    M->bytes += ggml_nbytes(d);
    return d;
}

// keep_embed_native: chunked mode never runs out_prod on embed_tokens (the
// dL/dhidden contraction is a plain forward mul_mat against a transposed copy),
// so the embedding can stay in its on-disk dtype and save H*V*2 bytes.
static bool spike_lm_weights_to_f32(Qwen3LM * lm, SpikeF32Mirror * M, bool keep_embed_native = false) {
    const int L = lm->cfg.n_layers;
    const int n = 2 + L * 11 + 16;
    ggml_init_params p = { (size_t) n * ggml_tensor_overhead() + 4096, nullptr, true };
    M->ctx = ggml_init(p);

    std::vector<ggml_tensor *>  srcs;
    std::vector<ggml_tensor **> slots;
    std::vector<ggml_tensor *>  dsts;

    auto add = [&](ggml_tensor ** slot, ggml_type ty = GGML_TYPE_F32) {
        ggml_tensor * d = s3_mirror_add(M, *slot, &srcs, &slots, slot, ty);
        dsts.push_back(d);
    };
    add(&lm->embed_tokens, keep_embed_native ? lm->embed_tokens->type : GGML_TYPE_F32);
    add(&lm->final_norm);
    for (int i = 0; i < L; i++) {
        Qwen3Layer & ly = lm->layers[i];
        add(&ly.input_layernorm); add(&ly.post_attn_layernorm);
        add(&ly.q_proj); add(&ly.k_proj); add(&ly.v_proj); add(&ly.o_proj);
        add(&ly.q_norm); add(&ly.k_norm);
        add(&ly.gate_proj); add(&ly.up_proj); add(&ly.down_proj);
    }

    M->buf = ggml_backend_alloc_ctx_tensors(M->ctx, lm->backend);
    if (!M->buf) { fprintf(stderr, "[S3] FATAL: F32 mirror alloc failed (%.1f MB)\n", M->bytes / 1048576.0); return false; }
    ggml_backend_buffer_set_usage(M->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<uint8_t> raw;
    std::vector<float>   f32;
    for (size_t i = 0; i < srcs.size(); i++) {
        ggml_tensor * s = srcs[i];
        ggml_tensor * d = dsts[i];
        const size_t  ne = (size_t) ggml_nelements(s);
        raw.resize(ggml_nbytes(s));
        ggml_backend_tensor_get(s, raw.data(), 0, raw.size());
        if (s->type == d->type) {
            ggml_backend_tensor_set(d, raw.data(), 0, raw.size());
        } else if (s->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(d, raw.data(), 0, raw.size());
        } else if (s->type == GGML_TYPE_BF16) {
            f32.resize(ne);
            const uint16_t * u = (const uint16_t *) raw.data();
            for (size_t j = 0; j < ne; j++) {
                uint32_t b = (uint32_t) u[j] << 16;
                float    v;
                memcpy(&v, &b, 4);
                f32[j] = v;
            }
            ggml_backend_tensor_set(d, f32.data(), 0, ne * sizeof(float));
        } else if (s->type == GGML_TYPE_F16) {
            f32.resize(ne);
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), f32.data(), (int64_t) ne);
            ggml_backend_tensor_set(d, f32.data(), 0, ne * sizeof(float));
        } else {
            fprintf(stderr, "[S3] FATAL: cannot mirror type %s (tensor %s) — spike needs BF16/F16/F32 base\n",
                    ggml_type_name(s->type), s->name);
            return false;
        }
        *slots[i] = d;
    }
    // Drop the original (BF16) weight buffer: the mirror fully replaces it.
    ggml_backend_buffer_free(lm->wctx.buffer);
    lm->wctx.buffer = nullptr;
    fprintf(stderr, "[S3] F32 weight mirror: %zu tensors, %.1f MB (BF16 buffer released)\n", srcs.size(),
            M->bytes / 1048576.0);
    return true;
}

static void spike_mirror_free(SpikeF32Mirror * M) {
    if (M->buf) ggml_backend_buffer_free(M->buf);
    if (M->ctx) ggml_free(M->ctx);
    *M = {};
}

// ─── sample loading (Side-Step lm_codes.jsonl schema) ───────────────────────

struct S3Sample {
    AcePrompt        prompt;
    std::vector<int> codes;
    std::string      file;
};

static bool s3_load_jsonl_row(const char * path, int row_idx, S3Sample * out) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[S3] cannot open %s\n", path); return false; }
    std::string line;
    int         idx = 0;
    char        buf[65536];
    bool        found = false;
    while (fgets(buf, sizeof(buf), f)) {
        line += buf;
        if (line.empty() || line.back() != '\n') continue;  // long line, keep reading
        if (idx == row_idx) { found = true; break; }
        idx++;
        line.clear();
    }
    fclose(f);
    if (!found) { fprintf(stderr, "[S3] row %d not found in %s\n", row_idx, path); return false; }

    yyjson_doc * doc = yyjson_read(line.data(), line.size(), 0);
    if (!doc) { fprintf(stderr, "[S3] bad JSON on row %d\n", row_idx); return false; }
    yyjson_val * r = yyjson_doc_get_root(doc);
    auto gs = [&](const char * k) -> std::string {
        yyjson_val * v = yyjson_obj_get(r, k);
        return (v && yyjson_is_str(v)) ? yyjson_get_str(v) : "";
    };
    auto gi = [&](const char * k, int fb) -> int {
        yyjson_val * v = yyjson_obj_get(r, k);
        return (v && yyjson_is_int(v)) ? (int) yyjson_get_int(v) : fb;
    };
    out->file                    = gs("file");
    out->prompt.caption          = gs("caption");
    out->prompt.lyrics           = gs("lyrics");
    out->prompt.keyscale         = gs("keyscale");
    out->prompt.timesignature    = gs("timesignature");
    out->prompt.vocal_language   = gs("language");
    out->prompt.bpm              = gi("bpm", 0);
    out->prompt.duration         = (float) gi("duration", 0);

    yyjson_val * c = yyjson_obj_get(r, "codes");
    if (c && yyjson_is_arr(c)) {
        size_t i, m; yyjson_val * v;
        yyjson_arr_foreach(c, i, m, v) { out->codes.push_back((int) yyjson_get_int(v)); }
    }
    yyjson_doc_free(doc);
    return !out->codes.empty();
}

// ─── training-sequence layout (mirrors Side-Step train_lm.py) ───────────────

struct S3Seq {
    std::vector<int32_t> tokens;  // full sequence
    int                  n_masked = 0;  // positions [0, n_masked) get no loss
};

// masked = prompt up to <think>; trained = think block + codes + im_end
// (Side-Step loss_on_cot=True). Truncated to max_len from the END of the codes.
static S3Seq s3_build_sequence(BPETokenizer & bpe, const S3Sample & s, int max_len) {
    std::string cot = build_cot_yaml(s.prompt);
    std::vector<int> pre = build_lm_prompt_with_cot(bpe, s.prompt, cot);

    int n_masked = (int) pre.size();
    for (size_t i = 0; i < pre.size(); i++) {
        if (pre[i] == TOKEN_THINK) { n_masked = (int) i; break; }
    }

    S3Seq q;
    q.tokens.assign(pre.begin(), pre.end());
    for (int c : s.codes) q.tokens.push_back(AUDIO_CODE_BASE + c);
    q.tokens.push_back(TOKEN_IM_END);
    q.n_masked = n_masked;

    fprintf(stderr, "[S3] untruncated: prompt=%zu tok (masked prefix=%d, think block=%zu), codes=%zu, total=%zu\n",
            pre.size(), n_masked, pre.size() - (size_t) n_masked, s.codes.size(), q.tokens.size());

    if ((int) q.tokens.size() > max_len) {
        q.tokens.resize(max_len);
    }
    if (q.n_masked >= (int) q.tokens.size()) {
        q.n_masked = (int) q.tokens.size() / 2;  // degenerate: keep something trainable
    }
    return q;
}

// ─── trunk graph ────────────────────────────────────────────────────────────

struct S3Graph {
    ggml_tensor * hidden = nullptr;  // [H, S] final-normed
    int           n_nodes = 0;
};

static ggml_tensor * s3_build_trunk(ggml_context * ctx, Qwen3LM * lm, ggml_tensor * tokens, ggml_tensor * pos,
                                    ggml_tensor * mask, int S, SpikeFwdOpts * fo) {
    const Qwen3LMConfig & c = lm->cfg;
    ggml_tensor * h = ggml_get_rows(ctx, lm->embed_tokens, tokens);  // [H, S]
    for (int l = 0; l < c.n_layers; l++) {
        h = spike_lm_layer(ctx, c, &lm->layers[l], h, pos, mask, S, fo);
    }
    return spike_rms(ctx, h, lm->final_norm, c.rms_norm_eps);
}

static ggml_opt_optimizer_params s3_opt_pars(void * ud) {
    return *(ggml_opt_optimizer_params *) ud;
}

// cosine schedule with 5% warmup + 0.1 floor, exactly Side-Step's lr_lambda
struct S3Sched {
    ggml_opt_optimizer_params base;
    int step = 0, total = 1, warmup = 1;
    ggml_opt_optimizer_params cur;
};

static ggml_opt_optimizer_params s3_sched_pars(void * ud) {
    S3Sched * s = (S3Sched *) ud;
    float mult;
    if (s->step < s->warmup) {
        mult = (float) (s->step + 1) / (float) s->warmup;
    } else {
        float p = (float) (s->step - s->warmup) / (float) std::max(1, s->total - s->warmup);
        mult = 0.1f + 0.9f * 0.5f * (1.0f + cosf(3.14159265f * p));
    }
    s->cur = s->base;
    s->cur.adamw.alpha = s->base.adamw.alpha * mult;
    return s->cur;
}

// ─── driver ─────────────────────────────────────────────────────────────────

static int spike_s3(const char * lm_path, const char * jsonl, int row, int max_seq, int rank, int nsteps,
                    int chunk, bool chunked) {
    fprintf(stderr, "\n=== S3: full 28-layer LM LoRA micro-train (%s) seq<=%d rank=%d steps=%d mode=%s ===\n",
            lm_path, max_seq, rank, nsteps, chunked ? "chunked" : "naive");

    g_qwen3_load_no_fuse = true;
    Qwen3LM lm;
    if (!qw3lm_load(&lm, lm_path, /*max_seq_len=*/64, /*n_kv_sets=*/1)) return 1;
    const Qwen3LMConfig & c = lm.cfg;

    const size_t vram_base = spike_vram_used_mb(lm.backend);
    fprintf(stderr, "[S3] VRAM after BF16 load: %zu MB\n", vram_base);

    SpikeF32Mirror M;
    if (!spike_lm_weights_to_f32(&lm, &M, /*keep_embed_native=*/chunked)) return 1;
    fprintf(stderr, "[S3] VRAM after F32 mirror: %zu MB\n", spike_vram_used_mb(lm.backend));

    // ── tokenizer + real sample ─────────────────────────────────────────
    BPETokenizer bpe;
    if (!load_bpe_from_gguf(&bpe, lm_path)) {
        fprintf(stderr, "[S3] FATAL: no tokenizer in %s\n", lm_path);
        return 1;
    }
    S3Sample smp;
    if (!s3_load_jsonl_row(jsonl, row, &smp)) return 1;
    S3Seq sq = s3_build_sequence(bpe, smp, max_seq);
    const int S    = (int) sq.tokens.size();
    const int S_tr = S - sq.n_masked - 1;  // last position has no next-token target
    fprintf(stderr, "[S3] sample '%s': %zu codes, seq=%d (masked prefix=%d, trained=%d)\n", smp.file.c_str(),
            smp.codes.size(), S, sq.n_masked, S_tr);
    if (S_tr <= 0) { fprintf(stderr, "[S3] FATAL: nothing trainable\n"); return 1; }
    if (nsteps <= 0) { fprintf(stderr, "[S3] dry run (steps=0), stopping after sequence stats\n"); return 0; }

    // ── LoRA on all layers ──────────────────────────────────────────────
    SpikeLora L;
    if (!spike_lora_init(&L, &lm, 0, c.n_layers, rank, (float) (2 * rank), 0xb0a7ULL, /*b_sigma=*/0.0f)) return 1;

    // ── static inputs ───────────────────────────────────────────────────
    ggml_context * ctx_static = nullptr;
    { ggml_init_params p = { 16 * ggml_tensor_overhead(), nullptr, true }; ctx_static = ggml_init(p); }
    ggml_tensor * t_tok = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, S);
    ggml_tensor * t_pos = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, S);
    ggml_tensor * t_msk = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, S, S);
    // Chunked-mode persistent tensors:
    //   t_G     dL/dhidden buffer that drives the surrogate trunk loss
    //   t_H     detached copy of the trunk output (PyTorch's h.detach())
    //   t_lab   dense one-hot labels for the trained span, resident (ggml has
    //           no index-target cross-entropy op — see report)
    //   t_embT  contiguous [V, H] transpose of embed_tokens, so dL/dhidden is a
    //           plain forward mul_mat instead of an F32-only out_prod
    //   t_gs    scalar upstream gradient handed to cross_entropy_loss_back
    ggml_tensor * t_G    = chunked ? ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, c.hidden_size, S) : nullptr;
    ggml_tensor * t_H    = chunked ? ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, c.hidden_size, S) : nullptr;
    ggml_tensor * t_lab  = chunked ? ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, c.vocab_size, S_tr) : nullptr;
    ggml_tensor * t_embT = chunked ? ggml_new_tensor_2d(ctx_static, lm.embed_tokens->type, c.vocab_size, c.hidden_size)
                                   : nullptr;
    ggml_tensor * t_gs   = chunked ? ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1) : nullptr;
    ggml_set_input(t_tok); ggml_set_input(t_pos); ggml_set_input(t_msk);
    if (t_G) { ggml_set_input(t_G); ggml_set_input(t_H); ggml_set_input(t_lab); ggml_set_input(t_gs); }
    ggml_backend_buffer_t buf_static = ggml_backend_alloc_ctx_tensors(ctx_static, lm.backend);
    if (!buf_static) { fprintf(stderr, "[S3] FATAL: static alloc failed\n"); return 1; }
    ggml_backend_tensor_set(t_tok, sq.tokens.data(), 0, (size_t) S * 4);
    { std::vector<int32_t> ip(S); for (int i = 0; i < S; i++) ip[i] = i;
      ggml_backend_tensor_set(t_pos, ip.data(), 0, (size_t) S * 4); }
    { std::vector<float> m = spike_causal_mask(S); spike_set(t_msk, m); }

    // targets: position p predicts token p+1
    std::vector<int32_t> targets(S_tr);
    for (int i = 0; i < S_tr; i++) targets[i] = sq.tokens[sq.n_masked + i + 1];

    // ── optimizer ───────────────────────────────────────────────────────
    S3Sched sch;
    sch.base = ggml_opt_get_default_optimizer_params(nullptr);
    sch.base.adamw.alpha = 1e-4f;
    sch.base.adamw.wd    = 0.01f;   // Side-Step uses AdamW(weight_decay=0.01)
    sch.total  = nsteps;
    sch.warmup = std::max(1, (int) (nsteps * 0.05));

    ggml_opt_params op = ggml_opt_default_params(lm.sched, chunked ? GGML_OPT_LOSS_TYPE_SUM
                                                                   : GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    op.ctx_compute = nullptr;  // DYNAMIC graphs: static path hardcodes
                               // GGML_DEFAULT_GRAPH_SIZE (2048) nodes, far too
                               // small for a 28-layer fwd+bwd graph.
    op.inputs      = nullptr;
    op.outputs     = nullptr;
    op.build_type  = GGML_OPT_BUILD_TYPE_OPT;
    op.opt_period  = 1;
    op.get_opt_pars = s3_sched_pars;
    op.get_opt_pars_ud = &sch;
    ggml_opt_context_t oc = ggml_opt_init(op);

    // dense one-hot labels ([V, S_tr] f32)
    std::vector<float> onehot;
    onehot.assign((size_t) c.vocab_size * S_tr, 0.0f);
    for (int i = 0; i < S_tr; i++) onehot[(size_t) i * c.vocab_size + targets[i]] = 1.0f;
    fprintf(stderr, "[S3] one-hot labels: %.1f MB (%s)\n", onehot.size() * 4.0 / 1048576.0,
            chunked ? "uploaded once, resident" : "re-uploaded every step into the graph's labels node");

    if (chunked) {
        // zero G once (chunk writes only touch trained columns), then re-upload
        // everything the wipe clobbered.
        ggml_backend_buffer_clear(buf_static, 0);
        ggml_backend_tensor_set(t_tok, sq.tokens.data(), 0, (size_t) S * 4);
        { std::vector<int32_t> ip(S); for (int i = 0; i < S; i++) ip[i] = i;
          ggml_backend_tensor_set(t_pos, ip.data(), 0, (size_t) S * 4); }
        { std::vector<float> m = spike_causal_mask(S); spike_set(t_msk, m); }
        ggml_backend_tensor_set(t_lab, onehot.data(), 0, onehot.size() * sizeof(float));
        onehot.clear();
        onehot.shrink_to_fit();
        // host transpose of embed_tokens [H,V] -> [V,H], same dtype
        {
            const size_t ts = ggml_type_size(lm.embed_tokens->type);
            std::vector<uint8_t> src(ggml_nbytes(lm.embed_tokens)), dst(ggml_nbytes(t_embT));
            ggml_backend_tensor_get(lm.embed_tokens, src.data(), 0, src.size());
            const int64_t H = c.hidden_size, V = c.vocab_size;
            for (int64_t v = 0; v < V; v++) {
                for (int64_t h = 0; h < H; h++) {
                    memcpy(&dst[(size_t) (h * V + v) * ts], &src[(size_t) (v * H + h) * ts], ts);
                }
            }
            ggml_backend_tensor_set(t_embT, dst.data(), 0, dst.size());
            fprintf(stderr, "[S3] embedT [%lld,%lld] %s: %.1f MB\n", (long long) V, (long long) H,
                    ggml_type_name(t_embT->type), ggml_nbytes(t_embT) / 1048576.0);
        }
    }

    const size_t ctx_bytes = (size_t) 64 * 1024 * 1024;
    std::vector<double> losses;
    size_t vram_peak = 0;
    long long ms_total = 0;

    long long ms_head = 0;
    for (int step = 0; step < nsteps; step++) {
        sch.step = step;
        double ce_report = 0.0;

        // ── chunked phase 1+2: trunk forward (detached) then per-chunk head ──
        if (chunked) {
            const int64_t th0 = ggml_time_ms();
            {   // phase 1: trunk forward only, result copied into t_H
                ggml_init_params cp = { ctx_bytes, nullptr, true };
                ggml_context *   cx = ggml_init(cp);
                ggml_cgraph *    g  = ggml_new_graph_custom(cx, 8192, false);
                SpikeFwdOpts fo; fo.cast_base_f32 = false;
                ggml_tensor * h = s3_build_trunk(cx, &lm, t_tok, t_pos, t_msk, S, &fo);
                ggml_build_forward_expand(g, ggml_cpy(cx, h, t_H));
                ggml_backend_sched_reset(lm.sched);
                ggml_backend_sched_graph_compute(lm.sched, g);
                ggml_free(cx);
            }
            // phase 2: chunked lm_head + CE + manual dL/dhidden into t_G
            for (int i = 0; i < S_tr; i += chunk) {
                const int    j  = std::min(i + chunk, S_tr);
                const int    Sc = j - i;
                const float  gs = (float) Sc / (float) S_tr;
                ggml_backend_tensor_set(t_gs, &gs, 0, sizeof(float));

                ggml_init_params cp = { 4u << 20, nullptr, true };
                ggml_context *   cx = ggml_init(cp);
                ggml_cgraph *    g  = ggml_new_graph_custom(cx, 256, false);

                ggml_tensor * hd = ggml_cont(cx, ggml_view_2d(cx, t_H, c.hidden_size, Sc, t_H->nb[1],
                                                              (size_t) (sq.n_masked + i) * t_H->nb[1]));
                ggml_tensor * lg = ggml_mul_mat(cx, lm.embed_tokens, hd);                       // [V, Sc]
                ggml_tensor * lb = ggml_view_2d(cx, t_lab, c.vocab_size, Sc, t_lab->nb[1],
                                                (size_t) i * t_lab->nb[1]);
                ggml_tensor * lc = ggml_cross_entropy_loss(cx, lg, lb);                          // scalar, mean/Sc
                ggml_set_output(lc);
                // dst = (softmax(lg) - lb) * (gs / Sc) = (softmax - onehot)/S_tr
                ggml_tensor * dl = ggml_cross_entropy_loss_back(cx, t_gs, lg, lb);               // [V, Sc]
                ggml_tensor * dh = ggml_mul_mat(cx, t_embT, dl);                                 // [H, Sc]
                ggml_tensor * gv = ggml_view_2d(cx, t_G, c.hidden_size, Sc, t_G->nb[1],
                                                (size_t) (sq.n_masked + i) * t_G->nb[1]);
                ggml_build_forward_expand(g, lc);
                ggml_build_forward_expand(g, ggml_cpy(cx, dh, gv));
                ggml_backend_sched_reset(lm.sched);
                ggml_backend_sched_graph_compute(lm.sched, g);
                float lv = 0;
                ggml_backend_tensor_get(lc, &lv, 0, sizeof(float));
                ce_report += (double) lv * (double) Sc / (double) S_tr;
                ggml_free(cx);
            }
            ggml_backend_sched_synchronize(lm.sched);
            ms_head += (long long) (ggml_time_ms() - th0);
        }

        ggml_init_params cp = { ctx_bytes, nullptr, true };
        ggml_context *   ctx = ggml_init(cp);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 32768, /*grads=*/true);

        SpikeFwdOpts fo; fo.cast_base_f32 = false;  // weights are already F32
        ggml_tensor * hidden = s3_build_trunk(ctx, &lm, t_tok, t_pos, t_msk, S, &fo);

        ggml_tensor * outputs = nullptr;
        if (!chunked) {
            ggml_tensor * hd = ggml_cont(ctx, ggml_view_2d(ctx, hidden, c.hidden_size, S_tr, hidden->nb[1],
                                                           (size_t) sq.n_masked * hidden->nb[1]));
            outputs = ggml_mul_mat(ctx, lm.embed_tokens, hd);  // [V, S_tr] logits
        } else {
            outputs = ggml_mul(ctx, hidden, t_G);              // surrogate: loss = sum(hidden * G)
        }
        ggml_set_output(outputs);

        ggml_build_forward_expand(gf, outputs);
        if (step == 0) fprintf(stderr, "[S3] forward graph: %d nodes\n", ggml_graph_n_nodes(gf));

        const int64_t t0 = ggml_time_ms();
        ggml_opt_prepare_alloc(oc, ctx, gf, t_tok, outputs);
        ggml_opt_alloc(oc, /*backward=*/true);
        if (!chunked) {
            ggml_tensor * lab = ggml_opt_labels(oc);
            ggml_backend_tensor_set(lab, onehot.data(), 0, onehot.size() * sizeof(float));
        }
        ggml_opt_eval(oc, nullptr);
        ggml_backend_sched_synchronize(lm.sched);
        const long long ms = (long long) (ggml_time_ms() - t0);
        ms_total += ms;

        float loss = 0;
        ggml_backend_tensor_get(ggml_opt_loss(oc), &loss, 0, sizeof(float));
        const double reported = chunked ? ce_report : (double) loss;
        losses.push_back(reported);
        size_t vm = spike_vram_used_mb(lm.backend);
        if (vm > vram_peak) vram_peak = vm;
        fprintf(stderr, "[S3] step %2d  CE=%.6f  lr=%.3e  %lld ms(bwd)  vram=%zu MB%s\n", step, reported,
                (double) sch.cur.adamw.alpha, ms, vm,
                chunked ? "  [+head, see ms_head]" : "");

        // NOTE: ggml_opt_grad_acc() is UNUSABLE with dynamic graphs — ggml_opt_eval
        // sets opt_ctx->gb_opt = nullptr on return (ggml-opt.cpp), so the accessor
        // dereferences null. No public API reaches the grad accumulators in dynamic
        // mode => no grad-norm logging and no gradient clipping inside ggml-opt.
        ggml_free(ctx);
    }

    const double l0 = losses.front(), ln = losses.back();
    const bool   ok = ln < l0 && std::isfinite(ln);
    fprintf(stderr,
            "\n[S3] SUMMARY mode=%s seq=%d trained=%d chunk=%d  CE %.4f -> %.4f  bwd %.0f ms/step  head %.0f ms/step  "
            "TOTAL %.0f ms/step  VRAM peak %zu MB -> %s\n",
            chunked ? "chunked" : "naive", S, S_tr, chunked ? chunk : 0, l0, ln, (double) ms_total / nsteps,
            (double) ms_head / nsteps, (double) (ms_total + ms_head) / nsteps, vram_peak, ok ? "PASS" : "FAIL");

    ggml_opt_free(oc);
    ggml_backend_buffer_free(buf_static);
    ggml_free(ctx_static);
    spike_lora_free(&L);
    spike_mirror_free(&M);
    return ok ? 0 : 1;
}
