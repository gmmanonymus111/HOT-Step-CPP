#pragma once
// train/mm3-lm-load.h — load the MiniMax-Music3 LM into the ACE LM TRAINER's
// model struct.
//
// HOT-Step file. Included by tools/ace-train.cpp only.
//
// ── WHY THIS IS A SHIM AND NOT A NEW TRAINER ────────────────────────────────
//
// `train/lm-graph.h` is already "a trainable, cache-free, unfused Qwen3 LM
// forward" — built for ACE's planner LM, but Qwen3 is Qwen3, and MM3's LM is
// Qwen3-8B with an extended audio vocabulary. Every property that graph was
// designed around holds here too: no KV cache (set_rows has no backward), no
// flash attention (no backward), no fused QKV/gate_up, swiglu_split, and the
// BF16 weight path from lm-bf16.h. So the MM3 LM trainer is a RETARGET of a
// validated trainer, and this file is the whole retarget of the LOAD side.
//
// Three things differ from an ACE planner LM, and all three live here:
//
//  1. TENSOR NAMES. convert-mm3.py deliberately writes the LM to llama.cpp
//     Qwen3 conventions ("the LM file is kept structurally pure Qwen3: a
//     llama.cpp-grade loader must be able to consume it"), so it is
//     `token_embd.weight` / `blk.N.attn_q.weight` / `output_norm.weight`.
//     The ACE loader wants HF names (`model.layers.N.self_attn.q_proj.weight`).
//     Mapped below, one table, no guessing.
//
//  2. CONFIG SOURCE. ACE reads an `acestep.config_json` blob; MM3 carries
//     first-class `qwen3.*` KV. Read straight across.
//
//  3. THE HEAD IS UNTIED. ACE's planner LM ties lm_head to the embedding
//     matrix; MM3 ships a separate `output.weight` [4096, 200000]. The trainer
//     hardcodes `mul_mat(lm.embed_tokens, h)` in two places, so the untied head
//     is returned SEPARATELY rather than smuggled into `embed_tokens` — a
//     caller that forgets it gets a compile-time miss, not a silently wrong
//     head.
//
// ── THE SEMANTIC SLICE, AND WHY IT IS FREE ──────────────────────────────────
//
// MM3 LM training supervises the SEMANTIC codebook only (books 1..7 belong to
// the depth decoder, which is not trained). The semantic tokens are a
// CONTIGUOUS ROW RANGE of the output matrix — [semantic_vocab_offset,
// +semantic_vocab_size) = 16384 of 200000 rows — so the sliced head is one
// `ggml_view_2d`, not a gather and not a copy. That turns the full-vocab logit
// tensor (1.12 GiB at S=1500, plus its gradient) into 0.09 GiB. `mm3_lm_train_
// sem_head()` below is that view; use it instead of the full head.
//
// ── WHAT THIS FILE DOES NOT DO ──────────────────────────────────────────────
//
// No KV cache, no graph arenas: `qw3lm_load` allocates both, and a cache-free
// trainer wants neither (the cache alone would be gigabytes at 36 layers).
// Loading is therefore open-coded here rather than delegated.

#include "qwen3-lm.h"
#include "minimax/mm3-model.h"   // GGUFModel/gf_*, WeightCtx, mm3_fmt

#include <string>
#include <vector>

struct MM3TrainLm {
    Qwen3LM       lm = {};            // the trainer's model struct
    ggml_tensor * lm_head = nullptr;  // [H, V] — UNTIED, not lm.embed_tokens
    // From the mm3.* KV: what the training loss needs and the ACE config has
    // no room for.
    uint32_t      semantic_vocab_offset = 0;
    uint32_t      semantic_vocab_size   = 0;
    uint32_t      acoustic_vocab_size   = 0;
    uint32_t      num_codebooks         = 0;
    uint32_t      eos_audio             = 0;
    float         embedding_scale       = 0.0f;  // 1/sqrt(8), applied to frame embeddings
};

// llama.cpp name -> the ACE trainer's slot, per layer. Kept as one table so a
// future rename shows up as a missing tensor at load, never as a wrong one.
struct MM3LmNameMap {
    const char * gguf;                       // printf template, one %d
    ggml_tensor * Qwen3Layer::* slot;
    bool         f32;                        // norms are kept F32 like the ACE path
};

static const MM3LmNameMap MM3_LM_LAYER_MAP[] = {
    { "blk.%d.attn_norm.weight",   &Qwen3Layer::input_layernorm,     true  },
    { "blk.%d.ffn_norm.weight",    &Qwen3Layer::post_attn_layernorm, true  },
    { "blk.%d.attn_q.weight",      &Qwen3Layer::q_proj,              false },
    { "blk.%d.attn_k.weight",      &Qwen3Layer::k_proj,              false },
    { "blk.%d.attn_v.weight",      &Qwen3Layer::v_proj,              false },
    { "blk.%d.attn_output.weight", &Qwen3Layer::o_proj,              false },
    { "blk.%d.attn_q_norm.weight", &Qwen3Layer::q_norm,              true  },
    { "blk.%d.attn_k_norm.weight", &Qwen3Layer::k_norm,              true  },
    { "blk.%d.ffn_gate.weight",    &Qwen3Layer::gate_proj,           false },
    { "blk.%d.ffn_up.weight",      &Qwen3Layer::up_proj,             false },
    { "blk.%d.ffn_down.weight",    &Qwen3Layer::down_proj,           false },
};

static void mm3_train_lm_free(MM3TrainLm * t) {
    Qwen3LM & m = t->lm;
    if (m.wctx.buffer) { ggml_backend_buffer_free(m.wctx.buffer); m.wctx.buffer = nullptr; }
    if (m.wctx.ctx)    { ggml_free(m.wctx.ctx); m.wctx.ctx = nullptr; }
    m.wctx = {};
    if (m.backend) {
        backend_release(m.backend, m.cpu_backend);
        m.backend = m.cpu_backend = nullptr;
    }
    *t = MM3TrainLm{};
}

/** Load mm3-lm-*.gguf into the trainer's Qwen3LM. No KV cache, no arenas.
 *  Returns false with `err` set; every shape is validated against the config,
 *  so a wrong-model file fails at load rather than at the first NaN. */
static bool mm3_train_lm_load(MM3TrainLm * t, const char * path, std::string * err) {
    GGUFModel gf = {};
    if (!gf_load(&gf, path)) {
        if (err) *err = std::string("cannot open ") + path;
        return false;
    }
    const std::string arch = gf_get_str(gf, "general.architecture");
    if (arch != "qwen3") {
        gf_close(&gf);
        if (err) *err = "not a qwen3 GGUF (general.architecture = '" + arch + "')";
        return false;
    }

    MM3LmConfig mc = {};
    mm3_parse_lm_config(gf, &mc);

    Qwen3LMConfig & c   = t->lm.cfg;
    c.vocab_size        = (int) mc.vocab_size;
    c.hidden_size       = (int) mc.embedding_length;
    c.intermediate_size = (int) mc.feed_forward_length;
    c.n_heads           = (int) mc.head_count;
    c.n_kv_heads        = (int) mc.head_count_kv;
    c.head_dim          = (int) mc.key_length;
    c.n_layers          = (int) mc.block_count;
    c.rope_theta        = mc.rope_freq_base;
    c.rms_norm_eps      = mc.rms_eps;
    c.tie_embeddings    = false;                 // MM3 ships output.weight
    c.max_seq_len       = (int) mc.context_length;

    t->semantic_vocab_offset = mc.semantic_vocab_offset;
    t->semantic_vocab_size   = mc.semantic_vocab_size;
    t->acoustic_vocab_size   = mc.acoustic_vocab_size;
    t->num_codebooks         = mc.num_codebooks;
    t->eos_audio             = mc.eos_audio;
    t->embedding_scale       = mc.ar_embedding_scale;

    if (c.n_layers <= 0 || c.n_layers > QW3LM_MAX_LAYERS) {
        gf_close(&gf);
        if (err) *err = "layer count " + std::to_string(c.n_layers) + " outside 1.." +
                        std::to_string(QW3LM_MAX_LAYERS);
        return false;
    }
    if (c.n_heads * c.head_dim != c.hidden_size) {
        gf_close(&gf);
        if (err) *err = "head_count * key_length != embedding_length";
        return false;
    }
    if (t->semantic_vocab_offset + t->semantic_vocab_size > (uint32_t) c.vocab_size) {
        gf_close(&gf);
        if (err) *err = "the semantic slice runs past the vocabulary";
        return false;
    }

    if (!t->lm.backend) {
        BackendPair bp  = backend_init("MM3TrainLM");
        t->lm.backend   = bp.backend;
        t->lm.cpu_backend = bp.cpu_backend;
    }

    const int64_t H = c.hidden_size, V = c.vocab_size, D = c.head_dim;
    const int64_t Nh = c.n_heads, Nkv = c.n_kv_heads, FF = c.intermediate_size;

    // embed + head + final norm + 11 per layer.
    wctx_init(&t->lm.wctx, 3 + c.n_layers * 11 + 8);

    std::vector<std::string> errs;
    MM3Loader                ld{ &t->lm.wctx, &gf, nullptr, &errs };

    t->lm.embed_tokens = ld.req("token_embd.weight", H, V);
    t->lm_head         = ld.req("output.weight", H, V);
    t->lm.final_norm   = ld.req("output_norm.weight", H);

    for (int i = 0; i < c.n_layers; i++) {
        Qwen3Layer & ly = t->lm.layers[i];
        ly = Qwen3Layer{};   // no fused slots: the trainer requires separate projections
        for (const MM3LmNameMap & nm : MM3_LM_LAYER_MAP) {
            const std::string name = mm3_fmt(nm.gguf, i);
            int64_t e0 = 0, e1 = 1;
            if      (nm.slot == &Qwen3Layer::input_layernorm ||
                     nm.slot == &Qwen3Layer::post_attn_layernorm) { e0 = H; }
            else if (nm.slot == &Qwen3Layer::q_norm ||
                     nm.slot == &Qwen3Layer::k_norm)              { e0 = D; }
            else if (nm.slot == &Qwen3Layer::q_proj)              { e0 = H;  e1 = Nh * D; }
            else if (nm.slot == &Qwen3Layer::k_proj ||
                     nm.slot == &Qwen3Layer::v_proj)              { e0 = H;  e1 = Nkv * D; }
            else if (nm.slot == &Qwen3Layer::o_proj)              { e0 = Nh * D; e1 = H; }
            else if (nm.slot == &Qwen3Layer::gate_proj ||
                     nm.slot == &Qwen3Layer::up_proj)             { e0 = H;  e1 = FF; }
            else if (nm.slot == &Qwen3Layer::down_proj)           { e0 = FF; e1 = H; }
            ly.*(nm.slot) = ld.req(name, e0, e1);
        }
        if (!errs.empty()) {
            break;   // one bad layer means the file is wrong; do not spam 36 copies
        }
    }

    if (!errs.empty()) {
        gf_close(&gf);
        mm3_train_lm_free(t);
        if (err) *err = errs[0];
        return false;
    }

    // Sum BEFORE the alloc: wctx_alloc consumes `pending`, so reading it
    // afterwards reports 0.00 GB for a 16 GB model.
    size_t bytes = 0;
    for (const auto & pc : t->lm.wctx.pending) {
        bytes += pc.nbytes;
    }
    if (!wctx_alloc(&t->lm.wctx, t->lm.backend)) {
        gf_close(&gf);
        mm3_train_lm_free(t);
        if (err) *err = "backend buffer allocation failed for the MM3 LM (out of VRAM?)";
        return false;
    }
    gf_close(&gf);

    fprintf(stderr, "[MM3TrainLM] %s: %dL H=%d V=%d Nh=%d Nkv=%d D=%d FF=%d, untied head, %.2f GB\n", path,
            c.n_layers, c.hidden_size, c.vocab_size, c.n_heads, c.n_kv_heads, c.head_dim,
            c.intermediate_size, (double) bytes / (1024.0 * 1024.0 * 1024.0));
    return true;
}

/** The semantic slice of the untied head, as a VIEW — no copy, no gather.
 *  This is the whole "chunked CE" lever: 16384 rows instead of 200000. */
static ggml_tensor * mm3_lm_train_sem_head(ggml_context * ctx, const MM3TrainLm & t) {
    ggml_tensor * W = t.lm_head;                        // [H, V]
    return ggml_view_2d(ctx, W, W->ne[0], (int64_t) t.semantic_vocab_size, W->nb[1],
                        (size_t) t.semantic_vocab_offset * W->nb[1]);
}
