// moss-model.h: GGUF loader for the MOSS-Music audio tower.
//
// Loads moss-aud-<quant>.gguf (arch "moss-aud", 502 tensors) produced by
// engine/tools/convert-moss.py. The LM half lives in moss-lm-<quant>.gguf and is
// structurally pure Qwen3, so it goes through the existing Qwen3 path rather
// than this file.
//
// Layout, mirroring the converter:
//   aud.conv{1,2,3}.{weight,bias}     conv stem, k3 s2 p1, 1->480->480->480
//   aud.stem_proj.{weight,bias}       Linear 7680 -> 1280   (480 * 16 mel rows)
//   aud.blk.N.*                       32 pre-norm Whisper encoder layers
//   aud.norm.{weight,bias}            final LayerNorm
//   aud.adapter.{gate,up,down}.weight SwiGLU 1280 -> 8192 -> 4096, no bias
//   aud.deepstack.K.{gate,up,down}    3 more of the same, fed from encoder
//                                     layers 8/16/24
//
// TWO THINGS THE LOADER DELIBERATELY DOES NOT DO
// ----------------------------------------------
// 1. There is no `aud.blk.N.attn_k.bias`. Whisper's k_proj genuinely has no
//    bias; the converter preserves that rather than synthesising a zero, so the
//    graph must skip the add instead of assuming a tensor is there.
// 2. There is no positional-embedding tensor. inv_timescales is registered
//    persistent=False upstream and is intentionally absent from the GGUF; the
//    graph recomputes sinusoids from moss.audio.max_timescale. See
//    engine/tools/moss-numpy-ref.py for the exact formula.

#pragma once

#include "gguf-weights.h"
#include "weight-ctx.h"

#include <cstdio>
#include <string>
#include <vector>

namespace moss {

struct AudioHParams {
    uint32_t n_layer            = 0;
    uint32_t d_model            = 0;
    uint32_t n_head             = 0;
    uint32_t n_ffn              = 0;
    uint32_t n_mels             = 0;
    uint32_t output_dim         = 0;
    uint32_t downsample_hidden  = 0;
    uint32_t downsample_rate    = 0;
    uint32_t adapter_hidden     = 0;
    uint32_t lm_embd            = 0;
    uint32_t sample_rate        = 0;
    uint32_t n_fft              = 0;
    uint32_t hop_length         = 0;
    uint32_t audio_token_id     = 0;
    uint32_t audio_bos_id       = 0;
    uint32_t audio_eos_id       = 0;
    float    layer_norm_eps     = 0.0f;
    float    tokens_per_second  = 0.0f;
    float    max_timescale      = 0.0f;
    std::vector<int32_t> deepstack_layers;   // encoder layers tapped, e.g. {8,16,24}
};

struct AudioBlock {
    ggml_tensor * attn_norm_w = nullptr;
    ggml_tensor * attn_norm_b = nullptr;
    ggml_tensor * q_w = nullptr;
    ggml_tensor * q_b = nullptr;
    ggml_tensor * k_w = nullptr;            // no bias, by design
    ggml_tensor * v_w = nullptr;
    ggml_tensor * v_b = nullptr;
    ggml_tensor * o_w = nullptr;
    ggml_tensor * o_b = nullptr;
    ggml_tensor * ffn_norm_w = nullptr;
    ggml_tensor * ffn_norm_b = nullptr;
    ggml_tensor * up_w = nullptr;
    ggml_tensor * up_b = nullptr;
    ggml_tensor * down_w = nullptr;
    ggml_tensor * down_b = nullptr;
};

struct SwiGLU {
    ggml_tensor * gate = nullptr;
    ggml_tensor * up   = nullptr;
    ggml_tensor * down = nullptr;
};

struct AudioTower {
    AudioHParams hp;

    ggml_tensor * conv_w[3] = {nullptr, nullptr, nullptr};
    ggml_tensor * conv_b[3] = {nullptr, nullptr, nullptr};
    ggml_tensor * stem_w = nullptr;
    ggml_tensor * stem_b = nullptr;
    ggml_tensor * norm_w = nullptr;
    ggml_tensor * norm_b = nullptr;

    std::vector<AudioBlock> blocks;
    SwiGLU adapter;
    std::vector<SwiGLU> deepstack;

    WeightCtx      wctx    = {};
    ggml_backend_t backend = nullptr;
};

namespace model_detail {

inline std::vector<int32_t> get_i32_array(const GGUFModel & gf, const char * key) {
    std::vector<int32_t> out;
    const int64_t idx = gguf_find_key(gf.gguf, key);
    if (idx < 0) {
        return out;
    }
    const size_t n = gguf_get_arr_n(gf.gguf, idx);
    const void * d = gguf_get_arr_data(gf.gguf, idx);
    const int32_t * p = (const int32_t *) d;
    for (size_t i = 0; i < n; ++i) {
        out.push_back(p[i]);
    }
    return out;
}

}  // namespace model_detail

inline void moss_free_audio_tower(AudioTower * m) {
    if (!m) {
        return;
    }
    wctx_free(&m->wctx);
    *m = {};
}

// Loads every tensor to `backend`. Returns false and leaves nothing allocated on
// any failure -- a partially-loaded tower is never handed back.
inline bool moss_load_audio_tower(AudioTower * m, const char * path,
                                  ggml_backend_t backend) {
    *m = {};
    m->backend = backend;

    GGUFModel gf;
    if (!gf_load(&gf, path)) {
        return false;
    }

    const char * arch = gf_get_str(gf, "general.architecture");
    if (!arch || std::string(arch) != "moss-aud") {
        fprintf(stderr, "[MOSS] %s: general.architecture is '%s', expected 'moss-aud'\n",
                path, arch ? arch : "(none)");
        gf_close(&gf);
        return false;
    }

    AudioHParams & hp = m->hp;
    hp.n_layer           = gf_get_u32(gf, "moss.audio.block_count");
    hp.d_model           = gf_get_u32(gf, "moss.audio.embedding_length");
    hp.n_head            = gf_get_u32(gf, "moss.audio.head_count");
    hp.n_ffn             = gf_get_u32(gf, "moss.audio.feed_forward_length");
    hp.n_mels            = gf_get_u32(gf, "moss.audio.n_mels");
    hp.output_dim        = gf_get_u32(gf, "moss.audio.output_dim");
    hp.downsample_hidden = gf_get_u32(gf, "moss.audio.downsample_hidden");
    hp.downsample_rate   = gf_get_u32(gf, "moss.audio.downsample_rate");
    hp.adapter_hidden    = gf_get_u32(gf, "moss.audio.adapter_hidden");
    hp.lm_embd           = gf_get_u32(gf, "moss.audio.lm_embedding_length");
    hp.sample_rate       = gf_get_u32(gf, "moss.audio.sample_rate");
    hp.n_fft             = gf_get_u32(gf, "moss.audio.n_fft");
    hp.hop_length        = gf_get_u32(gf, "moss.audio.hop_length");
    hp.audio_token_id    = gf_get_u32(gf, "moss.audio.audio_token_id");
    hp.audio_bos_id      = gf_get_u32(gf, "moss.audio.audio_bos_id");
    hp.audio_eos_id      = gf_get_u32(gf, "moss.audio.audio_eos_id");
    hp.layer_norm_eps    = gf_get_f32(gf, "moss.audio.layer_norm_eps");
    hp.tokens_per_second = gf_get_f32(gf, "moss.audio.tokens_per_second");
    hp.max_timescale     = gf_get_f32(gf, "moss.audio.max_timescale");
    hp.deepstack_layers  = model_detail::get_i32_array(gf, "moss.audio.deepstack_encoder_layers");

    if (hp.n_layer == 0 || hp.d_model == 0 || hp.n_head == 0) {
        fprintf(stderr, "[MOSS] %s: missing core hparams\n", path);
        gf_close(&gf);
        return false;
    }
    if (hp.deepstack_layers.empty()) {
        fprintf(stderr, "[MOSS] %s: no deepstack layer indexes\n", path);
        gf_close(&gf);
        return false;
    }

    // 11 stem/norm + 15 per block + 3 adapter + 3 per merger, with slack.
    wctx_init(&m->wctx, (int) (32 + 16 * hp.n_layer + 3 * (1 + hp.deepstack_layers.size())));

    bool ok = true;
    auto want = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = gf_load_tensor(&m->wctx, gf, name);
        if (!t) {
            fprintf(stderr, "[MOSS] %s: missing tensor %s\n", path, name.c_str());
            ok = false;
        }
        return t;
    };

    for (int i = 0; i < 3; ++i) {
        const std::string k = "aud.conv" + std::to_string(i + 1) + ".";
        m->conv_w[i] = want(k + "weight");
        m->conv_b[i] = want(k + "bias");
    }
    m->stem_w = want("aud.stem_proj.weight");
    m->stem_b = want("aud.stem_proj.bias");
    m->norm_w = want("aud.norm.weight");
    m->norm_b = want("aud.norm.bias");

    m->blocks.resize(hp.n_layer);
    for (uint32_t i = 0; i < hp.n_layer && ok; ++i) {
        const std::string p = "aud.blk." + std::to_string(i) + ".";
        AudioBlock & b = m->blocks[i];
        b.attn_norm_w = want(p + "attn_norm.weight");
        b.attn_norm_b = want(p + "attn_norm.bias");
        b.q_w         = want(p + "attn_q.weight");
        b.q_b         = want(p + "attn_q.bias");
        b.k_w         = want(p + "attn_k.weight");   // no matching bias, on purpose
        b.v_w         = want(p + "attn_v.weight");
        b.v_b         = want(p + "attn_v.bias");
        b.o_w         = want(p + "attn_out.weight");
        b.o_b         = want(p + "attn_out.bias");
        b.ffn_norm_w  = want(p + "ffn_norm.weight");
        b.ffn_norm_b  = want(p + "ffn_norm.bias");
        b.up_w        = want(p + "ffn_up.weight");
        b.up_b        = want(p + "ffn_up.bias");
        b.down_w      = want(p + "ffn_down.weight");
        b.down_b      = want(p + "ffn_down.bias");
    }

    m->adapter.gate = want("aud.adapter.gate.weight");
    m->adapter.up   = want("aud.adapter.up.weight");
    m->adapter.down = want("aud.adapter.down.weight");

    m->deepstack.resize(hp.deepstack_layers.size());
    for (size_t k = 0; k < m->deepstack.size() && ok; ++k) {
        const std::string p = "aud.deepstack." + std::to_string(k) + ".";
        m->deepstack[k].gate = want(p + "gate.weight");
        m->deepstack[k].up   = want(p + "up.weight");
        m->deepstack[k].down = want(p + "down.weight");
    }

    if (!ok || !wctx_alloc(&m->wctx, backend)) {
        fprintf(stderr, "[MOSS] %s: load failed\n", path);
        gf_close(&gf);
        moss_free_audio_tower(m);
        return false;
    }

    gf_close(&gf);  // safe: wctx_alloc has copied everything to the backend
    return true;
}

// ---------------------------------------------------------------------------
// LM half: moss-lm-<quant>.gguf, arch "qwen3", 399 tensors.
//
// Architecturally IDENTICAL to MM3's global LM (see minimax/mm3-lm-graph.h):
// 36 blocks, H=4096, 32 query / 8 KV heads x 128, SwiGLU 12288, RMSNorm 1e-6,
// per-head q/k RMSNorm BEFORE RoPE, NeoX RoPE at theta 1e6, untied head. Only
// the vocabulary differs (151936 here vs MM3's 200000 with audio codes).
// ---------------------------------------------------------------------------

struct LmLayer {
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * attn_q = nullptr;
    ggml_tensor * attn_k = nullptr;
    ggml_tensor * attn_v = nullptr;
    ggml_tensor * attn_output = nullptr;
    ggml_tensor * attn_q_norm = nullptr;   // Qwen3 QK-norm, per head, pre-RoPE
    ggml_tensor * attn_k_norm = nullptr;
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * ffn_gate = nullptr;
    ggml_tensor * ffn_up = nullptr;
    ggml_tensor * ffn_down = nullptr;
};

struct LmHParams {
    uint32_t n_layer = 0;
    uint32_t n_embd = 0;
    uint32_t n_ffn = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim = 0;
    uint32_t n_vocab = 0;
    uint32_t n_ctx = 0;
    float    rms_eps = 0.0f;
    float    rope_theta = 0.0f;
};

struct LmModel {
    LmHParams hp;
    ggml_tensor * token_embd = nullptr;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output = nullptr;         // untied head
    std::vector<LmLayer> layers;
    WeightCtx wctx = {};
    ggml_backend_t backend = nullptr;
};

inline void moss_free_lm(LmModel * m) {
    if (!m) {
        return;
    }
    wctx_free(&m->wctx);
    *m = {};
}

inline bool moss_load_lm(LmModel * m, const char * path, ggml_backend_t backend) {
    *m = {};
    m->backend = backend;

    GGUFModel gf;
    if (!gf_load(&gf, path)) {
        return false;
    }
    const char * arch = gf_get_str(gf, "general.architecture");
    if (!arch || std::string(arch) != "qwen3") {
        fprintf(stderr, "[MOSS] %s: general.architecture is '%s', expected 'qwen3'\n",
                path, arch ? arch : "(none)");
        gf_close(&gf);
        return false;
    }

    LmHParams & hp = m->hp;
    hp.n_layer    = gf_get_u32(gf, "qwen3.block_count");
    hp.n_embd     = gf_get_u32(gf, "qwen3.embedding_length");
    hp.n_ffn      = gf_get_u32(gf, "qwen3.feed_forward_length");
    hp.n_head     = gf_get_u32(gf, "qwen3.attention.head_count");
    hp.n_head_kv  = gf_get_u32(gf, "qwen3.attention.head_count_kv");
    hp.head_dim   = gf_get_u32(gf, "qwen3.attention.key_length");
    hp.n_vocab    = gf_get_u32(gf, "qwen3.vocab_size");
    hp.n_ctx      = gf_get_u32(gf, "qwen3.context_length");
    hp.rms_eps    = gf_get_f32(gf, "qwen3.attention.layer_norm_rms_epsilon");
    hp.rope_theta = gf_get_f32(gf, "qwen3.rope.freq_base");
    if (hp.n_layer == 0 || hp.n_embd == 0) {
        fprintf(stderr, "[MOSS] %s: missing core LM hparams\n", path);
        gf_close(&gf);
        return false;
    }

    wctx_init(&m->wctx, (int) (16 + 12 * hp.n_layer));
    bool ok = true;
    auto want = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = gf_load_tensor(&m->wctx, gf, name);
        if (!t) {
            fprintf(stderr, "[MOSS] %s: missing tensor %s\n", path, name.c_str());
            ok = false;
        }
        return t;
    };

    m->token_embd  = want("token_embd.weight");
    m->output_norm = want("output_norm.weight");
    // tie_word_embeddings is false upstream, so this is a real separate tensor.
    // A stock Qwen3 GGUF that omits it is NOT interchangeable with this file.
    m->output      = want("output.weight");

    m->layers.resize(hp.n_layer);
    for (uint32_t i = 0; i < hp.n_layer && ok; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        LmLayer & l = m->layers[i];
        l.attn_norm   = want(p + "attn_norm.weight");
        l.attn_q      = want(p + "attn_q.weight");
        l.attn_k      = want(p + "attn_k.weight");
        l.attn_v      = want(p + "attn_v.weight");
        l.attn_output = want(p + "attn_output.weight");
        l.attn_q_norm = want(p + "attn_q_norm.weight");
        l.attn_k_norm = want(p + "attn_k_norm.weight");
        l.ffn_norm    = want(p + "ffn_norm.weight");
        l.ffn_gate    = want(p + "ffn_gate.weight");
        l.ffn_up      = want(p + "ffn_up.weight");
        l.ffn_down    = want(p + "ffn_down.weight");
    }

    if (!ok || !wctx_alloc(&m->wctx, backend)) {
        fprintf(stderr, "[MOSS] %s: LM load failed\n", path);
        gf_close(&gf);
        moss_free_lm(m);
        return false;
    }
    gf_close(&gf);
    return true;
}

inline void moss_print_lm_hparams(const LmModel & m) {
    const LmHParams & h = m.hp;
    printf("[MOSS] LM: %u blocks, H %u, %u/%u heads x %u, ffn %u, vocab %u\n",
           h.n_layer, h.n_embd, h.n_head, h.n_head_kv, h.head_dim, h.n_ffn, h.n_vocab);
    printf("[MOSS]   rms_eps %.2e, rope_theta %.0f, ctx %u\n",
           h.rms_eps, h.rope_theta, h.n_ctx);
}

inline void moss_print_audio_hparams(const AudioTower & m) {
    const AudioHParams & h = m.hp;
    printf("[MOSS] audio tower: %u layers, d_model %u, %u heads, ffn %u\n",
           h.n_layer, h.d_model, h.n_head, h.n_ffn);
    printf("[MOSS]   mel %u bins @ %u Hz, n_fft %u, hop %u -> %.2f tokens/s\n",
           h.n_mels, h.sample_rate, h.n_fft, h.hop_length, h.tokens_per_second);
    printf("[MOSS]   conv stem %u ch / rate %u, adapter %u -> %u\n",
           h.downsample_hidden, h.downsample_rate, h.adapter_hidden, h.lm_embd);
    printf("[MOSS]   deepstack encoder layers:");
    for (int32_t l : h.deepstack_layers) {
        printf(" %d", l);
    }
    printf(" (injected into the first %zu LM layers)\n", h.deepstack_layers.size());
    printf("[MOSS]   audio token id %u, bos %u, eos %u, eps %.2e\n",
           h.audio_token_id, h.audio_bos_id, h.audio_eos_id, h.layer_norm_eps);
}

}  // namespace moss
