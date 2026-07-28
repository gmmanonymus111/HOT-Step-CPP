#pragma once
// mdx23c-ggml.h: native GGML implementation of MDX23C (TFC-TDF v3).
//
// Drives the SuperSep stage-3 drum split (kick/snare/toms/hh/ride/crash),
// replacing the ONNX Runtime path. Weights come from a GGUF produced by
// scripts/convert-mdx23c-gguf.py.
//
// Reference: ZFTurbo Music-Source-Separation-Training,
// models_without_stft/mdx23c_tfc_tdf_v3_no_stft.py — the STFT/iSTFT-amputated
// variant, because SuperSep does both natively in supersep-stft.h.
//
// TENSOR LAYOUT (the part that is easy to get wrong)
// --------------------------------------------------
// A torch conv tensor is [b, c, H, W] with W fastest. GGML is ne[0]-fastest,
// so the same data is ne = [W, H, c, b] and ggml_conv_2d's data layout
// [W, H, C, N] lines up with no shuffling. Likewise gguf reverses numpy dims
// on write, so a torch Conv2d weight (OC, IC, KH, KW) arrives as
// [KW, KH, IC, OC] — exactly ggml_conv_2d's kernel layout — and a
// ConvTranspose2d weight (IC, OC, KH, KW) arrives as [KW, KH, OC, IC], which
// is what ggml_conv_transpose_2d_p0 wants. Nothing needs transposing.
//
// The net runs its middle section transposed (torch `x.transpose(-1, -2)`),
// which in GGML is just swapping ne0/ne1, and matters because the TDF Linear
// operates on whatever ne0 is — the frequency axis.
//
// InstanceNorm2d(affine=True) normalises each (sample, channel) over its
// spatial extent. That is exactly ggml_group_norm with n_groups == n_channels,
// followed by a per-channel affine broadcast over ne0/ne1.
//
// Part of HOT-Step CPP. MIT license.

#ifndef HOT_STEP_MDX23C_GGML_H
#define HOT_STEP_MDX23C_GGML_H

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "backend.h"
#include "gguf-weights.h"
#include "weight-ctx.h"

#define MDX_NORM_EPS 1e-5f  // torch InstanceNorm2d default

// ── config ──────────────────────────────────────────────────────────────────

struct Mdx23cConfig {
    int n_scales = 0;
    int blocks_per_scale = 0;
    int num_channels = 0;   // c0
    int growth = 0;
    int bottleneck_factor = 0;
    int num_subbands = 0;
    int dim_c = 0;          // num_subbands * n_audio_channels * 2
    int dim_f = 0;
    int n_fft = 0;
    int hop_length = 0;
    int chunk_size = 0;
    int n_instruments = 0;
    int n_audio_channels = 0;
    int scale_h = 2, scale_w = 2;
    std::string instruments;
};

// ── weights ─────────────────────────────────────────────────────────────────

struct MdxTfcTdfBlock {
    struct ggml_tensor *tfc1_norm_w = nullptr, *tfc1_norm_b = nullptr, *tfc1_conv = nullptr;
    struct ggml_tensor *tdf_n1_w = nullptr, *tdf_n1_b = nullptr, *tdf_l1 = nullptr;
    struct ggml_tensor *tdf_n2_w = nullptr, *tdf_n2_b = nullptr, *tdf_l2 = nullptr;
    struct ggml_tensor *tfc2_norm_w = nullptr, *tfc2_norm_b = nullptr, *tfc2_conv = nullptr;
    struct ggml_tensor *shortcut = nullptr;
};

struct MdxScale {
    std::vector<MdxTfcTdfBlock> blocks;
    // encoder: downscale conv; decoder: upscale transposed conv
    struct ggml_tensor *rs_norm_w = nullptr, *rs_norm_b = nullptr, *rs_conv = nullptr;
};

struct Mdx23c {
    Mdx23cConfig cfg;

    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       cpu_backend = nullptr;
    ggml_backend_sched_t sched       = nullptr;
    WeightCtx            wctx        = {};

    struct ggml_tensor *          first_conv = nullptr;
    std::vector<MdxScale>         enc;
    std::vector<MdxTfcTdfBlock>   bot;
    std::vector<MdxScale>         dec;
    struct ggml_tensor *          final1 = nullptr;
    struct ggml_tensor *          final2 = nullptr;
};

// ── loading ─────────────────────────────────────────────────────────────────

static void mdx_load_tfc_tdf(Mdx23c * m, const GGUFModel & gf, const char * prefix,
                             std::vector<MdxTfcTdfBlock> & out, int n_blocks) {
    char nm[160];
    out.resize((size_t) n_blocks);
    for (int b = 0; b < n_blocks; b++) {
        MdxTfcTdfBlock & k = out[(size_t) b];
        auto L = [&](const char * suffix) {
            snprintf(nm, sizeof(nm), "%s.blk.%d.%s", prefix, b, suffix);
            return gf_load_tensor_f32(&m->wctx, gf, nm);
        };
        k.tfc1_norm_w = L("tfc1_norm_w");
        k.tfc1_norm_b = L("tfc1_norm_b");
        k.tfc1_conv   = L("tfc1_conv");
        k.tdf_n1_w    = L("tdf_n1_w");
        k.tdf_n1_b    = L("tdf_n1_b");
        k.tdf_l1      = L("tdf_l1");
        k.tdf_n2_w    = L("tdf_n2_w");
        k.tdf_n2_b    = L("tdf_n2_b");
        k.tdf_l2      = L("tdf_l2");
        k.tfc2_norm_w = L("tfc2_norm_w");
        k.tfc2_norm_b = L("tfc2_norm_b");
        k.tfc2_conv   = L("tfc2_conv");
        k.shortcut    = L("shortcut");
    }
}

static bool mdx_load(Mdx23c * m, const char * gguf_path) {
    BackendPair bp = backend_init("MDX23C");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched       = backend_sched_new(bp, 16384);

    GGUFModel gf = {};
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[MDX] FATAL: cannot load %s\n", gguf_path);
        return false;
    }

    Mdx23cConfig & c = m->cfg;
    c.n_scales          = (int) gf_get_u32(gf, "mdx23c.num_scales");
    c.blocks_per_scale  = (int) gf_get_u32(gf, "mdx23c.blocks_per_scale");
    c.num_channels      = (int) gf_get_u32(gf, "mdx23c.num_channels");
    c.growth            = (int) gf_get_u32(gf, "mdx23c.growth");
    c.bottleneck_factor = (int) gf_get_u32(gf, "mdx23c.bottleneck_factor");
    c.num_subbands      = (int) gf_get_u32(gf, "mdx23c.num_subbands");
    c.dim_c             = (int) gf_get_u32(gf, "mdx23c.dim_c");
    c.dim_f             = (int) gf_get_u32(gf, "mdx23c.dim_f");
    c.n_fft             = (int) gf_get_u32(gf, "mdx23c.n_fft");
    c.hop_length        = (int) gf_get_u32(gf, "mdx23c.hop_length");
    c.chunk_size        = (int) gf_get_u32(gf, "mdx23c.chunk_size");
    c.n_instruments     = (int) gf_get_u32(gf, "mdx23c.n_instruments");
    c.n_audio_channels  = (int) gf_get_u32(gf, "mdx23c.n_audio_channels");
    c.instruments       = gf_get_str(gf, "mdx23c.instruments");

    if (c.n_scales <= 0 || c.num_channels <= 0) {
        fprintf(stderr, "[MDX] FATAL: %s has no mdx23c.* metadata — was it "
                        "produced by scripts/convert-mdx23c-gguf.py?\n", gguf_path);
        gf_close(&gf);
        return false;
    }
    {
        int64_t idx = gguf_find_key(gf.gguf, "mdx23c.scale");
        if (idx >= 0 && gguf_get_arr_n(gf.gguf, idx) == 2) {
            const int32_t * a = (const int32_t *) gguf_get_arr_data(gf.gguf, idx);
            c.scale_h = a[0];
            c.scale_w = a[1];
        }
    }

    // 1 first_conv + 2 final + per-scale (3 resample + 13 per block) x2 sides
    const int per_block = 13;
    const int n_weights = 1 + 2
                        + 2 * c.n_scales * (3 + per_block * c.blocks_per_scale)
                        + per_block * c.blocks_per_scale;
    wctx_init(&m->wctx, n_weights + 32);

    char nm[160];
    m->first_conv = gf_load_tensor_f32(&m->wctx, gf, "first_conv");

    m->enc.resize((size_t) c.n_scales);
    for (int s = 0; s < c.n_scales; s++) {
        snprintf(nm, sizeof(nm), "enc.%d", s);
        mdx_load_tfc_tdf(m, gf, nm, m->enc[(size_t) s].blocks, c.blocks_per_scale);
        snprintf(nm, sizeof(nm), "enc.%d.down_norm_w", s);
        m->enc[(size_t) s].rs_norm_w = gf_load_tensor_f32(&m->wctx, gf, nm);
        snprintf(nm, sizeof(nm), "enc.%d.down_norm_b", s);
        m->enc[(size_t) s].rs_norm_b = gf_load_tensor_f32(&m->wctx, gf, nm);
        snprintf(nm, sizeof(nm), "enc.%d.down_conv", s);
        m->enc[(size_t) s].rs_conv = gf_load_tensor_f32(&m->wctx, gf, nm);
    }

    mdx_load_tfc_tdf(m, gf, "bot", m->bot, c.blocks_per_scale);

    m->dec.resize((size_t) c.n_scales);
    for (int s = 0; s < c.n_scales; s++) {
        snprintf(nm, sizeof(nm), "dec.%d.up_norm_w", s);
        m->dec[(size_t) s].rs_norm_w = gf_load_tensor_f32(&m->wctx, gf, nm);
        snprintf(nm, sizeof(nm), "dec.%d.up_norm_b", s);
        m->dec[(size_t) s].rs_norm_b = gf_load_tensor_f32(&m->wctx, gf, nm);
        snprintf(nm, sizeof(nm), "dec.%d.up_conv", s);
        m->dec[(size_t) s].rs_conv = gf_load_tensor_f32(&m->wctx, gf, nm);
        snprintf(nm, sizeof(nm), "dec.%d", s);
        mdx_load_tfc_tdf(m, gf, nm, m->dec[(size_t) s].blocks, c.blocks_per_scale);
    }

    m->final1 = gf_load_tensor_f32(&m->wctx, gf, "final1");
    m->final2 = gf_load_tensor_f32(&m->wctx, gf, "final2");

    if (!wctx_alloc(&m->wctx, m->backend)) {
        fprintf(stderr, "[MDX] FATAL: weight allocation failed\n");
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);

    fprintf(stderr, "[MDX] Loaded %s — scales=%d blocks/scale=%d c=%d growth=%d "
                    "subbands=%d instruments=%d (%s)\n",
            gguf_path, c.n_scales, c.blocks_per_scale, c.num_channels, c.growth,
            c.num_subbands, c.n_instruments, c.instruments.c_str());
    return true;
}

// ── graph helpers ───────────────────────────────────────────────────────────

// InstanceNorm2d(affine=True): per (sample, channel) over the spatial extent,
// then a per-channel scale/shift. x is [W, H, C].
static struct ggml_tensor * mdx_inorm(struct ggml_context * ctx, struct ggml_tensor * x,
                                      struct ggml_tensor * w, struct ggml_tensor * b) {
    const int64_t C = x->ne[2];
    x = ggml_group_norm(ctx, x, (int) C, MDX_NORM_EPS);
    struct ggml_tensor * wv = ggml_reshape_3d(ctx, w, 1, 1, C);
    struct ggml_tensor * bv = ggml_reshape_3d(ctx, b, 1, 1, C);
    return ggml_add(ctx, ggml_mul(ctx, x, wv), bv);
}

// norm -> GELU -> conv (the shape shared by tfc1, tfc2, downscale, upscale).
static struct ggml_tensor * mdx_norm_act(struct ggml_context * ctx, struct ggml_tensor * x,
                                         struct ggml_tensor * w, struct ggml_tensor * b) {
    return ggml_gelu(ctx, mdx_inorm(ctx, x, w, b));
}

// One residual TFC_TDF sub-block on x [F, T, C] (freq innermost — the net runs
// transposed through its middle, and the TDF Linear acts on ne0).
static struct ggml_tensor * mdx_tfc_tdf_block(struct ggml_context * ctx,
                                              const MdxTfcTdfBlock * k,
                                              struct ggml_tensor * x) {
    struct ggml_tensor * s = ggml_conv_2d(ctx, k->shortcut, x, 1, 1, 0, 0, 1, 1);

    // tfc1: norm -> act -> Conv2d 3x3 pad 1
    x = mdx_norm_act(ctx, x, k->tfc1_norm_w, k->tfc1_norm_b);
    x = ggml_conv_2d(ctx, k->tfc1_conv, x, 1, 1, 1, 1, 1, 1);

    // tdf: norm -> act -> Linear(f, f/bn) -> norm -> act -> Linear(f/bn, f)
    struct ggml_tensor * y = mdx_norm_act(ctx, x, k->tdf_n1_w, k->tdf_n1_b);
    y = ggml_mul_mat(ctx, k->tdf_l1, y);
    y = mdx_norm_act(ctx, y, k->tdf_n2_w, k->tdf_n2_b);
    y = ggml_mul_mat(ctx, k->tdf_l2, y);
    x = ggml_add(ctx, x, y);

    // tfc2: norm -> act -> Conv2d 3x3 pad 1
    x = mdx_norm_act(ctx, x, k->tfc2_norm_w, k->tfc2_norm_b);
    x = ggml_conv_2d(ctx, k->tfc2_conv, x, 1, 1, 1, 1, 1, 1);

    return ggml_add(ctx, x, s);
}

static struct ggml_tensor * mdx_tfc_tdf(struct ggml_context * ctx,
                                        const std::vector<MdxTfcTdfBlock> & blocks,
                                        struct ggml_tensor * x) {
    for (const MdxTfcTdfBlock & k : blocks) x = mdx_tfc_tdf_block(ctx, &k, x);
    return x;
}

// ── forward ─────────────────────────────────────────────────────────────────
//
// input:  complex spectrogram [T, dim_f, n_audio_channels*2] f32
//         (ne0 = time, matching torch [b, c, f, t])
// output: n_instruments * n_audio_channels*2 * dim_f * T, laid out per
//         instrument as [T, dim_f, ch*2] — i.e. torch (b, inst, c, f, t).
static void mdx_forward(Mdx23c * m, const float * input, int T, float * out) {
    const Mdx23cConfig & c = m->cfg;
    const int64_t k   = c.num_subbands;
    const int64_t Fs  = c.dim_f / k;             // folded freq
    const int64_t Cin = c.n_audio_channels * 2;  // 4

    const size_t ctx_size = (size_t) 32768 * ggml_tensor_overhead()
                          + ggml_graph_overhead_custom(32768, false);
    struct ggml_init_params gp = { ctx_size, NULL, true };
    struct ggml_context * ctx   = ggml_init(gp);
    struct ggml_cgraph *  graph = ggml_new_graph_custom(ctx, 32768, false);

    struct ggml_tensor * x_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, c.dim_f, Cin);
    ggml_set_name(x_in, "spec_in");
    ggml_set_input(x_in);

    // cac2cws: split the freq axis into (k outer, dim_f/k inner) and fold k
    // into channels. Contiguous reinterpretation only — merging ne2/ne3 gives
    // a combined channel index of c*k + k_idx, matching torch's reshape.
    struct ggml_tensor * x = ggml_reshape_4d(ctx, x_in, T, Fs, k, Cin);
    x = ggml_reshape_3d(ctx, x, T, Fs, k * Cin);   // [T, Fs, dim_c]
    struct ggml_tensor * mix = x;

    struct ggml_tensor * first_conv_out =
        ggml_conv_2d(ctx, m->first_conv, x, 1, 1, 0, 0, 1, 1);  // [T, Fs, c0]

    // transpose(-1,-2): torch [b,c,f,t] -> [b,c,t,f]; in GGML swap ne0/ne1.
    x = ggml_cont(ctx, ggml_permute(ctx, first_conv_out, 1, 0, 2, 3));  // [Fs, T, c0]

    std::vector<struct ggml_tensor *> skips;
    for (int s = 0; s < c.n_scales; s++) {
        x = mdx_tfc_tdf(ctx, m->enc[(size_t) s].blocks, x);
        skips.push_back(x);
        const MdxScale & sc = m->enc[(size_t) s];
        x = mdx_norm_act(ctx, x, sc.rs_norm_w, sc.rs_norm_b);
        x = ggml_conv_2d(ctx, sc.rs_conv, x, c.scale_w, c.scale_h, 0, 0, 1, 1);
    }

    x = mdx_tfc_tdf(ctx, m->bot, x);

    for (int s = 0; s < c.n_scales; s++) {
        const MdxScale & sc = m->dec[(size_t) s];
        x = mdx_norm_act(ctx, x, sc.rs_norm_w, sc.rs_norm_b);
        x = ggml_conv_transpose_2d_p0(ctx, sc.rs_conv, x, c.scale_w);
        x = ggml_concat(ctx, x, skips.back(), 2);  // channel concat
        skips.pop_back();
        x = mdx_tfc_tdf(ctx, sc.blocks, x);
    }

    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));  // back to [T, Fs, c0]
    x = ggml_mul(ctx, x, first_conv_out);                  // artifact reduction

    x = ggml_concat(ctx, mix, x, 2);                       // [T, Fs, dim_c + c0]
    x = ggml_conv_2d(ctx, m->final1, x, 1, 1, 0, 0, 1, 1);
    x = ggml_gelu(ctx, x);
    x = ggml_conv_2d(ctx, m->final2, x, 1, 1, 0, 0, 1, 1); // [T, Fs, n_inst*dim_c]

    // cws2cac: unfold the subbands back out of the channel axis.
    x = ggml_reshape_4d(ctx, x, T, Fs, k, (int64_t) c.n_instruments * Cin);
    x = ggml_reshape_3d(ctx, x, T, Fs * k, (int64_t) c.n_instruments * Cin);

    ggml_set_name(x, "out");
    ggml_set_output(x);
    ggml_build_forward_expand(graph, x);

    if (!ggml_backend_sched_alloc_graph(m->sched, graph)) {
        fprintf(stderr, "[MDX] FATAL: failed to allocate graph (T=%d)\n", T);
        exit(1);
    }

    ggml_backend_tensor_set(x_in, input, 0, (size_t) T * c.dim_f * Cin * sizeof(float));
    ggml_backend_sched_graph_compute(m->sched, graph);
    ggml_backend_tensor_get(x, out, 0, ggml_nbytes(x));

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
}

static void mdx_free(Mdx23c * m) {
    if (!m) return;
    if (m->sched) {
        ggml_backend_sched_free(m->sched);
        m->sched = nullptr;
    }
    wctx_free(&m->wctx);
    // Refcounted shared singleton — see the note in bs-roformer-ggml.h.
    backend_release(m->backend, m->cpu_backend);
    m->backend     = nullptr;
    m->cpu_backend = nullptr;
}

#endif  // HOT_STEP_MDX23C_GGML_H
