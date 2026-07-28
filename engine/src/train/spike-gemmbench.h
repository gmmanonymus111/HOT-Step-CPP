#pragma once
// spike-gemmbench.h — Rung `gemmbench`: BF16-GEMM lever measurement.
//
// EVIDENCE CODE, NOT PRODUCT CODE.
//
// WHY THIS EXISTS
// ---------------
// The LM trainer keeps the frozen base in BF16 but materialises a per-layer
// F32 "window" (lm-graph.h `lm_linear` -> `qwen3_f32`) because the mul_mat
// backward for the ACTIVATION gradient is, in ggml.c:6605:
//
//     ggml_out_prod(ctx, src0 /* = W */, ggml_transpose(ctx, grad))
//
// and GGML_OP_OUT_PROD on CUDA is F32-only:
//   * out-prod.cu:11-13  GGML_ASSERT(src0/src1/dst == GGML_TYPE_F32)
//   * out-prod.cu:63,76  cublasSgemm / cublasSgemmStridedBatched
//   * ggml-cuda.cu:5192  supports_op gates OUT_PROD to F32 on all three
//
// Because W must be F32 for the backward, the FORWARD mul_mat also sees an F32
// weight, so it lands in ggml_cuda_op_mul_mat_cublas's final `else` branch ->
// cublasSgemm -> TF32 tensor cores (common.cuh:1478 sets
// CUBLAS_TF32_TENSOR_OP_MATH on every handle).
//
// A BF16 weight would instead hit ggml-cuda.cu:1663 -> cublasGemmEx with
// CUDA_R_16BF / CUBLAS_COMPUTE_32F -> BF16 tensor cores.
//
// ggml.c:6601 itself records the alternative formulation for the activation
// gradient, commented out:
//     // ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, src0)), grad)
// which is dtype-agnostic and therefore works on a BF16 weight.
//
// This rung measures, on the trainer's REAL 4B shapes, whether the BF16
// forward + transpose-route backward actually beats the F32/TF32 status quo by
// enough to justify the plumbing.
//
// PATHS TIMED (per shape)
//   fwd_f32       mul_mat(W_f32[K,N],  x_f32[K,S])            -> [N,S]   (a)
//   fwd_bf16      mul_mat(W_bf16[K,N], x_f32[K,S])            -> [N,S]   (b)
//   bwd_outprod   out_prod(W_f32[K,N], transpose(g[N,S]))     -> [K,S]   (a)
//   bwd_mm_cont   mul_mat(cont(transpose(W_bf16)), g[N,S])    -> [K,S]   (c, incl. cont)
//   bwd_mm_pre    mul_mat(Wt_bf16[N,K], g[N,S])               -> [K,S]   (c, cont hoisted)
//   cast_f32      cpy(W_bf16 -> W_f32)   the CURRENT per-segment window cost
//   cont_t_bf16   cont(transpose(W_bf16))  the cont in isolation
//
// `bwd_mm_pre` is the realistic implementation: W is FROZEN, so its transpose
// is a loop invariant and can be built once per layer-window instead of per
// mul_mat, exactly like the F32 window is today.

#include "spike-common.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

// ─── bf16 conversion (self-contained; no dependency on ggml type-trait API) ──

static inline uint16_t gb_f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (uint16_t) ((u >> 16) | 64);  // NaN, keep it quiet
    }
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);  // round-to-nearest-even
}

static inline float gb_bf16_to_f32(uint16_t h) {
    uint32_t u = ((uint32_t) h) << 16;
    float    f;
    memcpy(&f, &u, 4);
    return f;
}

static inline void gb_set_bf16(ggml_tensor * t, const std::vector<float> & v) {
    GGML_ASSERT((int64_t) v.size() == ggml_nelements(t));
    std::vector<uint16_t> b(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        b[i] = gb_f32_to_bf16(v[i]);
    }
    ggml_backend_tensor_set(t, b.data(), 0, b.size() * sizeof(uint16_t));
}

// ─── shape table: Qwen3-4B geometry ─────────────────────────────────────────
//
// hidden_size H = 2560, intermediate F = 9728, n_heads 32, n_kv_heads 8,
// head_dim 128  =>  q 2560->4096, k/v 2560->1024, o 4096->2560,
//                   gate/up 2560->9728, down 9728->2560.
// ggml layout: mul_mat(W, x) with W = [K=in, N=out], x = [K, S] -> [N, S].

struct GbShape {
    const char * name;
    int64_t      K;  // in_features   (W->ne[0])
    int64_t      N;  // out_features  (W->ne[1])
};

struct GbResult {
    std::string name;
    int64_t     K = 0, N = 0, S = 0;
    double      fwd_f32 = 0, fwd_bf16 = 0;
    double      bwd_outprod = 0, bwd_mm_cont = 0, bwd_mm_pre = 0;
    double      cast_f32 = 0, cont_t_bf16 = 0;
    double      par_max_rel = 0, par_rms_rel = 0, par_cosine = 0;
    bool        par_finite = true;
};

// Time a single-op graph. Returns mean ms per graph execution.
static double gb_time_graph(ggml_backend_t backend, ggml_context * cctx, ggml_gallocr_t galloc, ggml_tensor * out,
                            int warmup, int reps) {
    ggml_cgraph * gf = ggml_new_graph(cctx);
    ggml_build_forward_expand(gf, out);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "[gemmbench] gallocr_alloc_graph FAILED\n");
        return -1.0;
    }
    for (int i = 0; i < warmup; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < reps; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    const int64_t t1 = ggml_time_us();
    return (double) (t1 - t0) / 1000.0 / (double) reps;
}

// Execute a single-op graph ONCE and read the result back to host.
static std::vector<float> gb_eval(ggml_backend_t backend, ggml_tensor * (*build)(ggml_context *, ggml_tensor **),
                                  ggml_tensor ** ins) {
    ggml_init_params cp = { GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   cctx   = ggml_init(cp);
    ggml_gallocr_t   galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_tensor *    out    = build(cctx, ins);
    ggml_cgraph *    gf     = ggml_new_graph(cctx);
    ggml_build_forward_expand(gf, out);
    std::vector<float> v;
    if (ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        v.resize((size_t) ggml_nelements(out));
        ggml_backend_tensor_get(out, v.data(), 0, v.size() * sizeof(float));
    }
    ggml_gallocr_free(galloc);
    ggml_free(cctx);
    return v;
}

// Gradient parity between the two activation-backward formulations.
struct GbParity {
    double max_rel = 0, rms_rel = 0, cosine = 0;
    bool   finite = true;
};

static GbParity gb_parity(const std::vector<float> & a, const std::vector<float> & b) {
    GbParity p;
    if (a.size() != b.size() || a.empty()) {
        p.finite = false;
        return p;
    }
    double dot = 0, na = 0, nb = 0, se = 0, sa = 0, amax = 0, dmax = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            p.finite = false;
        }
        const double x = a[i], y = b[i], d = x - y;
        dot += x * y; na += x * x; nb += y * y;
        se  += d * d; sa += x * x;
        amax = std::max(amax, fabs(x));
        dmax = std::max(dmax, fabs(d));
    }
    p.max_rel = dmax / std::max(amax, 1e-30);
    p.rms_rel = sqrt(se) / std::max(sqrt(sa), 1e-30);
    p.cosine  = dot / std::max(sqrt(na) * sqrt(nb), 1e-30);
    return p;
}

static void gb_print_row(const GbResult & r) {
    // 2*K*N*S FLOPs for every GEMM path here (out_prod included: same contraction volume).
    const double gflop = 2.0 * (double) r.K * (double) r.N * (double) r.S / 1e9;
    // GFLOP / ms  ==  1e9 FLOP / 1e-3 s  ==  1e12 FLOP/s  ==  TFLOP/s already.
    auto         tf    = [&](double ms) { return ms > 0 ? gflop / ms : 0.0; };
    printf("%-10s %6lld %6lld %6lld | %8.3f %7.1f | %8.3f %7.1f | %8.3f %7.1f | %8.3f %7.1f | %8.3f %7.1f | %8.3f %8.3f\n",
           r.name.c_str(), (long long) r.K, (long long) r.N, (long long) r.S,
           r.fwd_f32, tf(r.fwd_f32),
           r.fwd_bf16, tf(r.fwd_bf16),
           r.bwd_outprod, tf(r.bwd_outprod),
           r.bwd_mm_cont, tf(r.bwd_mm_cont),
           r.bwd_mm_pre, tf(r.bwd_mm_pre),
           r.cast_f32, r.cont_t_bf16);
}

static int spike_gemmbench(int64_t S_override, int reps_override) {
    BackendPair bp = backend_init("GEMMBENCH");
    if (!bp.has_gpu) {
        fprintf(stderr, "[gemmbench] no GPU backend; this rung is CUDA-only\n");
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }
    ggml_backend_t backend = bp.backend;

    printf("[gemmbench] backend: %s\n", ggml_backend_name(backend));
    printf("[gemmbench] VRAM in use at start: %zu MB\n", spike_vram_used_mb(backend));

    const GbShape shapes[] = {
        { "q_proj",    2560, 4096 },
        { "k_proj",    2560, 1024 },
        { "v_proj",    2560, 1024 },
        { "o_proj",    4096, 2560 },
        { "gate_proj", 2560, 9728 },
        { "up_proj",   2560, 9728 },
        { "down_proj", 9728, 2560 },
    };
    const int n_shapes = (int) (sizeof(shapes) / sizeof(shapes[0]));

    std::vector<int64_t> seqs;
    if (S_override > 0) {
        seqs.push_back(S_override);
    } else {
        seqs.push_back(512);
        seqs.push_back(2048);
        seqs.push_back(2944);
    }

    printf("\n%-10s %6s %6s %6s | %8s %7s | %8s %7s | %8s %7s | %8s %7s | %8s %7s | %8s %8s\n",
           "shape", "K", "N", "S",
           "fwdF32", "TF/s", "fwdBF16", "TF/s", "outprod", "TF/s", "mm+cont", "TF/s", "mm(pre)", "TF/s",
           "castF32", "contT");
    printf("%s\n", std::string(150, '-').c_str());

    std::vector<GbResult> all;

    for (size_t si = 0; si < seqs.size(); si++) {
        const int64_t S = seqs[si];
        for (int i = 0; i < n_shapes; i++) {
            const int64_t K = shapes[i].K, N = shapes[i].N;

            GbResult r;
            r.name = shapes[i].name;
            r.K = K; r.N = N; r.S = S;

            // ── static tensors ──────────────────────────────────────────
            ggml_init_params sp = { 16 * ggml_tensor_overhead(), nullptr, true };
            ggml_context *   sctx = ggml_init(sp);

            ggml_tensor * W_f32  = ggml_new_tensor_2d(sctx, GGML_TYPE_F32,  K, N);
            ggml_tensor * W_bf16 = ggml_new_tensor_2d(sctx, GGML_TYPE_BF16, K, N);
            ggml_tensor * Wt_bf16 = ggml_new_tensor_2d(sctx, GGML_TYPE_BF16, N, K);  // pre-transposed
            ggml_tensor * W_f32_dst = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, K, N); // cast target
            ggml_tensor * x      = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, K, S);
            ggml_tensor * g      = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, N, S);

            ggml_backend_buffer_t sbuf = ggml_backend_alloc_ctx_tensors(sctx, backend);
            if (!sbuf) {
                fprintf(stderr, "[gemmbench] static alloc FAILED for %s S=%lld\n", shapes[i].name, (long long) S);
                ggml_free(sctx);
                backend_release(bp.backend, bp.cpu_backend);
                return 1;
            }

            // ── fill ────────────────────────────────────────────────────
            {
                SpikeRng           rng = { 0x9E3779B97F4A7C15ull ^ (uint64_t) (i * 131 + S) };
                std::vector<float> vW((size_t) K * N), vx((size_t) K * S), vg((size_t) N * S);
                srng_fill_normal(&rng, vW, 1.0f / sqrtf((float) K));
                srng_fill_normal(&rng, vx, 1.0f);
                srng_fill_normal(&rng, vg, 1.0f);
                spike_set(W_f32, vW);
                gb_set_bf16(W_bf16, vW);
                spike_set(x, vx);
                spike_set(g, vg);
                // Wt_bf16 = transpose(W) in BF16
                std::vector<float> vWt((size_t) K * N);
                for (int64_t c = 0; c < N; c++) {
                    for (int64_t rr = 0; rr < K; rr++) {
                        vWt[(size_t) rr * N + c] = vW[(size_t) c * K + rr];
                    }
                }
                gb_set_bf16(Wt_bf16, vWt);
            }

            // Bigger shapes get fewer reps; keep each measurement ~0.1-1 s.
            const double gflop = 2.0 * (double) K * (double) N * (double) S / 1e9;
            int          reps  = reps_override > 0 ? reps_override : (int) std::max(10.0, std::min(200.0, 3000.0 / gflop));
            const int    warm  = std::max(3, reps / 10);

            // ── compute ctx + gallocr (one per measurement, so the graph
            //    allocator never reuses a buffer across differently-shaped ops)
            auto run = [&](ggml_tensor * (*build)(ggml_context *, ggml_tensor **), ggml_tensor ** ins) -> double {
                ggml_init_params cp = { GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead(),
                                        nullptr, true };
                ggml_context *   cctx   = ggml_init(cp);
                ggml_gallocr_t   galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                ggml_tensor *    out    = build(cctx, ins);
                double           ms     = gb_time_graph(backend, cctx, galloc, out, warm, reps);
                ggml_gallocr_free(galloc);
                ggml_free(cctx);
                return ms;
            };

            {  // fwd_f32: mul_mat(W_f32, x)
                ggml_tensor * ins[2] = { W_f32, x };
                r.fwd_f32 = run([](ggml_context * c, ggml_tensor ** in) { return ggml_mul_mat(c, in[0], in[1]); }, ins);
            }
            {  // fwd_bf16: mul_mat(W_bf16, x)
                ggml_tensor * ins[2] = { W_bf16, x };
                r.fwd_bf16 = run([](ggml_context * c, ggml_tensor ** in) { return ggml_mul_mat(c, in[0], in[1]); }, ins);
            }
            {  // bwd_outprod: out_prod(W_f32, transpose(g))   <- what ggml emits today
                ggml_tensor * ins[2] = { W_f32, g };
                r.bwd_outprod = run(
                    [](ggml_context * c, ggml_tensor ** in) {
                        return ggml_out_prod(c, in[0], ggml_transpose(c, in[1]));
                    },
                    ins);
            }
            {  // bwd_mm_cont: mul_mat(cont(transpose(W_bf16)), g)   <- incl. the cont
                ggml_tensor * ins[2] = { W_bf16, g };
                r.bwd_mm_cont = run(
                    [](ggml_context * c, ggml_tensor ** in) {
                        return ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, in[0])), in[1]);
                    },
                    ins);
            }
            {  // bwd_mm_pre: mul_mat(Wt_bf16, g)   <- cont hoisted out of the step
                ggml_tensor * ins[2] = { Wt_bf16, g };
                r.bwd_mm_pre = run([](ggml_context * c, ggml_tensor ** in) { return ggml_mul_mat(c, in[0], in[1]); }, ins);
            }
            {  // cast_f32: cpy(W_bf16 -> f32)   <- the CURRENT per-segment window cost
                ggml_tensor * ins[2] = { W_bf16, W_f32_dst };
                r.cast_f32 = run([](ggml_context * c, ggml_tensor ** in) { return ggml_cpy(c, in[0], in[1]); }, ins);
            }
            {  // cont_t_bf16: cont(transpose(W_bf16)) alone
                ggml_tensor * ins[2] = { W_bf16, nullptr };
                r.cont_t_bf16 =
                    run([](ggml_context * c, ggml_tensor ** in) { return ggml_cont(c, ggml_transpose(c, in[0])); }, ins);
            }

            {  // gradient parity: out_prod(W_f32, gT)  vs  mul_mat(Wt_bf16, g)
               // Both produce dL/dx of shape [K,S]. The F32/TF32 arm is the
               // reference (TF32 = 10-bit mantissa; BF16 = 8-bit).
                ggml_tensor *      ia[2] = { W_f32, g };
                std::vector<float> ref   = gb_eval(
                    backend,
                    [](ggml_context * c, ggml_tensor ** in) { return ggml_out_prod(c, in[0], ggml_transpose(c, in[1])); },
                    ia);
                ggml_tensor *      ib[2] = { Wt_bf16, g };
                std::vector<float> tst =
                    gb_eval(backend, [](ggml_context * c, ggml_tensor ** in) { return ggml_mul_mat(c, in[0], in[1]); }, ib);
                GbParity p    = gb_parity(ref, tst);
                r.par_max_rel = p.max_rel;
                r.par_rms_rel = p.rms_rel;
                r.par_cosine  = p.cosine;
                r.par_finite  = p.finite;
            }

            gb_print_row(r);
            fflush(stdout);
            all.push_back(r);

            ggml_backend_buffer_free(sbuf);
            ggml_free(sctx);
        }
        printf("%s\n", std::string(150, '-').c_str());
    }

    // ── aggregate: realistic per-layer step mixes ───────────────────────────
    //
    // A checkpointed segment runs forward TWICE (once in the outer no-grad
    // pass, once recomputed inside the segment) and the activation backward
    // ONCE, per layer per step.
    //
    //   current : 2*fwd_f32     + 1*bwd_outprod + 1*cast_f32       (F32 window)
    //   bf16    : 2*fwd_bf16    + 1*bwd_mm_pre  + 1*cont_t_bf16    (BF16 + pre-transposed window)
    //   bf16nc  : 2*fwd_bf16    + 1*bwd_mm_cont                    (cont inside the step, no window)
    // The trainer has TWO base-weight modes (lm-train-run.h:125-142):
    //   naive    : full F32 mirror allocated once, BF16 buffer released
    //              -> no per-segment cast, but 2x weight VRAM
    //   low-vram : BF16 base stays resident, per-segment F32 window emitted
    //              in-graph by qwen3_f32() (lm-graph.h:344)   <- the 4B path
    printf("\n[gemmbench] per-layer step mix (7 projections summed), ms per layer per step\n");
    printf("  cur_naive = 2*fwdF32  + outprod                (F32 mirror, 2x weight VRAM)\n");
    printf("  cur_lowvr = 2*fwdF32  + outprod  + castF32     (BF16 base + per-segment F32 window)\n");
    printf("  bf16_pre  = 2*fwdBF16 + mm(pre)  + contT       (BF16 base + per-segment BF16 transposed window)\n");
    printf("  bf16_cont = 2*fwdBF16 + mm+cont                (cont rebuilt inside every mul_mat)\n\n");
    printf("%-8s %11s %11s %11s %11s | %11s %11s\n", "S", "cur_naive", "cur_lowvr", "bf16_pre", "bf16_cont",
           "spd vs lowvr", "spd vs naive");
    for (size_t si = 0; si < seqs.size(); si++) {
        const int64_t S = seqs[si];
        double c_naive = 0, c_lowvr = 0, b_pre = 0, b_cont = 0;
        for (const GbResult & r : all) {
            if (r.S != S) continue;
            c_naive += 2.0 * r.fwd_f32  + r.bwd_outprod;
            c_lowvr += 2.0 * r.fwd_f32  + r.bwd_outprod + r.cast_f32;
            b_pre   += 2.0 * r.fwd_bf16 + r.bwd_mm_pre  + r.cont_t_bf16;
            b_cont  += 2.0 * r.fwd_bf16 + r.bwd_mm_cont;
        }
        printf("%-8lld %11.3f %11.3f %11.3f %11.3f | %10.2fx %10.2fx\n", (long long) S, c_naive, c_lowvr, b_pre,
               b_cont, b_pre > 0 ? c_lowvr / b_pre : 0.0, b_pre > 0 ? c_naive / b_pre : 0.0);
    }

    // GEMM-only view: strip every cast/transpose so the pure tensor-core delta
    // is visible (this is the ceiling the plumbing could ever approach).
    printf("\n[gemmbench] GEMM-only (2*fwd + bwd, no cast/cont), ms per layer per step\n");
    printf("%-8s %11s %11s %10s\n", "S", "f32/TF32", "bf16", "speedup");
    for (size_t si = 0; si < seqs.size(); si++) {
        const int64_t S = seqs[si];
        double f = 0, b = 0;
        for (const GbResult & r : all) {
            if (r.S != S) continue;
            f += 2.0 * r.fwd_f32  + r.bwd_outprod;
            b += 2.0 * r.fwd_bf16 + r.bwd_mm_pre;
        }
        printf("%-8lld %11.3f %11.3f %9.2fx\n", (long long) S, f, b, b > 0 ? f / b : 0.0);
    }

    // ── gradient parity ─────────────────────────────────────────────────────
    printf("\n[gemmbench] activation-gradient parity: out_prod(W_f32,gT) [TF32 ref] vs mul_mat(Wt_bf16,g)\n");
    printf("%-10s %6s | %12s %12s %14s %8s\n", "shape", "S", "max_rel", "rms_rel", "cosine", "finite");
    for (const GbResult & r : all) {
        printf("%-10s %6lld | %12.4e %12.4e %14.9f %8s\n", r.name.c_str(), (long long) r.S, r.par_max_rel,
               r.par_rms_rel, r.par_cosine, r.par_finite ? "yes" : "NO");
    }

    printf("\n[gemmbench] VRAM in use at end: %zu MB\n", spike_vram_used_mb(backend));
    backend_release(bp.backend, bp.cpu_backend);
    return 0;
}
