#pragma once
// mm3-f32-isolate.h — the F32-isolated gradient gate's model surgery.
//
// ── WHY THIS EXISTS ─────────────────────────────────────────────────────────
//
// `mm3-lm-train --fd-check` compares the same gradient computed two ways:
// whole-graph autodiff (the "naive" arm) and segmented recompute with a
// surrogate loss (the checkpointed arm). Those are structurally different
// pieces of machinery, so a wiring error in the two hooks the MM3 trainer added
// to lm-ckpt.h — the UNTIED SCORED HEAD and the FRAME-EMBEDDING ENTRY — shows
// up in that comparison and nowhere else. It is the DiT-trainer delta fiasco's
// exact bug class: the loss falls, and the deltas are wrong.
//
// Against the f16 base that comparison MEASURES but cannot DECIDE. Both routes
// accumulate through 36 layers of f16 matmul in different orders, and the
// disagreement that produces (~1.35e-02) is indistinguishable from the
// disagreement a real wiring bug would produce. ACE holds the same comparison
// to 2e-3 — but it trains against an F32 weight mirror, so its floor is F32
// reassociation rather than f16 rounding.
//
// An F32 mirror of MM3's 8.6B is ~34 GB and will never fit. This does the same
// job for ~2 GB by isolating only the part the gate actually reasons about.
//
// ── WHAT IS ISOLATED, AND WHAT DELIBERATELY IS NOT ──────────────────────────
//
//   TRUNCATED  the trunk to `n_layers` layers. The gate tests WIRING, not the
//              model: two layers exercise every LoRA slot, both norms, the
//              residual joins, attention and the MLP. Thirty-six of them
//              exercise the same code thirty-four more times, for 17x the F32
//              memory. The truncation is what makes the isolation affordable.
//
//   MIRRORED   every weight in those layers, plus final_norm, to F32, and the
//              SCORED HEAD SLICE ([H, 16389], 268 MB) to F32. These are the
//              tensors that carry ARITHMETIC — every matmul and every norm the
//              two arms could disagree about.
//
//   LEFT F16   embed_tokens and audio_embd. Not an oversight and not a
//              compromise: both are consumed ONLY by ggml_get_rows, a lookup
//              that widens f16 -> F32 exactly. There is no rounding to remove,
//              so mirroring them would cost 3.3 GB to change nothing.
//
//   NOT FREED  the original weight buffer. lm_build_f32_mirror() releases it,
//              which it can afford because it mirrors EVERYTHING. Here the
//              untouched layers, the full head and the audio embeddings all
//              still live in it, so freeing it would dangle three pointers.
//              The f16 base stays resident; at two layers there is room.
//
// The residual floor after this is F32 reassociation — the same floor ACE's
// 2e-3 bar was set against, which is what makes that bar transferable.

#include "train/lm-graph.h"
#include "train/mm3-lm-load.h"

#include <cstring>
#include <string>
#include <vector>

struct MM3F32Slice {
    ggml_context *        ctx       = nullptr;
    ggml_backend_buffer_t buf       = nullptr;
    size_t                bytes     = 0;
    int                   n_tensors = 0;
    int                   n_layers  = 0;
};

static void mm3_f32_isolate_free(MM3F32Slice * M) {
    if (M->buf) {
        ggml_backend_buffer_free(M->buf);
    }
    if (M->ctx) {
        ggml_free(M->ctx);
    }
    *M = MM3F32Slice{};
}

// Truncate to `n_layers` and mirror the arithmetic-carrying weights to F32.
//
// AFTERWARDS `t->lm_head` IS THE SCORED SLICE, not the full head, and
// `t->head_slice_row0` is 0 — so mm3_lm_train_out_slice() and LmCkptCfg::
// head_row0 keep addressing the same rows without either caller knowing. That
// indirection is the only reason this needs no change in the graph builders.
static bool mm3_f32_isolate(MM3TrainLm * t, int n_layers, MM3F32Slice * M, std::string * err) {
    Qwen3LM & lm = t->lm;
    if (n_layers < 1 || n_layers > lm.cfg.n_layers) {
        *err = "f32-isolate: layer count out of range";
        return false;
    }

    const int64_t H  = lm.cfg.hidden_size;
    const int64_t SL = mm3_lm_train_slice_size(*t);
    const int     n  = n_layers * 11 + 8;

    ggml_init_params p = { (size_t) n * ggml_tensor_overhead() + 4096, nullptr, true };
    M->ctx             = ggml_init(p);
    if (!M->ctx) {
        *err = "f32-isolate: context alloc failed";
        return false;
    }

    struct Slot {
        ggml_tensor ** field;
        ggml_tensor *  src;
        ggml_tensor *  dst;
        int64_t        row0;
        int64_t        rows;
    };
    std::vector<Slot> slots;

    // `rows` < 0 means "the whole tensor"; the head is the one partial copy.
    auto add = [&](ggml_tensor ** field, int64_t row0 = 0, int64_t rows = -1) {
        if (!field || !*field) {
            return;  // q_norm/k_norm are absent on some configs
        }
        ggml_tensor * s = *field;
        const int64_t r = rows < 0 ? s->ne[1] : rows;
        ggml_tensor * d = ggml_new_tensor_2d(M->ctx, GGML_TYPE_F32, s->ne[0], r);
        ggml_set_name(d, s->name);
        M->bytes += ggml_nbytes(d);
        slots.push_back({ field, s, d, row0, r });
    };

    for (int i = 0; i < n_layers; i++) {
        Qwen3Layer & ly = lm.layers[i];
        if (!ly.q_proj || !ly.k_proj || !ly.v_proj || !ly.o_proj || !ly.gate_proj || !ly.up_proj || !ly.down_proj) {
            char b[160];
            snprintf(b, sizeof(b), "f32-isolate: layer %d has fused projections — needs an unfused load", i);
            *err = b;
            return false;
        }
        add(&ly.input_layernorm);
        add(&ly.post_attn_layernorm);
        add(&ly.q_proj);
        add(&ly.k_proj);
        add(&ly.v_proj);
        add(&ly.o_proj);
        add(&ly.q_norm);
        add(&ly.k_norm);
        add(&ly.gate_proj);
        add(&ly.up_proj);
        add(&ly.down_proj);
    }
    add(&lm.final_norm);
    add(&t->lm_head, (int64_t) t->eos_audio, SL);

    M->buf = ggml_backend_alloc_ctx_tensors(M->ctx, lm.backend);
    if (!M->buf) {
        char b[160];
        snprintf(b, sizeof(b), "f32-isolate: allocation failed (%.1f MB)", M->bytes / 1048576.0);
        *err = b;
        return false;
    }
    ggml_backend_buffer_set_usage(M->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<uint8_t> raw;
    std::vector<float>   f32;
    for (Slot & sl : slots) {
        ggml_tensor * s = sl.src;
        // Copy only the rows being mirrored. For everything but the head that is
        // the whole tensor; for the head it is 16,389 rows out of 200,000, which
        // is the difference between 268 MB and 3.3 GB.
        const size_t row_bytes = ggml_row_size(s->type, s->ne[0]);
        const size_t off       = (size_t) sl.row0 * row_bytes;
        const size_t nbytes    = (size_t) sl.rows * row_bytes;
        const size_t ne        = (size_t) (sl.rows * s->ne[0]);
        raw.resize(nbytes);
        ggml_backend_tensor_get(s, raw.data(), off, nbytes);

        if (s->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(sl.dst, raw.data(), 0, nbytes);
        } else if (s->type == GGML_TYPE_F16) {
            f32.resize(ne);
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), f32.data(), (int64_t) ne);
            ggml_backend_tensor_set(sl.dst, f32.data(), 0, ne * sizeof(float));
        } else if (s->type == GGML_TYPE_BF16) {
            f32.resize(ne);
            const uint16_t * u = (const uint16_t *) raw.data();
            for (size_t j = 0; j < ne; j++) {
                const uint32_t b = (uint32_t) u[j] << 16;
                float          v;
                memcpy(&v, &b, 4);
                f32[j] = v;
            }
            ggml_backend_tensor_set(sl.dst, f32.data(), 0, ne * sizeof(float));
        } else {
            // A quantized base is a legitimate thing to TRAIN (the in-graph cast
            // handles it) but not a thing to ISOLATE: dequantizing here would
            // measure the quantizer, not the wiring.
            char b[200];
            snprintf(b, sizeof(b),
                     "f32-isolate: '%s' is %s — the gate needs an F16/BF16/F32 base "
                     "(run the gate on f16 even when training on q8_0)",
                     s->name, ggml_type_name(s->type));
            *err = b;
            return false;
        }
        *sl.field = sl.dst;
    }

    lm.cfg.n_layers    = n_layers;
    t->head_slice_row0 = 0;  // the head IS the slice now
    M->n_tensors       = (int) slots.size();
    M->n_layers        = n_layers;
    fprintf(stderr,
            "[mm3-fd] F32 isolation: %d layers, %d tensors, %.1f MB (head slice %lld x %lld F32; "
            "embeddings left F16 - get_rows is an exact widening)\n",
            n_layers, M->n_tensors, M->bytes / 1048576.0, (long long) H, (long long) SL);
    return true;
}
