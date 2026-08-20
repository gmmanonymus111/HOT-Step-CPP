#pragma once
// train/mm3-lm-train-run.h — MiniMax-Music3 LM LoRA trainer.
//
// HOT-Step file. Included by tools/ace-train.cpp only.
//
// ── WHAT IS NEW HERE, AND WHAT IS NOT ───────────────────────────────────────
//
// Almost nothing about the OPTIMISATION is new. `train/lm-graph.h` already
// carries a trainable cache-free unfused Qwen3 forward, `lm-optim.h` already
// carries AdamW *and Muon* with per-parameter rule selection, `lm-export.h`
// already writes PEFT safetensors, and MM3's LM is Qwen3-8B. So this file is
// the MM3-shaped parts only:
//
//   1. the DATA (codes + captions + lyrics -> a teacher-forced sequence),
//   2. the INPUT EMBEDDING (two tables, two files, summed and scaled),
//   3. the OUTPUT SLICE (semantic codes + EOS, not the 200k vocab),
//   4. the training loop that hangs those on the existing machinery.
//
// Everything in 1-3 was built and falsified first, without an optimiser
// attached, by `ace-train mm3-lm-loss` — see its header. Attaching a backward
// pass to an unverified sequence is how the DiT-trainer delta fiasco happened.
//
// ── MUON COMES FREE, AND THAT IS THE POINT ──────────────────────────────────
//
// `lm_optim_init` classifies EVERY parameter itself: a genuine 2-D matrix whose
// short side clears `muon.min_dim` goes on Muon, everything else falls to
// AdamW. A rank-256 LoRA on q/k/v/o + gate/up/down is 252 matrices of
// [in, 256] / [256, out] — all comfortably Muon-eligible. So `--optimizer muon`
// works here for exactly the same reason it works for ACE, with the same knobs
// (`--muon-lr-scale/-momentum/-ns-steps/-nesterov/-min-dim/-bucket`), and
// `opt.n_muon` is logged because a run where Muon silently classified zero
// parameters and trained as AdamW is the failure mode to watch for.
//
// ── THE OUTPUT SLICE: SEMANTIC + EOS ────────────────────────────────────────
//
// The AR loop masks its logits to "semantic codes + EOS" — 16,385 live
// candidates out of a 200,000-row head. Training over the full vocabulary would
// spend 1.12 GiB on logits (and the same again on their gradient) to supervise
// 8% of it.
//
// `eos_audio` (151670) sits just BELOW `semantic_vocab_offset` (151675), so
// [eos_audio, semantic_offset + semantic_size) is ONE CONTIGUOUS ROW RANGE of
// 16,389 rows — a single ggml_view_2d, no gather, no copy. The four rows
// between them are the caption/lyric delimiters; they ride along in the softmax
// denominator, which is a deliberate and stated approximation:
//
//   * at inference they are masked out, so the training distribution is very
//     slightly wider than the sampling one;
//   * the base model already puts negligible mass on a caption delimiter at an
//     audio position, and training only pushes it lower;
//   * the exact alternative is a concatenated [H, 16385] head built once
//     outside the graph (134 MB at f16) — the right fix if this is ever
//     measured to matter. It has not been.
//
// ── EOS SUPERVISION IS A CROP PROPERTY ──────────────────────────────────────
//
// The lm2 run trained on intros only, because `max_frames` truncated from the
// START; the random-crop patch fixed it and, just as importantly, restored EOS
// supervision by marking `has_audio_end` ONLY when the crop actually reaches
// the track's end. Both live here from day one:
//
//   crop reaches the end -> inputs are all K frames, targets are
//                           f[1..K-1] then EOS          (K+1 supervised)
//   crop does not        -> inputs are f[0..K-2],
//                           targets are f[0..K-1]       (K supervised)
//
// ── OUTPUT LAYOUT ───────────────────────────────────────────────────────────
//
// Each checkpoint is a PEFT directory `<out>/ckpt-<step>/` (adapter_config.json
// + adapter_model.safetensors), plus a `.json` sidecar beside the safetensors.
// Two consumers, one file:
//   * the python side (lm_sft_infer, SimpleTuner) reads a PEFT dir;
//   * pointing `--out` at `<adapters>/mm3-lm-adapters/<run>` makes the shipped
//     server lister find it with NO changes — it scans two directory levels and
//     reads `<file>.json` as the sidecar — so a finished checkpoint appears in
//     the MM3 adapter picker with its trigger word and recommended scales.
// The exported PEFT key names (`base_model.model.model.layers.N.<mod>.lora_A`)
// are already exactly what `minimax/mm3-lm-adapter.h` parses. Verified, not
// assumed: that parser strips `language_model.` and `base_model.model.` and
// then matches the same seven module strings `lm_slot_peft_name` emits.

#include "train/lm-graph.h"
#include "train/lm-optim.h"
#include "train/lm-data.h"
#include "train/lm-export.h"
#include "train/lm-ckpt.h"
#include "train/mm3-lm-load.h"
#include "minimax/mm3-request.h"
#include "minimax/mm3-tokenizer.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

struct MM3LmTrainArgs {
    std::string lm_path, depth_path, manifest, captions_dir, codes_dir, out_dir;
    int         rank = 256, alpha = 256;
    double      lr = 8e-5, weight_decay = 0.01, grad_clip = 1.0;
    int         steps = 800, save_every = 100, warmup = 0;
    int64_t     max_frames = 1500;
    std::string crop_mode = "random";     // random | beginning
    int         grad_accum = 1, seed = 42;
    std::string optimizer = "adamw";      // adamw | muon
    float       muon_lr_scale = 1.0f, muon_momentum = 0.95f;
    int         muon_ns_steps = 5, muon_min_dim = 16, muon_bucket = 16;
    bool        muon_nesterov = true;
    std::string trigger;                  // recorded in the sidecar, not prepended here
    std::string dataset_name;
    // Per-layer gradient checkpointing. ON by default and that is not a
    // preference: the MM3 prompt is ~1,100 tokens, so even a 128-frame crop
    // gives S > 1,200 and a naive fwd+bwd retains ~18 GB of activations on top
    // of a 16 GB f16 base. Measured: it spills into WDDM shared memory and a
    // step takes 38 s that should take under one.
    bool        ckpt       = true;
    int         ckpt_chunk = 128;
};

struct MM3LmSample {
    std::string          id;
    std::vector<int32_t> prompt;          // tokenised MM3 prompt
    std::vector<int32_t> codes;           // [n_frames * 8], warm-up row already dropped
    int64_t              n_frames = 0;
};

// ── data ────────────────────────────────────────────────────────────────────

static bool mm3_lm_read_file(const std::string & path, std::string * out) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out->assign((size_t) n, '\0');
    const bool ok = n == 0 || fread(&(*out)[0], 1, (size_t) n, f) == (size_t) n;
    fclose(f);
    return ok;
}

// ── the MM3 input embedding ─────────────────────────────────────────────────
//
// (token_embd[semantic + offset] + SUM_c audio_embd[code_c + (c-1)*1024])
// * num_codebooks^-0.5 — the reference's _embed_audio_frame, verbatim, for a
// whole crop at once, with the prompt's ordinary token embeddings in front.
//
// The acoustic indices are uploaded BOOK-MAJOR so the gather comes back as
// [H, Fin, NC] and the sum over books is NC-1 whole-slab adds rather than a
// strided gather.
struct MM3EmbedCtx {
    const MM3TrainLm * t   = nullptr;
    ggml_tensor *      t_prompt = nullptr, * t_sem = nullptr, * t_ac = nullptr;
    int64_t            P = 0, Fin = 0;
};

static ggml_tensor * mm3_lm_build_embed(ggml_context * ctx, const MM3EmbedCtx & e) {
    const MM3TrainLm & t   = *e.t;
    const int64_t      H   = t.lm.cfg.hidden_size;
    const int64_t      NC  = (int64_t) t.num_codebooks - 1;

    ggml_tensor * e_prompt = ggml_get_rows(ctx, t.lm.embed_tokens, ggml_view_1d(ctx, e.t_prompt, e.P, 0));
    if (e.Fin <= 0) {
        return e_prompt;
    }
    ggml_tensor * e_sem = ggml_get_rows(ctx, t.lm.embed_tokens, ggml_view_1d(ctx, e.t_sem, e.Fin, 0));
    ggml_tensor * e_ac  = ggml_reshape_3d(
        ctx, ggml_get_rows(ctx, t.audio_embd, ggml_view_1d(ctx, e.t_ac, e.Fin * NC, 0)), H, e.Fin, NC);
    ggml_tensor * acc = ggml_view_2d(ctx, e_ac, H, e.Fin, e_ac->nb[1], 0);
    for (int64_t k = 1; k < NC; k++) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, e_ac, H, e.Fin, e_ac->nb[1], (size_t) k * e_ac->nb[2]));
    }
    ggml_tensor * frames = ggml_scale(ctx, ggml_add(ctx, e_sem, acc), t.embedding_scale);
    return ggml_concat(ctx, e_prompt, frames, 1);
}

// The LmCkptRun hook. P1 calls this instead of get_rows on token ids.
static ggml_tensor * mm3_lm_ckpt_embed(ggml_context * ctx, LmCkptRun & r, int S) {
    const MM3EmbedCtx & e = *(const MM3EmbedCtx *) r.embed_user;
    GGML_ASSERT(e.P + e.Fin == (int64_t) S);
    return mm3_lm_build_embed(ctx, e);
}

// ── the run ─────────────────────────────────────────────────────────────────

static int mm3_lm_train_main(const MM3LmTrainArgs & a) {
    // TF32 off for the same reason every other MM3 training-data path turns it
    // off: this is gradient arithmetic against a frozen f16 base, and TF32's
    // ~1e-3 is not a trade worth taking for a few percent.
#ifdef _WIN32
    _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif

    std::string err;
    MM3TrainLm  t = {};
    if (!mm3_train_lm_load(&t, a.lm_path.c_str(), &err)) {
        fprintf(stderr, "[mm3-lm-train] LM load failed: %s\n", err.c_str());
        return 1;
    }
    if (!mm3_train_lm_load_audio_embd(&t, a.depth_path.c_str(), &err)) {
        fprintf(stderr, "[mm3-lm-train] audio_embd load failed: %s\n", err.c_str());
        mm3_train_lm_free(&t);
        return 1;
    }
    const Qwen3LMConfig & c   = t.lm.cfg;
    const int64_t         H   = c.hidden_size;
    const int64_t         NC  = (int64_t) t.num_codebooks - 1;
    const int64_t         AV  = t.acoustic_vocab_size;
    const int64_t         SL  = mm3_lm_train_slice_size(t);

    // ── samples ──
    std::vector<MM3LmSample> samples;
    {
        std::string jbuf;
        if (!mm3_lm_read_file(a.manifest, &jbuf)) {
            fprintf(stderr, "[mm3-lm-train] cannot read %s\n", a.manifest.c_str());
            mm3_train_lm_free(&t);
            return 1;
        }
        yyjson_doc * doc = yyjson_read(jbuf.c_str(), jbuf.size(), 0);
        yyjson_val * arr = doc ? yyjson_obj_get(yyjson_doc_get_root(doc), "samples") : nullptr;
        if (!arr || !yyjson_is_arr(arr)) {
            fprintf(stderr, "[mm3-lm-train] %s has no `samples` array\n", a.manifest.c_str());
            if (doc) yyjson_doc_free(doc);
            mm3_train_lm_free(&t);
            return 1;
        }
        MM3Model stub = {};
        stub.lm_file.found = true;
        stub.lm_file.path  = a.lm_path;
        stub.lm_file.name  = a.lm_path;
        stub.lm_cfg.semantic_vocab_offset = t.semantic_vocab_offset;
        MM3Tokenizer tok = {};
        if (!mm3_tokenizer_load(stub, &tok, &err)) {
            fprintf(stderr, "[mm3-lm-train] tokenizer: %s\n", err.c_str());
            yyjson_doc_free(doc);
            mm3_train_lm_free(&t);
            return 1;
        }

        yyjson_val *    s;
        yyjson_arr_iter it = yyjson_arr_iter_with(arr);
        while ((s = yyjson_arr_iter_next(&it))) {
            auto js = [&](const char * k) -> std::string {
                yyjson_val * v = yyjson_obj_get(s, k);
                return (v && yyjson_is_str(v)) ? std::string(yyjson_get_str(v)) : std::string();
            };
            const std::string id = js("id"), filename = js("filename"), lyrics = js("lyrics");
            std::string       stem = filename;
            const size_t      d    = stem.find_last_of('.');
            if (d != std::string::npos) stem = stem.substr(0, d);

            // The MM3 caption lives beside the audio as <stem>.mm3.txt, exactly
            // as mm3-condition reads it. REFUSE rather than fall back to the ACE
            // caption in the manifest: an ACE caption trains the wrong genre and
            // the failure is invisible.
            std::string caption;
            if (!mm3_lm_read_file(a.captions_dir + "/" + stem + ".mm3.txt", &caption)) {
                fprintf(stderr, "[mm3-lm-train] SKIP %s: no %s.mm3.txt\n", id.c_str(), stem.c_str());
                continue;
            }
            std::string cbuf;
            if (!mm3_lm_read_file(a.codes_dir + "/" + id + ".codes", &cbuf) || cbuf.size() < 16 * sizeof(int32_t)) {
                fprintf(stderr, "[mm3-lm-train] SKIP %s: no usable %s.codes\n", id.c_str(), id.c_str());
                continue;
            }
            const int64_t n_rows = (int64_t) (cbuf.size() / sizeof(int32_t)) / 8;
            MM3LmSample   sm;
            sm.id       = id;
            sm.n_frames = n_rows - 1;                        // drop the warm-up row
            sm.codes.resize((size_t) (sm.n_frames * 8));
            memcpy(sm.codes.data(), cbuf.data() + 8 * sizeof(int32_t), sm.codes.size() * sizeof(int32_t));
            mm3_tokenizer_encode(tok, mm3_assemble_prompt(caption, lyrics), &sm.prompt);
            if (sm.prompt.empty() || sm.n_frames < 8) {
                fprintf(stderr, "[mm3-lm-train] SKIP %s: empty prompt or %lld frames\n", id.c_str(),
                        (long long) sm.n_frames);
                continue;
            }
            samples.push_back(std::move(sm));
        }
        yyjson_doc_free(doc);
    }
    if (samples.empty()) {
        fprintf(stderr, "[mm3-lm-train] no usable samples\n");
        mm3_train_lm_free(&t);
        return 1;
    }

    int64_t max_prompt = 0;
    for (const auto & s : samples) max_prompt = std::max(max_prompt, (int64_t) s.prompt.size());
    const int64_t K_max = a.max_frames > 0 ? a.max_frames : 4096;
    const int64_t S_max = max_prompt + K_max;
    fprintf(stderr, "[mm3-lm-train] %zu songs, longest prompt %lld tok, crop <= %lld frames, seq <= %lld\n",
            samples.size(), (long long) max_prompt, (long long) K_max, (long long) S_max);

    // ── LoRA (attaches to the model) + optimizer ──
    LmLora lora;
    if (!lm_lora_init(&lora, &t.lm, 0, c.n_layers, a.rank, (float) a.alpha, (uint64_t) a.seed, 0.0f, &err)) {
        fprintf(stderr, "[mm3-lm-train] LoRA init failed: %s\n", err.c_str());
        mm3_train_lm_free(&t);
        return 1;
    }
    LmOptim opt;
    opt.optimizer     = a.optimizer;
    opt.muon.lr_scale = a.muon_lr_scale;
    opt.muon.momentum = a.muon_momentum;
    opt.muon.ns_steps = a.muon_ns_steps;
    opt.muon.nesterov = a.muon_nesterov;
    opt.muon.min_dim  = a.muon_min_dim;
    opt.muon.bucket   = a.muon_bucket;
    if (!lm_optim_init(&opt, lora.params, t.lm.backend, &err)) {
        fprintf(stderr, "[mm3-lm-train] optimizer init failed: %s\n", err.c_str());
        lm_lora_detach(&lora, &t.lm);
        lm_lora_free(&lora);
        mm3_train_lm_free(&t);
        return 1;
    }
    // A run where Muon classified ZERO parameters trains as AdamW and says
    // nothing about it. Print the split so that is visible.
    fprintf(stderr, "[mm3-lm-train] %zu LoRA tensors (rank %d, alpha %d) — %d on Muon in %zu buckets, %zu on AdamW\n",
            lora.params.size(), a.rank, a.alpha, opt.n_muon, opt.muon_buckets.size(),
            lora.params.size() - (size_t) opt.n_muon);

    // ── persistent tensors ──
    ggml_context * ctx_static = nullptr;
    {
        ggml_init_params p = { 32 * ggml_tensor_overhead(), nullptr, true };
        ctx_static         = ggml_init(p);
    }
    ggml_tensor * t_prompt = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, max_prompt);
    ggml_tensor * t_sem    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, K_max);
    ggml_tensor * t_ac     = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, K_max * NC);
    ggml_tensor * t_pos    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, S_max);
    ggml_tensor * t_msk    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, S_max * S_max);
    // The [SL, K_max+1] one-hot buffer is a NAIVE-path structure (98 MB at
    // K_max 1500). The checkpointed head chunks its own labels into a
    // [SL, chunk] buffer inside LmCkptState, so allocating this too would be
    // pure waste on the path that actually runs.
    ggml_tensor * t_lab    = a.ckpt ? nullptr
                                    : ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, SL, K_max + 1);
    ggml_tensor * t_adamw  = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 7);
    ggml_tensor * t_lg     = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_clip   = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_eps    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_gn2    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    // Checkpointed path only: the per-chunk upstream scalar and the segment
    // surrogate's loss gradient (which is exactly 1.0 — see lm-ckpt.h D9).
    ggml_tensor * t_gs     = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_one    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    // Sized but never read under checkpointing (the override supplies the
    // embedding); lm_ckpt_micro_step skips its upload.
    ggml_tensor * t_tok    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, S_max);
    for (ggml_tensor * x : { t_prompt, t_sem, t_ac, t_pos, t_msk, t_tok }) ggml_set_input(x);
    if (t_lab) ggml_set_input(t_lab);

    ggml_backend_buffer_t buf_static = ggml_backend_alloc_ctx_tensors(ctx_static, t.lm.backend);
    if (!buf_static) {
        fprintf(stderr, "[mm3-lm-train] static buffer allocation failed (lower --max-frames)\n");
        lm_lora_detach(&lora, &t.lm);
        lm_lora_free(&lora);
        mm3_train_lm_free(&t);
        return 1;
    }
    ggml_backend_buffer_clear(buf_static, 0);
    {
        const float lg = 1.0f / (float) std::max(1, a.grad_accum);
        const float cl = (float) a.grad_clip;
        const float ep = 1e-6f;
        ggml_backend_tensor_set(t_lg, &lg, 0, sizeof(float));
        ggml_backend_tensor_set(t_clip, &cl, 0, sizeof(float));
        ggml_backend_tensor_set(t_eps, &ep, 0, sizeof(float));
        const float one = 1.0f;
        ggml_backend_tensor_set(t_one, &one, 0, sizeof(float));
    }
    opt.t_adamw      = t_adamw;
    opt.t_lossgrad   = t_lg;
    opt.t_clip       = t_clip;
    opt.t_eps        = t_eps;
    opt.t_gnorm2     = t_gn2;
    opt.base_lr      = (float) a.lr;
    opt.weight_decay = (float) a.weight_decay;
    opt.grad_clip    = (float) a.grad_clip;
    opt.total_steps  = a.steps;
    opt.warmup_steps = a.warmup;

    // ── per-layer gradient checkpointing ──
    //
    // Not an optimisation here. A naive fwd+bwd retains every layer's
    // activations at once; with one segment per layer exactly ONE is live, and
    // the chunked CE head keeps the [16389, chunk] logits off the peak too.
    // The head override is what makes the second half work for MM3: an UNTIED
    // head, scored only over [eos_audio, semantic_offset + semantic_size).
    LmCkptState ckpt_st;
    LmCkptRun   ckpt_run;
    MM3EmbedCtx embed_ctx;
    if (a.ckpt) {
        if (!lm_ckpt_check_base(&t.lm, &err)) {
            fprintf(stderr, "[mm3-lm-train] %s\n", err.c_str());
            lm_lora_detach(&lora, &t.lm);
            lm_lora_free(&lora);
            mm3_train_lm_free(&t);
            return 1;
        }
        LmCkptCfg cc;
        cc.chunk     = a.ckpt_chunk;
        cc.s_max     = (int) S_max;
        cc.layer_lo  = 0;
        cc.layer_hi  = c.n_layers;
        cc.head_w    = t.lm_head;                       // UNTIED
        cc.head_row0 = (int64_t) t.eos_audio;           // slice starts at EOS
        cc.head_v    = (int) SL;
        if (!lm_ckpt_alloc(&ckpt_st, &t.lm, cc, &err) || !lm_ckpt_build_embed_t(&ckpt_st, &err)) {
            fprintf(stderr, "[mm3-lm-train] checkpoint setup failed: %s\n", err.c_str());
            lm_ckpt_free(&ckpt_st);
            lm_lora_detach(&lora, &t.lm);
            lm_lora_free(&lora);
            mm3_train_lm_free(&t);
            return 1;
        }
        embed_ctx.t        = &t;
        embed_ctx.t_prompt = t_prompt;
        embed_ctx.t_sem    = t_sem;
        embed_ctx.t_ac     = t_ac;

        ckpt_run.lm          = &t.lm;
        ckpt_run.opt         = &opt;
        ckpt_run.st          = &ckpt_st;
        ckpt_run.t_tok       = t_tok;
        ckpt_run.t_pos       = t_pos;
        ckpt_run.t_msk       = t_msk;
        ckpt_run.t_gs        = t_gs;
        ckpt_run.t_one       = t_one;
        ckpt_run.grad_accum  = std::max(1, a.grad_accum);
        ckpt_run.embed_build = mm3_lm_ckpt_embed;
        ckpt_run.embed_user  = &embed_ctx;
    }

    // ── graph sizing + scheduler ──
    // The scheduler is SHARED with the optimizer step, so it must be sized for
    // whichever graph is larger. This bit a real 4B Muon run: Muon's optimizer
    // graph is ~7-9k nodes while a segmented training graph was ~569, and ggml
    // asserts hash_set.size >= n_nodes + n_leafs mid-run. Do not "simplify".
    std::vector<uint8_t> arena((size_t) 512 << 20);
    int                  graph_nodes = 0;

    auto build_graph = [&](ggml_context * ctx, ggml_cgraph * gf, int64_t P, int64_t Fin, int64_t n_sup,
                           ggml_tensor ** out_loss) {
        const int64_t S = P + Fin;
        MM3EmbedCtx   ec{ &t, t_prompt, t_sem, t_ac, P, Fin };
        ggml_tensor * h_in = mm3_lm_build_embed(ctx, ec);

        ggml_tensor * hidden = lm_build_trunk_embeds(ctx, &t.lm, h_in, t_pos, t_msk, (int) S);
        // Supervised positions are a contiguous tail starting at P-1.
        ggml_tensor * hd = ggml_cont(
            ctx, ggml_view_2d(ctx, hidden, H, n_sup, hidden->nb[1], (size_t) (P - 1) * hidden->nb[1]));
        ggml_tensor * logits = ggml_mul_mat(ctx, mm3_lm_train_out_slice(ctx, t), hd);   // [SL, n_sup]
        ggml_tensor * labv   = ggml_view_2d(ctx, t_lab, SL, n_sup, t_lab->nb[1], 0);
        ggml_tensor * loss   = ggml_cross_entropy_loss(ctx, logits, labv);
        ggml_set_loss(loss);
        ggml_set_output(loss);
        ggml_build_forward_expand(gf, loss);
        *out_loss = loss;
    };

    if (a.ckpt) {
        // The worst checkpointed graph is ONE backward segment at S_max — the
        // trunk is never built whole, so sizing from it would over-allocate the
        // scheduler by ~L x. embed_ctx must describe a real crop first: the
        // probe builds P1, which calls the override.
        embed_ctx.P   = max_prompt;
        embed_ctx.Fin = K_max;
        ckpt_run.sched = nullptr;
        graph_nodes    = lm_ckpt_probe_segment_nodes(ckpt_run, (int) S_max);
    } else {
        ggml_init_params gip = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(gip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, true);
        ggml_tensor *    loss = nullptr;
        build_graph(ctx, gf, max_prompt, K_max, K_max + 1, &loss);
        std::vector<ggml_tensor *> gacc;
        lm_optim_fill_gacc(&opt, gf, &gacc);
        ggml_build_backward_expand(ctx, gf, gacc.data());
        graph_nodes = ggml_graph_n_nodes(gf);
        ggml_free(ctx);
    }
    fprintf(stderr, "[mm3-lm-train] %s graph: %d nodes\n",
            a.ckpt ? "worst backward segment" : "fwd+bwd", graph_nodes);

    BackendPair bp;
    bp.backend     = t.lm.backend;
    bp.cpu_backend = t.lm.cpu_backend;
    bp.has_gpu     = t.lm.backend != t.lm.cpu_backend;
    const int sched_nodes = std::max(std::max(8192, graph_nodes + graph_nodes / 2 + 2048),
                                     opt.est_nodes + opt.est_nodes / 4 + 1024);
    ggml_backend_sched_t sched = backend_sched_new(bp, sched_nodes);
    if (a.ckpt) {
        ckpt_run.sched = sched;
    }

    // ── training loop ──
    LmRng rng;
    lm_rng_seed(&rng, (uint64_t) a.seed ^ 0x9E3779B97F4A7C15ull);
    std::vector<int32_t> sem_in, ac_in, tgt, pos;
    std::vector<float>   msk;
    int                  last_mask_S = 0;
    double               running = 0.0;
    int                  n_micro = 0, rc = 0;
    LmStepStats          stats;
    const int64_t        t_start = ggml_time_ms();

    auto save_ckpt = [&](int step, double loss) {
        char sub[64];
        snprintf(sub, sizeof(sub), "ckpt-%d", step);
        const std::string dir = a.out_dir + "/" + sub;
        LmExportMeta      meta;
        meta.producer = "ace-train mm3-lm-train";
        meta.lm_path  = a.lm_path;
        meta.rank     = a.rank;
        meta.alpha    = a.alpha;
        meta.lr       = a.lr;
        meta.seed     = a.seed;
        meta.samples  = (int) samples.size();
        meta.trigger  = a.trigger;
        meta.saved_loss = loss;
        LmExportResult res;
        std::string    xerr;
        if (!lm_export_peft(lora, c, meta, dir, &res, &xerr)) {
            fprintf(stderr, "[mm3-lm-train] export failed: %s\n", xerr.c_str());
            return;
        }
        // The sidecar the shipped MM3 adapter picker reads. Written beside the
        // safetensors so `<out>` pointed at <adapters>/mm3-lm-adapters/<run>
        // makes the checkpoint appear in the UI with no install step.
        const std::string side = dir + "/adapter_model.safetensors.json";
        FILE *            sf   = hs_fopen(side, "wb");
        if (sf) {
            fprintf(sf,
                    "{\"name\":\"%s ckpt-%d\",\"trigger\":\"%s\",\"rank\":%d,\"dataset\":\"%s\","
                    "\"trainedSteps\":%d,\"recommendedScales\":{\"scaleMlp\":0.5},"
                    "\"notes\":\"ace-train mm3-lm-train, loss %.4f; render captions must carry the "
                    "artist's true bpm/tuning\"}\n",
                    a.dataset_name.empty() ? "MM3 LM" : a.dataset_name.c_str(), step, a.trigger.c_str(), a.rank,
                    a.dataset_name.c_str(), step, loss);
            fclose(sf);
        }
        fprintf(stderr, "[mm3-lm-train] saved %s (loss %.4f)\n", dir.c_str(), loss);
    };

    for (int step = 1; step <= a.steps && rc == 0; step++) {
        double acc_loss = 0.0;
        for (int micro = 0; micro < std::max(1, a.grad_accum) && rc == 0; micro++) {
            const MM3LmSample & s = samples[(size_t) (lm_rng_next(&rng) % samples.size())];
            const int64_t       P = (int64_t) s.prompt.size();

            // Random crop, fresh every time. `beginning` exists only to
            // reproduce the intros-only failure lm2 hit.
            int64_t K = std::min<int64_t>(K_max, s.n_frames);
            int64_t c0 = 0;
            if (a.crop_mode != "beginning" && s.n_frames > K) {
                c0 = (int64_t) (lm_rng_next(&rng) % (uint64_t) (s.n_frames - K + 1));
            }
            const bool    at_end = (c0 + K) >= s.n_frames;
            const int64_t Fin    = at_end ? K : K - 1;      // frames used as INPUT
            const int64_t n_sup  = at_end ? K + 1 : K;      // supervised positions
            const int64_t S      = P + Fin;

            sem_in.resize((size_t) Fin);
            ac_in.resize((size_t) (Fin * NC));
            for (int64_t i = 0; i < Fin; i++) {
                const int32_t * f = &s.codes[(size_t) ((c0 + i) * 8)];
                sem_in[(size_t) i] = f[0] + (int32_t) t.semantic_vocab_offset;
                for (int64_t k = 0; k < NC; k++) {
                    ac_in[(size_t) (k * Fin + i)] = f[1 + k] + (int32_t) (k * AV);
                }
            }
            // Targets: position P-1+j predicts frame c0+j, and the last one is
            // EOS when the crop really reached the end.
            tgt.resize((size_t) n_sup);
            for (int64_t j = 0; j < n_sup; j++) {
                tgt[(size_t) j] = (at_end && j == n_sup - 1)
                                    ? mm3_lm_train_slice_eos(t)
                                    : mm3_lm_train_slice_index(t, s.codes[(size_t) ((c0 + j) * 8)]);
            }

            if ((int) S != last_mask_S) {
                lm_causal_mask((int) S, &msk);
                ggml_backend_tensor_set(t_msk, msk.data(), 0, msk.size() * sizeof(float));
                last_mask_S = (int) S;
            }
            pos.resize((size_t) S);
            for (int64_t i = 0; i < S; i++) pos[(size_t) i] = (int32_t) i;
            ggml_backend_tensor_set(t_prompt, s.prompt.data(), 0, (size_t) P * sizeof(int32_t));
            ggml_backend_tensor_set(t_sem, sem_in.data(), 0, sem_in.size() * sizeof(int32_t));
            ggml_backend_tensor_set(t_ac, ac_in.data(), 0, ac_in.size() * sizeof(int32_t));
            ggml_backend_tensor_set(t_pos, pos.data(), 0, pos.size() * sizeof(int32_t));

            double ce = 0.0;
            bool   ok = false;
            if (a.ckpt) {
                // lm_ckpt_micro_step reads S from tokens.size() and the trained
                // span from (n_masked, s_tr); the ids themselves are never read
                // because the embedding is overridden.
                embed_ctx.P   = P;
                embed_ctx.Fin = Fin;
                LmSample smp;
                smp.tokens.assign((size_t) S, 0);
                smp.targets  = tgt;
                smp.n_masked = (int) P;
                smp.s_tr     = (int) n_sup;
                ok = lm_ckpt_micro_step(ckpt_run, smp, true, &ce);
            } else {
                ggml_init_params gip = { arena.size(), arena.data(), true };
                ggml_context *   ctx = ggml_init(gip);
                ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, true);
                ggml_tensor *    loss = nullptr;
                build_graph(ctx, gf, P, Fin, n_sup, &loss);
                std::vector<ggml_tensor *> gacc;
                lm_optim_fill_gacc(&opt, gf, &gacc);
                ggml_build_backward_expand(ctx, gf, gacc.data());
                {
                    LmLabelGuard guard(t_lab, tgt.data(), (int) n_sup, (int) SL);
                    ggml_backend_sched_reset(sched);
                    ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
                    if (ok) {
                        float lv = 0.0f;
                        ggml_backend_tensor_get(loss, &lv, 0, sizeof(float));
                        ce = (double) lv;
                    }
                }
                ggml_free(ctx);
            }
            if (!ok) {
                fprintf(stderr, "[mm3-lm-train] micro-step failed (lower --max-frames?)\n");
                rc = 1;
            } else {
                acc_loss += ce;
                running  += ce;
                n_micro++;
            }
        }
        if (rc) break;

        if (!lm_optim_step(&opt, sched, &stats)) {
            fprintf(stderr, "[mm3-lm-train] optimizer step failed\n");
            rc = 1;
            break;
        }
        const double win = acc_loss / std::max(1, a.grad_accum);
        if (step == 1 || step % 10 == 0 || step == a.steps) {
            fprintf(stderr, "[mm3-lm-train] step %d/%d loss %.4f lr %.3g |g| %.3f clip %.3f %lld s\n", step,
                    a.steps, win, (double) stats.lr, (double) stats.grad_norm, (double) stats.clip,
                    (long long) ((ggml_time_ms() - t_start) / 1000));
        }
        if (a.save_every > 0 && (step % a.save_every == 0 || step == a.steps)) {
            save_ckpt(step, win);
        }
    }

    fprintf(stderr, "[mm3-lm-train] %s after %d micro-steps, mean loss %.4f, %lld s\n",
            rc ? "STOPPED" : "done", n_micro, n_micro ? running / n_micro : 0.0,
            (long long) ((ggml_time_ms() - t_start) / 1000));

    ggml_backend_sched_free(sched);
    if (a.ckpt) {
        lm_ckpt_free(&ckpt_st);
    }
    ggml_backend_buffer_free(buf_static);
    ggml_free(ctx_static);
    lm_optim_free(&opt);
    lm_lora_detach(&lora, &t.lm);
    lm_lora_free(&lora);
    mm3_train_lm_free(&t);
    return rc;
}
