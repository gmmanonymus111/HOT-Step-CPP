#pragma once
// train/mm3-dit-train-run.h — MM3 flow-DiT LoRA trainer.
//
// Step 4b: the loop around mm3-dit-train-graph.h. Reads the two caches
// ace-train mm3-preprocess / mm3-condition produce and trains a LoRA.
//
// ── Scope of THIS increment ────────────────────────────────────────────────
//
// Deliberately the smallest thing that can be judged: forward -> loss ->
// backward -> AdamW step, with loss reported per step. It exists to clear
// validation-ladder rungs 1-4 from the design doc (forward parity, zero-adapter
// neutrality, finite differences, single-song overfit) — NOT to be a production
// trainer. No checkpoint/resume, no JSONL event stream, no Muon, no LoKR, no
// gradient accumulation beyond a simple counter, no export. Those all have
// working ACE implementations to lift once the numbers say this trains at all.
//
// ── The graph/backward pattern, copied from dit-train-run.h ────────────────
//
//   forward -> loss
//   ggml_set_loss(loss)
//   ggml_build_forward_expand(gf, loss)
//   lm_optim_fill_gacc(&opt, gf, &gacc)      <- AFTER the forward expansion:
//                                               gacc is indexed by forward-node
//                                               order and is wrong if built early
//   ggml_build_backward_expand(ctx, gf, gacc.data())
//   ggml_backend_sched_reset(sched)
//   ggml_backend_sched_graph_compute(sched, gf)
//
// ── The loss-sign gate, which runs BEFORE any training ─────────────────────
//
// The velocity target's direction is the one thing the design doc refused to
// settle on paper, because SimpleTuner states its convention against a different
// sampler and MM3 carries an output-negation trap. So `--sign-check` runs first:
// with the adapter at ZERO it measures the loss at high sigma against both
// candidate targets and reports which is lower. A model that already knows how
// to denoise must score better on the correct one. If they are within noise of
// each other, something else is wrong and training would be meaningless.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "train/mm3-dit-train-graph.h"
#include "train/lm-optim.h"

// One cached song: target latents + conditioning + where the seams are.
struct MM3TrainSong {
    std::string name;
    std::string latents_path;    // f32 [128, L]
    std::string cond_path;       // f16 [L', 2048]
    // TWO lengths, and confusing them silently corrupts every channel but the
    // first. The latents file is [128, L_audio] so L_audio is the ROW STRIDE;
    // L is how much of it is usable (conditioning is almost always shorter, both
    // from the rounding mismatch and from EOS-early rollouts). Seek with
    // L_audio, sample crops within L.
    int64_t     L_audio = 0;     // row stride of the latents file
    int64_t     L       = 0;     // usable = min(audio L, cond L')
    int64_t     cond_dim = 0;
    std::vector<int64_t> seams;  // latent indices where one rollout segment ends
};

struct MM3TrainArgs {
    std::string cache_dir, models_dir, out_dir;
    int64_t     rank       = 32;
    float       alpha      = 32.0f;
    float       lr         = 1e-4f;
    int64_t     steps      = 200;
    int64_t     crop       = 689;   // one 200-frame DiT window
    int64_t     grad_accum = 1;
    uint64_t    seed       = 42;
    float       logit_mean = 0.0f;
    float       logit_std  = 1.0f;
    bool        sign_check = false;
    std::string only_song;          // restrict to one song (overfit test)
};

// ── Cache loading ──────────────────────────────────────────────────────────

// Read the two manifests and pair them by id. A song is usable only if BOTH
// halves exist; conditioning can be shorter than the audio (see the design
// doc's rounding note and the EOS-early behaviour), so L is always the min.
static bool mm3_train_load_cache(const std::string & cache_dir, const std::string & only,
                                 std::vector<MM3TrainSong> * out, std::string * err) {
    auto slurp = [&](const std::string & p, std::string * dst) -> bool {
        FILE * f = hs_fopen(p, "rb");
        if (!f) { if (err) *err = "cannot open " + p; return false; }
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        dst->assign((size_t) n, '\0');
        const bool ok = fread(&(*dst)[0], 1, (size_t) n, f) == (size_t) n;
        fclose(f);
        if (!ok && err) *err = "short read on " + p;
        return ok;
    };
    std::string pre_s, cond_s;
    if (!slurp(cache_dir + "/mm3_preprocess.json", &pre_s)) return false;
    if (!slurp(cache_dir + "/mm3_condition.json", &cond_s)) return false;

    yyjson_doc * pd = yyjson_read(pre_s.c_str(), pre_s.size(), 0);
    yyjson_doc * cd = yyjson_read(cond_s.c_str(), cond_s.size(), 0);
    if (!pd || !cd) { if (err) *err = "malformed manifest JSON"; return false; }

    // id -> conditioning record
    std::map<std::string, yyjson_val *> cond_by_id;
    yyjson_val * carr = yyjson_obj_get(yyjson_doc_get_root(cd), "samples");
    yyjson_val * cv; yyjson_arr_iter cit = yyjson_arr_iter_with(carr);
    while ((cv = yyjson_arr_iter_next(&cit))) {
        yyjson_val * idv = yyjson_obj_get(cv, "id");
        if (idv && yyjson_is_str(idv)) cond_by_id[yyjson_get_str(idv)] = cv;
    }

    yyjson_val * parr = yyjson_obj_get(yyjson_doc_get_root(pd), "samples");
    yyjson_val * pv; yyjson_arr_iter pit = yyjson_arr_iter_with(parr);
    while ((pv = yyjson_arr_iter_next(&pit))) {
        auto gs = [&](yyjson_val * o, const char * k) -> std::string {
            yyjson_val * v = yyjson_obj_get(o, k);
            return (v && yyjson_is_str(v)) ? std::string(yyjson_get_str(v)) : std::string();
        };
        auto gi = [&](yyjson_val * o, const char * k) -> int64_t {
            yyjson_val * v = yyjson_obj_get(o, k);
            return v ? (int64_t) yyjson_get_int(v) : 0;
        };
        const std::string id = gs(pv, "id"), fn = gs(pv, "filename");
        if (!only.empty() && fn.find(only) == std::string::npos) continue;
        auto it = cond_by_id.find(id);
        if (it == cond_by_id.end()) {
            fprintf(stderr, "[mm3-train] %s has no conditioning, skipped\n", fn.c_str());
            continue;
        }
        MM3TrainSong s;
        s.name         = fn;
        s.latents_path = cache_dir + "/" + gs(pv, "latents");
        s.cond_path    = cache_dir + "/" + gs(it->second, "condition");
        s.cond_dim     = gi(it->second, "cond_dim");
        s.L_audio      = gi(pv, "latent_frames");
        s.L            = std::min<int64_t>(s.L_audio, gi(it->second, "cond_latents"));
        yyjson_val * sb = yyjson_obj_get(it->second, "segment_latent_starts");
        if (sb && yyjson_is_arr(sb)) {
            yyjson_val * b; yyjson_arr_iter bi = yyjson_arr_iter_with(sb);
            while ((b = yyjson_arr_iter_next(&bi))) s.seams.push_back((int64_t) yyjson_get_int(b));
        }
        out->push_back(std::move(s));
    }
    yyjson_doc_free(pd); yyjson_doc_free(cd);
    if (out->empty()) { if (err) *err = "no usable songs in the cache"; return false; }
    return true;
}

// Read one crop. latents are f32 [128, L] channel-major; conditioning is f16
// [L', 2048] latent-major, i.e. exactly what mm3_cond_encode emitted.
static bool mm3_train_read_crop(const MM3TrainSong & s, int64_t start, int64_t crop, int64_t IC,
                                std::vector<float> * lat, std::vector<float> * cond, std::string * err) {
    lat->assign((size_t) (IC * crop), 0.0f);
    FILE * f = hs_fopen(s.latents_path, "rb");
    if (!f) { if (err) *err = "cannot open " + s.latents_path; return false; }
    for (int64_t c = 0; c < IC; c++) {
        // channel c of the full [128, L] block, offset to the crop
        // stride is L_audio, NOT s.L — see the struct comment.
        if (fseek(f, (long) (((size_t) (c * s.L_audio + start)) * sizeof(float)), SEEK_SET) != 0 ||
            fread(lat->data() + c * crop, sizeof(float), (size_t) crop, f) != (size_t) crop) {
            fclose(f); if (err) *err = "short read on " + s.latents_path; return false;
        }
    }
    fclose(f);

    const int64_t CD = s.cond_dim;
    cond->assign((size_t) (CD * crop), 0.0f);
    FILE * g = hs_fopen(s.cond_path, "rb");
    if (!g) { if (err) *err = "cannot open " + s.cond_path; return false; }
    std::vector<ggml_fp16_t> half((size_t) (CD * crop));
    if (fseek(g, (long) (((size_t) (start * CD)) * sizeof(ggml_fp16_t)), SEEK_SET) != 0 ||
        fread(half.data(), sizeof(ggml_fp16_t), half.size(), g) != half.size()) {
        fclose(g); if (err) *err = "short read on " + s.cond_path; return false;
    }
    fclose(g);
    // f16 [crop, CD] -> f32 [CD, crop]: the graph wants channel-major.
    for (int64_t t = 0; t < crop; t++) {
        for (int64_t c = 0; c < CD; c++) {
            (*cond)[(size_t) (c * crop + t)] = ggml_fp16_to_fp32(half[(size_t) (t * CD + c)]);
        }
    }
    return true;
}

// ── Adapter parameters ─────────────────────────────────────────────────────

// Allocate LoRA A/B for every site. A is init'd small-random and B ZERO, so the
// adapter starts as an exact no-op — which is also validation rung 2: a fresh
// adapter must leave the forward bit-identical.
static bool mm3_train_make_adapters(ggml_context * ctx, const MM3Model & m, int64_t rank, float alpha,
                                    MM3TrainAdapters * ad, std::vector<ggml_tensor *> * params) {
    const MM3DitConfig & c = m.synth_cfg.dit;
    const int64_t E = (int64_t) c.embedding_length, FI = (int64_t) c.ff_inner;
    const int64_t CC = (int64_t) c.concat_channels, IC = (int64_t) c.in_channels;
    const float   sc = alpha / (float) rank;

    auto mk = [&](MM3TrainLora * lo, int64_t in, int64_t out, const char * tag) {
        lo->a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, rank);
        lo->b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, out);
        lo->scale = sc;
        ggml_set_name(lo->a, (std::string(tag) + ".A").c_str());
        ggml_set_name(lo->b, (std::string(tag) + ".B").c_str());
        ggml_set_param(lo->a);
        ggml_set_param(lo->b);
        params->push_back(lo->a);
        params->push_back(lo->b);
    };

    ad->blk.assign(m.synth.dit.blk.size(), MM3TrainBlockAdapters{});
    for (size_t i = 0; i < ad->blk.size(); i++) {
        const std::string p = "dit.blk." + std::to_string(i);
        mk(&ad->blk[i].qkv,      E,  3 * E,      (p + ".attn_qkv").c_str());
        mk(&ad->blk[i].attn_out, E,  E,          (p + ".attn_output").c_str());
        mk(&ad->blk[i].ff_in,    E,  2 * FI,     (p + ".ffn_in").c_str());
        mk(&ad->blk[i].ff_out,   FI, E,          (p + ".ffn_out").c_str());
    }
    mk(&ad->proj_in,  CC, E,  "dit.proj_in");
    mk(&ad->proj_out, E,  IC, "dit.proj_out");
    return true;
}

// ── Sigma / target ─────────────────────────────────────────────────────────
//
// Rectified flow, matching the scheduler this port already replicates:
//   sigma = 1 - sigmoid(logit_normal(mean, std)),  x_t = sigma*noise + (1-sigma)*x0
//
// The velocity target's SIGN is the one thing the design doc refused to settle
// on paper. mm3-dit-graph.h's Euler step is `x + (sigma_next - sigma) * v` with
// sigma RISING 0->1, i.e. integrating from data toward noise; so the field it
// wants is d(x)/d(sigma) = noise - x0. `--sign-check` verifies that empirically
// against the alternative before any training happens.
static inline float mm3_sigma_from(std::mt19937_64 & rng, float mean, float std_) {
    std::normal_distribution<double> nd(mean, std_);
    const double u = 1.0 / (1.0 + std::exp(-nd(rng)));
    return (float) std::min(1.0, std::max(0.0, 1.0 - u));
}

// target = (noise - x0) when `positive`, else (x0 - noise).
static void mm3_make_xt_target(const std::vector<float> & x0, const std::vector<float> & noise,
                               float sigma, bool positive, std::vector<float> * xt,
                               std::vector<float> * target) {
    xt->resize(x0.size());
    target->resize(x0.size());
    for (size_t i = 0; i < x0.size(); i++) {
        (*xt)[i]     = sigma * noise[i] + (1.0f - sigma) * x0[i];
        (*target)[i] = positive ? (noise[i] - x0[i]) : (x0[i] - noise[i]);
    }
}

// Deterministic Gaussian noise. NOT std::normal_distribution — its byte stream
// is stdlib-dependent, which is trap #8 in the mm3-backend skill and would make
// a "same seed" run unreproducible across builds. splitmix64 + Box-Muller.
static void mm3_fill_noise_train(std::vector<float> * v, uint64_t seed) {
    uint64_t s = seed;
    auto nxt = [&]() -> double {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= (z >> 31);
        return ((z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    };
    for (size_t i = 0; i < v->size(); i += 2) {
        const double u1 = std::max(1e-12, nxt()), u2 = nxt();
        const double r = std::sqrt(-2.0 * std::log(u1)), th = 6.283185307179586 * u2;
        (*v)[i] = (float) (r * std::cos(th));
        if (i + 1 < v->size()) (*v)[i + 1] = (float) (r * std::sin(th));
    }
}

// ── One micro-step ─────────────────────────────────────────────────────────
//
// Builds forward -> loss (and backward when training), computes, returns the
// loss. `backward == false` is the measurement path used by --sign-check and by
// zero-adapter neutrality.
static bool mm3_train_micro(MM3Model & m, MM3DitGraph & g, const MM3TrainAdapters & ad, LmOptim * opt,
                            const std::vector<float> & xt_h, const std::vector<float> & cond_h,
                            const std::vector<float> & tgt_h, const std::vector<float> & temb_h,
                            int64_t crop, bool backward, float * loss_out, std::string * err) {
    const MM3DitConfig & c  = m.synth_cfg.dit;
    const int64_t        IC = (int64_t) c.in_channels;
    const int64_t        CD = (int64_t) c.condition_dim;
    const int64_t        E  = (int64_t) c.embedding_length;

    const size_t     meta = ggml_tensor_overhead() * 262144 + ggml_graph_overhead_custom(262144, true);
    ggml_init_params ip   = { meta, nullptr, true };
    ggml_context *   ctx  = ggml_init(ip);
    if (!ctx) { if (err) *err = "ggml_init failed"; return false; }

    MM3TrainInputs in;
    in.xt        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, crop, IC);
    in.cond      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, crop, CD);
    in.temb      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, E);
    in.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, crop + 1);
    in.vtarget   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, crop, IC);
    ggml_set_input(in.xt); ggml_set_input(in.cond); ggml_set_input(in.temb);
    ggml_set_input(in.positions); ggml_set_input(in.vtarget);

    ggml_tensor * pred = mm3_dt_forward(ctx, m, ad, in);
    ggml_tensor * loss = mm3_dt_loss(ctx, pred, in.vtarget);
    if (backward) ggml_set_loss(loss);
    ggml_set_output(loss);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 262144, backward);
    ggml_build_forward_expand(gf, loss);
    if (backward) {
        // AFTER the forward expansion — gacc is indexed by forward-node order.
        std::vector<ggml_tensor *> gacc;
        lm_optim_fill_gacc(opt, gf, &gacc);
        ggml_build_backward_expand(ctx, gf, gacc.data());
    }

    ggml_backend_sched_reset(g.sched);
    if (!ggml_backend_sched_alloc_graph(g.sched, gf)) {
        ggml_free(ctx); if (err) *err = "sched_alloc_graph failed (out of VRAM?)"; return false;
    }
    ggml_backend_tensor_set(in.xt,      xt_h.data(),   0, xt_h.size()   * sizeof(float));
    ggml_backend_tensor_set(in.cond,    cond_h.data(), 0, cond_h.size() * sizeof(float));
    ggml_backend_tensor_set(in.vtarget, tgt_h.data(),  0, tgt_h.size()  * sizeof(float));
    ggml_backend_tensor_set(in.temb,    temb_h.data(), 0, temb_h.size() * sizeof(float));
    {
        // Time token holds position 0, so latent frames are 1..crop.
        std::vector<int32_t> pos((size_t) crop + 1);
        for (int64_t i = 0; i <= crop; i++) pos[(size_t) i] = (int32_t) i;
        ggml_backend_tensor_set(in.positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    }

    const bool ok = ggml_backend_sched_graph_compute(g.sched, gf) == GGML_STATUS_SUCCESS;
    if (ok && loss_out) ggml_backend_tensor_get(loss, loss_out, 0, sizeof(float));
    ggml_free(ctx);
    if (!ok && err) *err = "graph compute failed";
    return ok;
}
