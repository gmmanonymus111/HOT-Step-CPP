// ace-midi.cpp: MuScriptor audio->MIDI transcription (GGML port)
//
// Native port of the MuScriptor transcription model (Kyutai & Mirelo,
// arXiv:2607.08168, code MIT, weights CC BY-NC 4.0). Decoder-only causal
// transformer with mel-spectrogram prefix conditioning; MT3 event vocab.
// Design + validation plan: docs/plans/muscriptor-cpp-port.md.
//
// Phase 1 (this file, current state): weight loading + transformer prefill
// graph + logit-parity selftest against the Python oracle dumps produced by
// tools/ace-midi-validate.py.
//
//   ace-midi --model <dir> --validate <dir>
//     <dir>/model.safetensors + config.json ; validation dir with
//     prefix.bin / logits_bos.bin / manifest.json
//
// Later phases add: mel frontend, chunked greedy decode with KV cache +
// prelude forcing, note-event decode, MIDI writer, JSONL streaming.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "backend.h"
#include "ggml.h"
#include "safetensors.h"

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct MidiConfig {
    int dim        = 768;
    int num_heads  = 12;
    int num_layers = 14;
    int card       = 1393;
    int head_dim() const { return dim / num_heads; }
    int ffn_dim() const { return 4 * dim; }
    int bos_id() const { return card; }  // "initial token" = card
};

static int json_int_field(const char * json, const char * key, int fb) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char * p = strstr(json, needle);
    if (!p) return fb;
    p = strchr(p + strlen(needle), ':');
    if (!p) return fb;
    return atoi(p + 1);
}

static bool load_config(MidiConfig * c, const std::string & dir) {
    std::string path = dir + "/config.json";
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[ace-midi] cannot open %s\n", path.c_str());
        return false;
    }
    std::string j(65536, 0);
    size_t n = fread(j.data(), 1, j.size() - 1, f);
    fclose(f);
    j.resize(n);
    c->dim        = json_int_field(j.c_str(), "dim", c->dim);
    c->num_heads  = json_int_field(j.c_str(), "num_heads", c->num_heads);
    c->num_layers = json_int_field(j.c_str(), "num_layers", c->num_layers);
    c->card       = json_int_field(j.c_str(), "card", c->card);
    fprintf(stderr, "[ace-midi] config: dim=%d heads=%d layers=%d card=%d\n",
            c->dim, c->num_heads, c->num_layers, c->card);
    return true;
}

// ---------------------------------------------------------------------------
// Model weights
// ---------------------------------------------------------------------------

struct MidiLayer {
    ggml_tensor * norm1_w, * norm1_b;
    ggml_tensor * in_proj;    // [dim, 3*dim] (ggml: ne0=in)
    ggml_tensor * out_proj;   // [dim, dim]
    ggml_tensor * norm2_w, * norm2_b;
    ggml_tensor * ffn1;       // [dim, 4*dim]
    ggml_tensor * ffn2;       // [4*dim, dim]
};

struct MidiModel {
    MidiConfig             cfg;
    ggml_context *         wctx = nullptr;
    ggml_backend_buffer_t  wbuf = nullptr;
    ggml_tensor *          emb;         // [dim, card+1]
    std::vector<MidiLayer> layers;
    ggml_tensor *          out_norm_w, * out_norm_b;
    ggml_tensor *          head;        // [dim, card]

    ggml_backend_t       backend, cpu_backend;
    ggml_backend_sched_t sched;

    // BOS embedding row kept host-side for input assembly
    std::vector<float> bos_emb;
};

// Create a ggml tensor mirroring a safetensors entry (torch [out,in] row-major
// -> ggml [in, out]) and upload its data, converting BF16/F16 -> F32.
static ggml_tensor * load_tensor(MidiModel * m, ggml_context * ctx, const STFile & st,
                                 const std::string & name, int64_t ne0, int64_t ne1) {
    (void) m;
    const STEntry * e = st_find(st, name.c_str());
    if (!e) {
        fprintf(stderr, "[ace-midi] FATAL: missing tensor %s\n", name.c_str());
        exit(1);
    }
    // torch shape is [out, in] row-major -> ggml [ne0=in, ne1=out], same memory
    ggml_tensor * t = ne1 > 0
        ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1)
        : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    ggml_set_name(t, name.c_str());
    return t;
}

static void upload_tensor(MidiModel * m, const STFile & st, ggml_tensor * t) {
    const STEntry * e = st_find(st, t->name);
    size_t n = ggml_nelements(t);
    // verify element count matches
    int64_t st_n = 1;
    for (int i = 0; i < e->n_dims; i++) st_n *= e->shape[i];
    if ((int64_t) n != st_n) {
        fprintf(stderr, "[ace-midi] FATAL: %s shape mismatch (st=%lld ggml=%zu)\n",
                t->name, (long long) st_n, n);
        exit(1);
    }
    const void * src = st_data(st, *e);
    if (e->dtype == "F32") {
        ggml_backend_tensor_set(t, src, 0, n * 4);
    } else if (e->dtype == "BF16") {
        std::vector<float> tmp(n);
        const uint16_t * s = (const uint16_t *) src;
        for (size_t i = 0; i < n; i++) {
            uint32_t bits = (uint32_t) s[i] << 16;
            memcpy(&tmp[i], &bits, 4);
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, n * 4);
    } else if (e->dtype == "F16") {
        std::vector<float> tmp(n);
        const ggml_fp16_t * s = (const ggml_fp16_t *) src;
        for (size_t i = 0; i < n; i++) tmp[i] = ggml_fp16_to_fp32(s[i]);
        ggml_backend_tensor_set(t, tmp.data(), 0, n * 4);
    } else {
        fprintf(stderr, "[ace-midi] FATAL: %s unsupported dtype %s\n", t->name, e->dtype.c_str());
        exit(1);
    }
}

// Published checkpoints use the legacy multi-codebook key layout for the
// embedding and head (emb.0.* / linears.0.*) — same remap as the Python
// loader's _remap_single_codebook_keys.
static std::string resolve_key(const STFile & st, const std::string & canonical, const std::string & legacy) {
    if (st_find(st, canonical.c_str())) return canonical;
    if (st_find(st, legacy.c_str())) return legacy;
    return canonical;  // load_tensor will report it missing
}

static bool load_model(MidiModel * m, const std::string & dir) {
    if (!load_config(&m->cfg, dir)) return false;
    const MidiConfig & c = m->cfg;

    STFile st;
    std::string wpath = dir + "/model.safetensors";
    if (!st_open(&st, wpath.c_str())) return false;

    BackendPair bp = backend_init("MIDI");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched       = backend_sched_new(bp, 8192);

    int n_tensors = 3 /*emb, out_norm w/b*/ + 1 /*head*/ + c.num_layers * 8;
    ggml_init_params ip = { (size_t) n_tensors * ggml_tensor_overhead() + 4096, NULL, true };
    m->wctx = ggml_init(ip);

    const std::string emb_key  = resolve_key(st, "emb.weight", "emb.0.weight");
    const std::string head_key = resolve_key(st, "linear.weight", "linears.0.weight");
    m->emb        = load_tensor(m, m->wctx, st, emb_key, c.dim, c.card + 1);
    m->out_norm_w = load_tensor(m, m->wctx, st, "out_norm.weight", c.dim, 0);
    m->out_norm_b = load_tensor(m, m->wctx, st, "out_norm.bias", c.dim, 0);
    m->head       = load_tensor(m, m->wctx, st, head_key, c.dim, c.card);

    m->layers.resize(c.num_layers);
    for (int l = 0; l < c.num_layers; l++) {
        char base[96];
        snprintf(base, sizeof(base), "transformer.layers.%d.", l);
        MidiLayer & L = m->layers[l];
        L.norm1_w  = load_tensor(m, m->wctx, st, std::string(base) + "norm1.weight", c.dim, 0);
        L.norm1_b  = load_tensor(m, m->wctx, st, std::string(base) + "norm1.bias", c.dim, 0);
        L.in_proj  = load_tensor(m, m->wctx, st, std::string(base) + "self_attn.in_proj_weight", c.dim, 3 * c.dim);
        L.out_proj = load_tensor(m, m->wctx, st, std::string(base) + "self_attn.out_proj.weight", c.dim, c.dim);
        L.norm2_w  = load_tensor(m, m->wctx, st, std::string(base) + "norm2.weight", c.dim, 0);
        L.norm2_b  = load_tensor(m, m->wctx, st, std::string(base) + "norm2.bias", c.dim, 0);
        L.ffn1     = load_tensor(m, m->wctx, st, std::string(base) + "linear1.weight", c.dim, c.ffn_dim());
        L.ffn2     = load_tensor(m, m->wctx, st, std::string(base) + "linear2.weight", c.ffn_dim(), c.dim);
    }

    m->wbuf = ggml_backend_alloc_ctx_tensors(m->wctx, m->backend);
    if (!m->wbuf) {
        fprintf(stderr, "[ace-midi] FATAL: weight buffer alloc failed\n");
        return false;
    }
    for (ggml_tensor * t = ggml_get_first_tensor(m->wctx); t; t = ggml_get_next_tensor(m->wctx, t)) {
        upload_tensor(m, st, t);
    }

    // Keep the BOS embedding row host-side (input assembly happens on CPU)
    {
        const STEntry * e = st_find(st, emb_key.c_str());
        m->bos_emb.resize(c.dim);
        if (e->dtype == "F32") {
            const float * src = (const float *) st_data(st, *e) + (size_t) c.bos_id() * c.dim;
            memcpy(m->bos_emb.data(), src, (size_t) c.dim * 4);
        } else {
            const uint16_t * src = (const uint16_t *) st_data(st, *e) + (size_t) c.bos_id() * c.dim;
            for (int i = 0; i < c.dim; i++) {
                if (e->dtype == "BF16") {
                    uint32_t bits = (uint32_t) src[i] << 16;
                    memcpy(&m->bos_emb[i], &bits, 4);
                } else {
                    m->bos_emb[i] = ggml_fp16_to_fp32((ggml_fp16_t) src[i]);
                }
            }
        }
    }

    fprintf(stderr, "[ace-midi] loaded %d layers (%.1f MB weights)\n",
            c.num_layers, (double) ggml_backend_buffer_get_size(m->wbuf) / 1e6);
    st_close(&st);
    return true;
}

// ---------------------------------------------------------------------------
// Sinusoidal positions (transformer.py create_sin_embedding: cat([cos, sin]),
// exponent i/(half_dim - 1), max_period 10000)
// ---------------------------------------------------------------------------

static void add_sin_pos(float * x, int T, int dim, int pos0) {
    int half = dim / 2;
    for (int t = 0; t < T; t++) {
        double pos = (double) (pos0 + t);
        for (int i = 0; i < half; i++) {
            double phase = pos / pow(10000.0, (double) i / (double) (half - 1));
            x[(size_t) t * dim + i]        += (float) cos(phase);
            x[(size_t) t * dim + half + i] += (float) sin(phase);
        }
    }
}

// ---------------------------------------------------------------------------
// Prefill forward: input embeddings [dim, T] -> logits [card, T]
// ---------------------------------------------------------------------------

static ggml_tensor * build_layer_norm(ggml_context * ctx, ggml_tensor * x,
                                      ggml_tensor * w, ggml_tensor * b) {
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_mul(ctx, x, w);
    return ggml_add(ctx, x, b);
}

// Runs the full prefill graph and reads back the last position's logits.
static void forward_prefill(MidiModel * m, const float * input, int T, float * logits_last) {
    const MidiConfig & c = m->cfg;
    const int H = c.num_heads, D = c.head_dim();

    ggml_init_params ip = { ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false), NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.dim, T);
    ggml_set_name(inp, "inp");
    ggml_set_input(inp);

    ggml_tensor * x = inp;
    for (int l = 0; l < c.num_layers; l++) {
        MidiLayer & L = m->layers[l];

        // --- causal self-attention ---
        ggml_tensor * h   = build_layer_norm(ctx, x, L.norm1_w, L.norm1_b);
        ggml_tensor * qkv = ggml_mul_mat(ctx, L.in_proj, h);  // [3*dim, T]

        ggml_tensor * q = ggml_view_2d(ctx, qkv, c.dim, T, qkv->nb[1], 0);
        ggml_tensor * k = ggml_view_2d(ctx, qkv, c.dim, T, qkv->nb[1], (size_t) c.dim * 4);
        ggml_tensor * v = ggml_view_2d(ctx, qkv, c.dim, T, qkv->nb[1], (size_t) 2 * c.dim * 4);

        // packed layout per token is [h, d] (rearrange "(p h d)")
        ggml_tensor * Q = ggml_permute(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, q), D, H, T), 0, 2, 1, 3);  // [D, T, H]
        ggml_tensor * K = ggml_permute(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, k), D, H, T), 0, 2, 1, 3);  // [D, T, H]
        ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, v), D, H, T), 1, 2, 0, 3));  // [T, D, H]

        ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);                       // [T_k, T_q, H]
        kq = ggml_scale(ctx, kq, 1.0f / sqrtf((float) D));
        kq = ggml_diag_mask_inf(ctx, kq, 0);                              // causal (square)
        kq = ggml_soft_max(ctx, kq);

        ggml_tensor * kqv = ggml_mul_mat(ctx, V, kq);                     // [D, T_q, H]
        ggml_tensor * att = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // [D, H, T]
        att = ggml_reshape_2d(ctx, att, c.dim, T);
        att = ggml_mul_mat(ctx, L.out_proj, att);

        x = ggml_add(ctx, x, att);

        // --- FFN (exact GELU, matching torch F.gelu default) ---
        ggml_tensor * f = build_layer_norm(ctx, x, L.norm2_w, L.norm2_b);
        f = ggml_mul_mat(ctx, L.ffn1, f);
        f = ggml_gelu_erf(ctx, f);
        f = ggml_mul_mat(ctx, L.ffn2, f);
        x = ggml_add(ctx, x, f);
    }

    x = build_layer_norm(ctx, x, m->out_norm_w, m->out_norm_b);
    ggml_tensor * logits = ggml_mul_mat(ctx, m->head, x);  // [card, T]
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, logits);

    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {
        fprintf(stderr, "[ace-midi] FATAL: graph alloc failed\n");
        exit(1);
    }
    ggml_backend_tensor_set(inp, input, 0, (size_t) c.dim * T * 4);
    if (ggml_backend_sched_graph_compute(m->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[ace-midi] FATAL: graph compute failed\n");
        exit(1);
    }
    ggml_backend_tensor_get(logits, logits_last, (size_t) (T - 1) * logits->nb[1], (size_t) c.card * 4);

    ggml_free(ctx);
}

// ---------------------------------------------------------------------------
// Validation mode (Phase 1): logit parity vs the Python oracle
// ---------------------------------------------------------------------------

static std::vector<float> read_f32_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[ace-midi] cannot open %s\n", path.c_str());
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<float> v((size_t) sz / 4);
    size_t got = fread(v.data(), 4, v.size(), f);
    fclose(f);
    if (got != v.size()) {
        fprintf(stderr, "[ace-midi] short read on %s\n", path.c_str());
        exit(1);
    }
    return v;
}

// Parity gate: strict 1e-3 default is calibrated for the fp32 CPU backend
// (oracle is fp32 CPU torch). GPU backends drift ~5e-3 from TF32/accumulation
// order while still matching argmax — pass a looser --tol there; the real GPU
// acceptance test is greedy token-stream parity (Phase 2).
static int run_validate(MidiModel * m, const std::string & vdir, double tol) {
    const MidiConfig & c = m->cfg;

    std::vector<float> prefix = read_f32_file(vdir + "/prefix.bin");
    std::vector<float> ref    = read_f32_file(vdir + "/logits_bos.bin");
    if ((int) ref.size() != c.card) {
        fprintf(stderr, "[ace-midi] logits_bos.bin size %zu != card %d\n", ref.size(), c.card);
        return 1;
    }
    int T_prefix = (int) (prefix.size() / c.dim);
    int T        = T_prefix + 1;
    fprintf(stderr, "[ace-midi] validate: prefix %d frames + BOS\n", T_prefix);

    // Assemble input: [prefix | BOS] + sinusoidal positions
    std::vector<float> input((size_t) T * c.dim);
    memcpy(input.data(), prefix.data(), prefix.size() * 4);
    memcpy(input.data() + prefix.size(), m->bos_emb.data(), (size_t) c.dim * 4);
    add_sin_pos(input.data(), T, c.dim, 0);

    std::vector<float> logits(c.card);
    int64_t t0 = ggml_time_ms();
    forward_prefill(m, input.data(), T, logits.data());
    fprintf(stderr, "[ace-midi] prefill (%d tokens): %lld ms\n", T, (long long) (ggml_time_ms() - t0));

    // Compare
    double max_abs = 0, max_rel = 0;
    int    argmax_cpp = 0, argmax_ref = 0;
    for (int i = 0; i < c.card; i++) {
        double d = fabs((double) logits[i] - (double) ref[i]);
        if (d > max_abs) max_abs = d;
        double r = d / (fabs((double) ref[i]) + 1e-6);
        if (r > max_rel) max_rel = r;
        if (logits[i] > logits[argmax_cpp]) argmax_cpp = i;
        if (ref[i] > ref[argmax_ref]) argmax_ref = i;
    }
    printf("max_abs_diff = %.6g\nmax_rel_diff = %.6g\n", max_abs, max_rel);
    printf("argmax: cpp=%d (%.4f)  ref=%d (%.4f)\n", argmax_cpp, logits[argmax_cpp], argmax_ref, ref[argmax_ref]);

    bool pass = max_abs < tol && argmax_cpp == argmax_ref;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::string model_dir, validate_dir;
    double tol = 1e-3;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--validate") && i + 1 < argc) validate_dir = argv[++i];
        else if (!strcmp(argv[i], "--tol") && i + 1 < argc) tol = atof(argv[++i]);
    }
    if (model_dir.empty()) {
        fprintf(stderr,
                "ace-midi (Phase 1) — MuScriptor GGML port\n"
                "usage: ace-midi --model <dir> --validate <dir>\n"
                "  <model dir>    contains model.safetensors + config.json\n"
                "  <validate dir> oracle dumps from tools/ace-midi-validate.py\n");
        return 2;
    }

    ggml_time_init();
    MidiModel m;
    if (!load_model(&m, model_dir)) return 1;

    if (!validate_dir.empty()) {
        return run_validate(&m, validate_dir, tol);
    }
    fprintf(stderr, "[ace-midi] no mode given (Phase 1 supports --validate only)\n");
    return 2;
}
