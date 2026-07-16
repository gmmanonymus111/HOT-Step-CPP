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

// Constants fixed by the upstream model (transcription_model.py)
#define MIDI_SAMPLE_RATE   16000
#define MIDI_CHUNK_SAMPLES 80000   // 5 s
#define MIDI_N_FFT         2048
#define MIDI_HOP           160     // 100 Hz frame rate
#define MIDI_N_MELS        512
#define MIDI_MEL_FRAMES    501     // 1 + 80000/160 (center=True)
#define MIDI_MAX_GEN       2000    // max tokens per chunk
#define MIDI_EOS_ID        1

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

    // KV cache (batch=1): per layer K [D, max_seq, H], V [max_seq, D, H], f32
    ggml_context *             kv_ctx = nullptr;
    ggml_backend_buffer_t      kv_buf = nullptr;
    std::vector<ggml_tensor *> kv_k, kv_v;
    int                        max_seq = 0;

    // Host-side copies for CPU input assembly / mel frontend
    std::vector<float> emb_host;     // [card+1, dim] (row-major, incl. BOS row)
    std::vector<float> mel_window;   // [2048]
    std::vector<float> mel_fb;       // [1025, 512] row-major
    std::vector<float> mel_proj_w;   // [dim, 512] row-major (torch [out,in])
    std::vector<float> mel_proj_b;   // [dim]
    std::vector<float> ds_null_emb;  // dataset_name embed row 1 (None cond)
    std::vector<float> ig_null_emb;  // instrument_group embed row 1 (None cond)
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

    // Host-side copies: token embeddings (input assembly), mel frontend
    // weights, and the null class-conditioner rows. Conditioner tokenize
    // maps None -> -1, +1 in tokenize, +1 again in forward => row 1.
    auto read_host = [&](const char * name, std::vector<float> & out) {
        const STEntry * e = st_find(st, name);
        if (!e) {
            fprintf(stderr, "[ace-midi] FATAL: missing tensor %s\n", name);
            exit(1);
        }
        int64_t n = 1;
        for (int i = 0; i < e->n_dims; i++) n *= e->shape[i];
        out.resize((size_t) n);
        const void * src = st_data(st, *e);
        if (e->dtype == "F32") {
            memcpy(out.data(), src, (size_t) n * 4);
        } else {
            const uint16_t * s = (const uint16_t *) src;
            for (int64_t i = 0; i < n; i++) {
                if (e->dtype == "BF16") {
                    uint32_t bits = (uint32_t) s[i] << 16;
                    memcpy(&out[i], &bits, 4);
                } else {
                    out[i] = ggml_fp16_to_fp32((ggml_fp16_t) s[i]);
                }
            }
        }
    };
    read_host(emb_key.c_str(), m->emb_host);
    read_host("condition_provider.conditioners.self_wav.mel_spec_transform.spectrogram.window", m->mel_window);
    read_host("condition_provider.conditioners.self_wav.mel_spec_transform.mel_scale.fb", m->mel_fb);
    read_host("condition_provider.conditioners.self_wav.output_proj.weight", m->mel_proj_w);
    read_host("condition_provider.conditioners.self_wav.output_proj.bias", m->mel_proj_b);
    {
        std::vector<float> tmp;
        read_host("condition_provider.conditioners.dataset_name.embed.weight", tmp);
        m->ds_null_emb.assign(tmp.begin() + c.dim, tmp.begin() + 2 * c.dim);  // row 1
        read_host("condition_provider.conditioners.instrument_group.embed.weight", tmp);
        m->ig_null_emb.assign(tmp.begin() + c.dim, tmp.begin() + 2 * c.dim);  // row 1
    }

    // KV cache: prefix (mel 501 + 2 class conds) + BOS + tie prompt + max gen
    m->max_seq = MIDI_MEL_FRAMES + 2 + 1 + 300 + MIDI_MAX_GEN;
    {
        const int D = c.head_dim(), H = c.num_heads;
        ggml_init_params kp = { (size_t) c.num_layers * 2 * ggml_tensor_overhead() + 4096, NULL, true };
        m->kv_ctx = ggml_init(kp);
        m->kv_k.resize(c.num_layers);
        m->kv_v.resize(c.num_layers);
        for (int l = 0; l < c.num_layers; l++) {
            m->kv_k[l] = ggml_new_tensor_3d(m->kv_ctx, GGML_TYPE_F32, D, m->max_seq, H);
            m->kv_v[l] = ggml_new_tensor_3d(m->kv_ctx, GGML_TYPE_F32, m->max_seq, D, H);
            char nm[32];
            snprintf(nm, sizeof(nm), "kv_k_%d", l);
            ggml_set_name(m->kv_k[l], nm);
            snprintf(nm, sizeof(nm), "kv_v_%d", l);
            ggml_set_name(m->kv_v[l], nm);
        }
        m->kv_buf = ggml_backend_alloc_ctx_tensors(m->kv_ctx, m->backend);
        if (!m->kv_buf) {
            fprintf(stderr, "[ace-midi] FATAL: KV cache alloc failed\n");
            return false;
        }
    }

    fprintf(stderr, "[ace-midi] loaded %d layers (%.1f MB weights, %.1f MB KV cache)\n",
            c.num_layers, (double) ggml_backend_buffer_get_size(m->wbuf) / 1e6,
            (double) ggml_backend_buffer_get_size(m->kv_buf) / 1e6);
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

// Forward T tokens at cache position n_past; writes K/V into the cache and
// reads back the LAST position's logits. n_past=0 with T>1 is the prefill;
// T=1 with n_past>0 is a decode step. Caller advances n_past by T afterwards.
static void forward_tokens(MidiModel * m, const float * input, int T, int n_past, float * logits_last) {
    const MidiConfig & c = m->cfg;
    const int H = c.num_heads, D = c.head_dim();
    const int n_kv = n_past + T;

    ggml_init_params ip = { ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false), NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.dim, T);
    ggml_set_name(inp, "inp");
    ggml_set_input(inp);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    ggml_tensor * x = inp;
    for (int l = 0; l < c.num_layers; l++) {
        MidiLayer & L = m->layers[l];

        // --- causal self-attention with KV cache ---
        ggml_tensor * h   = build_layer_norm(ctx, x, L.norm1_w, L.norm1_b);
        ggml_tensor * qkv = ggml_mul_mat(ctx, L.in_proj, h);  // [3*dim, T]

        ggml_tensor * q = ggml_view_2d(ctx, qkv, c.dim, T, qkv->nb[1], 0);
        ggml_tensor * k = ggml_view_2d(ctx, qkv, c.dim, T, qkv->nb[1], (size_t) c.dim * 4);
        ggml_tensor * v = ggml_view_2d(ctx, qkv, c.dim, T, qkv->nb[1], (size_t) 2 * c.dim * 4);

        // packed layout per token is [h, d] (rearrange "(p h d)")
        ggml_tensor * k3 = ggml_reshape_3d(ctx, ggml_cont(ctx, k), D, H, T);
        ggml_tensor * v3 = ggml_reshape_3d(ctx, ggml_cont(ctx, v), D, H, T);

        // append current K rows: cache K layout [D, max_seq, H], slice dim1 [n_past, n_past+T)
        ggml_tensor * kc = m->kv_k[l];
        ggml_tensor * k_dst = ggml_view_3d(ctx, kc, D, T, H, kc->nb[1], kc->nb[2],
                                           (size_t) n_past * kc->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_permute(ctx, k3, 0, 2, 1, 3), k_dst));

        // append current V rows: cache V layout [max_seq, D, H], slice dim0 [n_past, n_past+T)
        ggml_tensor * vc = m->kv_v[l];
        ggml_tensor * v_dst = ggml_view_3d(ctx, vc, T, D, H, vc->nb[1], vc->nb[2],
                                           (size_t) n_past * vc->nb[0]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_permute(ctx, v3, 1, 2, 0, 3), v_dst));

        ggml_tensor * Q = ggml_permute(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, q), D, H, T), 0, 2, 1, 3);  // [D, T, H]
        ggml_tensor * K = ggml_view_3d(ctx, kc, D, n_kv, H, kc->nb[1], kc->nb[2], 0);   // [D, n_kv, H]
        ggml_tensor * V = ggml_view_3d(ctx, vc, n_kv, D, H, vc->nb[1], vc->nb[2], 0);   // [n_kv, D, H]

        ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);                       // [n_kv, T, H]
        kq = ggml_scale(ctx, kq, 1.0f / sqrtf((float) D));
        kq = ggml_diag_mask_inf(ctx, kq, n_past);                         // causal, bottom-right aligned
        kq = ggml_soft_max(ctx, kq);

        ggml_tensor * kqv = ggml_mul_mat(ctx, V, kq);                     // [D, T, H]
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
// Mel frontend (conditioners.py MelSpectrogramConditioner, torchaudio-equiv):
// magnitude STFT (2048/160, periodic hann from ckpt, center reflect pad) ->
// HTK mel fb from ckpt -> log(+1e-6) -> output_proj -> zero masked frames.
// ---------------------------------------------------------------------------

static void fft_radix2(float * re, float * im, int n) {
    // exact twiddle table (per stage), computed once — the naive multiplicative
    // twiddle recurrence drifts enough to visibly perturb the log-mel
    static std::vector<float> tw_re, tw_im;
    static int tw_n = 0;
    if (tw_n != n) {
        tw_re.assign((size_t) n, 0.0f);
        tw_im.assign((size_t) n, 0.0f);
        for (int len = 2, base = 0; len <= n; len <<= 1, base += len >> 2) {
            for (int j = 0; j < len / 2; j++) {
                double ang = -2.0 * 3.14159265358979323846 * j / len;
                tw_re[(size_t) base + j] = (float) cos(ang);
                tw_im[(size_t) base + j] = (float) sin(ang);
            }
        }
        tw_n = n;
    }

    // bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j |= bit;
        if (i < j) {
            float t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2, base = 0; len <= n; len <<= 1, base += len >> 2) {
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len / 2; j++) {
                int   a = i + j, b = i + j + len / 2;
                float cr = tw_re[(size_t) base + j], ci = tw_im[(size_t) base + j];
                float xr = re[b] * cr - im[b] * ci;
                float xi = re[b] * ci + im[b] * cr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr;        im[a] += xi;
            }
        }
    }
}

// Compute the full conditioning prefix for one 5 s chunk:
// [mel 501 | dataset_name(None) | instrument_group(None)] rows of dim floats.
// n_samples is the unpadded chunk length (masks trailing mel frames).
static std::vector<float> compute_prefix(MidiModel * m, const float * wav, int n_samples) {
    const MidiConfig & c = m->cfg;
    const int pad = MIDI_N_FFT / 2;
    const int n   = MIDI_CHUNK_SAMPLES;

    // reflect-padded chunk (zero-pad the tail to 5 s first, like F.pad)
    std::vector<float> padded(n + 2 * pad, 0.0f);
    auto sample_at = [&](int i) -> float {
        // reflect at both edges of the zero-padded 80000-sample chunk
        if (i < 0) i = -i;
        if (i >= n) i = 2 * n - 2 - i;
        return (i >= 0 && i < n_samples) ? wav[i] : 0.0f;
    };
    for (int i = 0; i < n + 2 * pad; i++) padded[i] = sample_at(i - pad);

    const int T = MIDI_MEL_FRAMES;
    std::vector<float> prefix((size_t) (T + 2) * c.dim, 0.0f);

    // frames masked at index >= n_samples/160.0 (length_to_mask semantics)
    const double frame_limit = (double) n_samples / (double) MIDI_HOP;

    std::vector<float>  re(MIDI_N_FFT), im(MIDI_N_FFT), logmel(MIDI_N_MELS);
    std::vector<double> melacc(MIDI_N_MELS);
    for (int f = 0; f < T; f++) {
        if ((double) f >= frame_limit) continue;  // masked -> zero row

        const float * frame = padded.data() + (size_t) f * MIDI_HOP;
        for (int i = 0; i < MIDI_N_FFT; i++) {
            re[i] = frame[i] * m->mel_window[i];
            im[i] = 0.0f;
        }
        fft_radix2(re.data(), im.data(), MIDI_N_FFT);

        // magnitude (power=1.0) -> mel -> log (double accumulation)
        const int n_freq = MIDI_N_FFT / 2 + 1;
        for (int mm = 0; mm < MIDI_N_MELS; mm++) melacc[mm] = 0.0;
        for (int k = 0; k < n_freq; k++) {
            double mag = sqrt((double) re[k] * re[k] + (double) im[k] * im[k]);
            if (mag == 0.0) continue;
            const float * fbrow = m->mel_fb.data() + (size_t) k * MIDI_N_MELS;
            for (int mm = 0; mm < MIDI_N_MELS; mm++) melacc[mm] += mag * fbrow[mm];
        }
        for (int mm = 0; mm < MIDI_N_MELS; mm++) logmel[mm] = logf((float) melacc[mm] + 1e-6f);

        // output_proj: [dim, 512] @ logmel + bias
        float * out = prefix.data() + (size_t) f * c.dim;
        for (int o = 0; o < c.dim; o++) {
            const float * wrow = m->mel_proj_w.data() + (size_t) o * MIDI_N_MELS;
            double acc = m->mel_proj_b[o];
            for (int i = 0; i < MIDI_N_MELS; i++) acc += (double) wrow[i] * logmel[i];
            out[o] = (float) acc;
        }
    }

    // class conditioner rows (always the None/null class at inference)
    memcpy(prefix.data() + (size_t) T * c.dim, m->ds_null_emb.data(), (size_t) c.dim * 4);
    memcpy(prefix.data() + (size_t) (T + 1) * c.dim, m->ig_null_emb.data(), (size_t) c.dim * 4);
    return prefix;
}

// ---------------------------------------------------------------------------
// Greedy chunk decode: prefill [prefix | BOS | prompt] then argmax steps.
// Emits every accepted token (prompt tokens included, EOS excluded) via cb.
// ---------------------------------------------------------------------------

static void greedy_argmax_range(const float * logits, int n_valid, int * out) {
    int best = 0;
    for (int i = 1; i < n_valid; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    *out = best;
}

template <typename TokenCb>
static void decode_chunk(MidiModel * m, const std::vector<float> & prefix,
                         const std::vector<int> & prompt, int max_gen, TokenCb cb) {
    const MidiConfig & c = m->cfg;
    const int n_valid = c.card < 1393 ? c.card : 1393;  // logits[1393:] masked upstream
    const int T_prefix = (int) (prefix.size() / c.dim);

    // prefill input: prefix + BOS + prompt tokens, sinusoidal positions from 0
    int T0 = T_prefix + 1 + (int) prompt.size();
    std::vector<float> input((size_t) T0 * c.dim);
    memcpy(input.data(), prefix.data(), prefix.size() * 4);
    memcpy(input.data() + prefix.size(), m->emb_host.data() + (size_t) c.bos_id() * c.dim, (size_t) c.dim * 4);
    for (size_t i = 0; i < prompt.size(); i++) {
        memcpy(input.data() + prefix.size() + (i + 1) * c.dim,
               m->emb_host.data() + (size_t) prompt[i] * c.dim, (size_t) c.dim * 4);
        cb(prompt[i]);  // teacher-forced tokens flow through the stream
    }
    add_sin_pos(input.data(), T0, c.dim, 0);

    std::vector<float> logits(c.card);
    forward_tokens(m, input.data(), T0, 0, logits.data());
    int n_past = T0;

    int tok;
    greedy_argmax_range(logits.data(), n_valid, &tok);

    std::vector<float> step(c.dim);
    for (int i = (int) prompt.size(); i < max_gen; i++) {
        if (tok == MIDI_EOS_ID) return;
        cb(tok);
        if (n_past + 1 > m->max_seq) {
            fprintf(stderr, "[ace-midi] WARNING: KV cache full at %d tokens\n", n_past);
            return;
        }
        memcpy(step.data(), m->emb_host.data() + (size_t) tok * c.dim, (size_t) c.dim * 4);
        add_sin_pos(step.data(), 1, c.dim, n_past);
        forward_tokens(m, step.data(), 1, n_past, logits.data());
        n_past++;
        greedy_argmax_range(logits.data(), n_valid, &tok);
    }
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
    memcpy(input.data() + prefix.size(), m->emb_host.data() + (size_t) c.bos_id() * c.dim, (size_t) c.dim * 4);
    add_sin_pos(input.data(), T, c.dim, 0);

    std::vector<float> logits(c.card);
    int64_t t0 = ggml_time_ms();
    forward_tokens(m, input.data(), T, 0, logits.data());
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

// Phase 2 validation: C++ mel prefix vs oracle prefix.bin, then greedy
// token-stream parity vs tokens_ref.json.
static int run_validate_decode(MidiModel * m, const std::string & vdir) {
    const MidiConfig & c = m->cfg;

    std::vector<float> wav        = read_f32_file(vdir + "/wav.bin");
    std::vector<float> prefix_ref = read_f32_file(vdir + "/prefix.bin");

    // 1. mel-prefix parity
    std::vector<float> prefix = compute_prefix(m, wav.data(), (int) wav.size());
    if (prefix.size() != prefix_ref.size()) {
        fprintf(stderr, "[ace-midi] prefix size mismatch: cpp %zu vs ref %zu\n", prefix.size(), prefix_ref.size());
        return 1;
    }
    double mel_max_abs = 0;
    for (size_t i = 0; i < prefix.size(); i++) {
        double d = fabs((double) prefix[i] - (double) prefix_ref[i]);
        if (d > mel_max_abs) mel_max_abs = d;
    }
    printf("prefix_max_abs_diff = %.6g\n", mel_max_abs);

    // 2. greedy token-stream parity (exact-match territory in fp32)
    std::vector<int> ref_tokens;
    {
        FILE * f = fopen((vdir + "/tokens_ref.json").c_str(), "rb");
        if (!f) {
            fprintf(stderr, "[ace-midi] cannot open tokens_ref.json\n");
            return 1;
        }
        std::string j(1 << 20, 0);
        size_t n = fread(j.data(), 1, j.size() - 1, f);
        fclose(f);
        j.resize(n);
        for (const char * p = j.c_str(); *p; p++) {
            if (*p >= '0' && *p <= '9') {
                ref_tokens.push_back(atoi(p));
                while (*p >= '0' && *p <= '9') p++;
            }
        }
    }

    std::vector<int> tokens;
    int64_t t0 = ggml_time_ms();
    decode_chunk(m, prefix, {}, 256, [&](int t) { tokens.push_back(t); });
    int64_t dt = ggml_time_ms() - t0;

    int first_div = -1;
    size_t n_cmp = tokens.size() < ref_tokens.size() ? tokens.size() : ref_tokens.size();
    for (size_t i = 0; i < n_cmp; i++) {
        if (tokens[i] != ref_tokens[i]) { first_div = (int) i; break; }
    }
    printf("tokens: cpp=%zu ref=%zu first_divergence=%d (%lld ms, %.1f tok/s)\n",
           tokens.size(), ref_tokens.size(), first_div,
           (long long) dt, tokens.empty() ? 0.0 : 1000.0 * (double) tokens.size() / (double) dt);
    if (first_div >= 0) {
        printf("  at %d: cpp=%d ref=%d\n", first_div, tokens[first_div], ref_tokens[first_div]);
    }

    bool pass = first_div < 0 && tokens.size() == ref_tokens.size();
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::string model_dir, validate_dir, validate_decode_dir;
    double tol = 1e-3;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--validate") && i + 1 < argc) validate_dir = argv[++i];
        else if (!strcmp(argv[i], "--validate-decode") && i + 1 < argc) validate_decode_dir = argv[++i];
        else if (!strcmp(argv[i], "--tol") && i + 1 < argc) tol = atof(argv[++i]);
    }
    if (model_dir.empty()) {
        fprintf(stderr,
                "ace-midi (Phase 2) — MuScriptor GGML port\n"
                "usage: ace-midi --model <dir> [--validate <dir>] [--validate-decode <dir>] [--tol <x>]\n"
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
    if (!validate_decode_dir.empty()) {
        return run_validate_decode(&m, validate_decode_dir);
    }
    fprintf(stderr, "[ace-midi] no mode given (--validate / --validate-decode)\n");
    return 2;
}
