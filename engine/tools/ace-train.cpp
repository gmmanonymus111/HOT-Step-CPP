// ace-train.cpp — HOT-Step training toolchain.
// Phase 2 ships one subcommand: `preprocess`.
//
// Standalone tool (no acestep-core): every engine module it uses is a
// header-only `static` API. Machine-readable JSONL on stdout (--jsonl),
// human logs on stderr. Exit: 0 ok, 1 runtime failure, 2 usage.
//
// docs/plans/2026-07-27-preprocess-implementation.md §3.1

#include "model-registry.h"
#include "train/dit-train-run.h"   // pulls in every dit-*.h (DiT LoRA trainer)
#include "train/lm-train-run.h"    // pulls in every lm-*.h (LM LoRA trainer)
#include "train/preprocess-run.h"  // pulls in st-write.h + preprocess-io.h
#include "train/spike-run.h"       // Phase-0 training spike (evidence code)

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <dirent.h>
#endif

// ─── JSONL emitter ──────────────────────────────────────────────────────────

static bool g_jsonl = false;

static void jl(const char * fmt, ...) {  // printf-style, caller writes the JSON
    if (!g_jsonl) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    putchar('\n');
    fflush(stdout);  // MANDATORY: Node reads line-by-line
}

static std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (size_t i = 0; i < s.size(); i++) {
        unsigned char c = (unsigned char) s[i];
        switch (c) {
            case '"':
                o += "\\\"";
                break;
            case '\\':
                o += "\\\\";
                break;
            case '\b':
                o += "\\b";
                break;
            case '\f':
                o += "\\f";
                break;
            case '\n':
                o += "\\n";
                break;
            case '\r':
                o += "\\r";
                break;
            case '\t':
                o += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char b[8];
                    snprintf(b, sizeof(b), "\\u%04x", c);
                    o += b;
                } else {
                    o.push_back((char) c);
                }
        }
    }
    return o;
}

static void jl_stage(const char * stage, const char * path, long long ms) {
    jl("{\"type\":\"model\",\"stage\":\"%s\",\"path\":\"%s\",\"ms\":%lld}", stage, json_escape(path).c_str(), ms);
    fprintf(stderr, "[ace-train] loaded %s (%lld ms)\n", stage, ms);
}

static void jl_warn(const char * msg) {
    jl("{\"type\":\"log\",\"level\":\"warn\",\"message\":\"%s\"}", json_escape(msg).c_str());
    fprintf(stderr, "[ace-train] WARN: %s\n", msg);
}

// ─── usage ──────────────────────────────────────────────────────────────────

static void print_usage(void) {
    fprintf(stderr,
            "ace-train %s — HOT-Step training toolchain\n"
            "\n"
            "usage: ace-train <subcommand> [options]\n"
            "\n"
            "Subcommands:\n"
            "  preprocess    Build per-song tensor caches from a dataset.\n"
            "  train-lm      Train a planner-LM LoRA from a preprocessed tensor cache.\n"
            "  train-dit     Train a DiT LoRA from a preprocessed tensor cache.\n"
            "\n"
            "ace-train preprocess  (all paths absolute; long options only; \"--flag value\" form)\n"
            "\n"
            "  Input (exactly one required):\n"
            "    --manifest <path>          preprocess_manifest.json (server path)\n"
            "    --dataset  <path>          Side-Step dataset.json (standalone path)\n"
            "\n"
            "  Output (required):\n"
            "    --out <dir>                output directory; created if missing\n"
            "\n"
            "  Models (required):\n"
            "    --models <dir>             root scanned for name lookups\n"
            "    --dit <name|path>          DiT GGUF/dir — cond-enc weights, silence_latent, variant identity\n"
            "    --vae <name|path>          VAE GGUF/safetensors (must NOT be .onnx)\n"
            "    --text-enc <name|path>     Qwen3-Embedding GGUF/dir (also supplies BPE vocab)\n"
            "\n"
            "  Options (defaults shown):\n"
            "    --max-duration <sec>          240      0 = no truncation\n"
            "    --normalize <none|peak>       peak\n"
            "    --target-db <db>              -1.0     peak-normalize target\n"
            "    --dtype <f32|bf16>            f32      storage dtype for every tensor\n"
            "    --compat <hotstep|sidestep>   hotstep\n"
            "    --timbre <silence|zeros>      silence\n"
            "    --max-caption-tokens <n>      256\n"
            "    --max-lyric-tokens <n>        512\n"
            "    --vae-chunk <latent-frames>   384\n"
            "    --vae-overlap <latent-frames> 48\n"
            "    --ffmpeg <path>               (none)   required for non-WAV/MP3 input\n"
            "    --overwrite                   off      re-encode songs that already have a cache\n"
            "    --limit <n>                   0        debug: stop after n songs (0 = all)\n"
            "    --jsonl                       off      emit machine-readable JSONL on stdout\n"
            "    --verify <file>               debug: re-open a written .safetensors and report it\n"
            "    -h, --help\n"
            "\n"
            "ace-train train-lm  (all paths absolute; long options only; \"--flag value\" form)\n"
            "\n"
            "  Stages (default: extract,train,export):\n"
            "    --stages <csv>              subset of extract,train,export in that order\n"
            "\n"
            "  Input:\n"
            "    --tensors <dir>             preprocess variant dir (required for `extract`;\n"
            "                                also the default location of the codes file)\n"
            "    --codes <path>              lm_codes.jsonl  (default <tensors>/lm_codes.jsonl)\n"
            "    --out <dir>                 adapter output dir (required for train/export)\n"
            "\n"
            "  Models:\n"
            "    --models <dir>              root scanned by registry_scan() for name lookups\n"
            "    --dit <name|path>           FSQ-tokenizer source for `extract`\n"
            "                                (default: dit_path from <tensors>/preprocess_meta.json)\n"
            "    --lm <name|path>            LM base GGUF (required for `train`)\n"
            "    --lm-size <0.6B|1.7B|4B>    label written into adapter_config/base_model_name_or_path\n"
            "                                (default: inferred from the LM's hidden_size/n_layers)\n"
            "\n"
            "  LoRA:\n"
            "    --rank <n>                  16\n"
            "    --alpha <n>                 32\n"
            "\n"
            "  Optimizer / schedule:\n"
            "    --lr <f>                    1e-4\n"
            "    --epochs <n>                16          hard cap; target-loss usually stops earlier\n"
            "    --grad-accum <n>            4\n"
            "    --warmup-ratio <f>          0.05\n"
            "    --grad-clip <f>             1.0         0 = disabled\n"
            "    --weight-decay <f>          0.01\n"
            "    --seed <n>                  42\n"
            "    --target-loss <f>           0.4         0 = disabled (run to the epoch cap)\n"
            "    --order <shuffle|fixed>     shuffle     `fixed` = file order (A/B parity runs)\n"
            "\n"
            "  Sequence / memory:\n"
            "    --max-len <n>               0           0 = auto-fit from free VRAM\n"
            "    --vram-reserve-mb <n>       1024        headroom left unallocated\n"
            "    --low-vram <auto|on|off>    auto        per-layer checkpointing + chunked CE;\n"
            "                                            auto = ON iff --lm-size is 4B, or the naive\n"
            "                                            auto-fit would yield max-len < 2048\n"
            "    --attn-head-block <n>       -1          heads per attention block inside a segment;\n"
            "                                            0 = whole-head attention (naive behaviour),\n"
            "                                            -1 = engine picks (<=16 heads -> 0, else 8).\n"
            "                                            Must divide n_heads AND n*n_kv/n_heads >= 1\n"
            "    --lm-chunk <n>              128         trained positions per CE chunk (low-vram only)\n"
            "    --loss-on-cot                           default ON\n"
            "    --no-loss-on-cot\n"
            "\n"
            "  Run management:\n"
            "    --milestone-step <f>        0.1         0 = disabled\n"
            "    --milestone-keep <n>        6\n"
            "    --no-milestones\n"
            "    --overwrite                 off         re-extract every song / overwrite <out>\n"
            "    --limit <n>                 0           debug: first n songs only\n"
            "    --self-test                 off         run the correctness gates and exit\n"
            "    --jsonl                     off         machine-readable JSONL on stdout\n"
            "    -h, --help\n"
            "\n"
            "ace-train train-dit  (all paths absolute; long options only; \"--flag value\" form)\n"
            "\n"
            "  Stages (default: train,export):\n"
            "    --stages <csv>              subset of train,export in that order\n"
            "\n"
            "  Input / output:\n"
            "    --tensors <dir>             preprocess variant dir (required)\n"
            "    --out <dir>                 adapter output dir (required)\n"
            "    --models <dir>              root scanned by registry_scan() for name lookups\n"
            "    --dit <name|path>           base DiT GGUF (default: dit_path from\n"
            "                                <tensors>/preprocess_meta.json — must match the\n"
            "                                variant the latents were made against)\n"
            "\n"
            "  Adapter:\n"
            "    --adapter-type <lora>       lora            (lokr reserved, D22 — rejected in v1)\n"
            "    --rank <n>                  16\n"
            "    --alpha <n>                 32\n"
            "    --target-mlp                off             also LoRA mlp gate/up/down\n"
            "    --layers <n>                0               0 = auto; else train the top n layers\n"
            "\n"
            "  Objective (design 4.5):\n"
            "    --loss-weighting <none|flow_snr>   flow_snr\n"
            "    --snr-gamma <f>             5.0\n"
            "    --t-bias <f>                0.5\n"
            "    --channel-balance / --no-channel-balance    default ON\n"
            "    --timestep-mu <f>           -0.4\n"
            "    --timestep-sigma <f>        1.0\n"
            "    --t-min <f>                 0.0             interval-expert window (rejection resample)\n"
            "    --t-max <f>                 1.0\n"
            "    --cfg-ratio <f>             0.15\n"
            "    --genre-ratio <n>           0               percent, 0-100\n"
            "\n"
            "  Optimizer / schedule:\n"
            "    --lr <f>                    5e-4\n"
            "    --epochs <n>                100         hard cap; target-loss usually stops earlier\n"
            "    --grad-accum <n>            4\n"
            "    --warmup-ratio <f>          0.05\n"
            "    --grad-clip <f>             1.0         0 = disabled\n"
            "    --weight-decay <f>          0.01\n"
            "    --seed <n>                  42\n"
            "    --target-loss <f>           0.4         0 = disabled (run to the epoch cap)\n"
            "    --order <shuffle|fixed>     shuffle\n"
            "\n"
            "  Crop / memory:\n"
            "    --crop <n>                  0           latent frames; 0 = auto-fit\n"
            "    --crop-min <n>              375\n"
            "    --crop-max <n>              1250\n"
            "    --vram-reserve-mb <n>       2048        desktop/OS headroom left unallocated\n"
            "    --vram-safety <f>           0.05        extra margin on the footprint model\n"
            "\n"
            "  Run management:\n"
            "    --milestone-step <f>        0.1         0 = disabled\n"
            "    --milestone-keep <n>        6\n"
            "    --no-milestones\n"
            "    --overwrite                 off\n"
            "    --limit <n>                 0           debug: first n songs only\n"
            "    --self-test                 off         run the correctness gates and exit\n"
            "    --jsonl                     off         machine-readable JSONL on stdout\n"
            "    -h, --help\n",
            ACE_VERSION);
}

// ─── path helpers ───────────────────────────────────────────────────────────

static bool looks_like_path(const std::string & v) {
    return v.find('/') != std::string::npos || v.find('\\') != std::string::npos || pm_path_exists(v);
}

// Resolve a model argument: explicit path, else exact `name` lookup in a registry bucket.
static bool resolve_model(const std::string & arg, const std::vector<ModelEntry> & bucket, bool non_onnx,
                          std::string & out, std::string & name_out) {
    if (arg.empty()) {
        return false;
    }
    if (looks_like_path(arg)) {
        if (!pm_path_exists(arg)) {
            return false;
        }
        // The registry branch below filters ONNX out via registry_find_non_onnx;
        // an explicit path must obey the same rule or a decoder-only ONNX VAE
        // reaches vae_enc_load(), which fatal-exits with no diagnostic.
        if (non_onnx && arg.size() >= 5 && pm_lower(arg.substr(arg.size() - 5)) == ".onnx") {
            return false;
        }
        out      = arg;
        name_out = pm_basename(arg);
        return true;
    }
    const ModelEntry * e = non_onnx ? registry_find_non_onnx(bucket, arg.c_str()) : registry_find(bucket, arg.c_str());
    if (!e) {
        return false;
    }
    out      = e->path;
    name_out = e->name;
    return true;
}

// Delete every file in `dir` whose name ends with `suffix` (non-recursive).
static void purge_suffix(const std::string & dir, const char * suffix) {
    std::vector<std::string> names;
    registry_list_dir(dir.c_str(), &names);
    const size_t slen = strlen(suffix);
    for (size_t i = 0; i < names.size(); i++) {
        if (names[i].size() >= slen && names[i].compare(names[i].size() - slen, slen, suffix) == 0) {
            remove((dir + "/" + names[i]).c_str());
        }
    }
}

static void purge_all(const std::string & dir) {
    std::vector<std::string> names;
    registry_list_dir(dir.c_str(), &names);
    for (size_t i = 0; i < names.size(); i++) {
        remove((dir + "/" + names[i]).c_str());
    }
}

// ─── --verify (debug round-trip reader, §6.2 assertion 6) ───────────────────

static int cmd_verify(const std::string & file) {
    STFile st;
    if (!st_open(&st, file.c_str())) {
        fprintf(stderr, "[verify] cannot open %s\n", file.c_str());
        return 1;
    }
    bool ok = true;
    fprintf(stderr, "[verify] %s — %zu tensors, data_offset=%zu, size=%zu\n", file.c_str(), st.entries.size(),
            st.data_offset, st.file_size);
    for (size_t i = 0; i < st.entries.size(); i++) {
        const STEntry & e = st.entries[i];
        long long       n = 1;
        for (int d = 0; d < e.n_dims; d++) {
            n *= e.shape[d];
        }
        const size_t esz   = (e.dtype == "F32") ? 4u : 2u;
        const size_t want  = (size_t) n * esz;
        const size_t have  = e.data_end - e.data_start;
        const bool   inrng = st.data_offset + e.data_end <= st.file_size;
        bool         fin   = true;
        double       mn = 0, mx = 0, sum = 0;
        const void * d = st_data(st, e);
        if (d && inrng && want == have) {
            for (long long k = 0; k < n; k++) {
                float v = (e.dtype == "F32") ? ((const float *) d)[k] : pm_bf16_to_f32(((const uint16_t *) d)[k]);
                if (!std::isfinite(v)) {
                    fin = false;
                }
                if (k == 0 || v < mn) {
                    mn = v;
                }
                if (k == 0 || v > mx) {
                    mx = v;
                }
                sum += v;
            }
        }
        const bool entry_ok = d && inrng && want == have && fin;
        ok                  = ok && entry_ok;
        fprintf(stderr, "[verify]   %-32s %-5s [", e.name.c_str(), e.dtype.c_str());
        for (int d2 = 0; d2 < e.n_dims; d2++) {
            fprintf(stderr, "%s%lld", d2 ? "," : "", (long long) e.shape[d2]);
        }
        fprintf(stderr, "] bytes=%zu/%zu finite=%s min=%.6g max=%.6g mean=%.6g %s\n", have, want, fin ? "yes" : "NO",
                mn, mx, n ? sum / (double) n : 0.0, entry_ok ? "OK" : "FAIL");
        if (g_jsonl) {
            std::string shape;
            for (int d2 = 0; d2 < e.n_dims; d2++) {
                char b[32];
                snprintf(b, sizeof(b), "%s%lld", d2 ? "," : "", (long long) e.shape[d2]);
                shape += b;
            }
            jl("{\"type\":\"verify\",\"name\":\"%s\",\"dtype\":\"%s\",\"shape\":[%s],\"bytes\":%zu,\"expected\":%zu,"
               "\"finite\":%s,\"min\":%.9g,\"max\":%.9g,\"mean\":%.9g,\"ok\":%s}",
               json_escape(e.name).c_str(), e.dtype.c_str(), shape.c_str(), have, want, fin ? "true" : "false", mn, mx,
               n ? sum / (double) n : 0.0, entry_ok ? "true" : "false");
        }
    }
    st_close(&st);
    return ok ? 0 : 1;
}

// ─── preprocess ─────────────────────────────────────────────────────────────

static int cmd_preprocess(int argc, char ** argv) {
    PreprocessOpts o;
    std::string    manifest_path, dataset_path, models_dir;
    std::string    dit_arg, vae_arg, text_arg, verify_file;
    std::string    normalize = "peak", dtype = "f32", compat = "hotstep", timbre = "silence";
    int            limit = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--manifest") && i + 1 < argc) manifest_path = argv[++i];
        else if (!strcmp(argv[i], "--dataset") && i + 1 < argc) dataset_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) o.out_dir = argv[++i];
        else if (!strcmp(argv[i], "--models") && i + 1 < argc) models_dir = argv[++i];
        else if (!strcmp(argv[i], "--dit") && i + 1 < argc) dit_arg = argv[++i];
        else if (!strcmp(argv[i], "--vae") && i + 1 < argc) vae_arg = argv[++i];
        else if (!strcmp(argv[i], "--text-enc") && i + 1 < argc) text_arg = argv[++i];
        else if (!strcmp(argv[i], "--max-duration") && i + 1 < argc) o.max_duration = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--normalize") && i + 1 < argc) normalize = argv[++i];
        else if (!strcmp(argv[i], "--target-db") && i + 1 < argc) o.target_db = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--dtype") && i + 1 < argc) dtype = argv[++i];
        else if (!strcmp(argv[i], "--compat") && i + 1 < argc) compat = argv[++i];
        else if (!strcmp(argv[i], "--timbre") && i + 1 < argc) timbre = argv[++i];
        else if (!strcmp(argv[i], "--max-caption-tokens") && i + 1 < argc) o.max_caption_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-lyric-tokens") && i + 1 < argc) o.max_lyric_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vae-chunk") && i + 1 < argc) o.vae_chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vae-overlap") && i + 1 < argc) o.vae_overlap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ffmpeg") && i + 1 < argc) o.ffmpeg = argv[++i];
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verify") && i + 1 < argc) verify_file = argv[++i];
        else if (!strcmp(argv[i], "--overwrite")) o.overwrite = true;
        else if (!strcmp(argv[i], "--jsonl")) g_jsonl = true;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { print_usage(); return 0; }
        else {
            fprintf(stderr, "ace-train: unknown option '%s'\n", argv[i]);
            print_usage();
            return 2;
        }
    }

    if (!verify_file.empty()) {
        return cmd_verify(verify_file);
    }

    // ── validate arguments ──────────────────────────────────────────────
    if (manifest_path.empty() && dataset_path.empty()) {
        fprintf(stderr, "ace-train preprocess: --manifest or --dataset is required\n");
        return 2;
    }
    if (o.out_dir.empty()) {
        fprintf(stderr, "ace-train preprocess: --out is required\n");
        return 2;
    }
    if (models_dir.empty()) {
        fprintf(stderr, "ace-train preprocess: --models is required\n");
        return 2;
    }
    if (dit_arg.empty() || vae_arg.empty() || text_arg.empty()) {
        fprintf(stderr, "ace-train preprocess: --dit, --vae and --text-enc are all required\n");
        return 2;
    }
    if (normalize != "none" && normalize != "peak") {
        fprintf(stderr, "ace-train preprocess: --normalize must be none|peak\n");
        return 2;
    }
    if (dtype != "f32" && dtype != "bf16") {
        fprintf(stderr, "ace-train preprocess: --dtype must be f32|bf16\n");
        return 2;
    }
    if (compat != "hotstep" && compat != "sidestep") {
        fprintf(stderr, "ace-train preprocess: --compat must be hotstep|sidestep\n");
        return 2;
    }
    if (timbre != "silence" && timbre != "zeros") {
        fprintf(stderr, "ace-train preprocess: --timbre must be silence|zeros\n");
        return 2;
    }
    if (o.max_duration < 0 || o.vae_chunk < 64 || o.vae_overlap < 0 || o.vae_overlap >= o.vae_chunk ||
        o.max_caption_tokens < 16 || o.max_lyric_tokens < 16) {
        fprintf(stderr, "ace-train preprocess: numeric option out of range\n");
        return 2;
    }

    o.normalize_peak  = (normalize == "peak");
    o.dtype           = (dtype == "bf16") ? STW_BF16 : STW_F32;
    o.sidestep_compat = (compat == "sidestep");
    o.timbre_zeros    = (timbre == "zeros");

    // ── resolve models ──────────────────────────────────────────────────
    ModelRegistry reg;
    registry_scan(&reg, models_dir.c_str());

    std::string dit_name, vae_name, text_name;
    if (!resolve_model(dit_arg, reg.dit, false, o.dit_path, dit_name)) {
        fprintf(stderr, "ace-train preprocess: cannot resolve --dit '%s' in %s\n", dit_arg.c_str(), models_dir.c_str());
        return 2;
    }
    if (!resolve_model(vae_arg, reg.vae, true, o.vae_path, vae_name)) {
        fprintf(stderr, "ace-train preprocess: cannot resolve --vae '%s' in %s (ONNX VAEs cannot encode)\n",
                vae_arg.c_str(), models_dir.c_str());
        return 2;
    }
    if (!resolve_model(text_arg, reg.text_enc, false, o.text_enc_path, text_name)) {
        fprintf(stderr, "ace-train preprocess: cannot resolve --text-enc '%s' in %s\n", text_arg.c_str(),
                models_dir.c_str());
        return 2;
    }
    o.model_variant = dit_name;

    // ── output dir ──────────────────────────────────────────────────────
    if (!pm_mkdir_p(o.out_dir)) {
        fprintf(stderr, "ace-train preprocess: cannot create output directory %s\n", o.out_dir.c_str());
        return 1;
    }
    pm_mkdir_p(o.out_dir + "/.tmp");
    purge_suffix(o.out_dir, ".__writing__");
    purge_suffix(o.out_dir, ".__old__");  // aside copies from an interrupted replace
    purge_all(o.out_dir + "/.tmp");

    // ── load the manifest ───────────────────────────────────────────────
    PManifest   mf;
    std::string err;
    bool        loaded = false;
    if (!manifest_path.empty()) {
        loaded = pm_load_manifest(manifest_path.c_str(), mf, err);  // --manifest wins if both given
    } else {
        loaded = pm_load_dataset_json(dataset_path.c_str(), mf, err);
    }
    if (!loaded) {
        fprintf(stderr, "ace-train preprocess: %s\n", err.c_str());
        return 1;
    }

    int total = (int) mf.samples.size();
    if (limit > 0 && limit < total) {
        total = limit;
    }

    ggml_time_init();
    const int64_t t_run0 = ggml_time_ms();

    jl("{\"type\":\"start\",\"total\":%d,\"out\":\"%s\",\"variant\":\"%s\",\"compat\":\"%s\",\"dtype\":\"%s\","
       "\"timbre\":\"%s\",\"maxDuration\":%d,\"normalize\":\"%s\"}",
       total, json_escape(o.out_dir).c_str(), json_escape(dit_name).c_str(), compat.c_str(), dtype.c_str(),
       timbre.c_str(), o.max_duration, normalize.c_str());
    fprintf(stderr, "[ace-train] preprocess: %d songs -> %s\n", total, o.out_dir.c_str());

    // ── load models (each stage timed) ──────────────────────────────────
    PreprocessCtx ctx;
    if (!pctx_load(ctx, o, err, jl_stage)) {
        fprintf(stderr, "ace-train preprocess: %s\n", err.c_str());
        jl("{\"type\":\"log\",\"level\":\"error\",\"message\":\"%s\"}", json_escape(err).c_str());
        pctx_free(ctx);
        return 1;
    }

    // ── driver loop ─────────────────────────────────────────────────────
    PreprocessMeta meta;
    meta.producer          = std::string("ace-train ") + ACE_VERSION;
    meta.created_at        = pm_iso8601_utc_now();
    meta.manifest          = manifest_path;
    meta.dataset_json      = manifest_path.empty() ? dataset_path : mf.dataset_json_path;
    meta.audio_dir         = mf.source_dir;
    meta.model_variant     = dit_name;
    meta.dit_path          = o.dit_path;
    meta.vae_path          = o.vae_path;
    meta.text_encoder_path = o.text_enc_path;
    meta.compat            = compat;
    meta.dtype             = (o.dtype == STW_F32) ? "F32" : "BF16";
    meta.timbre            = timbre;
    meta.normalize         = normalize;
    meta.target_db         = o.target_db;
    meta.max_duration      = o.max_duration;
    meta.max_caption_tokens = o.max_caption_tokens;
    meta.max_lyric_tokens   = o.max_lyric_tokens;
    meta.vae_chunk          = o.vae_chunk;
    meta.vae_overlap        = o.vae_overlap;
    meta.custom_tag         = mf.custom_tag;
    meta.tag_position       = mf.tag_position;
    meta.genre_ratio        = mf.genre_ratio;
    meta.total              = total;

    double    ch_sum[64] = { 0 }, ch_sqsum[64] = { 0 };
    long long n_frames  = 0;
    int       n_samples = 0;
    int       processed = 0, skipped = 0, failed = 0;

    for (int i = 0; i < total; i++) {
        const PSample & s     = mf.samples[(size_t) i];
        const std::string out = o.out_dir + "/" + s.out_stem + ".safetensors";

        PSampleMeta sm;
        sm.id           = s.id;
        sm.rel_path     = s.rel_path;
        sm.audio_path   = s.audio_path;
        sm.src_bytes    = s.src_bytes;
        sm.src_mtime_ms = s.src_mtime_ms;

        // Resume (P20): skip when the final file already exists — but ONLY when
        // it was produced with the settings this run is using. Skipping on
        // filename existence alone makes preprocess_meta.json (which the server
        // reports to the UI as provenance) claim the CURRENT compat / dtype /
        // timbre / max_duration for files whose own __metadata__ says otherwise;
        // a trainer that trusts the meta then trains on the wrong prompts.
        // A settings change re-encodes; an unchanged re-run is still instant.
        std::string resume_diff;
        if (!o.overwrite && pm_file_exists(out)) {
            PCacheSettings cs;
            if (pm_read_cached_settings(out.c_str(), cs)) {
                auto cmp_s = [&](const char * k, const std::string & had, const std::string & want) {
                    if (!had.empty() && had != want) {
                        if (!resume_diff.empty()) resume_diff += ", ";
                        resume_diff += std::string(k) + " " + had + "\xE2\x86\x92" + want;
                    }
                };
                auto cmp_i = [&](const char * k, int had, int want) {
                    if (had >= 0 && had != want) {
                        char b[96];
                        snprintf(b, sizeof(b), "%s %d\xE2\x86\x92%d", k, had, want);
                        if (!resume_diff.empty()) resume_diff += ", ";
                        resume_diff += b;
                    }
                };
                cmp_s("compat", cs.compat, compat);
                cmp_s("dtype", cs.dtype, (o.dtype == STW_F32) ? "F32" : "BF16");
                cmp_s("timbre", cs.timbre, timbre);
                cmp_s("normalize", cs.normalize, normalize);
                cmp_s("model_variant", cs.model_variant, dit_name);
                cmp_i("max_duration", cs.max_duration, o.max_duration);
                cmp_i("max_caption_tokens", cs.max_caption_tokens, o.max_caption_tokens);
                cmp_i("max_lyric_tokens", cs.max_lyric_tokens, o.max_lyric_tokens);
                cmp_i("vae_chunk", cs.vae_chunk, o.vae_chunk);
                cmp_i("vae_overlap", cs.vae_overlap, o.vae_overlap);
            }
            if (!resume_diff.empty()) {
                char wb[640];
                snprintf(wb, sizeof(wb), "%s: cache was built with different settings (%s) - re-encoding",
                         s.rel_path.c_str(), resume_diff.c_str());
                fprintf(stderr, "[ace-train] %s\n", wb);
                jl("{\"type\":\"log\",\"level\":\"warn\",\"message\":\"%s\"}", json_escape(wb).c_str());
            }
        }

        if (!o.overwrite && resume_diff.empty() && pm_file_exists(out)) {
            std::vector<float> cached;
            int                T = 0;
            if (pm_read_cached_latents(out.c_str(), cached, &T, &sm) && T > 0) {
                for (int t = 0; t < T; t++) {
                    const float * row = cached.data() + (size_t) t * 64;
                    for (int ch = 0; ch < 64; ch++) {
                        ch_sum[ch] += row[ch];
                        ch_sqsum[ch] += (double) row[ch] * (double) row[ch];
                    }
                }
                n_frames += T;
                n_samples++;
            }
            long long bytes = 0;
            pm_stat_file(out, &bytes, NULL);
            sm.file    = pm_basename(out);
            sm.bytes   = bytes;
            sm.ok      = true;
            sm.skipped = true;
            meta.samples.push_back(sm);
            skipped++;
            jl("{\"type\":\"song_skip\",\"index\":%d,\"id\":\"%s\",\"file\":\"%s\",\"reason\":\"exists\"}", i,
               json_escape(s.id).c_str(), json_escape(s.rel_path).c_str());
            jl("{\"type\":\"progress\",\"completed\":%d,\"total\":%d,\"processed\":%d,\"skipped\":%d,\"failed\":%d}",
               processed + skipped + failed, total, processed, skipped, failed);
            continue;
        }

        // A song that cannot be attempted is a SKIP, not a failure (§2.2). Left
        // as song_fail these count toward `failed`, and an all-FLAC dataset run
        // without --ffmpeg exits 1 — which the server surfaces as a hard job
        // failure with no hint that the cause is per-song and per-format.
        if (const char * skip_reason = pp_skip_reason(o, s)) {
            sm.file    = "";
            sm.ok      = false;
            sm.skipped = true;
            sm.error   = (strcmp(skip_reason, "missing-file") == 0)
                             ? ("audio file not found: " + s.audio_path)
                             : ("unsupported audio format " + pm_ext_of(s.audio_path) + "; ffmpeg not configured");
            meta.samples.push_back(sm);
            skipped++;
            fprintf(stderr, "[ace-train] SKIP %s: %s\n", s.rel_path.c_str(), sm.error.c_str());
            jl("{\"type\":\"song_skip\",\"index\":%d,\"id\":\"%s\",\"file\":\"%s\",\"reason\":\"%s\"}", i,
               json_escape(s.id).c_str(), json_escape(s.rel_path).c_str(), skip_reason);
            jl("{\"type\":\"log\",\"level\":\"warn\",\"message\":\"%s\"}", json_escape(sm.error).c_str());
            jl("{\"type\":\"progress\",\"completed\":%d,\"total\":%d,\"processed\":%d,\"skipped\":%d,\"failed\":%d}",
               processed + skipped + failed, total, processed, skipped, failed);
            continue;
        }

        jl("{\"type\":\"song_start\",\"index\":%d,\"id\":\"%s\",\"file\":\"%s\"}", i, json_escape(s.id).c_str(),
           json_escape(s.rel_path).c_str());
        fprintf(stderr, "[ace-train] [%d/%d] %s\n", i + 1, total, s.rel_path.c_str());

        const int64_t t0 = ggml_time_ms();
        SongResult    r  = preprocess_song(ctx, o, mf, s, out, jl_warn);
        const int64_t ms = ggml_time_ms() - t0;

        if (r.ok) {
            for (int t = 0; t < r.t_latent; t++) {
                const float * row = r.latents.data() + (size_t) t * 64;
                for (int ch = 0; ch < 64; ch++) {
                    ch_sum[ch] += row[ch];
                    ch_sqsum[ch] += (double) row[ch] * (double) row[ch];
                }
            }
            n_frames += r.t_latent;
            n_samples++;
            processed++;

            sm.file          = pm_basename(out);
            sm.bytes         = r.bytes;
            sm.t_latent      = r.t_latent;
            sm.s_total       = r.s_total;
            sm.s_text        = r.s_text;
            sm.s_lyric       = r.s_lyric;
            sm.s_total_genre = r.s_total_genre;
            sm.has_genre     = r.has_genre;
            sm.ok            = true;
            meta.samples.push_back(sm);

            jl("{\"type\":\"song_done\",\"index\":%d,\"id\":\"%s\",\"file\":\"%s\",\"out\":\"%s\",\"tLatent\":%d,"
               "\"sTotal\":%d,\"sText\":%d,\"sLyric\":%d,\"sTotalGenre\":%d,\"hasGenre\":%s,\"bytes\":%lld,"
               "\"ms\":%lld}",
               i, json_escape(s.id).c_str(), json_escape(s.rel_path).c_str(), json_escape(sm.file).c_str(), r.t_latent,
               r.s_total, r.s_text, r.s_lyric, r.s_total_genre, r.has_genre ? "true" : "false", (long long) r.bytes,
               (long long) ms);
        } else {
            failed++;
            sm.ok    = false;
            sm.error = r.error;
            meta.samples.push_back(sm);
            fprintf(stderr, "[ace-train] FAILED %s: %s\n", s.rel_path.c_str(), r.error.c_str());
            jl("{\"type\":\"song_fail\",\"index\":%d,\"id\":\"%s\",\"file\":\"%s\",\"error\":\"%s\"}", i,
               json_escape(s.id).c_str(), json_escape(s.rel_path).c_str(), json_escape(r.error).c_str());
        }

        jl("{\"type\":\"progress\",\"completed\":%d,\"total\":%d,\"processed\":%d,\"skipped\":%d,\"failed\":%d}",
           processed + skipped + failed, total, processed, skipped, failed);
    }

    // Free the GPU as early as possible — before the stats/meta writes.
    pctx_free(ctx);

    meta.processed = processed;
    meta.skipped   = skipped;
    meta.failed    = failed;

    if (n_frames > 0) {
        const std::string stats_path = o.out_dir + "/channel_stats.json";
        if (pm_write_channel_stats(stats_path.c_str(), ch_sum, ch_sqsum, n_frames, n_samples)) {
            jl("{\"type\":\"stats\",\"nSamples\":%d,\"nFrames\":%lld}", n_samples, (long long) n_frames);
        } else {
            fprintf(stderr, "[ace-train] WARNING: failed to write %s\n", stats_path.c_str());
        }
    }

    const std::string meta_path = o.out_dir + "/preprocess_meta.json";
    if (!pm_write_meta_json(meta_path.c_str(), meta)) {
        fprintf(stderr, "[ace-train] FATAL: failed to write %s\n", meta_path.c_str());
        return 1;
    }

    const long long run_ms = (long long) (ggml_time_ms() - t_run0);
    jl("{\"type\":\"done\",\"processed\":%d,\"skipped\":%d,\"failed\":%d,\"out\":\"%s\",\"ms\":%lld}", processed,
       skipped, failed, json_escape(o.out_dir).c_str(), run_ms);
    fprintf(stderr, "[ace-train] done: %d processed, %d skipped, %d failed (%.1f s)\n", processed, skipped, failed,
            (double) run_ms / 1000.0);

    if (!g_jsonl) {
        printf("preprocess: %d processed, %d skipped, %d failed -> %s\n", processed, skipped, failed, o.out_dir.c_str());
        fflush(stdout);
    }

    const int attempted = processed + failed;
    return (processed == 0 && attempted > 0) ? 1 : 0;
}

// ─── train-lm ───────────────────────────────────────────────────────────────
//
// docs/plans/2026-07-27-lm-trainer-implementation.md §2.1, §3.1

static int cmd_train_lm(int argc, char ** argv) {
    LmTrainArgs a;
    std::string stages_csv, dit_arg, lm_arg, lm_size_arg;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--stages") && i + 1 < argc) stages_csv = argv[++i];
        else if (!strcmp(argv[i], "--tensors") && i + 1 < argc) a.tensors_dir = argv[++i];
        else if (!strcmp(argv[i], "--codes") && i + 1 < argc) a.codes_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) a.out_dir = argv[++i];
        else if (!strcmp(argv[i], "--models") && i + 1 < argc) a.models_dir = argv[++i];
        else if (!strcmp(argv[i], "--dit") && i + 1 < argc) dit_arg = argv[++i];
        else if (!strcmp(argv[i], "--lm") && i + 1 < argc) lm_arg = argv[++i];
        else if (!strcmp(argv[i], "--lm-size") && i + 1 < argc) lm_size_arg = argv[++i];
        else if (!strcmp(argv[i], "--rank") && i + 1 < argc) a.rank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--alpha") && i + 1 < argc) a.alpha = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i + 1 < argc) a.lr = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--epochs") && i + 1 < argc) a.epochs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--grad-accum") && i + 1 < argc) a.grad_accum = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup-ratio") && i + 1 < argc) a.warmup_ratio = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--grad-clip") && i + 1 < argc) a.grad_clip = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--weight-decay") && i + 1 < argc) a.weight_decay = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) a.seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--target-loss") && i + 1 < argc) a.target_loss = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--order") && i + 1 < argc) a.order = argv[++i];
        else if (!strcmp(argv[i], "--max-len") && i + 1 < argc) a.max_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vram-reserve-mb") && i + 1 < argc) a.vram_reserve_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--low-vram") && i + 1 < argc) a.low_vram = argv[++i];
        else if (!strcmp(argv[i], "--attn-head-block") && i + 1 < argc) a.attn_head_block = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lm-chunk") && i + 1 < argc) a.chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--milestone-step") && i + 1 < argc) a.milestone_step = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--milestone-keep") && i + 1 < argc) a.milestone_keep = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) a.limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--loss-on-cot")) a.loss_on_cot = true;
        else if (!strcmp(argv[i], "--no-loss-on-cot")) a.loss_on_cot = false;
        else if (!strcmp(argv[i], "--no-milestones")) a.milestone_step = 0.0f;
        else if (!strcmp(argv[i], "--overwrite")) a.overwrite = true;
        else if (!strcmp(argv[i], "--self-test")) a.self_test = true;
        else if (!strcmp(argv[i], "--jsonl")) g_jsonl = true;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { print_usage(); return 0; }
        else {
            fprintf(stderr, "ace-train train-lm: unknown option '%s'\n", argv[i]);
            print_usage();
            return 2;
        }
    }

    // ── stages ──────────────────────────────────────────────────────────
    if (!stages_csv.empty()) {
        const bool want_ex = stages_csv.find("extract") != std::string::npos;
        const bool want_tr = stages_csv.find("train") != std::string::npos;
        const bool want_xp = stages_csv.find("export") != std::string::npos;
        a.stages.clear();
        if (want_ex) a.stages.push_back("extract");
        if (want_tr) a.stages.push_back("train");
        if (want_xp) a.stages.push_back("export");
        if (a.stages.empty()) {
            fprintf(stderr, "ace-train train-lm: --stages must name at least one of extract,train,export\n");
            return 2;
        }
    }

    // ── numeric sanity (the server clamps these too; be defensive) ───────
    if (a.rank < 1 || a.rank > 256 || a.alpha < 1 || a.alpha > 1024 || a.epochs < 1 || a.epochs > 200 ||
        a.grad_accum < 1 || a.grad_accum > 64 || a.lr <= 0.0f || a.lr > 1.0f || a.grad_clip < 0.0f ||
        a.warmup_ratio < 0.0f || a.warmup_ratio > 0.5f || a.weight_decay < 0.0f || a.weight_decay > 1.0f ||
        a.target_loss < 0.0f || a.target_loss > 20.0f || a.milestone_keep < 0 || a.milestone_keep > 64 ||
        a.vram_reserve_mb < 0 || (a.max_len != 0 && (a.max_len < 512 || a.max_len > 16384))) {
        fprintf(stderr, "ace-train train-lm: numeric option out of range\n");
        return 2;
    }
    if (a.order != "shuffle" && a.order != "fixed") {
        fprintf(stderr, "ace-train train-lm: --order must be shuffle|fixed\n");
        return 2;
    }
    if (a.low_vram != "auto" && a.low_vram != "on" && a.low_vram != "off") {
        fprintf(stderr, "ace-train train-lm: --low-vram must be auto|on|off\n");
        return 2;
    }
    if (a.attn_head_block < -1 || a.attn_head_block > 128 || a.chunk < 16 || a.chunk > 1024) {
        fprintf(stderr, "ace-train train-lm: --attn-head-block must be -1..128 and --lm-chunk 16..1024\n");
        return 2;
    }

    // ── model resolution (same rule as preprocess) ──────────────────────
    ModelRegistry reg;
    if (!a.models_dir.empty()) {
        registry_scan(&reg, a.models_dir.c_str());
    }
    if (!lm_arg.empty()) {
        std::string name;
        if (!resolve_model(lm_arg, reg.lm, false, a.lm_path, name)) {
            fprintf(stderr, "ace-train train-lm: cannot resolve --lm '%s' in %s\n", lm_arg.c_str(),
                    a.models_dir.c_str());
            return 2;
        }
        a.lm_name = name;
    }
    if (!dit_arg.empty()) {
        std::string name;
        if (!resolve_model(dit_arg, reg.dit, false, a.dit_path, name)) {
            fprintf(stderr, "ace-train train-lm: cannot resolve --dit '%s' in %s\n", dit_arg.c_str(),
                    a.models_dir.c_str());
            return 2;
        }
    }

    // ── required arguments per stage ────────────────────────────────────
    if (a.self_test) {
        if (a.lm_path.empty()) {
            fprintf(stderr, "ace-train train-lm --self-test: --lm is required\n");
            return 2;
        }
    } else {
        if (lm_has_stage(a, "extract") && a.tensors_dir.empty()) {
            fprintf(stderr, "ace-train train-lm: --tensors is required for the extract stage\n");
            return 2;
        }
        if (lm_has_stage(a, "train") && a.lm_path.empty()) {
            fprintf(stderr, "ace-train train-lm: --lm is required for the train stage\n");
            return 2;
        }
        if ((lm_has_stage(a, "train") || lm_has_stage(a, "export")) && a.out_dir.empty()) {
            fprintf(stderr, "ace-train train-lm: --out is required for the train/export stages\n");
            return 2;
        }
    }
    if (a.codes_path.empty()) {
        if (a.tensors_dir.empty()) {
            if (!a.self_test) {
                fprintf(stderr, "ace-train train-lm: --codes or --tensors is required\n");
                return 2;
            }
        } else {
            a.codes_path = lm_join(a.tensors_dir, "lm_codes.jsonl");
        }
    }

    // ── lm_size label ───────────────────────────────────────────────────
    if (!lm_size_arg.empty()) {
        a.lm_size = lm_size_arg;
    } else if (!a.lm_path.empty()) {
        const Qwen3LMConfig c = qw3lm_load_config(a.lm_path.c_str(), !ends_with_gguf(a.lm_path.c_str()));
        a.lm_size             = lm_size_label_from_config(c);
    }

    // MANDATORY: ggml_time_ms() divides by an uninitialised frequency otherwise.
    ggml_time_init();
    return lm_train_main(a);
}

// ─── train-dit (plan §2.1) ──────────────────────────────────────────────────

static int cmd_train_dit(int argc, char ** argv) {
    DitTrainArgs a;
    std::string  stages_csv, dit_arg;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--stages") && i + 1 < argc) stages_csv = argv[++i];
        else if (!strcmp(argv[i], "--tensors") && i + 1 < argc) a.tensors_dir = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) a.out_dir = argv[++i];
        else if (!strcmp(argv[i], "--models") && i + 1 < argc) a.models_dir = argv[++i];
        else if (!strcmp(argv[i], "--dit") && i + 1 < argc) dit_arg = argv[++i];
        else if (!strcmp(argv[i], "--adapter-type") && i + 1 < argc) a.adapter_type = argv[++i];
        else if (!strcmp(argv[i], "--rank") && i + 1 < argc) a.rank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--alpha") && i + 1 < argc) a.alpha = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) a.layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--target-mlp")) a.target_mlp = true;
        else if (!strcmp(argv[i], "--loss-weighting") && i + 1 < argc) a.loss_weighting = argv[++i];
        else if (!strcmp(argv[i], "--snr-gamma") && i + 1 < argc) a.snr_gamma = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--t-bias") && i + 1 < argc) a.t_bias = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--channel-balance")) a.channel_balance = true;
        else if (!strcmp(argv[i], "--no-channel-balance")) a.channel_balance = false;
        else if (!strcmp(argv[i], "--timestep-mu") && i + 1 < argc) a.timestep_mu = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--timestep-sigma") && i + 1 < argc) a.timestep_sigma = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--t-min") && i + 1 < argc) a.t_min = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--t-max") && i + 1 < argc) a.t_max = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--cfg-ratio") && i + 1 < argc) a.cfg_ratio = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--genre-ratio") && i + 1 < argc) a.genre_ratio = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i + 1 < argc) a.lr = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--epochs") && i + 1 < argc) a.epochs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--grad-accum") && i + 1 < argc) a.grad_accum = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup-ratio") && i + 1 < argc) a.warmup_ratio = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--grad-clip") && i + 1 < argc) a.grad_clip = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--weight-decay") && i + 1 < argc) a.weight_decay = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) a.seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--target-loss") && i + 1 < argc) a.target_loss = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--order") && i + 1 < argc) a.order = argv[++i];
        else if (!strcmp(argv[i], "--crop") && i + 1 < argc) a.crop = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--crop-min") && i + 1 < argc) a.crop_min = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--crop-max") && i + 1 < argc) a.crop_max = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vram-reserve-mb") && i + 1 < argc) a.vram_reserve_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vram-safety") && i + 1 < argc) a.vram_safety = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--milestone-step") && i + 1 < argc) a.milestone_step = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--milestone-keep") && i + 1 < argc) a.milestone_keep = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-milestones")) a.milestone_step = 0.0f;
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) a.limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--overwrite")) a.overwrite = true;
        else if (!strcmp(argv[i], "--self-test")) a.self_test = true;
        else if (!strcmp(argv[i], "--jsonl")) g_jsonl = true;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { print_usage(); return 0; }
        else {
            fprintf(stderr, "ace-train train-dit: unknown option '%s'\n", argv[i]);
            print_usage();
            return 2;
        }
    }

    // ── stages ──────────────────────────────────────────────────────────
    if (!stages_csv.empty()) {
        const bool want_tr = stages_csv.find("train") != std::string::npos;
        const bool want_xp = stages_csv.find("export") != std::string::npos;
        a.stages.clear();
        if (want_tr) a.stages.push_back("train");
        if (want_xp) a.stages.push_back("export");
        if (a.stages.empty()) {
            fprintf(stderr, "ace-train train-dit: --stages must name at least one of train,export\n");
            return 2;
        }
    }

    // ── numeric sanity (the server clamps these too; be defensive) ───────
    if (a.rank < 1 || a.rank > 256 || a.alpha < 1 || a.alpha > 1024 || a.epochs < 1 || a.epochs > 2000 ||
        a.grad_accum < 1 || a.grad_accum > 64 || a.lr <= 0.0f || a.lr > 1.0f || a.grad_clip < 0.0f ||
        a.grad_clip > 100.0f || a.warmup_ratio < 0.0f || a.warmup_ratio > 0.5f || a.weight_decay < 0.0f ||
        a.weight_decay > 1.0f || a.target_loss < 0.0f || a.target_loss > 20.0f || a.milestone_keep < 0 ||
        a.milestone_keep > 64 || a.milestone_step < 0.0f || a.milestone_step > 5.0f || a.vram_reserve_mb < 0 ||
        a.vram_reserve_mb > 16384 || a.vram_safety < 0.0f || a.vram_safety >= 1.0f || a.layers < 0 || a.layers > 64 ||
        (a.crop != 0 && (a.crop < 128 || a.crop > 8192)) || a.crop_min < 128 || a.crop_min > 8192 ||
        a.crop_max < a.crop_min || a.crop_max > 8192 || a.snr_gamma < 1.0f || a.snr_gamma > 100.0f ||
        a.t_bias < 0.0f || a.t_bias > 4.0f || a.timestep_mu < -4.0f || a.timestep_mu > 4.0f ||
        a.timestep_sigma <= 0.0f || a.timestep_sigma > 4.0f || a.t_min < 0.0f || a.t_max > 1.0f ||
        a.t_min >= a.t_max || a.cfg_ratio < 0.0f || a.cfg_ratio > 1.0f || a.genre_ratio < 0 || a.genre_ratio > 100) {
        fprintf(stderr, "ace-train train-dit: numeric option out of range\n");
        return 2;
    }
    if (a.order != "shuffle" && a.order != "fixed") {
        fprintf(stderr, "ace-train train-dit: --order must be shuffle|fixed\n");
        return 2;
    }
    if (a.loss_weighting != "none" && a.loss_weighting != "flow_snr") {
        fprintf(stderr, "ace-train train-dit: --loss-weighting must be none|flow_snr\n");
        return 2;
    }

    // ── required arguments ──────────────────────────────────────────────
    if (a.tensors_dir.empty()) {
        fprintf(stderr, "ace-train train-dit: --tensors is required\n");
        return 2;
    }
    if (!a.self_test && a.out_dir.empty()) {
        fprintf(stderr, "ace-train train-dit: --out is required\n");
        return 2;
    }

    // ── model resolution (same rule as preprocess/train-lm) ─────────────
    // The DiT used for training MUST be the one the latents were preprocessed
    // against — the encoder states and context latents are that model's outputs.
    if (dit_arg.empty()) {
        dit_arg = dit_meta_dit_path(a.tensors_dir.c_str());
        if (dit_arg.empty()) {
            fprintf(stderr, "ace-train train-dit: no --dit and no dit_path in %s/preprocess_meta.json\n",
                    a.tensors_dir.c_str());
            return 2;
        }
    }
    {
        ModelRegistry reg;
        if (!a.models_dir.empty()) {
            registry_scan(&reg, a.models_dir.c_str());
        }
        std::string name;
        if (!resolve_model(dit_arg, reg.dit, false, a.dit_path, name)) {
            fprintf(stderr, "ace-train train-dit: cannot resolve --dit '%s' in %s\n", dit_arg.c_str(),
                    a.models_dir.c_str());
            return 2;
        }
        a.dit_name = name;
    }

    // MANDATORY: ggml_time_ms() divides by an uninitialised frequency otherwise.
    ggml_time_init();
    return dit_train_main(a);
}

// ─── main ───────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage();
        return 2;
    }
    if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        print_usage();
        return 0;
    }
    if (!strcmp(argv[1], "preprocess")) {
        return cmd_preprocess(argc - 1, argv + 1);
    }
    if (!strcmp(argv[1], "train-lm")) {
        return cmd_train_lm(argc - 1, argv + 1);
    }
    if (!strcmp(argv[1], "train-dit")) {
        return cmd_train_dit(argc - 1, argv + 1);
    }
    if (!strcmp(argv[1], "spike")) {
        return cmd_spike(argc - 1, argv + 1);
    }
    fprintf(stderr, "ace-train: unknown subcommand '%s'\n", argv[1]);
    print_usage();
    return 2;
}
