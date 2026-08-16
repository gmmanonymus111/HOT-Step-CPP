// ace-train.cpp — HOT-Step training toolchain.
// Phase 2 ships one subcommand: `preprocess`.
//
// Standalone tool (no acestep-core): every engine module it uses is a
// header-only `static` API. Machine-readable JSONL on stdout (--jsonl),
// human logs on stderr. Exit: 0 ok, 1 runtime failure, 2 usage.
//
// docs/plans/2026-07-27-preprocess-implementation.md §3.1

// MM3 training path. mm3-dav-encode.h is training-only (audio -> target
// latents); the rest are the same headers ace-server uses, included here for the
// conditioning rollout (mm3-condition). All header-only static APIs.
#include "minimax/mm3-dav-encode.h"
#include "minimax/mm3-tokenizer.h"
#include "minimax/mm3-request.h"
#include "minimax/mm3-ar-loop.h"
#include "minimax/mm3-cond-graph.h"
#include "train/mm3-dit-train-run.h"   // MM3 flow-DiT LoRA trainer
#include "model-registry.h"
#include "train/dit-train-run.h"   // pulls in every dit-*.h (DiT LoRA trainer)
#include "train/lm-train-run.h"    // pulls in every lm-*.h (LM LoRA trainer)
#include "train/preprocess-run.h"  // pulls in st-write.h + preprocess-io.h
#include "train/spike-run.h"       // Phase-0 training spike (evidence code)

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
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
            "  mm3-encode    MiniMax-Music3 DAV encode: raw f32 audio -> flow latents.\n"
            "                --enc <mm3-enc-*.gguf> --audio <in.f32> [--channels 2]\n"
            "                [--out <out.f32>] [--ref <latents.f32> : parity-gate against it]\n"
            "                [--tf32 on|off]  default off — TF32 costs ~5e-3 on the targets\n"
            "  mm3-preprocess  MiniMax-Music3 dataset -> flow-DiT target latents.\n"
            "                --dataset <dataset.json> --enc <mm3-enc-*.gguf> --out <dir>\n"
            "                [--ffmpeg <path>] [--max-duration <sec>] [--tf32 on|off]\n"
            "                Audio is transcoded at 44100 Hz (NOT the ACE path's 48000).\n"
            "  mm3-condition MiniMax-Music3 AR rollout -> flow-DiT conditioning cache.\n"
            "                --manifest <mm3_preprocess.json> --models <dir> --captions <dataset dir>\n"
            "                [--seed N] [--lm-quant Q] [--dit-quant Q; default Q2_K, never executed]\n"
            "                [--segment-sec S] default 60; 0 = one rollout per song, which\n"
            "                UNDER-COVERS long songs (the LM hits EOS early: 65%% on a 280 s track)\n"
            "                [--codes <dir>] ALIGNED mode: teacher-force <id>.codes (int32\n"
            "                [n_iter, 8], row 0 = un-emitted warm-up) from the RVQ encoder\n"
            "                instead of sampling from the caption. One replay per song, so\n"
            "                no segments and no seams; --segment-sec is then ignored.\n"
            "                [--codes-mode full|semantic] default full. `semantic` pins only\n"
            "                the semantic code (content/structure) and SAMPLES the 7 acoustic\n"
            "                codebooks, leaving timbre unspecified so a style adapter must\n"
            "                learn it. Full alignment describes the target so completely that\n"
            "                the adapter learns nothing (measured: identical to base).\n"
            "  mm3-train-dit MiniMax-Music3 flow-DiT LoRA training.\n"
            "                --cache <dir from mm3-condition> --models <dir> --out <dir>\n"
            "                [--rank 32] [--alpha 32] [--lr 1e-4] [--steps 200] [--crop 689]\n"
            "                [--grad-accum 1] [--seed 42]\n"
            "                [--song <substring>] overfit one song. Matches a SUBSTRING OF THE\n"
            "                FILENAME, not the cache id -- passing an id gives the unhelpful\n"
            "                \"no usable songs in the cache\".\n"
            "                [--logit-mean M] default 0; POSITIVE biases sigma toward 0, i.e.\n"
            "                toward mostly-clean crops. Raise it (~+1.0..+1.5) or the run\n"
            "                spends most steps near pure noise learning only the genre mean.\n"
            "                [--logit-std S] default 1\n"
            "                [--eval-every N] fixed holdout, sigma on a stratified grid, forward\n"
            "                only. Comparable ACROSS runs; training loss is not, because\n"
            "                --logit-mean changes which sigmas it is measured at. Reports\n"
            "                three sigma bands — a fix that only moves the high band has\n"
            "                learned the genre marginal again.  [--eval-n K] default 24\n"
            "                [--eval-crop F] default 689; PINNED independently of --crop so a\n"
            "                run at a different crop is still comparable. 0 = follow --crop.\n"
            "                [--crop-mode random|beginning] default random, which covers the\n"
            "                whole song. `beginning` always starts at 0 — what SimpleTuner's\n"
            "                truncation_mode does, so it only ever sees each track's head.\n"
            "                [--ckpt-segments N] gradient checkpointing over the block stack.\n"
            "                Peak attention is n_blk*n_heads*S^2*4 B, so crop 2584 (30 s) needs\n"
            "                ~30.8 GB monolithic and is impossible without this; 6 segments\n"
            "                make it ~5.1 GB for ~1 extra forward. ggml's flash-attn has no\n"
            "                backward, so this is the ONLY lever.\n"
            "                [--ckpt-verify] run both paths on one micro-batch and compare.\n"
            "                Gate before trusting any checkpointed run; exits non-zero on\n"
            "                disagreement. With no --ckpt-segments it verifies and exits.\n"
            "                [--export-every N] snapshot mm3_lora_step<N>.safetensors along the\n"
            "                way. For the delta ladder: measure each with relnorm, ear-test the\n"
            "                1-5%% band (0.074%% proven inaudible, 17%% proven destroyed).\n"
            "                [--target all|mlpv] mlpv trains ONLY the timbre groups: MLPs +\n"
            "                attention V + to_out. No q,k (ablation-proven structure poison),\n"
            "                no proj heads (seed-dependent fuzz). Concentrates the whole delta\n"
            "                budget in what survives, instead of filtering at export.\n"
            "                [--sign-check] measure the velocity target's sign, train nothing\n"
            "                [--bwd outprod] restore the slow CPU mul_mat backward (510x slower)\n"
            "                [--tf32 on|off] default off\n"
            "                Writes <out>/mm3_lora.safetensors; load with MM3_ADAPTER=<path>.\n"
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
            "  Resume:\n"
            "    --init-adapter <dir>        continue training from an exported run (PEFT LoRA or\n"
            "                                LoKr dir). Identity hyperparams (adapter type, rank/\n"
            "                                alpha, lokr dim/alpha/factor, --weights, base size)\n"
            "                                are ADOPTED from its lm_train_log.json; an explicit\n"
            "                                contradicting flag is refused. Fresh warmup+cosine;\n"
            "                                optimizer momentum starts cold. Epoch 1 should land\n"
            "                                near the source's saved_loss (logged as a check).\n"
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
            "    --weights <f32-window|bf16> f32-window  projection GEMM dtype.\n"
            "                                            f32-window = the shipped per-segment F32 weight\n"
            "                                                         cast (cublasSgemm/TF32).\n"
            "                                            bf16       = BF16 projections + backward surgery\n"
            "                                                         (cublasGemmEx BF16). Needs a BF16\n"
            "                                                         base and low-VRAM mode, and CHANGES\n"
            "                                                         THE TRAINED WEIGHTS (the activation\n"
            "                                                         gradient is BF16-rounded at every\n"
            "                                                         layer). Not resume-compatible with\n"
            "                                                         an f32-window run.\n"
            "                                            EXPERIMENTAL, NOT ACCEPTED: its own parity gates\n"
            "                                            (--self-test T14/T15) measure OVER their bars —\n"
            "                                            max rel ~1.2e-1 vs a 2e-2 bar, cosine ~0.9995 vs\n"
            "                                            1-1e-4. Direction is close but not within spec;\n"
            "                                            judge a bf16 adapter by ear before trusting it.\n"
            "    --bwd <outprod|mm>          outprod     MUL_MAT activation-gradient formulation.\n"
            "                                            outprod = upstream ggml out_prod (F32-only on\n"
            "                                                      CUDA; forces an F32 weight, so the\n"
            "                                                      forward runs TF32).\n"
            "                                            mm      = mul_mat(cont(transpose(W)), grad) —\n"
            "                                                      identical maths, dtype-agnostic, so a\n"
            "                                                      BF16 weight uses BF16 tensor cores.\n"
            "                                                      ~1.7-1.8x per layer per step on an\n"
            "                                                      RTX 5090. Needs the vendored patch\n"
            "                                                      engine/patches/mm-backward.patch.\n"
            "    --optimizer <adamw|muon>    adamw       muon puts every 2-D parameter whose SHORT side is\n"
            "                                            >= --muon-min-dim on orthogonalized-momentum\n"
            "                                            (Newton-Schulz) updates. FOR A LoRA THE SHORT\n"
            "                                            SIDE IS THE RANK, so at rank 16 (the default)\n"
            "                                            every adapter matrix qualifies and at rank 8\n"
            "                                            none do — the startup line reports the split.\n"
            "                                            On the DiT this measured 1.41x fewer epochs to\n"
            "                                            target; it is untested on the LM.\n"
            "    --muon-lr-scale <f>         1.0         multiplies the shared LR schedule for Muon\n"
            "                                            params only. Muon's update is normalized, so its\n"
            "                                            LR does NOT mean AdamW's — the DiT needed ~20.\n"
            "    --muon-momentum <f>         0.95\n"
            "    --muon-ns-steps <n>         5           Newton-Schulz iterations\n"
            "    --muon-min-dim <n>          16          short-side floor for Muon eligibility\n"
            "    --muon-bucket <n>           16          params per batched Newton-Schulz (measured\n"
            "                                            optimum 8-16; larger is worse, the gather is\n"
            "                                            quadratic in bucket size)\n"
            "    --no-muon-nesterov                      use the plain momentum buffer, not g + mu*m\n"
            "    --batch <n|auto>            1           micro-batch size. NOT SUPPORTED in this build:\n"
            "                                            micro-batching was never written — the host\n"
            "                                            overhead it would amortise measures 9.3%% at 4B,\n"
            "                                            under the 10%% bar its build gate required.\n"
            "                                            Use --grad-accum to change effective batch size.\n"
            "    --loss-on-cot                           default ON\n"
            "    --no-loss-on-cot\n"
            "\n"
            "  Adapter identity:\n"
            "    --trigger <word>            \"\"          trigger word embedded in the adapter's\n"
            "                                            metadata. Default: custom_tag from the\n"
            "                                            variant's preprocess_meta.json.\n"
            "    --trigger-position <prepend|append>     where the trigger sat in the training\n"
            "                                            captions. Default: that file's tag_position.\n"
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
            "    --adapter-type <lora|lokr>  lora\n"
            "    --rank <n>                  128             lora only\n"
            "    --alpha <n>                 256             lora only\n"
            "    --lokr-dim <n>              512             lokr only, 4-4096\n"
            "    --lokr-alpha <f>            512             lokr only, 0-8192; 0 = dim. LyCORIS\n"
            "                                                forces alpha = dim (scale 1) wherever\n"
            "                                                both kron factors are monolithic.\n"
            "    --lokr-factor <n>           6               lokr only, -1 or 2-64\n"
            "    --lokr-decompose-both / --no-lokr-decompose-both   default ON (parity knob;\n"
            "                                                inert at dim 512 — w1 is always monolithic)\n"
            "    --target-mlp / --no-target-mlp   default ON  also LoRA mlp gate/up/down\n"
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
            "    --genre-ratio <n>           30              percent, 0-100\n"
            "\n"
            "  Optimizer / schedule:\n"
            "    --lr <f>                    5e-4\n"
            "    --epochs <n>                400         hard cap; target-loss usually stops earlier\n"
            "    --batch <n>                 1           1-16; crops per micro-batch, from that many\n"
            "                                            DIFFERENT songs. Reduced with a warn when the\n"
            "                                            variant has fewer songs, or when\n"
            "                                            n_kv_heads*max(S,enc_S)*B would exceed ggml's\n"
            "                                            CUDA repeat_back cap. Default 1 (off): measured\n"
            "                                            ~2.5x SLOWER at full depth on a 32 GB card, but\n"
            "                                            ~2.4x FASTER on shallow/partial-depth runs.\n"
            "    --grad-accum <n>            4           counts MICRO-BATCHES, so an optimizer step sees\n"
            "                                            batch x grad-accum songs (4 at the defaults).\n"
            "    --ckpt <n>                  1           gradient checkpointing: 0 = off (one monolithic\n"
            "                                            fwd+bwd graph), 1 = auto (the VRAM fit picks the\n"
            "                                            segment count), 2-32 = exactly that many segments.\n"
            "                                            Segments trade one extra no-grad forward (~+30-40%%\n"
            "                                            of forward time) for an activation set divided by\n"
            "                                            the segment count. Clamped to the trained depth.\n"
            "    --warmup-ratio <f>          0.05\n"
            "    --grad-clip <f>             1.0         0 = disabled\n"
            "    --weight-decay <f>          0.01\n"
            "    --seed <n>                  42\n"
            "    --target-loss <f>           0.4         0 = disabled (run to the epoch cap)\n"
            "    --order <shuffle|fixed>     shuffle\n"
            "\n"
            "  Resume:\n"
            "    --init-adapter <dir>        continue training from an exported run (PEFT LoRA or\n"
            "                                LoKr dir). Identity hyperparams (adapter type, rank/\n"
            "                                alpha, lokr dims, target-mlp, layers, base model) are\n"
            "                                ADOPTED from its dit_train_log.json; an explicit\n"
            "                                contradicting flag is refused. Layer-window coverage is\n"
            "                                tolerant: uncovered sites start at zero delta. mirror\n"
            "                                and --bwd may differ (measured ~7e-5 loss drift).\n"
            "\n"
            "  Crop / memory:\n"
            "    --crop <n>                  0           latent frames; 0 = auto-fit\n"
            "    --crop-min <n>              375\n"
            "    --crop-max <n>              1250\n"
            "    --vram-reserve-mb <n>       2048        desktop/OS headroom left unallocated\n"
            "    --vram-safety <f>           0.05        extra margin on the footprint model\n"
            "                                            (0.12 for --adapter-type lokr unless set)\n"
            "    --mirror <f32|bf16>         f32         frozen-weight mirror precision. bf16 keeps\n"
            "                                            the trainable layers' matmul weights in the\n"
            "                                            base's native BF16 instead of promoting them\n"
            "                                            to F32, roughly halving the mirror. CUDA only\n"
            "                                            (engine/patches/bf16-out-prod.patch);\n"
            "                                            EXPERIMENTAL, adapter quality unvalidated.\n"
            "    --bwd <outprod|mm>          outprod     MUL_MAT activation-gradient formulation.\n"
            "                                            outprod = upstream ggml out_prod (F32-only on\n"
            "                                                      CUDA; forces an F32 weight, so the\n"
            "                                                      forward runs TF32).\n"
            "                                            mm      = mul_mat(cont(transpose(W)), grad) —\n"
            "                                                      identical maths, dtype-agnostic, so a\n"
            "                                                      BF16 mirror uses BF16 tensor cores.\n"
            "                                                      ~1.7-1.8x per layer per step on an\n"
            "                                                      RTX 5090. Needs the vendored patch\n"
            "                                                      engine/patches/mm-backward.patch.\n"
            "\n"
            "  Optimizer:\n"
            "    --optimizer <adamw|muon>    adamw       muon puts every 2-D parameter whose SHORT side is\n"
            "                                            >= --muon-min-dim on orthogonalized-momentum\n"
            "                                            (Newton-Schulz) updates, and leaves the rest on\n"
            "                                            AdamW — a LoKR w1 is [4,5], where orthogonalizing\n"
            "                                            means nothing. Muon needs only ONE momentum\n"
            "                                            buffer where AdamW needs two (~900 MB at dim512).\n"
            "    --muon-lr-scale <f>         1.0         multiplies the shared LR schedule, for Muon\n"
            "                                            parameters only. Muon's update is normalized by\n"
            "                                            construction, so its LR does NOT mean AdamW's —\n"
            "                                            expect to retune before comparing.\n"
            "    --muon-momentum <f>         0.95\n"
            "    --muon-ns-steps <n>         5           Newton-Schulz iterations\n"
            "    --muon-min-dim <n>          16          short-side floor for Muon eligibility\n"
            "    --no-muon-nesterov                      use the plain momentum buffer instead of g + mu*m\n"
            "\n"
            "  Diagnostics:\n"
            "    --profile-step <n>          0           0 = off. n > 0 times every micro-step into\n"
            "                                            assemble / upload / build / backward / alloc /\n"
            "                                            compute / readback / free and prints the mean\n"
            "                                            every n steps, with graph node, leaf and\n"
            "                                            scheduler split counts. The run-mean lands in\n"
            "                                            dit_train_log.json under runtime.profile_ms.\n"
            "                                            Needs --ckpt 0 (the segmented path is not\n"
            "                                            instrumented). Off, the graph compute path is\n"
            "                                            byte-identical to a build without this flag.\n"
            "    --profile-ops               off         times ONE warmed-up micro-step node by node and\n"
            "                                            prints the top op/shape keys by share. The eval\n"
            "                                            callback serialises the graph, so absolute times\n"
            "                                            are inflated - read the shares, not the ms.\n"
            "\n"
            "  Adapter identity:\n"
            "    --trigger <word>            \"\"          trigger word embedded in the adapter's\n"
            "                                            metadata. Default: custom_tag from the\n"
            "                                            variant's preprocess_meta.json.\n"
            "    --trigger-position <prepend|append>     where the trigger sat in the training\n"
            "                                            captions. Default: that file's tag_position.\n"
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
            hs_remove(dir + "/" + names[i]);
        }
    }
}

static void purge_all(const std::string & dir) {
    std::vector<std::string> names;
    registry_list_dir(dir.c_str(), &names);
    for (size_t i = 0; i < names.size(); i++) {
        hs_remove(dir + "/" + names[i]);
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

                // Trigger pair — resolved exactly the way pp_caption_text does,
                // because it decides the CAPTION that got text-encoded into
                // this file. NOT cmp_s: that skips an empty `had`, and the
                // costliest change here (a cache written before the dataset had
                // a tag, or with the key absent) has precisely that shape. An
                // empty position means "prepend", and position is meaningless
                // without a tag, so both sides normalise before comparing.
                const std::string want_tag = s.custom_tag.empty() ? mf.custom_tag : s.custom_tag;
                const std::string want_pos_raw = s.tag_position.empty() ? mf.tag_position : s.tag_position;
                auto norm_pos = [](const std::string & tag, const std::string & pos) {
                    return tag.empty() ? std::string() : (pos.empty() ? std::string("prepend") : pos);
                };
                const std::string had_pos  = norm_pos(cs.custom_tag, cs.tag_position);
                const std::string want_pos = norm_pos(want_tag, want_pos_raw);
                if (cs.custom_tag != want_tag) {
                    if (!resume_diff.empty()) resume_diff += ", ";
                    resume_diff += "custom_tag " + (cs.custom_tag.empty() ? std::string("(none)") : cs.custom_tag)
                                 + "\xE2\x86\x92" + (want_tag.empty() ? std::string("(none)") : want_tag);
                }
                if (had_pos != want_pos) {
                    if (!resume_diff.empty()) resume_diff += ", ";
                    resume_diff += "tag_position " + (had_pos.empty() ? std::string("(none)") : had_pos)
                                 + "\xE2\x86\x92" + (want_pos.empty() ? std::string("(none)") : want_pos);
                }
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

// ─── --bwd: MUL_MAT activation-gradient formulation ─────────────────────────
//
// engine/patches/mm-backward.patch teaches ggml_compute_backward a second way to
// emit the gradient wrt src1 of a MUL_MAT:
//     mm      : mul_mat(cont(transpose(src0)), grad)
//     outprod : out_prod(src0, transpose(grad))          <- upstream ggml
// The two are shape- and maths-identical, but out_prod is F32-only on CUDA
// (cublasSgemm), which forces the frozen weight to F32 and drags the FORWARD
// mul_mat onto TF32 as well. The mul_mat arm is dtype-agnostic, so a BF16 weight
// rides real BF16 tensor cores in both directions. Measured on an RTX 5090:
// ~1.7-1.8x per layer per step (engine/src/train/spike-gemmbench.h).
//
// The patch reads GGML_BACKWARD_MM ONCE, at the first backward graph build, so
// this MUST run before any model load or graph construction — hence it lives in
// the argv handlers rather than inside the trainers, and covers --self-test too.
//
// `outprod` deliberately does NOT clear the variable. The self-test battery is
// A/B'd by exporting GGML_BACKWARD_MM in the environment, and clearing it here
// would make the default `--bwd outprod` silently defeat that.
static bool bwd_valid(const std::string & v) {
    return v == "outprod" || v == "mm";
}

static void bwd_apply(const std::string & v) {
    if (v != "mm") {
        return;
    }
#ifdef _WIN32
    _putenv("GGML_BACKWARD_MM=1");
#else
    setenv("GGML_BACKWARD_MM", "1", 1);
#endif
    fprintf(stderr, "[ace-train] --bwd mm: GGML_BACKWARD_MM=1 (mul_mat activation backward)\n");
}

// ─── train-lm ───────────────────────────────────────────────────────────────
//
// docs/plans/2026-07-27-lm-trainer-implementation.md §2.1, §3.1

// ─── mm3-encode ─────────────────────────────────────────────────────────────
//
// Bring-up + parity driver for the MM3 DAV encoder (training path step 2).
// Reads raw f32 planar audio (as dumped by mm3-weights/encode_ref.py) and
// writes raw f32 latents [128][L], so the two can be compared directly.
//
//   ace-train mm3-encode --enc <mm3-enc-*.gguf> --audio <in.f32> --out <out.f32>
//                        [--channels 2] [--ref <latents.f32>]
//
// With --ref it does the comparison itself and exits non-zero if the parity
// floor is missed, so it can be used as a gate and not just a dumper.
static int cmd_mm3_encode(int argc, char ** argv) {
    std::string enc_path, audio_path, out_path, ref_path;
    int         channels = 2;
    bool        tf32     = false;

    for (int i = 1; i < argc; i++) {
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "ace-train: %s needs a value\n", what); exit(2); }
            return argv[++i];
        };
        if      (!strcmp(argv[i], "--enc"))      enc_path   = next("--enc");
        else if (!strcmp(argv[i], "--audio"))    audio_path = next("--audio");
        else if (!strcmp(argv[i], "--out"))      out_path   = next("--out");
        else if (!strcmp(argv[i], "--ref"))      ref_path   = next("--ref");
        else if (!strcmp(argv[i], "--channels")) channels   = atoi(next("--channels"));
        else if (!strcmp(argv[i], "--tf32"))     tf32       = !strcmp(next("--tf32"), "on");
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { print_usage(); return 0; }
        else { fprintf(stderr, "ace-train: unknown option %s\n", argv[i]); return 2; }
    }

    // TF32 is OFF by default here, which is the opposite of the engine's usual
    // preference, and the reason is measured rather than assumed:
    //
    //   TF32 on : corr 0.999988843  rel-RMSE 4.725e-03  max-abs-diff 1.70e-01
    //   TF32 off: corr 1.000000000  rel-RMSE 7.493e-07  max-abs-diff 4.49e-05
    //
    // cuBLAS uses TF32 tensor cores for F32 GEMMs on Ampere+, and TF32 carries
    // 10 explicit mantissa bits (~1e-3 relative), which compounds over ~30
    // convolutions into ~5e-3 on the latents. These latents are the flow DiT's
    // TRAINING TARGETS: they are computed once and then re-read for hundreds of
    // epochs, so a systematic 0.5% error would be baked into every gradient for
    // the life of the dataset. Preprocessing is not the place to trade accuracy
    // for speed — and the speed it buys is a few seconds per dataset.
    //
    // Must be set BEFORE the CUDA backend initialises (mm3_enc_load ->
    // backend_init), since cuBLAS reads it at context creation. Same lever
    // dit-selftest.h uses for its finite-difference child process.
    if (!tf32) {
#ifdef _WIN32
        _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
        setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif
    }
    if (enc_path.empty() || audio_path.empty()) {
        fprintf(stderr, "ace-train mm3-encode: --enc and --audio are required\n");
        return 2;
    }
    if (channels != 1 && channels != 2) {
        fprintf(stderr, "ace-train mm3-encode: --channels must be 1 or 2\n");
        return 2;
    }

    // Planar f32: channel 0 in full, then channel 1.
    FILE * f = fopen(audio_path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", audio_path.c_str()); return 1; }
    fseek(f, 0, SEEK_END);
    const long   bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    const int64_t total = (int64_t) (bytes / (long) sizeof(float));
    const int64_t n     = total / channels;
    std::vector<float> raw((size_t) total);
    if (fread(raw.data(), sizeof(float), (size_t) total, f) != (size_t) total) {
        fclose(f); fprintf(stderr, "short read on %s\n", audio_path.c_str()); return 1;
    }
    fclose(f);
    fprintf(stderr, "[mm3-encode] %s: %lld samples x %d ch\n", audio_path.c_str(), (long long) n, channels);

    // Mono is duplicated, matching the reference's _prepare_waveform.
    const float * chans[2] = { raw.data(), channels == 2 ? raw.data() + n : raw.data() };

    MM3Enc      enc = {};
    std::string err;
    if (!mm3_enc_load(&enc, enc_path.c_str(), &err)) {
        fprintf(stderr, "[mm3-encode] load failed: %s\n", err.c_str());
        return 1;
    }

    std::vector<float> lat;
    int64_t            L = 0;
    const int64_t      t0 = ggml_time_ms();
    if (!mm3_enc_encode(&enc, chans, n, &lat, &L, &err)) {
        fprintf(stderr, "[mm3-encode] encode failed: %s\n", err.c_str());
        mm3_enc_free(&enc);
        return 1;
    }
    const int64_t ms = ggml_time_ms() - t0;
    fprintf(stderr, "[mm3-encode] -> [%u, %lld] in %lld ms\n", enc.cfg.latent_channels, (long long) L,
            (long long) ms);

    int rc = 0;

    if (!out_path.empty()) {
        FILE * o = fopen(out_path.c_str(), "wb");
        if (!o) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); rc = 1; }
        else {
            fwrite(lat.data(), sizeof(float), lat.size(), o);
            fclose(o);
            fprintf(stderr, "[mm3-encode] wrote %s (%zu floats)\n", out_path.c_str(), lat.size());
        }
    }

    if (!ref_path.empty()) {
        FILE * r = fopen(ref_path.c_str(), "rb");
        if (!r) { fprintf(stderr, "cannot open ref %s\n", ref_path.c_str()); mm3_enc_free(&enc); return 1; }
        fseek(r, 0, SEEK_END);
        const size_t rn = (size_t) (ftell(r) / (long) sizeof(float));
        fseek(r, 0, SEEK_SET);
        std::vector<float> ref(rn);
        if (fread(ref.data(), sizeof(float), rn, r) != rn) {
            fclose(r); fprintf(stderr, "short read on ref\n"); mm3_enc_free(&enc); return 1;
        }
        fclose(r);
        if (rn != lat.size()) {
            fprintf(stderr, "[mm3-encode] PARITY FAIL: ref has %zu floats, ours %zu\n", rn, lat.size());
            mm3_enc_free(&enc);
            return 1;
        }
        // Correlation + relative RMSE, the standing MM3 per-module bar.
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, sd = 0, sr = 0, amax = 0;
        for (size_t i = 0; i < rn; i++) {
            const double a = ref[i], b = lat[i], d = b - a;
            sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
            sd += d * d; sr += a * a;
            amax = std::max(amax, std::abs(d));
        }
        const double nn   = (double) rn;
        const double cov  = sab / nn - (sa / nn) * (sb / nn);
        const double va   = saa / nn - (sa / nn) * (sa / nn);
        const double vb   = sbb / nn - (sb / nn) * (sb / nn);
        const double corr = cov / std::sqrt(va * vb);
        const double rel  = std::sqrt(sd / sr);
        fprintf(stderr, "[mm3-encode] PARITY corr=%.9f rel-RMSE=%.3e max-abs-diff=%.4e\n", corr, rel, amax);
        // The bar for an fp32-vs-fp32 module port: this is not a bf16 dump
        // comparison, both sides are f32, so it should be tight.
        if (!(corr >= 0.9999 && rel <= 1e-3)) {
            fprintf(stderr, "[mm3-encode] PARITY FAIL (want corr >= 0.9999 and rel-RMSE <= 1e-3)\n");
            rc = 1;
        } else {
            fprintf(stderr, "[mm3-encode] PARITY OK\n");
        }
    }

    mm3_enc_free(&enc);
    return rc;
}

// ─── mm3-preprocess ─────────────────────────────────────────────────────────
//
// Training path step 3a: dataset -> flow-DiT TARGET latents, one file per song.
//
//   ace-train mm3-preprocess --dataset <dataset.json> --enc <mm3-enc-*.gguf>
//                            --out <dir> [--ffmpeg <path>] [--max-duration <s>]
//
// Consumes the EXISTING Training Studio dataset format unchanged — the same
// dataset.json + `<stem>.txt` sidecars the ACE path uses. MM3 needs only three
// fields per sample (audio, caption, lyrics); duration comes from the audio.
// Caption and lyrics are copied into the manifest for step 3b (the AR
// conditioning rollout), which is the other half of a preprocessed sample.
//
// Audio goes through ffmpeg at **44100 Hz**, not the ACE path's 48000: the DAV
// hop is 512 samples and 44100/512 = 86.1328125 Hz is the latent rate the flow
// DiT and the condition resampler are both built around. Resampling in ffmpeg
// rather than in-engine keeps the one high-quality resampler we already depend
// on, and means read_wav_buf can take the result verbatim.
static int cmd_mm3_preprocess(int argc, char ** argv) {
    std::string ds_path, enc_path, out_dir, ffmpeg = "ffmpeg";
    int         max_duration = 0;
    bool        tf32         = false;

    for (int i = 1; i < argc; i++) {
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "ace-train: %s needs a value\n", what); exit(2); }
            return argv[++i];
        };
        if      (!strcmp(argv[i], "--dataset"))      ds_path      = next("--dataset");
        else if (!strcmp(argv[i], "--enc"))          enc_path     = next("--enc");
        else if (!strcmp(argv[i], "--out"))          out_dir      = next("--out");
        else if (!strcmp(argv[i], "--ffmpeg"))       ffmpeg       = next("--ffmpeg");
        else if (!strcmp(argv[i], "--max-duration")) max_duration = atoi(next("--max-duration"));
        else if (!strcmp(argv[i], "--tf32"))         tf32         = !strcmp(next("--tf32"), "on");
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { print_usage(); return 0; }
        else { fprintf(stderr, "ace-train: unknown option %s\n", argv[i]); return 2; }
    }
    if (ds_path.empty() || enc_path.empty() || out_dir.empty()) {
        fprintf(stderr, "ace-train mm3-preprocess: --dataset, --enc and --out are required\n");
        return 2;
    }
    // Same reasoning as mm3-encode: these are training targets, computed once
    // and re-read for hundreds of epochs. See cmd_mm3_encode for the numbers.
    if (!tf32) {
#ifdef _WIN32
        _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
        setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif
    }

    // ── read dataset.json ──
    FILE * df = hs_fopen(ds_path, "rb");
    if (!df) { fprintf(stderr, "cannot open %s\n", ds_path.c_str()); return 1; }
    fseek(df, 0, SEEK_END);
    const long dsz = ftell(df);
    fseek(df, 0, SEEK_SET);
    std::string dbuf((size_t) dsz, '\0');
    if (fread(&dbuf[0], 1, (size_t) dsz, df) != (size_t) dsz) {
        fclose(df); fprintf(stderr, "short read on %s\n", ds_path.c_str()); return 1;
    }
    fclose(df);

    yyjson_doc * doc = yyjson_read(dbuf.c_str(), dbuf.size(), 0);
    if (!doc) { fprintf(stderr, "%s is not valid JSON\n", ds_path.c_str()); return 1; }
    yyjson_val * root    = yyjson_doc_get_root(doc);
    yyjson_val * samples = yyjson_obj_get(root, "samples");
    if (!samples || !yyjson_is_arr(samples)) {
        yyjson_doc_free(doc);
        fprintf(stderr, "%s has no `samples` array\n", ds_path.c_str());
        return 1;
    }
    auto jstr = [](yyjson_val * o, const char * k) -> std::string {
        yyjson_val * v = yyjson_obj_get(o, k);
        return (v && yyjson_is_str(v)) ? std::string(yyjson_get_str(v)) : std::string();
    };
    auto jnum = [](yyjson_val * o, const char * k) -> double {
        yyjson_val * v = yyjson_obj_get(o, k);
        return (v && yyjson_is_num(v)) ? yyjson_get_num(v) : 0.0;
    };

    if (!pm_mkdir_p(out_dir) || !pm_mkdir_p(out_dir + "/latents")) {
        yyjson_doc_free(doc);
        fprintf(stderr, "cannot create %s\n", out_dir.c_str());
        return 1;
    }

    MM3Enc      enc = {};
    std::string err;
    if (!mm3_enc_load(&enc, enc_path.c_str(), &err)) {
        yyjson_doc_free(doc);
        fprintf(stderr, "[mm3-preprocess] encoder load failed: %s\n", err.c_str());
        return 1;
    }
    const int64_t hop = (int64_t) enc.cfg.total_downsample;
    const int64_t SR  = (int64_t) enc.cfg.sampling_rate;

    yyjson_mut_doc * mdoc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * mroot = yyjson_mut_obj(mdoc);
    yyjson_mut_doc_set_root(mdoc, mroot);
    yyjson_mut_obj_add_str(mdoc, mroot, "kind", "mm3_preprocess");
    yyjson_mut_obj_add_int(mdoc, mroot, "version", 1);
    yyjson_mut_obj_add_uint(mdoc, mroot, "sample_rate", (uint64_t) SR);
    yyjson_mut_obj_add_uint(mdoc, mroot, "hop", (uint64_t) hop);
    yyjson_mut_obj_add_uint(mdoc, mroot, "latent_channels", enc.cfg.latent_channels);
    yyjson_mut_obj_add_real(mdoc, mroot, "latent_rate_hz", (double) SR / (double) hop);
    yyjson_mut_obj_add_strcpy(mdoc, mroot, "dataset", ds_path.c_str());
    yyjson_mut_val * marr = yyjson_mut_arr(mdoc);
    yyjson_mut_obj_add_val(mdoc, mroot, "samples", marr);

    const std::string tmp_wav = out_dir + "/_mm3_tmp.wav";
    size_t            idx = 0, ok_n = 0, fail_n = 0;
    double            total_sec = 0.0;
    const size_t      n_samples = yyjson_arr_size(samples);

    yyjson_val * s;
    yyjson_arr_iter it = yyjson_arr_iter_with(samples);
    while ((s = yyjson_arr_iter_next(&it))) {
        idx++;
        const std::string id       = jstr(s, "id");
        const std::string filename = jstr(s, "filename");
        const std::string audio    = jstr(s, "audio_path");
        if (audio.empty()) {
            fprintf(stderr, "[mm3-preprocess] %zu/%zu SKIP %s: no audio_path\n", idx, n_samples, filename.c_str());
            fail_n++;
            continue;
        }

        char cmd[4096];
        if (max_duration > 0) {
            snprintf(cmd, sizeof(cmd),
                     "\"%s\" -y -v error -i \"%s\" -ac 2 -ar %lld -t %d -c:a pcm_f32le -f wav \"%s\"",
                     ffmpeg.c_str(), audio.c_str(), (long long) SR, max_duration, tmp_wav.c_str());
        } else {
            snprintf(cmd, sizeof(cmd),
                     "\"%s\" -y -v error -i \"%s\" -ac 2 -ar %lld -c:a pcm_f32le -f wav \"%s\"",
                     ffmpeg.c_str(), audio.c_str(), (long long) SR, tmp_wav.c_str());
        }
#ifdef _WIN32
        const std::string wrapped = "\"" + std::string(cmd) + "\"";  // cmd.exe strips the outer pair
        const int         rc      = hs_system(wrapped);
#else
        const int rc = hs_system(cmd);
#endif
        if (rc != 0 || !pm_file_exists(tmp_wav)) {
            fprintf(stderr, "[mm3-preprocess] %zu/%zu FAIL %s: ffmpeg exit %d\n", idx, n_samples,
                    filename.c_str(), rc);
            fail_n++;
            continue;
        }

        // Read the transcode back. It is already 44.1 kHz stereo f32, so there
        // is no in-engine resample.
        //
        // USE audio_io_read_wav_buf, NOT read_wav_buf. The raw reader returns
        // INTERLEAVED [T,2] (see the header comment in wav.h); the encoder
        // wants PLANAR [L:T][R:T]. This call site used to take the raw reader's
        // output and slice it as { p, p + T }, which silently made "left" the
        // FIRST HALF of the song with L and R alternating, and "right" the
        // second half. Because L ~= R in most music, alternating them
        // duplicates every sample -- an exact 2x time stretch, exactly one
        // octave down. Every cached training target was the song in slow
        // motion, and five LoRA runs learned to produce slow motion.
        //
        // Nothing caught it for a day because T is PER-CHANNEL frames, so
        // latent_frames / duration still came out at exactly 86.1328 Hz and
        // every arithmetic check passed. The DAV encoder's own parity gate
        // passed too -- it is fed encode_ref.py's dump, which really is planar.
        // The encoder was never wrong; only this wiring into it was.
        //
        // Judge preprocessing by DECODING A TARGET AND LISTENING TO IT. That
        // found this in one listen after metadata checks found nothing.
        FILE * wf = hs_fopen(tmp_wav, "rb");
        if (!wf) { fprintf(stderr, "[mm3-preprocess] cannot reopen temp wav\n"); fail_n++; continue; }
        fseek(wf, 0, SEEK_END);
        const long wsz = ftell(wf);
        fseek(wf, 0, SEEK_SET);
        std::vector<uint8_t> wbuf((size_t) wsz);
        const bool           wread = fread(wbuf.data(), 1, (size_t) wsz, wf) == (size_t) wsz;
        fclose(wf);
        int      T = 0, got_sr = 0;
        float *  planar = wread ? audio_io_read_wav_buf(wbuf.data(), wbuf.size(), &T, &got_sr) : nullptr;
        if (!planar || T <= 0) {
            fprintf(stderr, "[mm3-preprocess] %zu/%zu FAIL %s: cannot decode transcode\n", idx, n_samples,
                    filename.c_str());
            free(planar);
            fail_n++;
            continue;
        }
        if (got_sr != (int) SR) {
            fprintf(stderr, "[mm3-preprocess] %zu/%zu FAIL %s: ffmpeg produced %d Hz, wanted %lld\n", idx,
                    n_samples, filename.c_str(), got_sr, (long long) SR);
            free(planar);
            fail_n++;
            continue;
        }

        const float * chans[2] = { planar, planar + T };
        std::vector<float> lat;
        int64_t            L = 0;
        if (!mm3_enc_encode(&enc, chans, (int64_t) T, &lat, &L, &err)) {
            fprintf(stderr, "[mm3-preprocess] %zu/%zu FAIL %s: %s\n", idx, n_samples, filename.c_str(),
                    err.c_str());
            free(planar);
            fail_n++;
            continue;
        }
        free(planar);

        const std::string stem = id.empty() ? std::to_string(idx) : id;
        const std::string lp   = out_dir + "/latents/" + stem + ".f32";
        FILE *            lf   = hs_fopen(lp, "wb");
        if (!lf) { fprintf(stderr, "[mm3-preprocess] cannot write %s\n", lp.c_str()); fail_n++; continue; }
        fwrite(lat.data(), sizeof(float), lat.size(), lf);
        fclose(lf);

        const double secs = (double) T / (double) SR;
        total_sec += secs;
        ok_n++;

        yyjson_mut_val * m = yyjson_mut_obj(mdoc);
        yyjson_mut_obj_add_strcpy(mdoc, m, "id", stem.c_str());
        yyjson_mut_obj_add_strcpy(mdoc, m, "filename", filename.c_str());
        yyjson_mut_obj_add_strcpy(mdoc, m, "latents", ("latents/" + stem + ".f32").c_str());
        yyjson_mut_obj_add_int(mdoc, m, "latent_frames", (int64_t) L);
        yyjson_mut_obj_add_int(mdoc, m, "n_samples", (int64_t) T);
        yyjson_mut_obj_add_real(mdoc, m, "duration_sec", secs);
        // Carried for step 3b (the AR conditioning rollout), so that stage does
        // not have to re-open the dataset or re-resolve sidecars.
        yyjson_mut_obj_add_strcpy(mdoc, m, "caption", jstr(s, "caption").c_str());
        yyjson_mut_obj_add_strcpy(mdoc, m, "lyrics", jstr(s, "lyrics").c_str());
        yyjson_mut_obj_add_real(mdoc, m, "bpm", jnum(s, "bpm"));
        yyjson_mut_obj_add_strcpy(mdoc, m, "keyscale", jstr(s, "keyscale").c_str());
        yyjson_mut_obj_add_strcpy(mdoc, m, "timesignature", jstr(s, "timesignature").c_str());
        yyjson_mut_arr_append(marr, m);

        fprintf(stderr, "[mm3-preprocess] %zu/%zu ok %-44s %6.1fs -> [%u, %lld]\n", idx, n_samples,
                filename.c_str(), secs, enc.cfg.latent_channels, (long long) L);
    }

    hs_remove(tmp_wav);
    yyjson_doc_free(doc);
    mm3_enc_free(&enc);

    yyjson_mut_obj_add_uint(mdoc, mroot, "n_ok", (uint64_t) ok_n);
    yyjson_mut_obj_add_uint(mdoc, mroot, "n_failed", (uint64_t) fail_n);
    yyjson_mut_obj_add_real(mdoc, mroot, "total_audio_sec", total_sec);

    const std::string mpath = out_dir + "/mm3_preprocess.json";
    size_t            mlen  = 0;
    char *            mjson = yyjson_mut_write(mdoc, YYJSON_WRITE_PRETTY, &mlen);
    int               rc    = 0;
    if (mjson) {
        FILE * mf = hs_fopen(mpath, "wb");
        if (mf) { fwrite(mjson, 1, mlen, mf); fclose(mf); }
        else    { fprintf(stderr, "cannot write %s\n", mpath.c_str()); rc = 1; }
        free(mjson);
    } else {
        rc = 1;
    }
    yyjson_mut_doc_free(mdoc);

    fprintf(stderr, "[mm3-preprocess] done: %zu ok, %zu failed, %.1f min of audio -> %s\n", ok_n, fail_n,
            total_sec / 60.0, mpath.c_str());
    return (fail_n && !ok_n) ? 1 : rc;
}

// ─── mm3-condition ──────────────────────────────────────────────────────────
//
// Training path step 3b: the flow DiT's CONDITIONING, cached per song.
//
//   ace-train mm3-condition --manifest <mm3_preprocess.json> --models <dir>
//                           --captions <dataset dir> [--seed N] [--dit-quant Q]
//
// Per song: caption + lyrics -> AR rollout (LM + depth decoder) -> frame_hiddens
// -> condition encoder -> condition, stored f16 beside the latents.
//
// ── The thing to understand before trusting this output ─────────────────────
//
// The rollout is SAMPLED FROM THE CAPTION. It is not, and cannot be, derived
// from the training audio: MiniMax ships no audio->code encoder, so there are no
// ground-truth codes for a real song and the conditioning cannot be
// teacher-forced (see docs/plans/2026-08-14-mm3-training-feasibility.md). The
// conditioning therefore matches the target only in caption, lyrics and
// duration — never in musical content. That is a property of the release, not a
// shortcut here, and it is why the DiT can learn the target's timbre/production
// marginal but not its note content.
//
// ── Why the DiT is loaded at a tiny quant ───────────────────────────────────
//
// mm3_load_parts' `rest` covers cond+dit+voc, and only `cond` is used here. The
// DiT is dead weight, so its role is pointed at the smallest quant on disk:
// f16 is 4.8 GB against Q2_K's 0.83 GB, and at f16 the LM (17 GB) + depth +
// DiT + KV (288 kB/position, ~2 GB at 6990 frames) + compute headroom does not
// fit the card alongside a desktop. Nothing here ever executes a DiT graph.
// Load a teacher-forcing code stream: int32 little-endian, [n_iter, 8], one row
// per AR ITERATION with columns [semantic, acoustic1..7].
//
// Row 0 is the WARM-UP iteration: it is fed back into the LM but emits no audio,
// so the file carries one more row than there are frames (audio frame j pairs
// with row j+1). That is the reference's own indexing, and getting it wrong
// degrades conditioning parity 49x while passing every per-module check — see
// trap 4 and the off-by-one note in mm3-ar-loop.h.
static bool load_forced_codes(const std::string & path, int64_t NC, std::vector<int32_t> & sem,
                              std::vector<int32_t> & ac, int64_t * n_iter, std::string * err) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) { *err = "cannot open " + path; return false; }
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    const int64_t stride = NC + 1;
    if (sz <= 0 || (size_t) sz % (sizeof(int32_t) * (size_t) stride) != 0) {
        fclose(f);
        *err = path + ": size " + std::to_string(sz) + " is not a multiple of " + std::to_string(stride) +
               " int32 columns";
        return false;
    }
    std::vector<int32_t> raw((size_t) sz / sizeof(int32_t));
    const bool           ok = fread(raw.data(), 1, (size_t) sz, f) == (size_t) sz;
    fclose(f);
    if (!ok) { *err = "short read on " + path; return false; }

    const int64_t rows = (int64_t) raw.size() / stride;
    if (rows < 2) { *err = path + " has " + std::to_string(rows) + " rows, needs >= 2 (warm-up + 1 frame)"; return false; }
    sem.resize((size_t) rows);
    ac.resize((size_t) (rows * NC));
    for (int64_t r = 0; r < rows; r++) {
        sem[(size_t) r] = raw[(size_t) (r * stride)];
        for (int64_t c = 0; c < NC; c++) {
            ac[(size_t) (r * NC + c)] = raw[(size_t) (r * stride + 1 + c)];
        }
    }
    *n_iter = rows;
    return true;
}

static int cmd_mm3_condition(int argc, char ** argv) {
    std::string manifest_path, models_dir, captions_dir, codes_dir, codes_mode = "full";
    std::string dit_quant = "Q2_K", lm_quant;
    int64_t     seed        = 42;
    double      segment_sec = 60.0;   // 0 = one rollout for the whole song
    bool        tf32        = false;

    for (int i = 1; i < argc; i++) {
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "ace-train: %s needs a value\n", what); exit(2); }
            return argv[++i];
        };
        if      (!strcmp(argv[i], "--manifest"))  manifest_path = next("--manifest");
        else if (!strcmp(argv[i], "--models"))    models_dir    = next("--models");
        else if (!strcmp(argv[i], "--captions"))  captions_dir  = next("--captions");
        else if (!strcmp(argv[i], "--codes"))     codes_dir     = next("--codes");
        else if (!strcmp(argv[i], "--codes-mode")) codes_mode   = next("--codes-mode");
        else if (!strcmp(argv[i], "--dit-quant")) dit_quant     = next("--dit-quant");
        else if (!strcmp(argv[i], "--lm-quant"))  lm_quant      = next("--lm-quant");
        else if (!strcmp(argv[i], "--seed"))      seed          = atoll(next("--seed"));
        else if (!strcmp(argv[i], "--segment-sec")) segment_sec = atof(next("--segment-sec"));
        else if (!strcmp(argv[i], "--tf32"))      tf32          = !strcmp(next("--tf32"), "on");
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { print_usage(); return 0; }
        else { fprintf(stderr, "ace-train: unknown option %s\n", argv[i]); return 2; }
    }
    if (manifest_path.empty() || models_dir.empty() || captions_dir.empty()) {
        fprintf(stderr, "ace-train mm3-condition: --manifest, --models and --captions are required\n");
        return 2;
    }
    if (!tf32) {
#ifdef _WIN32
        _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
        setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif
    }

    // ── manifest ──
    FILE * mf = hs_fopen(manifest_path, "rb");
    if (!mf) { fprintf(stderr, "cannot open %s\n", manifest_path.c_str()); return 1; }
    fseek(mf, 0, SEEK_END);
    const long msz = ftell(mf);
    fseek(mf, 0, SEEK_SET);
    std::string mbuf((size_t) msz, '\0');
    if (fread(&mbuf[0], 1, (size_t) msz, mf) != (size_t) msz) { fclose(mf); fprintf(stderr, "short read\n"); return 1; }
    fclose(mf);

    yyjson_doc * doc = yyjson_read(mbuf.c_str(), mbuf.size(), 0);
    if (!doc) { fprintf(stderr, "%s is not valid JSON\n", manifest_path.c_str()); return 1; }
    yyjson_val * mroot = yyjson_doc_get_root(doc);
    yyjson_val * arr   = yyjson_obj_get(mroot, "samples");
    if (!arr || !yyjson_is_arr(arr)) {
        yyjson_doc_free(doc);
        fprintf(stderr, "manifest has no `samples` array\n");
        return 1;
    }
    std::string out_dir = manifest_path;
    {
        size_t s = out_dir.find_last_of("/\\");
        out_dir  = (s == std::string::npos) ? std::string(".") : out_dir.substr(0, s);
    }
    if (!pm_mkdir_p(out_dir + "/condition")) {
        yyjson_doc_free(doc);
        fprintf(stderr, "cannot create %s/condition\n", out_dir.c_str());
        return 1;
    }

    // ── model: LM + depth + rest, DiT pinned tiny ──
    static MM3Model model;
    std::string     roles[MM3_R_COUNT];
    roles[MM3_R_DIT] = dit_quant;
    mm3_discover(&model, models_dir.c_str(), lm_quant, roles);
    std::string err;
    if (!mm3_load_parts(&model, /*lm*/ true, /*depth*/ true, /*rest*/ true, &err)) {
        yyjson_doc_free(doc);
        fprintf(stderr, "[mm3-condition] load failed: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "[mm3-condition] lm=%s dit=%s (unused) cond=%s\n",
            model.lm_file.name.c_str(), model.role_file[MM3_R_DIT].name.c_str(),
            model.role_file[MM3_R_COND].name.c_str());

    MM3Tokenizer tok;
    if (!mm3_tokenizer_load(model, &tok, &err)) {
        fprintf(stderr, "[mm3-condition] tokenizer: %s\n", err.c_str());
        return 1;
    }

    const int64_t H   = (int64_t) model.lm_cfg.embedding_length;
    const int64_t LAY = (int64_t) model.lm_cfg.num_codebooks;   // 8 = LM hidden + 7 depth
    const int64_t FPS = (int64_t) model.lm_cfg.frame_rate;

    yyjson_mut_doc * odoc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * oroot = yyjson_mut_obj(odoc);
    yyjson_mut_doc_set_root(odoc, oroot);
    yyjson_mut_val * oarr = yyjson_mut_arr(odoc);

    size_t idx = 0, n_ok = 0, n_fail = 0;
    double ar_ms_total = 0.0;
    const size_t n_total = yyjson_arr_size(arr);

    yyjson_val *    s;
    yyjson_arr_iter it = yyjson_arr_iter_with(arr);
    while ((s = yyjson_arr_iter_next(&it))) {
        idx++;
        auto js = [&](const char * k) -> std::string {
            yyjson_val * v = yyjson_obj_get(s, k);
            return (v && yyjson_is_str(v)) ? std::string(yyjson_get_str(v)) : std::string();
        };
        const std::string id       = js("id");
        const std::string filename = js("filename");
        const std::string lyrics   = js("lyrics");
        yyjson_val *      vL       = yyjson_obj_get(s, "latent_frames");
        const int64_t     L_target = vL ? (int64_t) yyjson_get_int(vL) : 0;
        yyjson_val *      vD       = yyjson_obj_get(s, "duration_sec");
        const double      dur      = vD ? yyjson_get_num(vD) : 0.0;

        // The MM3 caption lives beside the audio as <stem>.mm3.txt, NOT in the
        // sidecar (see mm3-caption-restructure.py). Refuse rather than silently
        // fall back to the ACE caption: an ACE caption produces the wrong genre
        // (measured), so a silent fallback would poison the cache invisibly.
        std::string stem = filename;
        {
            size_t d = stem.find_last_of('.');
            if (d != std::string::npos) stem = stem.substr(0, d);
        }
        const std::string cap_path = captions_dir + "/" + stem + ".mm3.txt";
        FILE *            cf       = hs_fopen(cap_path, "rb");
        if (!cf) {
            fprintf(stderr, "[mm3-condition] %zu/%zu SKIP %s: no %s.mm3.txt\n", idx, n_total,
                    filename.c_str(), stem.c_str());
            n_fail++;
            continue;
        }
        std::string caption;
        {
            char   b[8192];
            size_t r;
            while ((r = fread(b, 1, sizeof(b), cf)) > 0) caption.append(b, r);
            fclose(cf);
        }

        // ── segmented AR rollout ──
        //
        // A single rollout for a whole song UNDER-COVERS it: the LM emits EOS
        // long before a 4-minute request is satisfied. Measured on this dataset,
        // one-shot: sadie asked for 6990 frames and stopped at 4540 (65 % of the
        // song), mercy_me 83 %, burn 84 %. Nothing is corrupt, but a third of
        // some songs would silently have no conditioning and so could never be
        // trained on. MM3's reference generates 30-40 s clips; 4.5 minutes in one
        // unbroken rollout is far outside what it sustains.
        //
        // So roll out in segments and concatenate. This costs nothing musically:
        // the conditioning is caption-derived and content-misaligned with the
        // target audio by construction, so a seam every --segment-sec is no less
        // faithful than the middle of a segment. Segment boundaries ARE recorded,
        // so the trainer can decline to sample a window that straddles one.
        //
        // Lyrics are split proportionally across segments rather than repeated:
        // handing every 60 s segment the whole song's lyrics would make each one
        // try to sing all of them, giving implausibly dense vocal conditioning.
        const int64_t frames_want = std::min<int64_t>((int64_t) llround(dur * (double) FPS),
                                                      (int64_t) model.lm_cfg.max_audio_frames);
        const int64_t seg_frames  = (segment_sec > 0.0)
                                        ? std::max<int64_t>(1, (int64_t) llround(segment_sec * (double) FPS))
                                        : frames_want;
        const int64_t n_seg       = (frames_want + seg_frames - 1) / seg_frames;

        std::vector<std::string> lyric_parts;
        {
            std::vector<std::string> lines;
            size_t                   p = 0;
            while (p <= lyrics.size()) {
                size_t nl = lyrics.find('\n', p);
                if (nl == std::string::npos) { lines.push_back(lyrics.substr(p)); break; }
                lines.push_back(lyrics.substr(p, nl - p));
                p = nl + 1;
            }
            lyric_parts.assign((size_t) n_seg, std::string());
            for (size_t li = 0; li < lines.size(); li++) {
                const size_t sidx = std::min<size_t>((size_t) n_seg - 1, li * (size_t) n_seg / std::max<size_t>(1, lines.size()));
                if (!lyric_parts[sidx].empty()) lyric_parts[sidx] += "\n";
                lyric_parts[sidx] += lines[li];
            }
        }

        std::vector<float>   hiddens;      // [F_total, LAY, H]
        std::vector<int64_t> seg_bounds;   // frame index of each segment start
        int64_t              F        = 0;
        int64_t              n_eos    = 0;
        bool                 song_ok  = true;
        const int64_t        t0       = ggml_time_ms();

        // ── ALIGNED (teacher-forced) conditioning ────────────────────────────
        //
        // With --codes the conditioning stops being sampled from the caption and
        // becomes derived from THE ACTUAL AUDIO, via an RVQ encoder run offline
        // (docs/plans/2026-08-16-mm3-rvq-encoder-plan.md). Frame j of the
        // conditioning now describes frame j of the target instead of merely
        // sharing its caption, which is the entire point of the exercise.
        //
        // Three consequences, all good:
        //   - ONE replay covers the whole song (max_audio_frames 9000 ~ 6 min),
        //     so there are no segments and no seams. The seam machinery below
        //     and trap 17's reject-and-retry become vestigial for these caches.
        //   - No EOS risk: the loop is driven by the code stream, not by the LM
        //     deciding it is finished, so the under-coverage that forced
        //     segmentation in the sampled path cannot occur.
        //   - Lyrics are passed WHOLE rather than split proportionally; there is
        //     no segment to apportion them across.
        if (!codes_dir.empty()) {
            std::vector<int32_t> fsem, fac;
            int64_t              n_iter = 0;
            const int64_t        NC     = LAY - 1;   // 8 layers = LM hidden + 7 acoustic
            if (!load_forced_codes(codes_dir + "/" + id + ".codes", NC, fsem, fac, &n_iter, &err)) {
                fprintf(stderr, "[mm3-condition] %zu/%zu SKIP %s: %s\n", idx, n_total, filename.c_str(),
                        err.c_str());
                n_fail++;
                continue;
            }
            // The KV cache is 288 kB/position, so an over-long stream is a VRAM
            // failure rather than a quality one. Clamp and say so.
            if (n_iter - 1 > (int64_t) model.lm_cfg.max_audio_frames) {
                fprintf(stderr, "[mm3-condition] %s: %lld frames clamped to max_audio_frames %lld\n",
                        filename.c_str(), (long long) (n_iter - 1),
                        (long long) model.lm_cfg.max_audio_frames);
                n_iter = (int64_t) model.lm_cfg.max_audio_frames + 1;
            }

            MM3SynthRequest req;
            req.caption    = caption;
            req.lyrics     = lyrics;
            req.duration   = (double) (n_iter - 1) / (double) FPS;
            req.prompt     = mm3_assemble_prompt(req.caption, req.lyrics, &req.instrumental);
            req.max_frames = n_iter - 1;
            req.seed_in    = seed;
            if (!mm3_request_tokenize(model, &tok, &req, &err)) {
                fprintf(stderr, "[mm3-condition] %zu/%zu FAIL %s: tokenize: %s\n", idx, n_total,
                        filename.c_str(), err.c_str());
                n_fail++;
                continue;
            }

            MM3ArOptions aopt;
            aopt.max_frames      = n_iter - 1;
            aopt.seed            = (uint64_t) seed;
            aopt.collect_hiddens = true;
            aopt.forced_semantic = fsem.data();
            // --codes-mode semantic pins CONTENT (what happens when) to the real
            // audio but lets the depth decoder sample TIMBRE from its own
            // distribution. Full alignment describes the target so completely
            // that the base DiT can render it without the adapter learning
            // anything -- measured 2026-08-16: aligned conditioning trained an
            // adapter that was audibly identical to base. Leaving timbre
            // unspecified is what forces the adapter to store it.
            aopt.forced_acoustic = (codes_mode == "semantic") ? nullptr : fac.data();
            aopt.forced_len      = n_iter;
            MM3ArResult arf;
            if (!mm3_ar_plan(model, req.gen.ids_cond.data(), req.gen.ids_uncond.data(),
                             (int64_t) req.gen.ids_cond.size(), aopt, &arf, &err)) {
                fprintf(stderr, "[mm3-condition] %zu/%zu FAIL %s: forced AR: %s\n", idx, n_total,
                        filename.c_str(), err.c_str());
                n_fail++;
                continue;
            }
            seg_bounds.push_back(0);
            hiddens.swap(arf.frame_hiddens);
            F = arf.n_frames;
        }

        for (int64_t sg = 0; sg < n_seg && song_ok && codes_dir.empty(); sg++) {
            const int64_t want = std::min<int64_t>(seg_frames, frames_want - sg * seg_frames);
            if (want <= 0) break;

            MM3SynthRequest req;
            req.caption  = caption;
            req.lyrics   = lyric_parts[(size_t) sg];
            req.duration = (double) want / (double) FPS;
            req.prompt   = mm3_assemble_prompt(req.caption, req.lyrics, &req.instrumental);
            req.max_frames = want;
            req.seed_in    = seed + sg;
            if (!mm3_request_tokenize(model, &tok, &req, &err)) {
                fprintf(stderr, "[mm3-condition] %zu/%zu FAIL %s seg %lld: %s\n", idx, n_total,
                        filename.c_str(), (long long) sg, err.c_str());
                song_ok = false;
                break;
            }

            MM3ArOptions aopt;
            aopt.max_frames      = want;
            // Per-segment seed: identical segments would otherwise produce
            // identical conditioning, which is a degenerate signal to train on.
            aopt.seed            = (uint64_t) (seed + sg);
            aopt.collect_hiddens = true;
            MM3ArResult ar;
            if (!mm3_ar_plan(model, req.gen.ids_cond.data(), req.gen.ids_uncond.data(),
                             (int64_t) req.gen.ids_cond.size(), aopt, &ar, &err)) {
                fprintf(stderr, "[mm3-condition] %zu/%zu FAIL %s seg %lld: AR: %s\n", idx, n_total,
                        filename.c_str(), (long long) sg, err.c_str());
                song_ok = false;
                break;
            }
            seg_bounds.push_back(F);
            hiddens.insert(hiddens.end(), ar.frame_hiddens.begin(), ar.frame_hiddens.end());
            F += ar.n_frames;
            if (ar.eos_hit) n_eos++;
        }
        if (!song_ok || F <= 0) {
            n_fail++;
            continue;
        }
        ar_ms_total += (double) (ggml_time_ms() - t0);
        MM3ArResult ar;              // aggregate stand-in for the fields used below
        ar.n_frames     = F;
        ar.eos_hit      = n_eos > 0;
        ar.total_ms     = (double) (ggml_time_ms() - t0);
        ar.frame_hiddens.swap(hiddens);

        // ── condition encode, chunked ──
        //
        // MM3_COND_MAX_FRAMES is 4096 and a 280 s song is ~6990 frames, so long
        // songs need more than one pass. Chunks carry OVERLAP frames of context
        // on each side and keep only their core; the frame->latent map is
        // non-integer (x3.4453125, truncated), so each chunk's destination
        // offset is taken from mm3_cond_latent_length() of its core start
        // rather than computed by multiplication.
        // F is the total across all segments, set by the rollout loop above.
        const int64_t L_all = mm3_cond_latent_length(model.synth_cfg.cond, F);
        const int64_t CHUNK = 3072, OVER = 128;
        std::vector<float> cond_all((size_t) (L_all * (int64_t) model.synth_cfg.cond.out_dim), 0.0f);
        const int64_t      CD = (int64_t) model.synth_cfg.cond.out_dim;
        bool               cond_ok = true;

        for (int64_t start = 0; start < F && cond_ok; start += CHUNK) {
            const int64_t lead  = std::min<int64_t>(OVER, start);
            const int64_t begin = start - lead;
            const int64_t want  = std::min<int64_t>(CHUNK + lead + OVER, F - begin);
            std::vector<float> part;
            int64_t            pl = 0;
            if (!mm3_cond_encode(model, ar.frame_hiddens.data() + (size_t) (begin * LAY * H), want, part, &pl,
                                 &err)) {
                fprintf(stderr, "[mm3-condition] %zu/%zu FAIL %s: cond: %s\n", idx, n_total, filename.c_str(),
                        err.c_str());
                cond_ok = false;
                break;
            }
            const int64_t l_dst  = mm3_cond_latent_length(model.synth_cfg.cond, start);
            const int64_t l_skip = mm3_cond_latent_length(model.synth_cfg.cond, begin + lead) -
                                   mm3_cond_latent_length(model.synth_cfg.cond, begin);
            const int64_t l_core = std::min<int64_t>(pl - l_skip, L_all - l_dst);
            if (l_core > 0) {
                memcpy(cond_all.data() + (size_t) (l_dst * CD), part.data() + (size_t) (l_skip * CD),
                       (size_t) (l_core * CD) * sizeof(float));
            }
        }
        if (!cond_ok) {
            n_fail++;
            continue;
        }

        // f16 on disk: this is conditioning, not a gradient target, and it
        // halves a cache that is already the largest artifact in the pipeline.
        std::vector<ggml_fp16_t> half(cond_all.size());
        ggml_fp32_to_fp16_row(cond_all.data(), half.data(), (int) cond_all.size());
        const std::string cp = out_dir + "/condition/" + id + ".f16";
        FILE *            of = hs_fopen(cp, "wb");
        if (!of) { fprintf(stderr, "[mm3-condition] cannot write %s\n", cp.c_str()); n_fail++; continue; }
        fwrite(half.data(), sizeof(ggml_fp16_t), half.size(), of);
        fclose(of);

        yyjson_mut_val * e = yyjson_mut_obj(odoc);
        yyjson_mut_obj_add_strcpy(odoc, e, "id", id.c_str());
        yyjson_mut_obj_add_strcpy(odoc, e, "filename", filename.c_str());
        yyjson_mut_obj_add_strcpy(odoc, e, "condition", ("condition/" + id + ".f16").c_str());
        yyjson_mut_obj_add_int(odoc, e, "cond_frames", F);
        yyjson_mut_obj_add_int(odoc, e, "cond_latents", L_all);
        yyjson_mut_obj_add_int(odoc, e, "cond_dim", CD);
        yyjson_mut_obj_add_int(odoc, e, "latent_frames", L_target);
        yyjson_mut_obj_add_bool(odoc, e, "eos_hit", ar.eos_hit);
        // Latent index of each segment start. A training window that straddles
        // one of these spans two independent rollouts, so the trainer should
        // either skip it or accept the seam knowingly.
        yyjson_mut_val * sb = yyjson_mut_arr(odoc);
        for (int64_t fb : seg_bounds) {
            yyjson_mut_arr_add_int(odoc, sb, mm3_cond_latent_length(model.synth_cfg.cond, fb));
        }
        yyjson_mut_obj_add_val(odoc, e, "segment_latent_starts", sb);
        yyjson_mut_obj_add_int(odoc, e, "n_segments", (int64_t) seg_bounds.size());
        // The trainer must be able to tell the two cache kinds apart without
        // guessing: an aligned cache pairs conditioning with the SAME music as
        // the target, a sampled one only shares its caption. They warrant
        // different expectations, and mixing them silently would be its own trap.
        yyjson_mut_obj_add_bool(odoc, e, "aligned", !codes_dir.empty());
        yyjson_mut_arr_append(oarr, e);
        n_ok++;

        const double cover = L_target > 0 ? 100.0 * (double) std::min<int64_t>(L_all, L_target) / (double) L_target
                                          : 0.0;
        fprintf(stderr, "[mm3-condition] %zu/%zu ok %-42s F=%-5lld cond L=%-6lld (audio L=%-6lld %5.1f%%) "
                        "%lld seg %.1fs\n",
                idx, n_total, filename.c_str(), (long long) F, (long long) L_all, (long long) L_target, cover,
                (long long) seg_bounds.size(), ar.total_ms / 1000.0);
    }

    yyjson_doc_free(doc);

    yyjson_mut_obj_add_str(odoc, oroot, "kind", "mm3_condition");
    yyjson_mut_obj_add_int(odoc, oroot, "version", 1);
    yyjson_mut_obj_add_int(odoc, oroot, "seed", seed);
    yyjson_mut_obj_add_strcpy(odoc, oroot, "lm", model.lm_file.name.c_str());
    yyjson_mut_obj_add_str(odoc, oroot, "dtype", "f16");
    yyjson_mut_obj_add_uint(odoc, oroot, "n_ok", (uint64_t) n_ok);
    yyjson_mut_obj_add_uint(odoc, oroot, "n_failed", (uint64_t) n_fail);
    yyjson_mut_obj_add_val(odoc, oroot, "samples", oarr);

    const std::string opath = out_dir + "/mm3_condition.json";
    size_t            olen  = 0;
    char *            ojson = yyjson_mut_write(odoc, YYJSON_WRITE_PRETTY, &olen);
    int               rc    = 0;
    if (ojson) {
        FILE * o = hs_fopen(opath, "wb");
        if (o) { fwrite(ojson, 1, olen, o); fclose(o); } else { rc = 1; }
        free(ojson);
    } else {
        rc = 1;
    }
    yyjson_mut_doc_free(odoc);

    fprintf(stderr, "[mm3-condition] done: %zu ok, %zu failed, %.1f min of AR -> %s\n", n_ok, n_fail,
            ar_ms_total / 60000.0, opath.c_str());
    return (n_fail && !n_ok) ? 1 : rc;
}


// ─── mm3-train-dit ──────────────────────────────────────────────────────────
static int cmd_mm3_train_dit(int argc, char ** argv) {
    MM3TrainArgs a;
    bool tf32 = false;
    bool bwd_outprod = false;   // --bwd outprod restores the slow CPU path
    for (int i = 1; i < argc; i++) {
        auto next = [&](const char * w) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "ace-train: %s needs a value\n", w); exit(2); }
            return argv[++i];
        };
        if      (!strcmp(argv[i], "--cache"))      a.cache_dir  = next("--cache");
        else if (!strcmp(argv[i], "--models"))     a.models_dir = next("--models");
        else if (!strcmp(argv[i], "--out"))        a.out_dir    = next("--out");
        else if (!strcmp(argv[i], "--rank"))       a.rank       = atoll(next("--rank"));
        else if (!strcmp(argv[i], "--alpha"))      a.alpha      = (float) atof(next("--alpha"));
        else if (!strcmp(argv[i], "--lr"))         a.lr         = (float) atof(next("--lr"));
        else if (!strcmp(argv[i], "--steps"))      a.steps      = atoll(next("--steps"));
        else if (!strcmp(argv[i], "--crop"))       a.crop       = atoll(next("--crop"));
        else if (!strcmp(argv[i], "--grad-accum")) a.grad_accum = atoll(next("--grad-accum"));
        else if (!strcmp(argv[i], "--seed"))       a.seed       = (uint64_t) atoll(next("--seed"));
        else if (!strcmp(argv[i], "--song"))       a.only_song  = next("--song");
        // Sigma is 1 - sigmoid(N(mean, std)), so a POSITIVE mean pushes sigma
        // toward 0, i.e. toward crops that are mostly REAL AUDIO. That is the
        // knob that decides what the run can learn at all. At mean 0 most steps
        // land near sigma 1 (near-pure noise), where the only signal available
        // is the caption marginal -- and because our conditioning is sampled
        // from the caption rather than teacher-forced from the target, those
        // steps can only ever teach "what does this genre average to". Run 01
        // did exactly that: loss fell 2% over 3000 steps and the adapter was
        // only usable at scale 0.2 because most of what it learned was the mean.
        else if (!strcmp(argv[i], "--logit-mean")) a.logit_mean = (float) atof(next("--logit-mean"));
        else if (!strcmp(argv[i], "--logit-std"))  a.logit_std  = (float) atof(next("--logit-std"));
        // Fixed stratified-sigma holdout. This is the ONLY number comparable
        // between runs -- training loss is measured at whatever sigmas
        // logit_mean happens to draw, so it changes meaning when that changes.
        else if (!strcmp(argv[i], "--eval-every")) a.eval_every = atoll(next("--eval-every"));
        else if (!strcmp(argv[i], "--eval-n"))     a.eval_n     = atoll(next("--eval-n"));
        // Keep this PINNED across crop experiments. Default 689 so every run
        // stays comparable to runs 06/07; 0 means "follow --crop", which makes
        // the eval move with the variable under test.
        else if (!strcmp(argv[i], "--eval-crop"))  a.eval_crop  = atoll(next("--eval-crop"));
        else if (!strcmp(argv[i], "--crop-mode"))  a.crop_mode  = next("--crop-mode");
        else if (!strcmp(argv[i], "--ckpt-segments")) a.ckpt_segments = atoll(next("--ckpt-segments"));
        else if (!strcmp(argv[i], "--ckpt-verify"))   a.ckpt_verify   = true;
        else if (!strcmp(argv[i], "--export-every"))  a.export_every  = atoll(next("--export-every"));
        else if (!strcmp(argv[i], "--target")) {
            a.target = next("--target");
            if (a.target != "all" && a.target != "mlpv") {
                fprintf(stderr, "ace-train: --target must be all or mlpv\n"); return 2;
            }
        }
        else if (!strcmp(argv[i], "--sign-check")) a.sign_check = true;
        else if (!strcmp(argv[i], "--tf32"))       tf32         = !strcmp(next("--tf32"), "on");
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { print_usage(); return 0; }
        else { fprintf(stderr, "ace-train: unknown option %s\n", argv[i]); return 2; }
    }
    if (a.cache_dir.empty() || a.models_dir.empty()) {
        fprintf(stderr, "ace-train mm3-train-dit: --cache and --models are required\n");
        return 2;
    }
    if (!tf32) {
#ifdef _WIN32
        _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
        setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif
    }
    // THE mul_mat BACKWARD REFORMULATION. Without it every LoRA weight gradient
    // is an OUT_PROD, ggml-cuda cannot take it, and all 146 of them run on the
    // CPU with a round trip each way -- measured as 292 backend crossings and
    // 46 s per step at crop 344 while the GPU sat at 6%. engine/patches/
    // mm-backward.patch rewrites out_prod(W, transpose(grad)) into the provably
    // identical mul_mat(cont(transpose(W)), grad), which CUDA does implement.
    //
    // Must be set BEFORE any backward is built: the patch latches it into a
    // static on first use. Same call ACE's train-lm/train-dit make for --bwd mm.
    if (!bwd_outprod) {
#ifdef _WIN32
        _putenv("GGML_BACKWARD_MM=1");
#else
        setenv("GGML_BACKWARD_MM", "1", 1);
#endif
    }
    return mm3_train_dit_run(a);
}

static int cmd_train_lm(int argc, char ** argv) {
    LmTrainArgs      a;
    LmResumeExplicit saw;   // which identity flags were typed (resume adopt-or-refuse)
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
        else if (!strcmp(argv[i], "--init-adapter") && i + 1 < argc) a.init_adapter = argv[++i];
        else if (!strcmp(argv[i], "--rank") && i + 1 < argc) { a.rank = atoi(argv[++i]); saw.rank = true; }
        else if (!strcmp(argv[i], "--alpha") && i + 1 < argc) { a.alpha = atoi(argv[++i]); saw.alpha = true; }
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
        else if (!strcmp(argv[i], "--weights") && i + 1 < argc) { a.weights = argv[++i]; saw.weights = true; }
        else if (!strcmp(argv[i], "--optimizer") && i + 1 < argc) a.optimizer = argv[++i];
        else if (!strcmp(argv[i], "--muon-lr-scale") && i + 1 < argc) a.muon_lr_scale = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--muon-momentum") && i + 1 < argc) a.muon_momentum = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--muon-ns-steps") && i + 1 < argc) a.muon_ns_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--muon-min-dim") && i + 1 < argc) a.muon_min_dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--muon-bucket") && i + 1 < argc) a.muon_bucket = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-muon-nesterov")) a.muon_nesterov = false;
        else if (!strcmp(argv[i], "--adapter-type") && i + 1 < argc) { a.adapter_type = argv[++i]; saw.adapter_type = true; }
        else if (!strcmp(argv[i], "--lokr-dim") && i + 1 < argc) { a.lokr_dim = atoi(argv[++i]); saw.lokr_dim = true; }
        else if (!strcmp(argv[i], "--lokr-alpha") && i + 1 < argc) { a.lokr_alpha = (float) atof(argv[++i]); saw.lokr_alpha = true; }
        else if (!strcmp(argv[i], "--lokr-factor") && i + 1 < argc) { a.lokr_factor = atoi(argv[++i]); saw.lokr_factor = true; }
        else if (!strcmp(argv[i], "--no-lokr-decompose-both")) a.lokr_decompose_both = false;
        else if (!strcmp(argv[i], "--bwd") && i + 1 < argc) a.bwd = argv[++i];
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) a.batch = argv[++i];
        else if (!strcmp(argv[i], "--trigger") && i + 1 < argc) a.trigger = argv[++i];
        else if (!strcmp(argv[i], "--trigger-position") && i + 1 < argc) a.trigger_position = argv[++i];
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

    // ── resume (--init-adapter): adopt the source run's identity ─────────
    //
    // BEFORE the sanity/lever checks so they validate the ADOPTED values, and
    // before model resolution so the lm_size refuse below can use the source's
    // recorded size. Explicit CLI contradictions exit 2 inside prepare.
    LmResumeSource resume_src;
    if (!a.init_adapter.empty()) {
        std::string err;
        if (!lm_resume_prepare(&a, saw, &resume_src, &err)) {
            fprintf(stderr, "ace-train train-lm: %s\n", err.c_str());
            return 2;
        }
        a.init_from_loss = resume_src.saved_loss;
        if (!lm_size_arg.empty() && !resume_src.lm_size.empty() && lm_size_arg != resume_src.lm_size) {
            fprintf(stderr,
                    "ace-train train-lm: --init-adapter %s was trained on a %s base but --lm-size says %s — "
                    "a resumed adapter must stay on its own base size\n",
                    a.init_adapter.c_str(), resume_src.lm_size.c_str(), lm_size_arg.c_str());
            return 2;
        }
        fprintf(stderr,
                "[train-lm] resuming %s (%s, saved_loss %.4f @ epoch %d) — identity adopted from its log\n",
                a.init_adapter.c_str(), resume_src.adapter_type.c_str(), resume_src.saved_loss,
                resume_src.saved_epoch);
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
    // "replace" never applies the tag during preprocessing (preprocess-run.h:203),
    // so an adapter trained from it carried no trigger and must not claim one.
    if (!a.trigger_position.empty() && a.trigger_position != "prepend" && a.trigger_position != "append") {
        fprintf(stderr, "ace-train train-lm: --trigger-position must be prepend|append\n");
        return 2;
    }
    if (a.trigger.size() > 128) {
        fprintf(stderr, "ace-train train-lm: --trigger must be 128 characters or fewer\n");
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

    // ── speed levers (2026-07-28 plan §2.1) ─────────────────────────────
    //
    // Every one of these is raised BEFORE any model load, so the server never
    // pays an engine stop/restart for a bad flag.
    if (a.weights != "f32-window" && a.weights != "bf16") {
        fprintf(stderr, "ace-train train-lm: --weights must be f32-window|bf16\n");
        return 2;
    }
    if (!bwd_valid(a.bwd)) {
        fprintf(stderr, "ace-train train-lm: --bwd must be outprod|mm\n");
        return 2;
    }
    // COLLISION, refused rather than coerced.
    //
    // `--weights bf16` (lm-bf16.h, Lever A) reaches the SAME mul_mat backward by
    // a different route: it lets ggml build the out_prod form and then rewrites
    // those nodes in place, asserting EXACTLY 7 rewrites / 0 skipped / 0 residual
    // per segment (its S18 tripwire). `--bwd mm` makes ggml emit mul_mat directly,
    // so the surgery finds nothing to rewrite and the tripwire GGML_ABORTs — which
    // is the tripwire doing its job, but it would abort mid-run, after the model
    // load, rather than here.
    //
    // There is no version of this pair worth building: on the f32-window path
    // `--bwd mm` has nothing to gain either, because the weight it transposes is
    // the F32 window, so the GEMM stays F32/TF32 and the extra `cont` is pure
    // cost. The LM already solved this problem its own way; `--bwd mm` is a win
    // for train-dit, whose bf16 mirror has no such surgery.
    if (a.weights == "bf16" && a.bwd == "mm") {
        fprintf(stderr,
                "ace-train train-lm: --weights bf16 and --bwd mm are two routes to the SAME\n"
                "  mul_mat backward and cannot be combined. --weights bf16 rewrites ggml's out_prod\n"
                "  nodes in place and asserts it found exactly 7 per segment (lm-bf16.h S18); --bwd mm\n"
                "  makes ggml emit mul_mat directly, so that surgery finds nothing and aborts the run.\n"
                "  Use --weights bf16 (it already gives you the BF16 tensor-core backward) or pair\n"
                "  --bwd mm with --weights f32-window.\n");
        return 2;
    }
    int batch_n = 1;
    if (a.batch != "auto") {
        char *     end = nullptr;
        const long bv  = strtol(a.batch.c_str(), &end, 10);
        if (!end || *end != '\0' || bv < 1 || bv > 8) {
            fprintf(stderr, "ace-train train-lm: --batch must be 1..8 or auto\n");
            return 2;
        }
        batch_n = (int) bv;
    }
    // bf16 has no F32 mirror to fall back on, so it is only meaningful under
    // checkpointing. An explicit `--low-vram off` is a direct contradiction,
    // not something to silently override.
    if (a.weights == "bf16" && a.low_vram == "off") {
        fprintf(stderr, "ace-train train-lm: --weights bf16 requires low-VRAM mode "
                        "(the naive path mirrors every weight to F32) — drop --low-vram off\n");
        return 2;
    }
    // LEVER B IS NOT BUILT — see the §6.1 Phase-0 measurement, and read the
    // numbers rather than the headline, because they are more equivocal than a
    // one-word verdict suggests. Batching cannot make the GEMMs cheaper (S7);
    // its whole win is spreading per-micro-step HOST cost over B samples, so the
    // design made it conditional on that cost reaching 10 %. Measured
    // (HOTSTEP_LM_PHASE_TIMER=1, 20 micro-steps, S=2427):
    //
    //   4B  low-VRAM: build 2.35 % + reset 0.01 % + plan 0.20 % + submit 6.75 %
    //                 = 9.31 % — under the bar even counting all of submit.
    //   0.6B low-VRAM: 7.43 + 0.03 + 0.35 + 12.43 = 20.24 %, BUT `submit` is an
    //                 upper bound (async launches absorb GPU wait when the queue
    //                 backs up), so the true figure is bracketed [7.81 %,
    //                 20.24 %] — the bar sits INSIDE the bracket. Inconclusive,
    //                 not a miss.
    //
    // So: definitively closed at 4B (the case that costs real wall clock), and
    // unresolved at 0.6B low-VRAM — which is the narrowest slice there is, since
    // 0.6B runs naive by default and is FASTER that way (257 vs 368 ms). Not
    // worth ~600 lines through the hottest code in the trainer on that basis.
    // Accepting the flag and silently packing batches of 1 is exactly the
    // failure mode S18 forbids, so this refuses loudly instead.
    //
    // THIS CHECK RUNS BEFORE THE --low-vram INTERACTION BELOW, deliberately.
    // `--batch 2 --low-vram off` used to answer "--batch >1 requires low-VRAM
    // mode", which sends the user off to fix something that still would not
    // work. The unconditional refusal is the true reason, so it speaks first.
    if (a.batch == "auto" || batch_n > 1) {
        fprintf(stderr,
                "ace-train train-lm: --batch %s is not supported in this build.\n"
                "  Micro-batching was never written. It cannot make the maths cheaper — it only\n"
                "  spreads per-micro-step host overhead over B samples, and that overhead measures\n"
                "  9.3%% at 4B (under the 10%% bar the design required) and at most 20%% at 0.6B in\n"
                "  low-VRAM mode, which is a mode 0.6B does not need. Use --grad-accum instead — it\n"
                "  changes the effective batch size with none of the padding waste.\n",
                a.batch.c_str());
        return 2;
    }
    // Kept for the day Lever B is built: at that point this is the rule, and
    // until then it is unreachable-by-construction rather than wrong.
    if ((a.batch == "auto" || batch_n > 1) && a.low_vram == "off") {
        fprintf(stderr, "ace-train train-lm: --batch >1 requires low-VRAM mode — drop --low-vram off\n");
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
        // Resume base check: same SIZE is a hard rule (the shapes differ);
        // a different file of the same size (quant/bf16 variant) only warns.
        if (!a.init_adapter.empty() && !resume_src.lm_size.empty()) {
            const std::string token = "-" + resume_src.lm_size + "-";
            if (a.lm_name.find(token) == std::string::npos) {
                fprintf(stderr,
                        "ace-train train-lm: --init-adapter %s was trained on a %s base but --lm resolved to %s — "
                        "a resumed adapter must stay on its own base size\n",
                        a.init_adapter.c_str(), resume_src.lm_size.c_str(), a.lm_name.c_str());
                return 2;
            }
            const std::string src_base = resume_src.lm_path.substr(resume_src.lm_path.find_last_of("/\\") + 1);
            if (!src_base.empty() && src_base != a.lm_name) {
                fprintf(stderr, "[train-lm] note: resuming on %s; the source run used %s (same size — proceeding)\n",
                        a.lm_name.c_str(), src_base.c_str());
            }
        }
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

    // MUST precede every model load and graph build — the ggml patch latches
    // GGML_BACKWARD_MM on the first backward it emits.
    bwd_apply(a.bwd);

    // MANDATORY: ggml_time_ms() divides by an uninitialised frequency otherwise.
    ggml_time_init();
    return lm_train_main(a);
}

// ─── train-dit (plan §2.1) ──────────────────────────────────────────────────

static int cmd_train_dit(int argc, char ** argv) {
    DitTrainArgs      a;
    DitResumeExplicit saw;   // which identity flags were typed (resume adopt-or-refuse)
    std::string  stages_csv, dit_arg;
    bool         safety_user = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--stages") && i + 1 < argc) stages_csv = argv[++i];
        else if (!strcmp(argv[i], "--tensors") && i + 1 < argc) a.tensors_dir = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) a.out_dir = argv[++i];
        else if (!strcmp(argv[i], "--models") && i + 1 < argc) a.models_dir = argv[++i];
        else if (!strcmp(argv[i], "--dit") && i + 1 < argc) dit_arg = argv[++i];
        else if (!strcmp(argv[i], "--init-adapter") && i + 1 < argc) a.init_adapter = argv[++i];
        else if (!strcmp(argv[i], "--adapter-type") && i + 1 < argc) { a.adapter_type = argv[++i]; saw.adapter_type = true; }
        else if (!strcmp(argv[i], "--rank") && i + 1 < argc) { a.rank = atoi(argv[++i]); saw.rank = true; }
        else if (!strcmp(argv[i], "--alpha") && i + 1 < argc) { a.alpha = atoi(argv[++i]); saw.alpha = true; }
        else if (!strcmp(argv[i], "--lokr-dim") && i + 1 < argc) { a.lokr_dim = atoi(argv[++i]); saw.lokr_dim = true; }
        else if (!strcmp(argv[i], "--lokr-alpha") && i + 1 < argc) { a.lokr_alpha = (float) atof(argv[++i]); saw.lokr_alpha = true; }
        else if (!strcmp(argv[i], "--lokr-factor") && i + 1 < argc) { a.lokr_factor = atoi(argv[++i]); saw.lokr_factor = true; }
        else if (!strcmp(argv[i], "--lokr-decompose-both")) a.lokr_decompose_both = true;
        else if (!strcmp(argv[i], "--no-lokr-decompose-both")) a.lokr_decompose_both = false;
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) { a.layers = atoi(argv[++i]); saw.layers = true; }
        else if (!strcmp(argv[i], "--target-mlp")) { a.target_mlp = true; saw.target_mlp = true; }
        else if (!strcmp(argv[i], "--no-target-mlp")) { a.target_mlp = false; saw.target_mlp = true; }
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
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) a.batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ckpt") && i + 1 < argc) a.ckpt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--crop") && i + 1 < argc) a.crop = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--crop-min") && i + 1 < argc) a.crop_min = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--crop-max") && i + 1 < argc) a.crop_max = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vram-reserve-mb") && i + 1 < argc) a.vram_reserve_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vram-safety") && i + 1 < argc) { a.vram_safety = (float) atof(argv[++i]); safety_user = true; }
        else if (!strcmp(argv[i], "--mirror") && i + 1 < argc) a.mirror = argv[++i];
        else if (!strcmp(argv[i], "--bwd") && i + 1 < argc) a.bwd = argv[++i];
        else if (!strcmp(argv[i], "--profile-step") && i + 1 < argc) a.profile_step = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--profile-ops")) a.profile_ops = true;
        else if (!strcmp(argv[i], "--optimizer") && i + 1 < argc) a.optimizer = argv[++i];
        else if (!strcmp(argv[i], "--muon-lr-scale") && i + 1 < argc) a.muon_lr_scale = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--muon-momentum") && i + 1 < argc) a.muon_momentum = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--muon-ns-steps") && i + 1 < argc) a.muon_ns_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--muon-min-dim") && i + 1 < argc) a.muon_min_dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--muon-bucket") && i + 1 < argc) a.muon_bucket = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-muon-nesterov")) a.muon_nesterov = false;
        else if (!strcmp(argv[i], "--trigger") && i + 1 < argc) a.trigger = argv[++i];
        else if (!strcmp(argv[i], "--trigger-position") && i + 1 < argc) a.trigger_position = argv[++i];
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

    // ── resume (--init-adapter): adopt the source run's identity ─────────
    // BEFORE the sanity checks so they validate the ADOPTED values. Explicit
    // CLI contradictions exit 2 inside prepare. Same shape as train-lm's.
    DitResumeSource resume_src;
    if (!a.init_adapter.empty()) {
        std::string err;
        if (!dit_resume_prepare(&a, saw, &resume_src, &err)) {
            fprintf(stderr, "ace-train train-dit: %s\n", err.c_str());
            return 2;
        }
        a.init_from_ma5 = resume_src.saved_ma5;
        fprintf(stderr, "[train-dit] resuming %s (%s, saved ma5 %.4f @ epoch %d) — identity adopted from its log\n",
                a.init_adapter.c_str(), resume_src.adapter_type.c_str(), resume_src.saved_ma5,
                resume_src.saved_epoch);
    }

    // ── numeric sanity (the server clamps these too; be defensive) ───────
    if (a.rank < 1 || a.rank > 256 || a.alpha < 1 || a.alpha > 1024 || a.epochs < 1 || a.epochs > 2000 ||
        a.grad_accum < 1 || a.grad_accum > 64 || a.lr <= 0.0f || a.lr > 1.0f || a.grad_clip < 0.0f ||
        a.grad_clip > 100.0f || a.warmup_ratio < 0.0f || a.warmup_ratio > 0.5f || a.weight_decay < 0.0f ||
        a.weight_decay > 1.0f || a.target_loss < 0.0f || a.target_loss > 20.0f || a.milestone_keep < 0 ||
        a.batch < 1 || a.batch > 16 || a.ckpt < 0 || a.ckpt > 32 ||
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
    // "replace" never applies the tag during preprocessing (preprocess-run.h:203),
    // so an adapter trained from it carried no trigger and must not claim one.
    if (!a.trigger_position.empty() && a.trigger_position != "prepend" && a.trigger_position != "append") {
        fprintf(stderr, "ace-train train-dit: --trigger-position must be prepend|append\n");
        return 2;
    }
    if (a.trigger.size() > 128) {
        fprintf(stderr, "ace-train train-dit: --trigger must be 128 characters or fewer\n");
        return 2;
    }
    if (a.loss_weighting != "none" && a.loss_weighting != "flow_snr") {
        fprintf(stderr, "ace-train train-dit: --loss-weighting must be none|flow_snr\n");
        return 2;
    }
    // Validated HERE (§2.1): dit_train_main keeps a belt-and-braces check, but a
    // typo must not cost a model load first.
    if (a.adapter_type != "lora" && a.adapter_type != "lokr") {
        fprintf(stderr, "ace-train train-dit: --adapter-type must be lora|lokr\n");
        return 2;
    }
    if (a.lokr_dim < 4 || a.lokr_dim > 4096 || a.lokr_alpha < 0.0f || a.lokr_alpha > 8192.0f ||
        (a.lokr_factor != -1 && (a.lokr_factor < 2 || a.lokr_factor > 64))) {
        fprintf(stderr, "ace-train train-dit: --lokr-dim 4-4096, --lokr-alpha 0-8192, --lokr-factor -1 or 2-64\n");
        return 2;
    }
    if (a.mirror != "f32" && a.mirror != "bf16") {
        fprintf(stderr, "ace-train train-dit: --mirror must be f32|bf16\n");
        return 2;
    }
    if (!bwd_valid(a.bwd)) {
        fprintf(stderr, "ace-train train-dit: --bwd must be outprod|mm\n");
        return 2;
    }
    // LoKR's est-vs-peak gap measured ~13 % (the kron-matvec intermediates the
    // fitted arena polynomial never saw — see dit-vram.h's K10 note), so 5 % is
    // not enough margin for it. An explicit --vram-safety always wins.
    if (!safety_user && a.adapter_type == "lokr") {
        a.vram_safety = 0.12f;
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
        // Resume base check: a DiT adapter is per-base (cross-base transfer is
        // its own basin-sensitive minefield — docs/plans lokr-cross-base) so a
        // different base file is a hard refuse, not a warn.
        if (!a.init_adapter.empty() && !resume_src.dit_name.empty() && resume_src.dit_name != a.dit_name) {
            fprintf(stderr,
                    "ace-train train-dit: --init-adapter %s was trained on %s but --dit resolved to %s — "
                    "a resumed adapter must stay on its own base\n",
                    a.init_adapter.c_str(), resume_src.dit_name.c_str(), a.dit_name.c_str());
            return 2;
        }
    }

    // MUST precede every model load and graph build — the ggml patch latches
    // GGML_BACKWARD_MM on the first backward it emits.
    bwd_apply(a.bwd);

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
    if (!strcmp(argv[1], "mm3-encode")) {
        return cmd_mm3_encode(argc - 1, argv + 1);
    }
    if (!strcmp(argv[1], "mm3-preprocess")) {
        return cmd_mm3_preprocess(argc - 1, argv + 1);
    }
    if (!strcmp(argv[1], "mm3-condition")) {
        return cmd_mm3_condition(argc - 1, argv + 1);
    }
    if (!strcmp(argv[1], "mm3-train-dit")) {
        return cmd_mm3_train_dit(argc - 1, argv + 1);
    }
    if (!strcmp(argv[1], "spike")) {
        return cmd_spike(argc - 1, argv + 1);
    }
    fprintf(stderr, "ace-train: unknown subcommand '%s'\n", argv[1]);
    print_usage();
    return 2;
}
