#pragma once
// spike-run.h — `ace-train spike` subcommand dispatch (Phase-0 evidence).

#include "spike-bf16layer.h"
#include "spike-dit2.h"
#include "spike-gemmbench.h"
#include "spike-s1.h"
#include "spike-s2.h"
#include "spike-s3.h"
#include "spike-s4.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>

static void spike_usage(void) {
    fprintf(stderr,
            "ace-train spike <rung> [options]\n"
            "  s1    toy LoRA convergence + finite-difference gradient check\n"
            "  s2    one real Qwen3 layer: fwd+bwd+AdamW step   --lm <gguf|dir>\n"
            "  s3    full LM LoRA micro-train                    --lm <gguf|dir> [--steps N] [--seq N]\n"
            "  s4    one DiT decoder block                       --dit <gguf> [--sample <dir>]\n"
            "  dit2  ROUND 2: full 32-layer UNFUSED DiT LoRA micro-train on a real crop\n"
            "        --dit <gguf> --sample <safetensors> --stub <path>\n"
            "        [--crop 750] [--crop-start 0] [--rank 16] [--alpha 32] [--steps 48]\n"
            "        [--layer-lo 0] [--lr 1e-4] [--grad-clip 1.0] [--seed 42]\n"
            "        [--mu -0.4] [--sigma 1.0] [--cfg-ratio 0.15] [--fixed-t <t>]\n"
            "        [--sweep] [--mirror-ca-kv]\n"
            "  gemmbench  BF16-GEMM lever: time the real 4B trainer GEMM shapes\n"
            "        F32/TF32 vs BF16 vs transpose-route   [--seq N] [--reps N]\n"
            "  bf16layer  P2 prototype on ONE REAL layer: F32-window vs backward surgery\n"
            "        --lm <gguf|dir> [--seq N] [--rank N] [--layer N] [--reps N]\n");
}

static int cmd_spike(int argc, char ** argv) {
    if (argc < 2) {
        spike_usage();
        return 2;
    }
    ggml_time_init();  // ggml_time_ms() divides by timer_freq; 0 until this runs
    const char * rung = argv[1];

    std::string lm_path, dit_path, sample_dir, jsonl;
    int  seq = 256, rank = 16, steps = 20, row = 0, chunk = 512;
    int  reps = 0;  // gemmbench: 0 = auto-scale by shape
    bool no_cast = false, chunked = false;
    D2Args d2;
    d2.rank = 16;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--stub") && i + 1 < argc)        d2.stub_path = argv[++i];
        else if (!strcmp(argv[i], "--export") && i + 1 < argc) d2.export_dir = argv[++i];
        else if (!strcmp(argv[i], "--crop") && i + 1 < argc)   d2.crop = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--crop-start") && i + 1 < argc) d2.crop_start = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--alpha") && i + 1 < argc)  d2.alpha = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layer-lo") && i + 1 < argc) d2.layer_lo = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i + 1 < argc)     d2.lr = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--grad-clip") && i + 1 < argc) d2.grad_clip = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)   d2.seed = (uint64_t) atoll(argv[++i]);
        else if (!strcmp(argv[i], "--mu") && i + 1 < argc)     d2.mu = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--sigma") && i + 1 < argc)  d2.sigma = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--cfg-ratio") && i + 1 < argc) d2.cfg_ratio = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--fixed-t") && i + 1 < argc) d2.fixed_t = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--sweep"))                  d2.sweep = true;
        else if (!strcmp(argv[i], "--control"))                { d2.sweep = true; d2.control = true; }
        else if (!strcmp(argv[i], "--mirror-ca-kv"))           d2.mirror_ca_kv = true;
        else if (!strcmp(argv[i], "--shuffle-target"))         d2.shuffle_target = true;  // VERIFIER ADDITION
        else if (!strcmp(argv[i], "--merge-adapter"))          d2.merge_adapter = true;   // VERIFIER ADDITION
        else if (!strcmp(argv[i], "--lm") && i + 1 < argc)     lm_path = argv[++i];
        else if (!strcmp(argv[i], "--dit") && i + 1 < argc)    dit_path = argv[++i];
        else if (!strcmp(argv[i], "--sample") && i + 1 < argc) sample_dir = argv[++i];
        else if (!strcmp(argv[i], "--codes") && i + 1 < argc)  jsonl = argv[++i];
        else if (!strcmp(argv[i], "--row") && i + 1 < argc)    row = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq") && i + 1 < argc)    seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rank") && i + 1 < argc)   rank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc)  steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunk") && i + 1 < argc)  chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc)   reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunked"))                chunked = true;
        else if (!strcmp(argv[i], "--no-cast"))                no_cast = true;
        else { fprintf(stderr, "ace-train spike: unknown option '%s'\n", argv[i]); return 2; }
    }
    (void) dit_path; (void) sample_dir;

    if (!strcmp(rung, "bf16layer")) {
        if (lm_path.empty()) { fprintf(stderr, "spike bf16layer: --lm required\n"); return 2; }
        return spike_bf16layer(lm_path.c_str(), seq == 256 ? 1024 : seq, rank, row, reps > 0 ? reps : 20);
    }
    if (!strcmp(rung, "gemmbench")) {
        // `--seq 0` (the default here) means "sweep 512/2048/2944".
        return spike_gemmbench(seq == 256 ? 0 : (int64_t) seq, reps);
    }
    if (!strcmp(rung, "s1")) {
        return spike_s1();
    }
    if (!strcmp(rung, "s2")) {
        if (lm_path.empty()) { fprintf(stderr, "spike s2: --lm required\n"); return 2; }
        return spike_s2(lm_path.c_str(), seq, rank, !no_cast);
    }
    if (!strcmp(rung, "s3")) {
        if (lm_path.empty() || jsonl.empty()) { fprintf(stderr, "spike s3: --lm and --codes required\n"); return 2; }
        return spike_s3(lm_path.c_str(), jsonl.c_str(), row, seq, rank, steps, chunk, chunked);
    }
    if (!strcmp(rung, "s4")) {
        if (dit_path.empty() || sample_dir.empty()) { fprintf(stderr, "spike s4: --dit and --sample required\n"); return 2; }
        return spike_s4(dit_path.c_str(), sample_dir.c_str(), seq, chunk, rank);
    }
    if (!strcmp(rung, "dit2")) {
        if (dit_path.empty() || sample_dir.empty()) {
            fprintf(stderr, "spike dit2: --dit and --sample required\n");
            return 2;
        }
        if (d2.stub_path.empty()) { fprintf(stderr, "spike dit2: --stub <path> required\n"); return 2; }
        d2.dit_path    = dit_path;
        d2.sample_path = sample_dir;
        d2.rank        = rank;
        d2.steps       = steps;
        return spike_dit2(d2);
    }
    spike_usage();
    return 2;
}
