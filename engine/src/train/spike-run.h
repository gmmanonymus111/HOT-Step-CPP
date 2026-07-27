#pragma once
// spike-run.h — `ace-train spike` subcommand dispatch (Phase-0 evidence).

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
            "  s4    one DiT decoder block                       --dit <gguf> [--sample <dir>]\n");
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
    bool no_cast = false, chunked = false;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--lm") && i + 1 < argc)          lm_path = argv[++i];
        else if (!strcmp(argv[i], "--dit") && i + 1 < argc)    dit_path = argv[++i];
        else if (!strcmp(argv[i], "--sample") && i + 1 < argc) sample_dir = argv[++i];
        else if (!strcmp(argv[i], "--codes") && i + 1 < argc)  jsonl = argv[++i];
        else if (!strcmp(argv[i], "--row") && i + 1 < argc)    row = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq") && i + 1 < argc)    seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rank") && i + 1 < argc)   rank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc)  steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunk") && i + 1 < argc)  chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunked"))                chunked = true;
        else if (!strcmp(argv[i], "--no-cast"))                no_cast = true;
        else { fprintf(stderr, "ace-train spike: unknown option '%s'\n", argv[i]); return 2; }
    }
    (void) dit_path; (void) sample_dir;

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
    spike_usage();
    return 2;
}
