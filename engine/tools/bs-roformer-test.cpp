// bs-roformer-test.cpp: numeric validation of bs-roformer-ggml.h.
//
// Compares the GGML graph against PyTorch reference activations dumped by
// scripts/dump_bs_roformer_goldens.py, stage by stage, so a divergence can be
// localised to a layer instead of only showing up as bad audio.
//
// The goldens .npz is converted to a flat .bin by scripts/goldens_to_bin.py
// (npz parsing in C++ is not worth the code).
//
// Usage:
//   bs-roformer-test <model.gguf> <goldens.bin> [--no-flash]
//
// Part of HOT-Step CPP. MIT license.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bs-roformer-ggml.h"

namespace {

// Flat golden file layout (little-endian):
//   magic "BSRG"           4 bytes
//   int32 T, in_dim, depth, dim, n_bands, n_stems
//   f32   input      [in_dim * T]
//   f32   band_split [dim * n_bands * T]
//   f32   layer_00   [dim * n_bands * T]
//   f32   layer_01   [dim * n_bands * T]
//   f32   layer_last [dim * n_bands * T]
//   f32   final_norm [dim * n_bands * T]
//   f32   mask       [n_stems * in_dim * T]
struct Goldens {
    int32_t            T = 0, in_dim = 0, depth = 0, dim = 0, n_bands = 0, n_stems = 0;
    std::vector<float> input, band_split, layer_00, layer_01, layer_last, final_norm, mask;
};

bool read_block(FILE * f, std::vector<float> & v, size_t n) {
    v.resize(n);
    return fread(v.data(), sizeof(float), n, f) == n;
}

bool load_goldens(const char * path, Goldens & g) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "BSRG", 4) != 0) {
        fprintf(stderr, "%s: bad magic\n", path);
        fclose(f);
        return false;
    }
    int32_t hdr[6];
    if (fread(hdr, sizeof(int32_t), 6, f) != 6) {
        fclose(f);
        return false;
    }
    g.T = hdr[0]; g.in_dim = hdr[1]; g.depth = hdr[2];
    g.dim = hdr[3]; g.n_bands = hdr[4]; g.n_stems = hdr[5];

    const size_t hidden = (size_t) g.dim * g.n_bands * g.T;
    bool ok = read_block(f, g.input, (size_t) g.in_dim * g.T)
           && read_block(f, g.band_split, hidden)
           && read_block(f, g.layer_00, hidden)
           && read_block(f, g.layer_01, hidden)
           && read_block(f, g.layer_last, hidden)
           && read_block(f, g.final_norm, hidden)
           && read_block(f, g.mask, (size_t) g.n_stems * g.in_dim * g.T);
    fclose(f);
    if (!ok) fprintf(stderr, "%s: truncated\n", path);
    return ok;
}

struct Cmp {
    double max_abs = 0.0, mean_abs = 0.0, ref_peak = 0.0, got_peak = 0.0, rel = 0.0;
    size_t worst_idx = 0;
    bool   finite = true;
};

Cmp compare(const std::vector<float> & got, const std::vector<float> & ref) {
    Cmp c;
    const size_t n = ref.size() < got.size() ? ref.size() : got.size();
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(got[i])) c.finite = false;
        double d = std::fabs((double) got[i] - (double) ref[i]);
        if (d > c.max_abs) { c.max_abs = d; c.worst_idx = i; }
        sum += d;
        double a = std::fabs((double) ref[i]);
        if (a > c.ref_peak) c.ref_peak = a;
        double b = std::fabs((double) got[i]);
        if (b > c.got_peak) c.got_peak = b;
    }
    c.mean_abs = n ? sum / (double) n : 0.0;
    c.rel = c.ref_peak > 0.0 ? c.max_abs / c.ref_peak : c.max_abs;
    return c;
}

// Tolerances are per-stage and deliberately loose at the tail.
//
// F32 GGML vs F32 PyTorch across 16 layers of a residual stream accumulates
// genuine reordering drift, and it is much worse for the instrumental
// checkpoint than the vocal one: its activations reach absmax ~321 (vs ~32),
// and final_norm then divides that down by ~90, so a ~1% relative difference
// at layer_last lands intact on the mask.
//
// That is conditioning, not a bug — the same graph on the vocal weights
// matches to 2e-3, and the flash and materialised-F32 paths agree with each
// other to better than they each agree with PyTorch. The checkpoints were
// trained with use_amp=true, so they are not precision-critical at 1e-3.
// A 1% mask error is ~0.09 dB of amplitude — well below audibility.
bool report(const char * label, const Cmp & c, double tol) {
    const bool pass = c.finite && c.rel <= tol;
    printf("  %-12s max %.3e  mean %.3e  ref_pk %.3e  got_pk %.3e  rel %.2e  %s\n",
           label, c.max_abs, c.mean_abs, c.ref_peak, c.got_peak, c.rel,
           !c.finite ? "NON-FINITE" : (pass ? "ok" : "FAIL"));
    return pass;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <model.gguf> <goldens.bin> [--no-flash]\n", argv[0]);
        return 2;
    }
    const char * gguf_path = argv[1];
    const char * gold_path = argv[2];
    bool no_flash = false;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--no-flash") == 0) no_flash = true;
    }

    Goldens g;
    if (!load_goldens(gold_path, g)) return 1;
    printf("goldens: T=%d in_dim=%d depth=%d dim=%d bands=%d stems=%d\n",
           g.T, g.in_dim, g.depth, g.dim, g.n_bands, g.n_stems);

    BsRoformer m;
    if (!bsr_load(&m, gguf_path)) return 1;
    m.use_flash_attn = !no_flash;
    printf("attention: %s\n\n", m.use_flash_attn ? "flash" : "materialised f32");

    if (m.cfg.in_dim != g.in_dim || m.cfg.depth != g.depth ||
        m.cfg.dim != g.dim || m.cfg.n_bands != g.n_bands) {
        fprintf(stderr, "model/goldens mismatch\n");
        return 1;
    }

    const size_t hidden = (size_t) g.dim * g.n_bands * g.T;
    std::vector<float> out((size_t) g.n_stems * g.in_dim * g.T);
    bool all_ok = true;

    struct Stage {
        int                        stage;
        const char *               label;
        const std::vector<float> * ref;
        double                     tol;
    };
    const Stage stages[] = {
        { 0,            "band_split", &g.band_split, 2e-3 },
        { 1,            "layer_00",   &g.layer_00,   5e-3 },
        { 2,            "layer_01",   &g.layer_01,   5e-3 },
        { g.depth,      "layer_last", &g.layer_last, 2e-2 },
        { g.depth + 1,  "final_norm", &g.final_norm, 2e-2 },
    };

    for (const Stage & s : stages) {
        m.debug_stage = s.stage;
        bsr_forward(&m, g.input.data(), g.T, out.data());
        if (m.debug_out.size() != hidden) {
            fprintf(stderr, "  %-12s debug buffer %zu != %zu\n",
                    s.label, m.debug_out.size(), hidden);
            all_ok = false;
            continue;
        }
        all_ok &= report(s.label, compare(m.debug_out, *s.ref), s.tol);
    }

    m.debug_stage = -1;
    bsr_forward(&m, g.input.data(), g.T, out.data());
    // 8e-2 accommodates the instrumental checkpoint's conditioning (see above);
    // the vocal one lands at ~2e-3 against the same bar.
    all_ok &= report("mask", compare(out, g.mask), 8e-2);

    bsr_free(&m);

    // Regression guard: load/free/load again in the same process. The engine
    // runs the vocal then the instrumental checkpoint back to back, and
    // backend_init hands out a refcounted shared singleton — a bsr_free that
    // frees it outright leaves the second load holding a dangling backend and
    // segfaults. A single-model test cannot see that, so do it explicitly.
    {
        printf("\nreload check (shared-backend refcount)...\n");
        BsRoformer m2;
        if (!bsr_load(&m2, gguf_path)) {
            fprintf(stderr, "  reload FAILED — backend teardown is wrong\n");
            return 1;
        }
        std::vector<float> out2((size_t) g.n_stems * g.in_dim * g.T);
        m2.debug_stage = -1;
        bsr_forward(&m2, g.input.data(), g.T, out2.data());
        all_ok &= report("mask(reload)", compare(out2, g.mask), 8e-2);
        bsr_free(&m2);
    }

    printf("\n%s\n", all_ok ? "PASS — GGML matches the PyTorch reference"
                            : "FAIL — see stages above");
    return all_ok ? 0 : 1;
}
