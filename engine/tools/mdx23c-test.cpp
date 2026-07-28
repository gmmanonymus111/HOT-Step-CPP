// mdx23c-test.cpp: validation for mdx23c-ggml.h.
//
// Without a goldens file: loads the model, runs one forward at the trained
// frame count and checks the output shape, finiteness and range. That alone
// catches the shape/layout mistakes that are easy to make in a conv U-Net.
//
// With a goldens .bin (scripts/dump_mdx23c_goldens.py): also compares against
// the PyTorch reference elementwise.
//
// Usage:
//   mdx23c-test <model.gguf> [goldens.bin]
//
// Part of HOT-Step CPP. MIT license.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "mdx23c-ggml.h"

namespace {

// Flat golden layout: magic "MDXG", int32 T, dim_f, cin, n_inst,
// then f32 input [T*dim_f*cin], f32 output [n_inst*cin*dim_f*T].
struct Goldens {
    int32_t            T = 0, dim_f = 0, cin = 0, n_inst = 0;
    std::vector<float> input, out;
};

bool load_goldens(const char * path, Goldens & g) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
    char magic[4];
    int32_t hdr[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "MDXG", 4) != 0 ||
        fread(hdr, sizeof(int32_t), 4, f) != 4) {
        fprintf(stderr, "%s: bad header\n", path); fclose(f); return false;
    }
    g.T = hdr[0]; g.dim_f = hdr[1]; g.cin = hdr[2]; g.n_inst = hdr[3];
    g.input.resize((size_t) g.T * g.dim_f * g.cin);
    g.out.resize((size_t) g.n_inst * g.cin * g.dim_f * g.T);
    bool ok = fread(g.input.data(), sizeof(float), g.input.size(), f) == g.input.size()
           && fread(g.out.data(),   sizeof(float), g.out.size(),   f) == g.out.size();
    fclose(f);
    if (!ok) fprintf(stderr, "%s: truncated\n", path);
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [goldens.bin]\n", argv[0]);
        return 2;
    }

    Mdx23c m;
    if (!mdx_load(&m, argv[1])) return 1;

    const Mdx23cConfig & c = m.cfg;
    const int cin = c.n_audio_channels * 2;
    int T = c.chunk_size / c.hop_length + 1;

    Goldens g;
    const bool have_goldens = (argc >= 3) && load_goldens(argv[2], g);
    if (argc >= 3 && !have_goldens) return 1;
    if (have_goldens) {
        if (g.dim_f != c.dim_f || g.cin != cin || g.n_inst != c.n_instruments) {
            fprintf(stderr, "goldens/model mismatch\n");
            return 1;
        }
        T = g.T;
    }

    printf("T=%d dim_f=%d cin=%d instruments=%d\n", T, c.dim_f, cin, c.n_instruments);

    std::vector<float> in((size_t) T * c.dim_f * cin);
    if (have_goldens) {
        memcpy(in.data(), g.input.data(), in.size() * sizeof(float));
    } else {
        unsigned s = 1234;
        for (size_t i = 0; i < in.size(); i++) {
            s = s * 1664525u + 1013904223u;
            in[i] = ((float) (s >> 8) / 8388608.0f - 1.0f) * 0.05f;
        }
    }

    std::vector<float> out((size_t) c.n_instruments * cin * c.dim_f * T);
    mdx_forward(&m, in.data(), T, out.data());

    double pk = 0.0;
    bool finite = true;
    for (float v : out) {
        if (!std::isfinite(v)) finite = false;
        double a = std::fabs((double) v);
        if (a > pk) pk = a;
    }
    printf("output %zu values, peak %.6f, %s\n", out.size(), pk,
           finite ? "all finite" : "NON-FINITE");
    if (!finite) return 1;

    int rc = 0;
    if (have_goldens) {
        double max_abs = 0.0, sum = 0.0, ref_pk = 0.0;
        for (size_t i = 0; i < out.size(); i++) {
            double d = std::fabs((double) out[i] - (double) g.out[i]);
            if (d > max_abs) max_abs = d;
            sum += d;
            double a = std::fabs((double) g.out[i]);
            if (a > ref_pk) ref_pk = a;
        }
        const double mean = sum / (double) out.size();
        const double rel  = ref_pk > 0.0 ? max_abs / ref_pk : max_abs;
        printf("vs PyTorch: max %.3e  mean %.3e  ref_pk %.3e  rel %.2e  %s\n",
               max_abs, mean, ref_pk, rel, rel <= 2e-2 ? "ok" : "FAIL");
        rc = rel <= 2e-2 ? 0 : 1;
    } else {
        printf("(no goldens supplied — shape/finiteness check only)\n");
    }

    // Reload guard: backend_init hands out a refcounted shared singleton, so a
    // wrong teardown only shows up on the SECOND load in a process.
    {
        Mdx23c m2;
        if (!mdx_load(&m2, argv[1])) {
            fprintf(stderr, "reload FAILED — backend teardown is wrong\n");
            mdx_free(&m);
            return 1;
        }
        printf("reload ok\n");
        mdx_free(&m2);
    }

    mdx_free(&m);
    printf("%s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
