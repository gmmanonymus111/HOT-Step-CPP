// moss-ggml-test.cpp: parity tests for the MOSS-Music GGML port vs fp32 fixtures.
//
// Same shape as sa3-ggml-test.cpp: run a component, compare against a golden
// dump, report correlation + max abs diff, exit nonzero if anything misses.
//
// Usage:
//   moss-ggml-test --fixtures <dir> [--models <dir>] [--component mel|load|all]
//
// SET GGML_BACKEND=CPU FOR PARITY RUNS. backend_init() picks the best device,
// which means a bare invocation quietly takes ~1.6 GB of VRAM just to hold
// weights it never runs a graph on. Parity work does not need the GPU and this
// box shares one card between two agents, so:
//   GGML_BACKEND=CPU moss-ggml-test --fixtures ... --models ...
//
// The fixture dir is produced by capture_fixtures.py and holds one subdir per
// track, each with raw little-endian f32 dumps plus manifest.json:
//   mel.f32                 [128, frames]   from HF WhisperFeatureExtractor
//   encoder_out.f32         [tokens, 1280]
//   encoder_deepstack_N.f32 [tokens, 1280]  encoder layers 8/16/24
//   adapter_out.f32         [tokens, 4096]
//   deepstack_merger_N.f32  [tokens, 4096]
//   logits_last.f32         [151936]
//
// The mel component additionally needs the source audio as 16 kHz mono WAV, in
// <fixtures>/../tracks/wav16k/<track>.wav (the manifest records the stem).
//
// Bars, from the mm3-backend validation policy: >=0.999 correlation per module
// against a bf16 dump, >=0.9999 against an fp32 CPU rerun. These fixtures ARE
// the fp32 rerun, so the bar here is 0.9999 -- except for mel, whose reference
// is stored bf16 (MelConfig.mel_dtype), where 0.9999 is still comfortably met.

// <string> MUST precede backend.h: backend.h uses std::wstring in its Windows
// CUDA-diagnostics path but does not include <string> itself, so it only
// compiles when a prior header happens to pull it in. Keeping the std headers
// first here avoids depending on that accident.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "backend.h"
#include "moss/moss-encoder-graph.h"
#include "moss/moss-lm-graph.h"
#include "moss/moss-mel.h"
#include "moss/moss-model.h"

static const double PASS_CORR = 0.9999;

static bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize((size_t) std::max(0L, n));
    const size_t got = out.empty() ? 0 : fread(out.data(), 1, out.size(), f);
    fclose(f);
    return got == out.size();
}

static bool read_f32(const std::string & path, std::vector<float> & out) {
    std::vector<uint8_t> raw;
    if (!read_file(path, raw)) {
        return false;
    }
    out.resize(raw.size() / 4);
    memcpy(out.data(), raw.data(), out.size() * 4);
    return true;
}

// Minimal 16-bit PCM WAV reader; the fixtures' audio is always 16 kHz mono s16.
static bool read_wav_mono16(const std::string & path, std::vector<float> & pcm,
                            int * sample_rate) {
    std::vector<uint8_t> b;
    if (!read_file(path, b) || b.size() < 44) {
        return false;
    }
    if (memcmp(b.data(), "RIFF", 4) != 0 || memcmp(b.data() + 8, "WAVE", 4) != 0) {
        return false;
    }
    size_t p = 12;
    int channels = 1, bits = 16, sr = 16000;
    while (p + 8 <= b.size()) {
        char id[5] = {0};
        memcpy(id, &b[p], 4);
        uint32_t sz = 0;
        memcpy(&sz, &b[p + 4], 4);
        const size_t body = p + 8;
        if (memcmp(id, "fmt ", 4) == 0 && body + 16 <= b.size()) {
            uint16_t ch = 0, bp = 0;
            uint32_t rate = 0;
            memcpy(&ch, &b[body + 2], 2);
            memcpy(&rate, &b[body + 4], 4);
            memcpy(&bp, &b[body + 14], 2);
            channels = ch ? ch : 1;
            sr = (int) rate;
            bits = bp ? bp : 16;
        } else if (memcmp(id, "data", 4) == 0) {
            if (bits != 16) {
                fprintf(stderr, "moss-ggml-test: only 16-bit PCM WAV supported\n");
                return false;
            }
            const size_t n = std::min((size_t) sz, b.size() - body) / 2;
            pcm.resize(n / (size_t) channels);
            for (size_t i = 0; i < pcm.size(); ++i) {
                int16_t s = 0;
                memcpy(&s, &b[body + (i * (size_t) channels) * 2], 2);
                pcm[i] = (float) s / 32768.0f;
            }
            if (sample_rate) {
                *sample_rate = sr;
            }
            return true;
        }
        p = body + sz + (sz & 1);
    }
    return false;
}

static void stats(const std::vector<float> & ref, const std::vector<float> & got,
                  double * corr, double * maxabs, double * relrmse) {
    const size_t n = std::min(ref.size(), got.size());
    double mr = 0, mg = 0;
    for (size_t i = 0; i < n; ++i) {
        mr += ref[i];
        mg += got[i];
    }
    mr /= (double) n;
    mg /= (double) n;
    double num = 0, dr = 0, dg = 0, se = 0, sr = 0, mx = 0;
    for (size_t i = 0; i < n; ++i) {
        const double a = ref[i] - mr, b = got[i] - mg;
        num += a * b;
        dr += a * a;
        dg += b * b;
        const double d = (double) ref[i] - (double) got[i];
        se += d * d;
        sr += (double) ref[i] * (double) ref[i];
        mx = std::max(mx, std::fabs(d));
    }
    *corr = (dr > 0 && dg > 0) ? num / std::sqrt(dr * dg) : 0.0;
    *maxabs = mx;
    *relrmse = std::sqrt(se / (double) n) / (std::sqrt(sr / (double) n) + 1e-30);
}

// Pull one string field out of manifest.json without a JSON dependency.
static std::string json_str(const std::string & js, const std::string & key) {
    const std::string pat = "\"" + key + "\"";
    size_t p = js.find(pat);
    if (p == std::string::npos) {
        return "";
    }
    p = js.find(':', p + pat.size());
    if (p == std::string::npos) {
        return "";
    }
    p = js.find('"', p);
    if (p == std::string::npos) {
        return "";
    }
    const size_t e = js.find('"', p + 1);
    if (e == std::string::npos) {
        return "";
    }
    return js.substr(p + 1, e - p - 1);
}

static double json_num(const std::string & js, const std::string & key) {
    const std::string pat = "\"" + key + "\"";
    size_t p = js.find(pat);
    if (p == std::string::npos) {
        return 0.0;
    }
    p = js.find(':', p + pat.size());
    return (p == std::string::npos) ? 0.0 : atof(js.c_str() + p + 1);
}

static bool list_dirs(const std::string & root, std::vector<std::string> & out);

#ifdef _WIN32
#  include <windows.h>
static bool list_dirs(const std::string & root, std::vector<std::string> & out) {
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((root + "\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }
    do {
        if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && fd.cFileName[0] != '.') {
            out.emplace_back(fd.cFileName);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    return true;
}
#else
#  include <dirent.h>
#  include <sys/stat.h>
static bool list_dirs(const std::string & root, std::vector<std::string> & out) {
    DIR * d = opendir(root.c_str());
    if (!d) {
        return false;
    }
    while (struct dirent * e = readdir(d)) {
        if (e->d_name[0] == '.') {
            continue;
        }
        struct stat st;
        if (stat((root + "/" + e->d_name).c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            out.emplace_back(e->d_name);
        }
    }
    closedir(d);
    return true;
}
#endif

static bool test_mel(const std::string & fixtures, const std::string & track) {
#ifdef _WIN32
    const char sep = '\\';
#else
    const char sep = '/';
#endif
    const std::string dir = fixtures + sep + track;
    std::vector<uint8_t> mraw;
    if (!read_file(dir + sep + "manifest.json", mraw)) {
        printf("  SKIP %-24s (no manifest.json)\n", track.c_str());
        return true;
    }
    const std::string js((const char *) mraw.data(), mraw.size());
    const std::string stem = json_str(js, "_track");
    const double clip = json_num(js, "_clip_seconds");

    std::vector<float> ref;
    if (!read_f32(dir + sep + "mel.f32", ref)) {
        printf("  SKIP %-24s (no mel.f32)\n", track.c_str());
        return true;
    }

    const std::string wav = fixtures + sep + ".." + sep + "tracks" + sep + "wav16k" +
                            sep + stem + ".wav";
    std::vector<float> pcm;
    int sr = 0;
    if (!read_wav_mono16(wav, pcm, &sr)) {
        printf("  SKIP %-24s (cannot read %s)\n", track.c_str(), wav.c_str());
        return true;
    }
    if (clip > 0) {
        const size_t want = (size_t) (clip * (double) sr);
        if (pcm.size() > want) {
            pcm.resize(want);
        }
    }

    moss::MelParams p;
    int frames = 0;
    std::vector<float> got = moss::log_mel(pcm.data(), pcm.size(), p, &frames);

    const size_t n_mels = (size_t) p.n_mels;
    const size_t ref_frames = ref.size() / n_mels;
    const size_t use = std::min((size_t) frames, ref_frames);
    // Both are row-major [n_mels, frames]; compare the common prefix.
    std::vector<float> a, b;
    a.reserve(n_mels * use);
    b.reserve(n_mels * use);
    for (size_t m = 0; m < n_mels; ++m) {
        for (size_t t = 0; t < use; ++t) {
            a.push_back(ref[m * ref_frames + t]);
            b.push_back(got[m * (size_t) frames + t]);
        }
    }

    double corr = 0, mx = 0, rel = 0;
    stats(a, b, &corr, &mx, &rel);
    const bool ok = corr >= PASS_CORR;
    printf("  %s mel  %-22s ref[%zu,%zu] got[%d,%d]  corr=%.7f relRMSE=%.2e maxabs=%.2e\n",
           ok ? "OK  " : "FAIL", track.c_str(), n_mels, ref_frames, p.n_mels, frames,
           corr, rel, mx);
    return ok;
}

static bool test_load(const std::string & aud_gguf) {
    // backend_init hands out a REFCOUNTED shared backend. Releasing it with
    // ggml_backend_free would leave the cached pointer dangling for the next
    // caller -- see the note in bs-roformer-ggml.h. Always backend_release.
    BackendPair bp = backend_init("MOSS");
    if (!bp.backend) {
        fprintf(stderr, "  FAIL could not init a backend\n");
        return false;
    }
    moss::AudioTower tower;
    bool sane = false;
    if (moss::moss_load_audio_tower(&tower, aud_gguf.c_str(), bp.backend)) {
        moss::moss_print_audio_hparams(tower);
        const moss::AudioHParams & h = tower.hp;
        sane = h.n_layer == 32 && h.d_model == 1280 && h.n_head == 20 &&
               h.n_ffn == 5120 && h.n_mels == 128 && h.deepstack_layers.size() == 3 &&
               tower.blocks.size() == 32 && tower.deepstack.size() == 3 &&
               h.tokens_per_second > 12.49f && h.tokens_per_second < 12.51f;
        printf("  %s load  %zu blocks, %zu deepstack mergers, all tensors resident\n",
               sane ? "OK  " : "FAIL", tower.blocks.size(), tower.deepstack.size());
        moss::moss_free_audio_tower(&tower);
    }
    backend_release(bp.backend, bp.cpu_backend);
    return sane;
}

// Feeds the FIXTURE mel rather than recomputing it, so the encoder is tested in
// isolation from any frontend difference (the stored mel is bf16; recomputing in
// fp32 would inject a ~1 ULP delta that has nothing to do with this graph).
static bool test_encoder(const std::string & fixtures, const std::string & track,
                         const moss::AudioTower & tower, ggml_backend_t backend) {
#ifdef _WIN32
    const char sep = '\\';
#else
    const char sep = '/';
#endif
    const std::string dir = fixtures + sep + track;
    std::vector<float> mel;
    if (!read_f32(dir + sep + "mel.f32", mel)) {
        printf("  SKIP %-24s (no mel.f32)\n", track.c_str());
        return true;
    }
    const int n_mels = (int) tower.hp.n_mels;
    const int T = (int) (mel.size() / (size_t) n_mels);

    moss::EncoderOutput eo;
    if (!moss::moss_encode_audio(tower, mel.data(), n_mels, T, backend, &eo)) {
        printf("  FAIL %-24s (encode failed)\n", track.c_str());
        return false;
    }

    bool ok = true;
    auto cmp = [&](const char * name, const std::vector<float> & got) {
        std::vector<float> ref;
        if (!read_f32(dir + sep + name + ".f32", ref)) {
            printf("  SKIP %-20s %-22s (no fixture)\n", name, track.c_str());
            return;
        }
        if (ref.size() != got.size()) {
            printf("  FAIL %-20s %-22s size ref=%zu got=%zu\n", name, track.c_str(),
                   ref.size(), got.size());
            ok = false;
            return;
        }
        double corr = 0, mx = 0, rel = 0;
        stats(ref, got, &corr, &mx, &rel);
        // The GGUF is f16 against fp32 dumps, so the bar is 0.999 (numpy, reading
        // the same GGUF, reached 1.000000).
        const bool good = corr >= 0.999;
        printf("  %s %-20s %-22s corr=%.7f relRMSE=%.2e maxabs=%.2e\n",
               good ? "OK  " : "FAIL", name, track.c_str(), corr, rel, mx);
        ok = ok && good;
    };

    cmp("encoder_out", eo.encoder_out);
    for (size_t k = 0; k < eo.deepstack_taps.size(); ++k) {
        cmp(("encoder_deepstack_" + std::to_string(k)).c_str(), eo.deepstack_taps[k]);
    }
    cmp("adapter_out", eo.adapter_out);
    for (size_t k = 0; k < eo.merger_out.size(); ++k) {
        cmp(("deepstack_merger_" + std::to_string(k)).c_str(), eo.merger_out[k]);
    }
    return ok;
}

// End-to-end through the audio->LM seam: encoder -> splice + deepstack -> prefill,
// scored against the reference's last-position logits. This is the check that
// catches a mis-wired deepstack, which is the most likely silent break in the port
// (mergers come from ENCODER layers 8/16/24 and land in LM layers 0/1/2).
static bool test_logits(const std::string & fixtures, const std::string & track,
                        const moss::AudioTower & tower, const moss::LmModel & lm,
                        ggml_backend_t backend) {
#ifdef _WIN32
    const char sep = '\\';
#else
    const char sep = '/';
#endif
    const std::string dir = fixtures + sep + track;

    std::vector<float> mel, ids_f, ref;
    if (!read_f32(dir + sep + "mel.f32", mel) ||
        !read_f32(dir + sep + "input_ids.f32", ids_f) ||
        !read_f32(dir + sep + "logits_last.f32", ref)) {
        printf("  SKIP %-24s (missing mel/input_ids/logits_last)\n", track.c_str());
        return true;
    }

    const int n_mels = (int) tower.hp.n_mels;
    const int T_mel = (int) (mel.size() / (size_t) n_mels);
    moss::EncoderOutput eo;
    if (!moss::moss_encode_audio(tower, mel.data(), n_mels, T_mel, backend, &eo)) {
        printf("  FAIL %-24s (encode failed)\n", track.c_str());
        return false;
    }

    const int64_t T = (int64_t) ids_f.size();
    const int64_t H = (int64_t) lm.hp.n_embd;
    std::vector<int32_t> ids((size_t) T);
    std::vector<float> text_mask((size_t) T, 1.0f);
    std::vector<float> audio_vals((size_t) (H * T), 0.0f);
    std::vector<std::vector<float>> merge(eo.merger_out.size(),
                                          std::vector<float>((size_t) (H * T), 0.0f));

    int64_t seen = 0;
    for (int64_t i = 0; i < T; ++i) {
        ids[(size_t) i] = (int32_t) ids_f[(size_t) i];
        if ((uint32_t) ids[(size_t) i] == tower.hp.audio_token_id) {
            text_mask[(size_t) i] = 0.0f;
            if (seen < (int64_t) eo.n_tokens) {
                memcpy(&audio_vals[(size_t) (i * H)],
                       &eo.adapter_out[(size_t) (seen * H)], (size_t) H * sizeof(float));
                for (size_t k = 0; k < merge.size(); ++k) {
                    memcpy(&merge[k][(size_t) (i * H)],
                           &eo.merger_out[k][(size_t) (seen * H)], (size_t) H * sizeof(float));
                }
            }
            ++seen;
        }
    }
    if (seen != eo.n_tokens) {
        printf("  FAIL %-24s: %lld audio slots in ids but encoder made %d tokens\n",
               track.c_str(), (long long) seen, eo.n_tokens);
        return false;
    }

    std::vector<float> got;
    if (!moss::moss_lm_prefill_logits(lm, ids, audio_vals, merge, text_mask, backend, &got)) {
        printf("  FAIL %-24s (prefill failed)\n", track.c_str());
        return false;
    }

    double corr = 0, mx = 0, rel = 0;
    stats(ref, got, &corr, &mx, &rel);
    // argmax agreement is the thing that actually decides the first emitted token.
    size_t ai = 0, bi = 0;
    for (size_t i = 1; i < ref.size(); ++i) {
        if (ref[i] > ref[ai]) ai = i;
    }
    for (size_t i = 1; i < got.size(); ++i) {
        if (got[i] > got[bi]) bi = i;
    }
    const bool good = corr >= 0.999 && ai == bi;
    printf("  %s logits_last %-22s corr=%.7f relRMSE=%.2e argmax ref=%zu got=%zu%s\n",
           good ? "OK  " : "FAIL", track.c_str(), corr, rel, ai, bi,
           ai == bi ? "" : "  ** ARGMAX DIFFERS **");
    return good;
}

int main(int argc, char ** argv) {
    std::string fixtures, component = "all", models;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--fixtures" && i + 1 < argc) {
            fixtures = argv[++i];
        } else if (a == "--component" && i + 1 < argc) {
            component = argv[++i];
        } else if (a == "--models" && i + 1 < argc) {
            models = argv[++i];
        } else {
            fprintf(stderr, "usage: moss-ggml-test --fixtures <dir> [--models <dir>] "
                            "[--component mel|load|all]\n");
            return 2;
        }
    }
    if (fixtures.empty()) {
        fprintf(stderr, "usage: moss-ggml-test --fixtures <dir> [--models <dir>] "
                        "[--component mel|load|all]\n");
        return 2;
    }

    std::vector<std::string> tracks;
    if (!list_dirs(fixtures, tracks) || tracks.empty()) {
        fprintf(stderr, "moss-ggml-test: no track subdirs under %s\n", fixtures.c_str());
        return 2;
    }
    std::sort(tracks.begin(), tracks.end());

    bool ok = true;
    if (component == "mel" || component == "all") {
        printf("== mel frontend (bar: corr >= %.4f) ==\n", PASS_CORR);
        for (const std::string & t : tracks) {
            ok = test_mel(fixtures, t) && ok;
        }
    }
    if (component == "load" || component == "all") {
        if (models.empty()) {
            printf("\n== loader == SKIP (pass --models <dir with moss-aud-f16.gguf>)\n");
        } else {
#ifdef _WIN32
            const char sep = '\\';
#else
            const char sep = '/';
#endif
            printf("\n== audio tower loader ==\n");
            ok = test_load(models + sep + "moss-aud-f16.gguf") && ok;
        }
    }
    if (component == "encoder" || component == "all") {
        if (models.empty()) {
            printf("\n== encoder == SKIP (pass --models <dir with moss-aud-f16.gguf>)\n");
        } else {
#ifdef _WIN32
            const char sep = '\\';
#else
            const char sep = '/';
#endif
            printf("\n== encoder graph (bar: corr >= 0.999) ==\n");
            BackendPair bp = backend_init("MOSS");
            moss::AudioTower tower;
            if (bp.backend &&
                moss::moss_load_audio_tower(&tower, (models + sep + "moss-aud-f16.gguf").c_str(),
                                            bp.backend)) {
                for (const std::string & t : tracks) {
                    ok = test_encoder(fixtures, t, tower, bp.backend) && ok;
                }
                moss::moss_free_audio_tower(&tower);
            } else {
                printf("  FAIL could not load the audio tower\n");
                ok = false;
            }
            backend_release(bp.backend, bp.cpu_backend);
        }
    }
    if (component == "logits" || component == "all") {
        if (models.empty()) {
            printf("\n== logits == SKIP (pass --models <dir with both GGUFs>)\n");
        } else {
#ifdef _WIN32
            const char sep = '\\';
#else
            const char sep = '/';
#endif
            printf("\n== audio->LM seam: prefill logits (bar: corr >= 0.999 AND argmax match) ==\n");
            BackendPair bp = backend_init("MOSS");
            moss::AudioTower tower;
            moss::LmModel lm;
            const std::string aud = models + sep + "moss-aud-f16.gguf";
            std::string lmf = models + sep + "moss-lm-q8_0.gguf";
            {   // fall back to an f16 LM if that is what is on disk
                FILE * probe = fopen(lmf.c_str(), "rb");
                if (probe) {
                    fclose(probe);
                } else {
                    lmf = models + sep + "moss-lm-f16.gguf";
                }
            }
            if (bp.backend &&
                moss::moss_load_audio_tower(&tower, aud.c_str(), bp.backend) &&
                moss::moss_load_lm(&lm, lmf.c_str(), bp.backend)) {
                moss::moss_print_lm_hparams(lm);
                for (const std::string & t : tracks) {
                    ok = test_logits(fixtures, t, tower, lm, bp.backend) && ok;
                }
            } else {
                printf("  FAIL could not load the audio tower and/or the LM\n");
                ok = false;
            }
            moss::moss_free_lm(&lm);
            moss::moss_free_audio_tower(&tower);
            backend_release(bp.backend, bp.cpu_backend);
        }
    }
    printf("\n%s\n", ok ? "ALL COMPONENTS PASSED" : "** PARITY FAILURE **");
    return ok ? 0 : 1;
}
