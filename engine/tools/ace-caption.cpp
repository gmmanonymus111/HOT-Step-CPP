// ace-caption.cpp: MOSS-Music captioning CLI. Audio in, caption out.
//
// Usage:
//   ace-caption --models <dir> --src-audio <file> [--mode prose|mm3|lyrics[,...]]
//               [--prompt "..."] [--prompt-file [<mode>=]<path>]... [-o <out.txt>]
//               [--max-tokens N] [--max-seconds S]
//               [--temperature T] [--rep-penalty R] [--freq-penalty F]
//
// Shaped like ace-understand.cpp (audio -> metadata/lyrics), which it
// complements: ace-understand goes through ACE's own VAE+FSQ+LM, this one runs
// the MOSS-Music tower.
//
// ── Prompt sources, and why --prompt-file exists ─────────────────────────────
//
// The ACE-Step 1.5 caption format is defined by CAPTION_SENTENCE_PLAN in
// server/src/services/training/captionPrompt.ts, which is byte-identical to
// Side-Step's caption_config.py so a caption produced here is indistinguishable
// from one Side-Step produced. Duplicating that wording in C++ would guarantee
// it drifts. So the built-in modes cover prose/mm3/lyrics, and the server passes
// the AS1.5 prompt through --prompt-file, keeping one source of truth.
//
// --prompt-file is repeatable and takes an optional `<mode>=` prefix. A bare path
// overrides every mode; `prose=<path>` overrides just that one, leaving the other
// modes on their built-ins. That is what lets a single run emit the AS1.5 block
// AND an MM3 Structured Caption off one encode — the Training Studio's whole ask.
// A bare Windows path still parses as the blanket form: the split only happens
// when the text before the first '=' is literally prose, mm3 or lyrics.
//
// ── Cost shape ───────────────────────────────────────────────────────────────
//
// The encoder is prompt-independent: mel + 32 Whisper layers depend only on the
// audio. Only the LM sees the prompt. So --mode takes a COMMA-SEPARATED list and
// runs one encode followed by N decodes:
//
//   ace-caption --mode prose,mm3,lyrics -o out.txt
//     -> out.prose.txt, out.mm3.txt, out.lyrics.txt
//
// On a 4-minute track the encode is the expensive half, so asking for all three
// together is far cheaper than three invocations. This is the shape the Training
// Studio pipeline should use: dataset tracks want the AS1.5 caption AND the MM3
// Structured Caption, and there is no reason to pay for the audio twice.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "audio-io.h"
#include "audio-resample.h"
#include "backend.h"
#include "bpe.h"
#include "moss/moss-encoder-graph.h"
#include "moss/moss-lm-graph.h"
#include "moss/moss-mel.h"
#include "moss/moss-model.h"

static const char * PROMPT_PROSE =
    "Analyse this music track and describe it in detail: genre and energy, the drum and "
    "percussion pattern, bass, harmony and lead instrumentation, vocal style and timbre, "
    "production and mix character, and how the arrangement develops from intro to ending. "
    "State the tempo in BPM, the key and scale, and the time signature. Be specific and "
    "concrete about what you actually hear.";

// Plain-text section labels, NOT markdown headings: the 1000 official templates
// in .claude/skills/mm3-captioning/upstream/templates/ use bare labels, and MM3
// was trained on that. Lyrics are forbidden inside an MM3 caption.
static const char * PROMPT_MM3 =
    "Listen to this track and write a Structured Caption in exactly the format below.\n\n"
    "Use these three section labels on their own lines, with no markdown, no '#', and no "
    "extra sections:\n\n"
    "Global Metadata\n"
    "Basic Attributes: bpm is <N>. key is <X>, and scale is <major|minor>. <Genre / Subgenre>.\n"
    "Global Emotional Progression: <how the emotional arc moves across the song>\n"
    "Application Scenarios & Imagery: <where this music would be used; concrete visual imagery>\n"
    "Sonics & Production Profile: <soundstage, density, frequency balance, dynamics, mix character>\n"
    "Vocal Details\n"
    "Vocal Gender & Timbre: <Singer A (Male/Female). timbre, register, texture>\n"
    "Vocal Style: <delivery and how it evolves across sections>\n"
    "Harmony/Backing Vocals: <layering, doubling, gang vocals, call-and-response, or state none>\n"
    "Vocal FX: <reverb, delay, distortion, pitch correction, or state minimal>\n"
    "Arrangement\n"
    "Instrument Lifecycle Description (Primary/Secondary Layering):\n"
    "Primary: <the instruments carrying the track and how they change>\n"
    "Secondary: <supporting instruments and when they enter or drop>\n"
    "Groove & Foundation Progression: <rhythm section, feel changes, section by section>\n"
    "Embellishments, Textures & Spatial FX: <risers, sweeps, reverse reverb, glitches, ambience>\n\n"
    "Rules:\n"
    "- Describe only what you actually hear. Do not invent an exact BPM or key if unsure.\n"
    "- If the track is instrumental, say so under Vocal Details and name the lead instrument.\n"
    "- Do NOT quote or summarise the lyrics anywhere.\n"
    "- Write flowing prose inside each field.";

static const char * PROMPT_LYRICS =
    "Transcribe the complete lyrics of this song.\n\n"
    "Mark every section with a bracketed tag on its own line: [Intro], [Verse], "
    "[Pre-Chorus], [Chorus], [Bridge], [Breakdown], [Instrumental], [Outro]. Put the sung "
    "words underneath each tag, one line per sung line. Use [Instrumental] for passages "
    "with no vocals. Transcribe only what is actually sung; do not invent lines, translate, "
    "or add commentary.\n\n"
    "If the track has no vocals at all, reply with exactly: [Instrumental]";

static bool ends_with_ci(const std::string & s, const char * suf) {
    const size_t n = strlen(suf);
    if (s.size() < n) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        if (tolower((unsigned char) s[s.size() - n + i]) != tolower((unsigned char) suf[i])) {
            return false;
        }
    }
    return true;
}

// audio-io.h reads WAV and MP3 only. Everything else goes through ffmpeg, which
// is the same contract ace-train uses ("--ffmpeg <path> required for non-WAV/MP3
// input") -- worth matching rather than vendoring a second decode path, since
// dataset tracks are overwhelmingly FLAC.
static bool transcode_to_wav16k(const std::string & ffmpeg, const std::string & src,
                                std::string & out_path) {
    char tmpl[L_tmpnam_s ? L_tmpnam_s : 260];
#ifdef _WIN32
    if (tmpnam_s(tmpl, sizeof(tmpl)) != 0) {
        return false;
    }
#else
    if (!tmpnam(tmpl)) {
        return false;
    }
#endif
    out_path = std::string(tmpl) + ".wav";
    const std::string cmd = "\"\"" + ffmpeg + "\" -y -loglevel error -i \"" + src +
                            "\" -ac 1 -ar 16000 -c:a pcm_s16le \"" + out_path + "\"\"";
    const int rc = system(cmd.c_str());
    if (rc != 0) {
        fprintf(stderr, "ace-caption: ffmpeg failed (%d) on %s\n", rc, src.c_str());
        return false;
    }
    return true;
}

static bool read_text_file(const std::string & path, std::string & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize((size_t) std::max(0L, n));
    const size_t got = out.empty() ? 0 : fread(&out[0], 1, out.size(), f);
    fclose(f);
    return got == out.size();
}

static void push_text(const BPETokenizer & tok, const std::string & s,
                      std::vector<int32_t> & ids) {
    for (int id : bpe_encode(&tok, s, false)) {
        ids.push_back((int32_t) id);
    }
}

static bool push_special(const BPETokenizer & tok, const char * s,
                         std::vector<int32_t> & ids) {
    auto it = tok.vocab.find(s);
    if (it == tok.vocab.end()) {
        fprintf(stderr, "ace-caption: tokenizer has no '%s'\n", s);
        return false;
    }
    ids.push_back((int32_t) it->second);
    return true;
}

// GPT-2 byte-level decode: map each token's UTF-8 surrogate chars back to bytes.
static std::string detokenize(const BPETokenizer & tok, const std::vector<int32_t> & ids) {
    std::unordered_map<std::string, unsigned char> str2byte;
    for (int b = 0; b < 256; ++b) {
        str2byte[tok.byte2str[b]] = (unsigned char) b;
    }
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || (size_t) id >= tok.id_to_str.size()) {
            continue;
        }
        const std::string & piece = tok.id_to_str[(size_t) id];
        // Walk UTF-8 codepoints; each maps back to exactly one byte.
        size_t i = 0;
        while (i < piece.size()) {
            int adv = 1;
            utf8_codepoint(piece.c_str() + i, &adv);
            const std::string ch = piece.substr(i, (size_t) adv);
            auto it = str2byte.find(ch);
            if (it != str2byte.end()) {
                out.push_back((char) it->second);
            } else {
                out += ch;  // a special token or something unmapped: pass through
            }
            i += (size_t) adv;
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    std::string models, src, mode = "prose", prompt, out_path, ffmpeg;
    std::vector<std::string> prompt_files;   // repeatable; see the per-mode note below
    int max_tokens = 1024;
    int top_k_debug = 0;
    double max_seconds = 420.0;   // hard cap, see below
    moss::SamplerParams sp;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "ace-caption: %s needs a value\n", what);
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--models") { models = next("--models"); }
        else if (a == "--src-audio") { src = next("--src-audio"); }
        else if (a == "--mode") { mode = next("--mode"); }
        else if (a == "--prompt") { prompt = next("--prompt"); }
        else if (a == "--prompt-file") { prompt_files.push_back(next("--prompt-file")); }
        else if (a == "-o") { out_path = next("-o"); }
        else if (a == "--ffmpeg") { ffmpeg = next("--ffmpeg"); }
        else if (a == "--top-k-debug") { top_k_debug = atoi(next("--top-k-debug").c_str()); }
        else if (a == "--max-tokens") { max_tokens = atoi(next("--max-tokens").c_str()); }
        else if (a == "--max-seconds") { max_seconds = atof(next("--max-seconds").c_str()); }
        else if (a == "--temperature") { sp.temperature = (float) atof(next("--temperature").c_str()); }
        else if (a == "--rep-penalty") { sp.repetition_penalty = (float) atof(next("--rep-penalty").c_str()); }
        else if (a == "--freq-penalty") { sp.frequency_penalty = (float) atof(next("--freq-penalty").c_str()); }
        else {
            fprintf(stderr,
                    "usage: ace-caption --models <dir> --src-audio <file>\n"
                    "       [--mode prose|mm3|lyrics[,...]] [--prompt \"...\"]\n"
                    "       [--prompt-file [prose=|mm3=|lyrics=]<path>]   (repeatable)\n"
                    "       [-o <out.txt>] [--max-tokens N] [--max-seconds S]\n"
                    "       [--temperature T] [--rep-penalty R] [--freq-penalty F]\n");
            return 2;
        }
    }
    if (models.empty() || src.empty()) {
        fprintf(stderr, "ace-caption: --models and --src-audio are required\n");
        return 2;
    }

    // --prompt / --prompt-file override every mode; otherwise each mode picks its
    // own built-in prompt. --modes takes a comma-separated list so all formats
    // come off ONE encode (see the cost-shape note at the top).
    //
    // --prompt-file also takes a PER-MODE form, `--prompt-file prose=<path>`,
    // which is what makes "one encode, two different caption formats" possible.
    // The Training Studio needs Side-Step's 5-line `caption:/genre:/bpm:/key:/
    // signature:` block AND an MM3 Structured Caption from the same audio; with
    // only the blanket override those are two full encodes, and the encode is
    // the expensive half of the run.
    std::string prompt_override = prompt;
    std::unordered_map<std::string, std::string> mode_prompt;
    for (const std::string & spec : prompt_files) {
        // Split on the FIRST '=' only when the left side names a mode, so a
        // bare Windows path (`D:\x\p.txt`) still parses as the blanket form.
        const size_t eq = spec.find('=');
        const std::string head = eq == std::string::npos ? std::string() : spec.substr(0, eq);
        const bool per_mode = head == "prose" || head == "mm3" || head == "lyrics";
        std::string text;
        if (!read_text_file(per_mode ? spec.substr(eq + 1) : spec, text)) {
            fprintf(stderr, "ace-caption: cannot read %s\n",
                    (per_mode ? spec.substr(eq + 1) : spec).c_str());
            return 1;
        }
        if (per_mode) { mode_prompt[head] = text; } else { prompt_override = text; }
    }
    std::vector<std::string> modes;
    {
        std::string cur;
        for (char c : mode) {
            if (c == ',') {
                if (!cur.empty()) { modes.push_back(cur); }
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
        if (!cur.empty()) { modes.push_back(cur); }
        if (modes.empty()) { modes.push_back("prose"); }
    }
    // Precedence: per-mode override > blanket override > the mode's built-in.
    auto prompt_for_mode = [&](const std::string & m) -> std::string {
        const auto it = mode_prompt.find(m);
        if (it != mode_prompt.end()) { return it->second; }
        if (!prompt_override.empty()) { return prompt_override; }
        if (m == "mm3") { return PROMPT_MM3; }
        if (m == "lyrics") { return PROMPT_LYRICS; }
        return PROMPT_PROSE;
    };

    // ---- audio: any format -> 16 kHz mono ---------------------------------
    std::string audio_path = src, tmp_wav;
    if (!ends_with_ci(src, ".wav") && !ends_with_ci(src, ".mp3")) {
        if (ffmpeg.empty()) {
            fprintf(stderr, "ace-caption: %s is not WAV/MP3; pass --ffmpeg <path>\n",
                    src.c_str());
            return 1;
        }
        if (!transcode_to_wav16k(ffmpeg, src, tmp_wav)) {
            return 1;
        }
        audio_path = tmp_wav;
    }
    int T_in = 0, sr = 0;
    float * planar = audio_read(audio_path.c_str(), &T_in, &sr);  // planar [L:T][R:T]
    if (!tmp_wav.empty()) {
        remove(tmp_wav.c_str());
    }
    if (!planar || T_in <= 0) {
        fprintf(stderr, "ace-caption: cannot read audio %s\n", src.c_str());
        return 1;
    }
    std::vector<float> pcm;
    {
        int T16 = T_in;
        float * res = planar;
        bool owned = false;
        if (sr != 16000) {
            res = audio_resample(planar, T_in, sr, 16000, 2, &T16);
            owned = true;
            if (!res) {
                fprintf(stderr, "ace-caption: resample %d -> 16000 failed\n", sr);
                free(planar);
                return 1;
            }
        }
        // A 9:57 track pushed 78 GB through WDDM shared-memory spill on a 32 GB
        // card and never finished. On Windows that fails as silent slowdown, not
        // an error, so the cap is enforced here rather than trusted to the user.
        const int max_samples = (int) (max_seconds * 16000.0);
        const int T_use = std::min(T16, max_samples);
        if (T_use < T16) {
            fprintf(stderr, "[MOSS] input is %.0f s; using the first %.0f s (--max-seconds)\n",
                    (double) T16 / 16000.0, max_seconds);
        }
        pcm.resize((size_t) T_use);
        for (int t = 0; t < T_use; ++t) {
            pcm[(size_t) t] = 0.5f * (res[t] + res[T16 + t]);   // planar L/R -> mono
        }
        if (owned) {
            free(res);
        }
    }
    free(planar);

    // ---- models -----------------------------------------------------------
    const std::string sep =
#ifdef _WIN32
        "\\";
#else
        "/";
#endif
    BackendPair bp = backend_init("MOSS");
    if (!bp.backend) {
        fprintf(stderr, "ace-caption: no backend\n");
        return 1;
    }

    moss::AudioTower tower;
    moss::LmModel lm;
    std::string lm_path = models + sep + "moss-lm-q8_0.gguf";
    {
        FILE * probe = fopen(lm_path.c_str(), "rb");
        if (probe) {
            fclose(probe);
        } else {
            lm_path = models + sep + "moss-lm-f16.gguf";
        }
    }
    if (!moss::moss_load_audio_tower(&tower, (models + sep + "moss-aud-f16.gguf").c_str(),
                                     bp.backend) ||
        !moss::moss_load_lm(&lm, lm_path.c_str(), bp.backend)) {
        moss::moss_free_lm(&lm);
        moss::moss_free_audio_tower(&tower);
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }

    BPETokenizer tok;
    if (!load_bpe_from_gguf(&tok, lm_path.c_str())) {
        fprintf(stderr, "ace-caption: tokenizer load failed\n");
        moss::moss_free_lm(&lm);
        moss::moss_free_audio_tower(&tower);
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }

    // ---- mel + encode -----------------------------------------------------
    moss::MelParams mp;
    mp.sample_rate = (int) tower.hp.sample_rate;
    mp.n_fft = (int) tower.hp.n_fft;
    mp.hop_length = (int) tower.hp.hop_length;
    mp.n_mels = (int) tower.hp.n_mels;

    int frames = 0;
    std::vector<float> mel = moss::log_mel(pcm.data(), pcm.size(), mp, &frames);
    moss::EncoderOutput eo;
    if (!moss::moss_encode_audio(tower, mel.data(), mp.n_mels, frames, bp.backend, &eo)) {
        fprintf(stderr, "ace-caption: encode failed\n");
        return 1;
    }
    fprintf(stderr, "[MOSS] %.1f s audio -> %d mel frames -> %d audio tokens (%.2f/s)\n",
            (double) pcm.size() / 16000.0, frames, eo.n_tokens,
            (double) eo.n_tokens / ((double) pcm.size() / 16000.0));

    // ---- prompt assembly (per mode) ---------------------------------------
    // Everything from here down runs once PER OUTPUT MODE, but `eo` above is
    // computed once and shared: the encoder is prompt-independent, so N formats
    // cost one encode plus N decodes rather than N full passes. On a 4-minute
    // track the encode is the expensive half, so asking for prose+mm3+lyrics
    // together is far cheaper than three invocations.
    auto build_ids = [&](const std::string & user_prompt, std::vector<int32_t> & ids) -> bool {
    // Mirrors processing_moss_music.py::_build_default_prompt.
    bool ok = true;
    ok = ok && push_special(tok, "<|im_start|>", ids);
    push_text(tok, "system\nYou are a helpful assistant.", ids);
    ok = ok && push_special(tok, "<|im_end|>", ids);
    push_text(tok, "\n", ids);
    ok = ok && push_special(tok, "<|im_start|>", ids);
    push_text(tok, "user\n", ids);
    // The audio markers are pushed BY ID, not by looking up "<|audio_bos|>" etc.
    // Those strings are NOT in the vocabulary: processing_moss_music.py monkey-
    // patches convert_tokens_to_ids with an alias map precisely because the
    // tokenizer cannot resolve them. Ids come from the GGUF KVs instead.
    ids.push_back((int32_t) tower.hp.audio_bos_id);
    {
        // TIME MARKERS. Every `every_s` seconds the elapsed second count is
        // written into the audio stream as ordinary digit tokens, splitting the
        // <|AUDIO|> run into segments.
        //
        // THIS IS NOT OPTIONAL. SGLang's processor -- the runtime MOSS's authors
        // recommend -- always emits them. The HF processor has the identical
        // routine but `from_pretrained` pops `enable_time_marker` defaulting to
        // FALSE (its __init__ signature says True), so the Transformers path
        // silently feeds the model an out-of-distribution token stream. That is
        // what makes the HF path confidently wrong rather than noisy: on a 30 s
        // clip of Daft Punk it answers "Arabic pop" where SGLang answers
        // "Eurodance". Omit these and the port reproduces the wrong reference.
        const int every_s = 2;
        const double tps = (double) tower.hp.tokens_per_second;   // 12.5
        const int every_tok = (int) (tps * every_s);              // 25
        const int total_s = (int) ((double) eo.n_tokens / tps);
        int consumed = 0;
        for (int second = every_s; second <= total_s; second += every_s) {
            const int marker_pos = (second / every_s) * every_tok;
            for (int i = consumed; i < marker_pos && i < eo.n_tokens; ++i) {
                ids.push_back((int32_t) tower.hp.audio_token_id);
            }
            consumed = std::min(marker_pos, eo.n_tokens);
            // The second count as decimal digits, each its own ordinary token.
            const std::string digits = std::to_string(second);
            for (char d : digits) {
                push_text(tok, std::string(1, d), ids);
            }
        }
        for (int i = consumed; i < eo.n_tokens; ++i) {
            ids.push_back((int32_t) tower.hp.audio_token_id);
        }
    }
    ids.push_back((int32_t) tower.hp.audio_eos_id);
    push_text(tok, "\n" + user_prompt, ids);
    ok = ok && push_special(tok, "<|im_end|>", ids);
    push_text(tok, "\n", ids);
    ok = ok && push_special(tok, "<|im_start|>", ids);
    push_text(tok, "assistant\n", ids);
    return ok;
    };  // build_ids

    // Runs one output mode against the shared EncoderOutput.
    auto run_mode = [&](const std::string & user_prompt, std::string & out_text) -> bool {
    std::vector<int32_t> ids;
    if (!build_ids(user_prompt, ids)) {
        return false;
    }

    // ---- join tensors -----------------------------------------------------
    const int64_t T = (int64_t) ids.size();
    const int64_t H = (int64_t) lm.hp.n_embd;
    std::vector<float> text_mask((size_t) T, 1.0f);
    std::vector<float> audio_vals((size_t) (H * T), 0.0f);
    std::vector<std::vector<float>> merge(eo.merger_out.size(),
                                          std::vector<float>((size_t) (H * T), 0.0f));
    int64_t seen = 0;
    for (int64_t i = 0; i < T; ++i) {
        if ((uint32_t) ids[(size_t) i] == tower.hp.audio_token_id) {
            text_mask[(size_t) i] = 0.0f;
            memcpy(&audio_vals[(size_t) (i * H)], &eo.adapter_out[(size_t) (seen * H)],
                   (size_t) H * sizeof(float));
            for (size_t k = 0; k < merge.size(); ++k) {
                memcpy(&merge[k][(size_t) (i * H)], &eo.merger_out[k][(size_t) (seen * H)],
                       (size_t) H * sizeof(float));
            }
            ++seen;
        }
    }

    // ---- generate ---------------------------------------------------------
    moss::LmKv kv;
    if (!moss::moss_lm_kv_init(&kv, lm, T + max_tokens + 8, bp.backend)) {
        fprintf(stderr, "ace-caption: kv init failed (needs %lld positions)\n",
                (long long) (T + max_tokens + 8));
        return false;
    }
    // The cache is per-mode: each mode has its own prompt and therefore its own
    // KV contents. Only `eo` is shared.
    struct KvGuard {
        moss::LmKv * p;
        ~KvGuard() { moss::moss_lm_kv_free(p); }
    } kv_guard{ &kv };

    std::vector<float> logits;
    if (!moss::moss_lm_eval(lm, kv, ids, audio_vals, merge, text_mask, bp.backend, &logits)) {
        return false;
    }

    // --top-k-debug: dump the first generated position's top-N as a stable,
    // comparable distribution. Free-text agreement is the WRONG parity bar for a
    // 400-token greedy generation -- two runs that differ in the last bits will
    // fork at the first genuinely ambiguous token and never reconverge. The
    // distribution at step 0 is the thing that actually says whether the graphs
    // agree, and it is directly comparable with SGLang's return_logprob output.
    if (top_k_debug > 0) {
        std::vector<int> idx((size_t) logits.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            idx[i] = (int) i;
        }
        const int n = std::min<int>(top_k_debug, (int) idx.size());
        std::partial_sort(idx.begin(), idx.begin() + n, idx.end(),
                          [&](int a, int b) { return logits[(size_t) a] > logits[(size_t) b]; });
        printf("TOPK %d\n", n);
        for (int i = 0; i < n; ++i) {
            const std::string piece = detokenize(tok, { (int32_t) idx[(size_t) i] });
            printf("%d\t%.6f\t%s\n", idx[(size_t) i], logits[(size_t) idx[(size_t) i]],
                   piece.c_str());
        }
        out_text.clear();
        return true;   // caller exits; --top-k-debug is a diagnostic, not a caption
    }

    const int32_t eos = (int32_t) 151645;   // <|im_end|>
    std::vector<int32_t> counts((size_t) lm.hp.n_vocab, 0);
    std::vector<int32_t> gen;
    for (int step = 0; step < max_tokens; ++step) {
        const int32_t next = moss::moss_sample(logits, counts, sp);
        if (next == eos) {
            break;
        }
        gen.push_back(next);
        if ((size_t) next < counts.size()) {
            counts[(size_t) next]++;
        }
        if (step + 1 >= max_tokens) {
            break;
        }
        if (!moss::moss_lm_eval(lm, kv, { next }, {}, {}, {}, bp.backend, &logits)) {
            break;
        }
    }

    out_text = detokenize(tok, gen);
    return true;
    };  // run_mode

    // ---- run every requested mode off the one encode ----------------------
    int rc = 0;
    for (size_t mi = 0; mi < modes.size(); ++mi) {
        const std::string & mname = modes[mi];
        const std::string p = prompt_for_mode(mname);   // handles both override forms
        std::string text;
        const double t0 = (double) clock() / CLOCKS_PER_SEC;
        if (!run_mode(p, text)) {
            fprintf(stderr, "ace-caption: mode '%s' failed\n", mname.c_str());
            rc = 1;
            break;
        }
        const double el = (double) clock() / CLOCKS_PER_SEC - t0;
        if (top_k_debug > 0) {
            break;   // diagnostic already printed
        }

        if (out_path.empty()) {
            if (modes.size() > 1) {
                printf("===== %s =====\n", mname.c_str());
            }
            printf("%s\n", text.c_str());
        } else {
            // One mode -> -o is the file. Several -> -o is a prefix, and each
            // mode appends its own suffix, so a caller can ask for all formats
            // in one invocation and get predictable filenames.
            std::string dst = out_path;
            if (modes.size() > 1) {
                const size_t dot = out_path.find_last_of('.');
                const std::string stem = (dot == std::string::npos) ? out_path
                                                                    : out_path.substr(0, dot);
                const std::string ext = (dot == std::string::npos) ? ".txt"
                                                                   : out_path.substr(dot);
                dst = stem + "." + mname + ext;
            }
            FILE * f = fopen(dst.c_str(), "wb");
            if (!f) {
                fprintf(stderr, "ace-caption: cannot write %s\n", dst.c_str());
                rc = 1;
                break;
            }
            fwrite(text.data(), 1, text.size(), f);
            fputc('\n', f);
            fclose(f);
            fprintf(stderr, "[MOSS] %-7s -> %s (%.1fs)\n", mname.c_str(), dst.c_str(), el);
        }
    }

    moss::moss_free_lm(&lm);
    moss::moss_free_audio_tower(&tower);
    backend_release(bp.backend, bp.cpu_backend);
    return rc;
}
