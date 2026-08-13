#pragma once
// minimax/mm3-server.h — HTTP surface for the MiniMax-Music3 backend.
//
// HOT-Step file (does not exist upstream). This is the ONE hook include that
// wires the whole MiniMax subsystem into an upstream-derived file:
//
//     engine/tools/hot-step-server.cpp:   #include "minimax/mm3-server.h"
//                                         mm3_register_routes(svr, models_dir);
//
// Nothing else in engine/src/ or engine/tools/ refers to minimax/, so an
// upstream sync can only break this in a way `engine/verify-hooks.ps1` catches
// (and it fails loudly at compile time, not silently).
//
// Endpoints (all family-scoped under /mm3/ so they can never collide with the
// ACE-Step surface):
//
//   GET  /mm3/props       discovery + per-component config summary + residency + VRAM
//   POST /mm3/warm        load both GGUFs into VRAM (idempotent)
//   POST /mm3/unload      free them (idempotent)
//   POST /mm3/voc-decode  DEBUG: vocoder-only decode, latents in -> WAV out
//
// SCOPE: residency, metadata, and the standalone vocoder. There is no
// /mm3/synth yet — the LM / depth / cond / DiT stages arrive in later
// increments.
//
// Concurrency: the ACE pipeline runs GPU work on a single worker thread while
// these handlers run on httplib threads. g_mm3_mutex serialises MM3 load,
// unload and vocoder decode against each other. /mm3/voc-decode is a bring-up
// and parity-validation endpoint, NOT a production path: it does real GPU work
// on an httplib thread, so it can contend with the ACE worker. When the full
// /mm3/synth lands it must go through the existing job queue like /synth does,
// so MM3 compute shares the one GPU worker rather than racing it.

#include "mm3-model.h"
#include "mm3-vocoder-graph.h"

#include "audio-io.h"
#include "yyjson.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wshadow"
#endif
#include "httplib.h"
#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

#include <mutex>
#include <string>

static MM3Model  g_mm3;
static std::mutex g_mm3_mutex;

// Local copy of the server's error-response shape (hot-step-server.cpp's
// json_error is a static defined after this include, so it cannot be reused).
static void mm3_json_error(httplib::Response & res, int status, const std::string & msg) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_strcpy(doc, root, "error", msg.c_str());
    char * json = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    res.status = status;
    res.set_content(json ? json : "{\"error\":\"unknown\"}", "application/json");
    if (json) {
        free(json);
    }
}

static void mm3_json_add_file(yyjson_mut_doc * doc, yyjson_mut_val * parent, const char * key,
                              const MM3FileInfo & fi) {
    yyjson_mut_val * o = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, parent, key, o);
    yyjson_mut_obj_add_bool(doc, o, "found", fi.found);
    yyjson_mut_obj_add_bool(doc, o, "probe_ok", fi.probe_ok);
    yyjson_mut_obj_add_strcpy(doc, o, "name", fi.name.c_str());
    yyjson_mut_obj_add_strcpy(doc, o, "path", fi.path.c_str());
    yyjson_mut_obj_add_strcpy(doc, o, "arch", fi.arch.c_str());
    yyjson_mut_obj_add_strcpy(doc, o, "general_name", fi.general_name.c_str());
    yyjson_mut_obj_add_strcpy(doc, o, "license", fi.license.c_str());
    yyjson_mut_obj_add_strcpy(doc, o, "source_layout", fi.source_layout.c_str());
    yyjson_mut_obj_add_uint(doc, o, "converter_version", fi.converter_version);
    yyjson_mut_obj_add_uint(doc, o, "file_type", fi.file_type);
    yyjson_mut_obj_add_uint(doc, o, "file_bytes", fi.file_bytes);
    yyjson_mut_obj_add_uint(doc, o, "tensor_bytes", fi.tensor_bytes);
    yyjson_mut_obj_add_int(doc, o, "n_tensors", fi.n_tensors);
    if (!fi.probe_error.empty()) {
        yyjson_mut_obj_add_strcpy(doc, o, "error", fi.probe_error.c_str());
    }
}

static void mm3_json_add_config(yyjson_mut_doc * doc, yyjson_mut_val * root) {
    yyjson_mut_val * cfg = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "config", cfg);

    // ── lm ──
    {
        const MM3LmConfig & c = g_mm3.lm_cfg;
        yyjson_mut_val *    o = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, cfg, "lm", o);
        yyjson_mut_obj_add_uint(doc, o, "block_count", c.block_count);
        yyjson_mut_obj_add_uint(doc, o, "context_length", c.context_length);
        yyjson_mut_obj_add_uint(doc, o, "embedding_length", c.embedding_length);
        yyjson_mut_obj_add_uint(doc, o, "feed_forward_length", c.feed_forward_length);
        yyjson_mut_obj_add_uint(doc, o, "head_count", c.head_count);
        yyjson_mut_obj_add_uint(doc, o, "head_count_kv", c.head_count_kv);
        yyjson_mut_obj_add_uint(doc, o, "key_length", c.key_length);
        yyjson_mut_obj_add_uint(doc, o, "value_length", c.value_length);
        yyjson_mut_obj_add_real(doc, o, "rms_eps", c.rms_eps);
        yyjson_mut_obj_add_real(doc, o, "rope_freq_base", c.rope_freq_base);
        yyjson_mut_obj_add_uint(doc, o, "vocab_size", c.vocab_size);
        yyjson_mut_obj_add_uint(doc, o, "semantic_vocab_offset", c.semantic_vocab_offset);
        yyjson_mut_obj_add_uint(doc, o, "semantic_vocab_size", c.semantic_vocab_size);
        yyjson_mut_obj_add_uint(doc, o, "acoustic_vocab_size", c.acoustic_vocab_size);
        yyjson_mut_obj_add_uint(doc, o, "num_codebooks", c.num_codebooks);
        yyjson_mut_obj_add_uint(doc, o, "eos_audio", c.eos_audio);
        yyjson_mut_obj_add_uint(doc, o, "frame_rate", c.frame_rate);
        yyjson_mut_obj_add_uint(doc, o, "max_audio_frames", c.max_audio_frames);
        yyjson_mut_obj_add_uint(doc, o, "max_prompt_tokens", c.max_prompt_tokens);
        yyjson_mut_obj_add_real(doc, o, "ar_cfg_scale", c.ar_cfg_scale);
        yyjson_mut_obj_add_uint(doc, o, "ar_top_k", c.ar_top_k);
        yyjson_mut_obj_add_real(doc, o, "ar_embedding_scale", c.ar_embedding_scale);

        yyjson_mut_val * tok = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, o, "tokens", tok);
        yyjson_mut_obj_add_uint(doc, tok, "im_start", c.tok_im_start);
        yyjson_mut_obj_add_uint(doc, tok, "im_end", c.tok_im_end);
        yyjson_mut_obj_add_uint(doc, tok, "audio_cfg", c.tok_audio_cfg);
        yyjson_mut_obj_add_uint(doc, tok, "audio_start", c.tok_audio_start);
        yyjson_mut_obj_add_uint(doc, tok, "audio_end", c.tok_audio_end);
        yyjson_mut_obj_add_uint(doc, tok, "caption_start", c.tok_caption_start);
        yyjson_mut_obj_add_uint(doc, tok, "caption_end", c.tok_caption_end);
        yyjson_mut_obj_add_uint(doc, tok, "lyrics_start", c.tok_lyrics_start);
        yyjson_mut_obj_add_uint(doc, tok, "lyrics_end", c.tok_lyrics_end);
    }

    const MM3SynthConfig & s = g_mm3.synth_cfg;

    yyjson_mut_val * comps = yyjson_mut_arr(doc);
    for (const auto & c : s.components) {
        yyjson_mut_arr_add_strcpy(doc, comps, c.c_str());
    }
    yyjson_mut_obj_add_val(doc, cfg, "synth_components", comps);

    // ── depth ──
    {
        const MM3DepthConfig & c = s.depth;
        yyjson_mut_val *       o = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, cfg, "depth", o);
        yyjson_mut_obj_add_uint(doc, o, "block_count", c.block_count);
        yyjson_mut_obj_add_uint(doc, o, "embedding_length", c.embedding_length);
        yyjson_mut_obj_add_uint(doc, o, "feed_forward_length", c.feed_forward_length);
        yyjson_mut_obj_add_uint(doc, o, "head_count", c.head_count);
        yyjson_mut_obj_add_uint(doc, o, "head_dim", c.head_dim);
        yyjson_mut_obj_add_uint(doc, o, "max_position", c.max_position);
        yyjson_mut_obj_add_real(doc, o, "rms_eps", c.rms_eps);
        yyjson_mut_obj_add_uint(doc, o, "num_codebooks", c.num_codebooks);
        yyjson_mut_obj_add_uint(doc, o, "audio_vocab_size", c.audio_vocab_size);
        yyjson_mut_obj_add_uint(doc, o, "audio_embd_rows", c.audio_embd_rows);
        yyjson_mut_obj_add_bool(doc, o, "causal", c.causal);
        yyjson_mut_obj_add_bool(doc, o, "rope", c.rope);
    }

    // ── cond ──
    {
        const MM3CondConfig & c = s.cond;
        yyjson_mut_val *      o = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, cfg, "cond", o);
        yyjson_mut_obj_add_uint(doc, o, "num_layers", c.num_layers);
        yyjson_mut_obj_add_uint(doc, o, "hidden_dim", c.hidden_dim);
        yyjson_mut_obj_add_uint(doc, o, "out_dim", c.out_dim);
        yyjson_mut_obj_add_uint(doc, o, "kernel_size", c.kernel_size);
        yyjson_mut_obj_add_uint(doc, o, "padding", c.padding);
        yyjson_mut_obj_add_uint(doc, o, "input_sampling_rate", c.input_sampling_rate);
        yyjson_mut_obj_add_uint(doc, o, "input_hop_length", c.input_hop_length);
        yyjson_mut_obj_add_uint(doc, o, "output_sampling_rate", c.output_sampling_rate);
        yyjson_mut_obj_add_uint(doc, o, "output_hop_length", c.output_hop_length);
        yyjson_mut_obj_add_strcpy(doc, o, "interpolation", c.interpolation.c_str());
        yyjson_mut_obj_add_strcpy(doc, o, "layer_mix", c.layer_mix.c_str());
    }

    // ── dit ──
    {
        const MM3DitConfig & c = s.dit;
        yyjson_mut_val *     o = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, cfg, "dit", o);
        yyjson_mut_obj_add_uint(doc, o, "block_count", c.block_count);
        yyjson_mut_obj_add_uint(doc, o, "embedding_length", c.embedding_length);
        yyjson_mut_obj_add_uint(doc, o, "head_count", c.head_count);
        yyjson_mut_obj_add_uint(doc, o, "head_dim", c.head_dim);
        yyjson_mut_obj_add_uint(doc, o, "ff_inner", c.ff_inner);
        yyjson_mut_obj_add_uint(doc, o, "in_channels", c.in_channels);
        yyjson_mut_obj_add_uint(doc, o, "condition_dim", c.condition_dim);
        yyjson_mut_obj_add_uint(doc, o, "concat_channels", c.concat_channels);
        yyjson_mut_obj_add_real(doc, o, "layer_norm_eps", c.layer_norm_eps);
        yyjson_mut_obj_add_uint(doc, o, "rope_dim", c.rope_dim);
        yyjson_mut_obj_add_real(doc, o, "rope_theta", c.rope_theta);
        yyjson_mut_obj_add_strcpy(doc, o, "rope_type", c.rope_type.c_str());
        yyjson_mut_obj_add_uint(doc, o, "fourier_dim", c.fourier_dim);
        yyjson_mut_obj_add_strcpy(doc, o, "glu_order", c.glu_order.c_str());
        yyjson_mut_obj_add_bool(doc, o, "output_negated", c.output_negated);
        yyjson_mut_obj_add_bool(doc, o, "timestep_token_prepended", c.timestep_token_prepended);
        yyjson_mut_obj_add_bool(doc, o, "pre_post_conv_residual", c.pre_post_conv_residual);
        yyjson_mut_obj_add_bool(doc, o, "attn_bias", c.attn_bias);
        yyjson_mut_obj_add_uint(doc, o, "window_frames", c.window_frames);
        yyjson_mut_obj_add_uint(doc, o, "hop_frames", c.hop_frames);
        yyjson_mut_obj_add_uint(doc, o, "window_latents", c.window_latents);
        yyjson_mut_obj_add_uint(doc, o, "hop_latents", c.hop_latents);
    }

    // ── flow ──
    {
        const MM3FlowConfig & c = s.flow;
        yyjson_mut_val *      o = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, cfg, "flow", o);
        yyjson_mut_obj_add_strcpy(doc, o, "scheduler", c.scheduler.c_str());
        yyjson_mut_obj_add_uint(doc, o, "steps", c.steps);
        yyjson_mut_obj_add_real(doc, o, "cfg_scale", c.cfg_scale);
        yyjson_mut_obj_add_bool(doc, o, "invert_sigmas", c.invert_sigmas);
        yyjson_mut_obj_add_real(doc, o, "shift", c.shift);
        yyjson_mut_obj_add_uint(doc, o, "num_train_timesteps", c.num_train_timesteps);
    }

    // ── voc ──
    {
        const MM3VocConfig & c = s.voc;
        yyjson_mut_val *     o = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, cfg, "voc", o);
        yyjson_mut_obj_add_uint(doc, o, "latent_channels", c.latent_channels);
        yyjson_mut_obj_add_uint(doc, o, "fold_channels", c.fold_channels);
        yyjson_mut_obj_add_uint(doc, o, "dec_in_dim", c.dec_in_dim);
        yyjson_mut_obj_add_uint(doc, o, "hidden_dim", c.hidden_dim);
        yyjson_mut_val * ups = yyjson_mut_arr(doc);
        for (int32_t r : c.upsample_rates) {
            yyjson_mut_arr_add_int(doc, ups, r);
        }
        yyjson_mut_obj_add_val(doc, o, "upsample_rates", ups);
        yyjson_mut_val * dil = yyjson_mut_arr(doc);
        for (int32_t r : c.res_dilations) {
            yyjson_mut_arr_add_int(doc, dil, r);
        }
        yyjson_mut_obj_add_val(doc, o, "res_dilations", dil);
        yyjson_mut_obj_add_uint(doc, o, "total_upsample", c.total_upsample);
        yyjson_mut_obj_add_uint(doc, o, "sampling_rate", c.sampling_rate);
        yyjson_mut_obj_add_uint(doc, o, "channels", c.channels);
        yyjson_mut_obj_add_real(doc, o, "snake_eps", c.snake_eps);
        yyjson_mut_obj_add_bool(doc, o, "final_tanh", c.final_tanh);
        yyjson_mut_obj_add_bool(doc, o, "weight_norm_folded", c.weight_norm_folded);
        yyjson_mut_obj_add_strcpy(doc, o, "snake", c.snake.c_str());
    }
}

// GET /mm3/props
static void mm3_handle_props(const httplib::Request &, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_mm3_mutex);

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    yyjson_mut_obj_add_strcpy(doc, root, "backend", "minimax-m3");
    yyjson_mut_obj_add_strcpy(doc, root, "model", "MiniMax-Music3");
    yyjson_mut_obj_add_bool(doc, root, "available", mm3_available(g_mm3));
    yyjson_mut_obj_add_bool(doc, root, "loaded", g_mm3.loaded);
    yyjson_mut_obj_add_strcpy(doc, root, "models_dir", g_mm3.models_dir.c_str());

    yyjson_mut_val * dirs = yyjson_mut_arr(doc);
    for (const auto & d : g_mm3.search_dirs) {
        yyjson_mut_arr_add_strcpy(doc, dirs, d.c_str());
    }
    yyjson_mut_obj_add_val(doc, root, "search_dirs", dirs);

    yyjson_mut_val * files = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "files", files);
    mm3_json_add_file(doc, files, "lm", g_mm3.lm_file);
    mm3_json_add_file(doc, files, "synth", g_mm3.synth_file);

    mm3_json_add_config(doc, root);

    yyjson_mut_val * vram = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "vram", vram);
    yyjson_mut_obj_add_uint(doc, vram, "lm_bytes", g_mm3.vram_lm);
    yyjson_mut_obj_add_uint(doc, vram, "synth_bytes", g_mm3.vram_synth);
    yyjson_mut_obj_add_uint(doc, vram, "total_bytes", mm3_vram_bytes(g_mm3));
    yyjson_mut_obj_add_real(doc, vram, "total_mb", (double) mm3_vram_bytes(g_mm3) / (1024.0 * 1024.0));
    yyjson_mut_obj_add_real(doc, vram, "load_ms", g_mm3.load_ms);
    yyjson_mut_obj_add_uint(doc, vram, "lm_tensors", g_mm3.tmap_lm.size());
    yyjson_mut_obj_add_uint(doc, vram, "synth_tensors", g_mm3.tmap_synth.size());

    yyjson_mut_val * errs = yyjson_mut_arr(doc);
    for (const auto & e : g_mm3.meta_errors) {
        yyjson_mut_arr_add_strcpy(doc, errs, e.c_str());
    }
    yyjson_mut_obj_add_val(doc, root, "errors", errs);

    char * json = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
}

// POST /mm3/warm — load both files. Idempotent.
static void mm3_handle_warm(const httplib::Request &, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_mm3_mutex);

    const bool  was_loaded = g_mm3.loaded;
    std::string err;
    if (!mm3_load(&g_mm3, &err)) {
        mm3_json_error(res, 500, err.empty() ? "MM3 load failed" : err);
        return;
    }

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_bool(doc, root, "loaded", true);
    yyjson_mut_obj_add_bool(doc, root, "already_loaded", was_loaded);
    yyjson_mut_obj_add_uint(doc, root, "lm_bytes", g_mm3.vram_lm);
    yyjson_mut_obj_add_uint(doc, root, "synth_bytes", g_mm3.vram_synth);
    yyjson_mut_obj_add_uint(doc, root, "total_bytes", mm3_vram_bytes(g_mm3));
    yyjson_mut_obj_add_real(doc, root, "total_mb", (double) mm3_vram_bytes(g_mm3) / (1024.0 * 1024.0));
    yyjson_mut_obj_add_real(doc, root, "load_ms", g_mm3.load_ms);
    yyjson_mut_obj_add_uint(doc, root, "lm_tensors", g_mm3.tmap_lm.size());
    yyjson_mut_obj_add_uint(doc, root, "synth_tensors", g_mm3.tmap_synth.size());
    char * json = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
}

// POST /mm3/unload — free VRAM. Idempotent.
static void mm3_handle_unload(const httplib::Request &, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_mm3_mutex);

    const bool   was_loaded = g_mm3.loaded;
    const size_t freed      = mm3_vram_bytes(g_mm3);
    // Vocoder graph state holds derived weights + a scheduler that point into
    // the model's buffers and hold a backend reference — tear it down first.
    mm3_vocoder_free(&g_mm3_voc);
    mm3_unload(&g_mm3);

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_bool(doc, root, "unloaded", was_loaded);
    yyjson_mut_obj_add_bool(doc, root, "loaded", false);
    yyjson_mut_obj_add_uint(doc, root, "freed_bytes", was_loaded ? freed : 0);
    yyjson_mut_obj_add_real(doc, root, "freed_mb", was_loaded ? (double) freed / (1024.0 * 1024.0) : 0.0);
    char * json = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
}

// FNV-1a over the raw output bytes. Used only by the selftest response, as a
// crisp "byte-identical across runs" signal that rms/peak alone cannot give.
static uint64_t mm3_fnv1a(const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t        h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// POST /mm3/voc-decode — vocoder-only bring-up + parity endpoint.
//
//   ?frames=L        body is raw little-endian F32, exactly 128*L*4 bytes,
//                    channel-major (channel c at offset c*L) — i.e. the memory
//                    order of a torch [1, 128, L] contiguous tensor.
//                    Returns 16-bit PCM stereo WAV at mm3.voc.sampling_rate.
//
//   ?selftest=1      ignores the body. Generates L=256 deterministic
//                    pseudo-random latents (std::mt19937 seeded 20260813,
//                    N(0, 0.5)) and returns JSON statistics instead of audio.
//                    Re-running must reproduce the same hash.
//
// 503 unless MM3 is warm.
static void mm3_handle_voc_decode(const httplib::Request & req, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_mm3_mutex);

    if (!g_mm3.loaded) {
        mm3_json_error(res, 503, "MiniMax-Music3 is not warm — POST /mm3/warm first");
        return;
    }

    const MM3VocConfig & vc = g_mm3.synth_cfg.voc;
    const int64_t        LC = (int64_t) vc.latent_channels;
    const int            sr = (int) vc.sampling_rate;

    const bool selftest = req.has_param("selftest") && req.get_param_value("selftest") != "0";

    int64_t L = 0;
    if (selftest) {
        L = 256;
    } else {
        if (!req.has_param("frames")) {
            mm3_json_error(res, 400, "missing ?frames=<latent frame count>");
            return;
        }
        L = strtoll(req.get_param_value("frames").c_str(), nullptr, 10);
    }
    if (L <= 0 || L > 8192) {
        mm3_json_error(res, 400, "frames must be in 1..8192");
        return;
    }

    std::vector<float> latents((size_t) (LC * L));
    if (selftest) {
        std::mt19937                          rng(20260813u);
        std::normal_distribution<float>       dist(0.0f, 0.5f);
        for (size_t i = 0; i < latents.size(); i++) {
            latents[i] = dist(rng);
        }
    } else {
        const size_t want = (size_t) (LC * L) * sizeof(float);
        if (req.body.size() != want) {
            char buf[192];
            snprintf(buf, sizeof(buf), "body is %zu bytes, expected %zu (= %lld channels * %lld frames * 4)",
                     req.body.size(), want, (long long) LC, (long long) L);
            mm3_json_error(res, 400, buf);
            return;
        }
        memcpy(latents.data(), req.body.data(), want);
    }

    std::vector<float> audio;
    std::string        err;
    const auto         t0 = std::chrono::steady_clock::now();
    const bool         ok = mm3_vocoder_decode(g_mm3, latents.data(), L, audio, &err);
    const double       ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) {
        mm3_json_error(res, 500, err.empty() ? "vocoder decode failed" : err);
        return;
    }

    const int T = (int) (audio.size() / 2);
    fprintf(stderr, "[MM3-Voc] Decoded %lld latent frames -> %d samples/ch (%.2fs @ %d Hz) in %.0f ms\n",
            (long long) L, T, (double) T / (double) (sr > 0 ? sr : 1), sr, ms);

    if (!selftest) {
        std::string wav = audio_encode_wav_s16(audio.data(), T, sr);
        res.set_content(wav, "audio/wav");
        return;
    }

    double sum_sq  = 0.0;
    float  peak    = 0.0f;
    bool   has_nan = false;
    for (float v : audio) {
        if (std::isnan(v) || std::isinf(v)) {
            has_nan = true;
            continue;
        }
        sum_sq += (double) v * (double) v;
        float a = std::fabs(v);
        if (a > peak) {
            peak = a;
        }
    }
    const double rms = audio.empty() ? 0.0 : std::sqrt(sum_sq / (double) audio.size());

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_bool(doc, root, "selftest", true);
    yyjson_mut_obj_add_int(doc, root, "frames", L);
    yyjson_mut_obj_add_int(doc, root, "n_samples", T);
    yyjson_mut_obj_add_int(doc, root, "channels", 2);
    yyjson_mut_obj_add_int(doc, root, "sample_rate", sr);
    yyjson_mut_obj_add_real(doc, root, "rms", rms);
    yyjson_mut_obj_add_real(doc, root, "peak", (double) peak);
    yyjson_mut_obj_add_bool(doc, root, "has_nan", has_nan);
    yyjson_mut_obj_add_real(doc, root, "ms", ms);
    yyjson_mut_obj_add_uint(doc, root, "hash",
                            mm3_fnv1a(audio.data(), audio.size() * sizeof(float)));
    char * json = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
}

// The single entry point the server calls. Discovers + probes the GGUFs (cheap:
// mmap + header parse, no weight reads) and registers the /mm3/* routes.
static void mm3_register_routes(httplib::Server & svr, const char * models_dir) {
    mm3_discover(&g_mm3, models_dir);

    svr.Get("/mm3/props", mm3_handle_props);
    svr.Post("/mm3/warm", mm3_handle_warm);
    svr.Post("/mm3/unload", mm3_handle_unload);
    svr.Post("/mm3/voc-decode", mm3_handle_voc_decode);
}
