#pragma once
// yue2/yue2-server.h — /yue2/* HTTP routes. Milestones M7/M8,
// docs/plans/yue2/06-engine-port-plan.md §7.
//
// HOT-Step file (no acestep.cpp analog). Mirrors minimax/mm3-server.h's
// props/warm/unload/select-model shape, PLUS the production POST /yue2/synth
// endpoint mm3-job.h keeps in its own separate file — collapsed into one
// file/one include/one call here because YuE2 registers no separate job
// route (docs/plans/yue2/06-engine-port-plan.md §7: "YuE2 registers no new
// job routes... progress/result are read through the SHARED GET/POST /job").
//
// Included MID-FILE in hot-step-server.cpp (next to minimax/mm3-job.h, not
// beside minimax/mm3-server.h at the top) because this file's own
// yue2_handle_synth needs job_create()/work_push() (via yue2-job.h), both
// defined earlier in that file's job system — see the include site's own
// comment for why a literal top-of-file position (as the engine-port-plan's
// prose describes) is not achievable without forward-declaring the whole job
// system, which nothing else in this codebase does either.
//
// Endpoints:
//   GET  /yue2/props            <- yue2_handle_props
//   POST /yue2/warm             <- yue2_handle_warm
//   POST /yue2/unload           <- yue2_handle_unload
//   POST /yue2/select-model     <- yue2_handle_select_model   (VAE variant picker)
//   POST /yue2/tokenize-check   <- yue2_handle_tokenize_check (bring-up, cheap)
//   POST /yue2/synth            <- yue2_handle_synth          (production; returns the
//                                   shared engine job id — poll/fetch via GET/POST /job)
//
// NOT implemented (rough edge, listed rather than built for time's sake —
// task rule): POST /yue2/vae-decode and POST /yue2/abc-plan, the plan's own
// "lower priority... useful for isolated fixture validation" bring-up
// endpoints. yue2-probe.cpp's --nar-parity/--vae-parity subcommands already
// cover that standalone-validation need from the CLI side; the HTTP
// equivalents were not built this pass since nothing in the M10 gate
// (POST /yue2/synth end-to-end) needs them.

#include "yue2-job.h"
#include "yue2-model.h"
#include "yue2-request.h"
#include "yue2-tokenizer.h"

#include "httplib.h"
#include "yyjson.h"

#include <mutex>
#include <string>

static void yue2_json_error(httplib::Response & res, int code, const std::string & msg) {
    res.status = code;
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_strcpy(doc, root, "error", msg.c_str());
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{\"error\":\"unknown\"}", "application/json");
    yyjson_mut_doc_free(doc);
    if (json) {
        free(json);
    }
}

static void yue2_json_add_file(yyjson_mut_doc * doc, yyjson_mut_val * parent, const char * key,
                                const Yue2FileInfo & fi) {
    yyjson_mut_val * o = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, parent, key, o);
    yyjson_mut_obj_add_bool(doc, o, "found", fi.found);
    yyjson_mut_obj_add_bool(doc, o, "probe_ok", fi.probe_ok);
    yyjson_mut_obj_add_strcpy(doc, o, "name", fi.name.c_str());
    yyjson_mut_obj_add_uint(doc, o, "bytes", fi.file_bytes);
    if (!fi.probe_error.empty()) {
        yyjson_mut_obj_add_strcpy(doc, o, "error", fi.probe_error.c_str());
    }
}

static void yue2_handle_props(const httplib::Request &, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_yue2_mutex);

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    yyjson_mut_obj_add_strcpy(doc, root, "backend", "yue2");
    yyjson_mut_obj_add_bool(doc, root, "available", yue2_available(g_yue2));
    yyjson_mut_obj_add_bool(doc, root, "lm_resident", g_yue2.lm_resident);
    yyjson_mut_obj_add_bool(doc, root, "vae_resident", g_yue2.vae_resident);
    yyjson_mut_obj_add_strcpy(doc, root, "vae_variant_loaded", YUE2_VAE_VARIANT_NAME[g_yue2.vae_loaded_variant]);
    yyjson_mut_obj_add_strcpy(doc, root, "models_dir", g_yue2.models_dir.c_str());

    yyjson_mut_val * files = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "files", files);
    yue2_json_add_file(doc, files, "lm", g_yue2.lm_file);
    yue2_json_add_file(doc, files, "vae_standard", g_yue2.vae_file[YUE2_VAE_STANDARD]);
    yue2_json_add_file(doc, files, "vae_legacy", g_yue2.vae_file[YUE2_VAE_LEGACY]);

    yyjson_mut_val * vram = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "vram", vram);
    yyjson_mut_obj_add_uint(doc, vram, "lm_bytes", g_yue2.vram_lm);
    yyjson_mut_obj_add_uint(doc, vram, "vae_bytes", g_yue2.vram_vae);
    yyjson_mut_obj_add_uint(doc, vram, "total_bytes", yue2_vram_bytes(g_yue2));
    yyjson_mut_obj_add_real(doc, vram, "total_mb", (double) yue2_vram_bytes(g_yue2) / (1024.0 * 1024.0));
    yyjson_mut_obj_add_real(doc, vram, "load_ms", g_yue2.load_ms);

    yyjson_mut_val * errs = yyjson_mut_arr(doc);
    for (const auto & e : g_yue2.meta_errors) {
        yyjson_mut_arr_add_strcpy(doc, errs, e.c_str());
    }
    yyjson_mut_obj_add_val(doc, root, "errors", errs);

    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    yyjson_mut_doc_free(doc);
    if (json) {
        free(json);
    }
}

// POST /yue2/warm — load LM + the currently-selected VAE variant. Idempotent.
static void yue2_handle_warm(const httplib::Request &, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_yue2_mutex);

    const bool  was_loaded = g_yue2.lm_resident && g_yue2.vae_resident;
    std::string err;
    if (!yue2_load_parts(&g_yue2, true, true, g_yue2.vae_loaded_variant, false, &err)) {
        yue2_json_error(res, 500, err.empty() ? "YuE2 warm failed" : err);
        return;
    }
    if (!yue2_ensure_tokenizer(&err)) {
        yue2_json_error(res, 500, err);
        return;
    }

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_bool(doc, root, "loaded", true);
    yyjson_mut_obj_add_bool(doc, root, "already_loaded", was_loaded);
    yyjson_mut_obj_add_uint(doc, root, "total_bytes", yue2_vram_bytes(g_yue2));
    yyjson_mut_obj_add_real(doc, root, "total_mb", (double) yue2_vram_bytes(g_yue2) / (1024.0 * 1024.0));
    yyjson_mut_obj_add_real(doc, root, "load_ms", g_yue2.load_ms);
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    yyjson_mut_doc_free(doc);
    if (json) {
        free(json);
    }
}

// POST /yue2/unload — free all YuE2 VRAM unconditionally. Idempotent.
static void yue2_handle_unload(const httplib::Request &, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_yue2_mutex);
    yue2_unload(&g_yue2);
    res.set_content("{\"unloaded\":true}", "application/json");
}

// POST /yue2/select-model — {"vae_variant": "standard"|"legacy"}. Only
// active variant is ever resident (yue2_load_parts's own contract); if a VAE
// is currently loaded this reloads it, otherwise it just records the pick
// for the next warm/synth.
static void yue2_handle_select_model(const httplib::Request & req, httplib::Response & res) {
    std::string variant_str;
    if (!req.body.empty()) {
        yyjson_doc * d = yyjson_read(req.body.data(), req.body.size(), 0);
        if (d) {
            yyjson_val * v = yyjson_obj_get(yyjson_doc_get_root(d), "vae_variant");
            if (v && yyjson_is_str(v)) {
                variant_str = yyjson_get_str(v);
            }
            yyjson_doc_free(d);
        }
    }
    Yue2VaeVariant variant = YUE2_VAE_STANDARD;
    if (variant_str == "legacy") {
        variant = YUE2_VAE_LEGACY;
    } else if (!variant_str.empty() && variant_str != "standard") {
        yue2_json_error(res, 400, "vae_variant must be \"standard\"|\"legacy\"");
        return;
    }

    std::lock_guard<std::mutex> lock(g_yue2_mutex);
    const bool                   want_vae = g_yue2.vae_resident;  // only reload if one is already resident
    std::string                  err;
    if (want_vae) {
        if (!yue2_load_parts(&g_yue2, false, true, variant, false, &err)) {
            yue2_json_error(res, 500, err.empty() ? "YuE2 select-model failed" : err);
            return;
        }
    } else {
        g_yue2.vae_loaded_variant = variant;  // recorded for the next warm/synth
    }
    std::string body = "{\"selected\":true,\"vae_variant\":\"" + std::string(YUE2_VAE_VARIANT_NAME[variant]) + "\"}";
    res.set_content(body, "application/json");
}

// POST /yue2/tokenize-check — bring-up: assemble the prefix and report its
// token count + resolved defaults, no GPU work. Safe to call on a cold
// server (only needs the LM GGUF's tokenizer KV, not resident weights).
static void yue2_handle_tokenize_check(const httplib::Request & req, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_yue2_mutex);
    std::string                  err;
    if (!yue2_ensure_tokenizer(&err)) {
        yue2_json_error(res, 500, err);
        return;
    }
    Yue2Request preq;
    if (!yue2_parse_request(req.body, &preq, &err)) {
        yue2_json_error(res, 400, err);
        return;
    }
    yue2_request_resolve_defaults(&preq, g_yue2.lm_cfg);

    std::vector<int32_t> abc_ids;
    std::vector<int32_t> prefix;
    try {
        if (preq.abc_provided) {
            abc_ids = yue2_bpe_encode(&g_yue2_tok, preq.abc);
        }
        prefix = yue2_token_prefixes(&g_yue2_tok, preq.style, preq.lyrics, preq.cot,
                                     preq.abc_provided ? &abc_ids : nullptr);
    } catch (const std::exception & e) {
        yue2_json_error(res, 400, e.what());
        return;
    }

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_uint(doc, root, "n_prefix_tokens", prefix.size());
    yyjson_mut_obj_add_strcpy(doc, root, "cot", yue2_cot_name(preq.cot));
    yyjson_mut_obj_add_real(doc, root, "cfg_scale", preq.cfg_scale);
    yyjson_mut_obj_add_int(doc, root, "ode_steps", preq.ode_steps);
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    yyjson_mut_doc_free(doc);
    if (json) {
        free(json);
    }
}

// POST /yue2/synth — the production endpoint. Parses the request, creates a
// job on the SHARED job system, and hands the render to the one GPU worker
// thread; returns immediately with the job id (same shape ACE/MM3 already
// use: poll GET /job?id=, fetch with GET /job?id=&result=1).
static void yue2_handle_synth(const httplib::Request & req, httplib::Response & res) {
    Yue2Request preq;
    std::string err;
    if (!yue2_parse_request(req.body, &preq, &err)) {
        yue2_json_error(res, 400, err);
        return;
    }
    auto job = job_create();
    work_push([job, preq]() mutable { yue2_synth_worker(job, std::move(preq)); });
    res.set_content("{\"id\":\"" + job->id + "\"}", "application/json");
}

static void yue2_register_routes(httplib::Server & svr, const char * models_dir) {
    {
        std::lock_guard<std::mutex> lock(g_yue2_mutex);
        yue2_discover(&g_yue2, models_dir);
    }
    svr.Get("/yue2/props", yue2_handle_props);
    svr.Post("/yue2/warm", yue2_handle_warm);
    svr.Post("/yue2/unload", yue2_handle_unload);
    svr.Post("/yue2/select-model", yue2_handle_select_model);
    svr.Post("/yue2/tokenize-check", yue2_handle_tokenize_check);
    svr.Post("/yue2/synth", yue2_handle_synth);
    fprintf(stderr, "[Server] YuE2 routes registered (models_dir=%s, available=%s)\n", models_dir,
            yue2_available(g_yue2) ? "yes" : "no");
}
