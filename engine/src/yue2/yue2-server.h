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
//   POST /yue2/imatrix          <- yue2_handle_imatrix        (arm/disarm/save activation-
//                                   importance collection for quantize --imatrix; mirrors
//                                   minimax/mm3-server.h's POST /mm3/imatrix — see
//                                   yue2-imatrix.h for what this collects and why)
//
// NOT implemented (rough edge, listed rather than built for time's sake —
// task rule): POST /yue2/vae-decode and POST /yue2/abc-plan, the plan's own
// "lower priority... useful for isolated fixture validation" bring-up
// endpoints. yue2-probe.cpp's --nar-parity/--vae-parity subcommands already
// cover that standalone-validation need from the CLI side; the HTTP
// equivalents were not built this pass since nothing in the M10 gate
// (POST /yue2/synth end-to-end) needs them.

#include "yue2-imatrix.h"
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

    // Quant catalogue — what the UI's LM quant picker is built from. Mirrors
    // minimax/mm3-server.h's own "variants" block (add_role), scoped to just
    // the LM (YuE2 has no per-role VAE quant ladder, only the fixed
    // standard/legacy variant pick handled by /yue2/select-model's
    // vae_variant field). "selected" is the type actually in force (the
    // resolved pick, not the request), so a fallback from a deleted file is
    // visible rather than silent.
    {
        yyjson_mut_val * variants = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, root, "variants", variants);
        yyjson_mut_val * lm_v = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, variants, "lm", lm_v);
        yyjson_mut_val * arr = yyjson_mut_arr(doc);
        yyjson_mut_obj_add_val(doc, lm_v, "available", arr);
        std::string selected;
        for (const auto & v : g_yue2.lm_variants) {
            yyjson_mut_val * e = yyjson_mut_obj(doc);
            yyjson_mut_obj_add_strcpy(doc, e, "type", v.type.c_str());
            yyjson_mut_obj_add_strcpy(doc, e, "filename", v.name.c_str());
            yyjson_mut_obj_add_uint(doc, e, "bytes", v.bytes);
            yyjson_mut_arr_add_val(arr, e);
            if (v.name == g_yue2.lm_file.name) {
                selected = v.type;
            }
        }
        yyjson_mut_obj_add_strcpy(doc, lm_v, "selected", selected.c_str());
        yyjson_mut_obj_add_strcpy(doc, lm_v, "requested", g_yue2.lm_type_want.c_str());
    }

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

// POST /yue2/select-model — {"vae_variant": "standard"|"legacy", "lm_type": "<token>"}.
// Both fields optional/independent. vae_variant: only the active variant is
// ever resident (yue2_load_parts's own contract); if a VAE is currently
// loaded this reloads it, otherwise it just records the pick for the next
// warm/synth. lm_type: "" (or omitted) means auto/best-first
// (yue2_quant_rank order); a specific token (e.g. "Q4_K_M", "Q4_K_M-imat")
// pins discovery to yue2-lm-<token>.gguf. Changing it from the current pick
// unloads the model (yue2_unload has no LM-only free — mirrors mm3's
// full-teardown-then-lazy-reload contract) and re-discovers; the next
// warm/synth loads the new file. Not yet exposed by any UI; added for
// standalone-server quant A/B (docs/plans/yue2/07-quant-ladder.md).
static void yue2_handle_select_model(const httplib::Request & req, httplib::Response & res) {
    std::string variant_str;
    std::string lm_type_str;
    bool        lm_type_given = false;
    if (!req.body.empty()) {
        yyjson_doc * d = yyjson_read(req.body.data(), req.body.size(), 0);
        if (d) {
            yyjson_val * root = yyjson_doc_get_root(d);
            yyjson_val * v    = yyjson_obj_get(root, "vae_variant");
            if (v && yyjson_is_str(v)) {
                variant_str = yyjson_get_str(v);
            }
            yyjson_val * lt = yyjson_obj_get(root, "lm_type");
            if (lt && yyjson_is_str(lt)) {
                lm_type_str   = yyjson_get_str(lt);
                lm_type_given = true;
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

    if (lm_type_given && lm_type_str != g_yue2.lm_type_want) {
        // Full teardown: yue2_unload() drops LM+VAE together (no LM-only
        // free exists), then re-discover pins the new LM file. VAE residency
        // is lost too, but yue2_load_parts's need_vae check reloads it on the
        // next warm/synth same as a cold start.
        yue2_unload(&g_yue2);
        g_yue2.lm_type_want = lm_type_str;
        yue2_discover(&g_yue2, g_yue2.models_dir.c_str(), lm_type_str.empty() ? nullptr : lm_type_str.c_str());
    }

    const bool   want_vae = g_yue2.vae_resident;  // only reload if one is already resident
    std::string  err;
    if (want_vae) {
        if (!yue2_load_parts(&g_yue2, false, true, variant, false, &err)) {
            yue2_json_error(res, 500, err.empty() ? "YuE2 select-model failed" : err);
            return;
        }
    } else {
        g_yue2.vae_loaded_variant = variant;  // recorded for the next warm/synth
    }

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_bool(doc, root, "selected", true);
    yyjson_mut_obj_add_strcpy(doc, root, "vae_variant", YUE2_VAE_VARIANT_NAME[variant]);
    yyjson_mut_obj_add_strcpy(doc, root, "lm_type_want", g_yue2.lm_type_want.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "lm_file", g_yue2.lm_file.found ? g_yue2.lm_file.name.c_str() : "");
    yyjson_mut_obj_add_bool(doc, root, "lm_found", g_yue2.lm_file.found);
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    yyjson_mut_doc_free(doc);
    if (json) {
        free(json);
    }
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

// POST /yue2/imatrix — activation-importance collection for quantization.
//
// See yue2-imatrix.h for what an imatrix is, why 2-3 bit quants need one, and
// why IQ2_XXS/IQ2_XS refuse to run without it. Mirrors
// minimax/mm3-server.h's mm3_handle_imatrix exactly (same action set, same
// response shape) — see that function's own comment for the full rationale;
// noted here only where YuE2 diverges.
//
//   {"action":"start"}                  arm; clears anything already collected
//   {"action":"start","keep":true}      arm and ADD to what is already there
//   {"action":"stop"}                   disarm, keep the accumulator
//   {"action":"status"}                 counts, no state change
//   {"action":"reset"}                  disarm and throw the accumulator away
//   {"action":"save","path":"out.gguf"} write it out (does NOT disarm)
//
// The calibration loop is: start -> N x POST /yue2/synth (poll GET /job?id=
// to completion each) -> save. Unlike MM3's lm-plan driver, a YuE2 /synth
// call IS the whole LM (plan + semantic AR + NAR flow all share one GGUF and
// one tmap_lm), so there is no cheaper LM-only endpoint to prefer here — see
// yue2-pipeline.h's yue2_pipeline_run, which counts one call as one chunk.
static void yue2_handle_imatrix(const httplib::Request & req, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_yue2_mutex);

    yyjson_doc * doc  = req.body.empty() ? nullptr : yyjson_read(req.body.data(), req.body.size(), 0);
    yyjson_val * root = doc ? yyjson_doc_get_root(doc) : nullptr;
    if (!req.body.empty() && (!root || !yyjson_is_obj(root))) {
        if (doc) {
            yyjson_doc_free(doc);
        }
        yue2_json_error(res, 400, "body must be a JSON object");
        return;
    }
    struct DocGuard {
        yyjson_doc * d;
        ~DocGuard() {
            if (d) {
                yyjson_doc_free(d);
            }
        }
    } guard{ doc };

    yyjson_val * action_v = root ? yyjson_obj_get(root, "action") : nullptr;
    std::string  action   = (action_v && yyjson_is_str(action_v)) ? yyjson_get_str(action_v) : "";
    if (action.empty()) {
        action = "status";
    }

    std::vector<std::string> warnings;
    std::string              saved_path;

    if (action == "start") {
        if (!g_yue2.lm_resident) {
            yue2_json_error(res, 503, "the LM is not resident — POST /yue2/warm first");
            return;
        }
        yyjson_val * keep_v = root ? yyjson_obj_get(root, "keep") : nullptr;
        const bool   keep   = keep_v && yyjson_is_true(keep_v);
        if (!keep) {
            yue2_imatrix_reset();
        }

        // Only names that came out of the YuE2 LM GGUF are collectable, and
        // only the 2-D ones can ever be a matmul weight — derived from
        // tmap_lm rather than a name pattern so LoRA factors (if any ever
        // exist) or KV views can't drift into the set. AR blocks
        // (blk.N.attn_*/ffn_*), NAR twins (nar_blk.N.nar_attn_*/nar_ffn_*)
        // and the shared token_embd/output/latent_pos_embed/vae2llm/llm2vae/
        // time_embd tensors are all in this one tmap (05-gguf-layout.md §3).
        g_yue2_imatrix.allow.clear();
        for (const auto & kv : g_yue2.tmap_lm) {
            if (kv.second && ggml_n_dims(kv.second) >= 2) {
                g_yue2_imatrix.allow.insert(kv.first);
            }
        }
        if (g_yue2_imatrix.allow.empty()) {
            yue2_json_error(res, 500, "no LM weight names to collect — tmap_lm is empty");
            return;
        }

        if (g_yue2.lm.output && ggml_is_quantized(g_yue2.lm.output->type)) {
            warnings.push_back(std::string("the resident LM is ") + ggml_type_name(g_yue2.lm.output->type) +
                               " — collect on the f16 or bf16 LM, or the imatrix describes this "
                               "checkpoint's quantization damage instead of the model");
        }
        g_yue2_imatrix.armed = true;
        fprintf(stderr, "[YUE2-IMAT] Armed over %zu LM tensors (LM = %s). Expect synth to run much slower.\n",
                g_yue2_imatrix.allow.size(), g_yue2.lm_file.found ? g_yue2.lm_file.name.c_str() : "?");
    } else if (action == "stop") {
        g_yue2_imatrix.armed = false;
    } else if (action == "reset") {
        g_yue2_imatrix.armed = false;
        yue2_imatrix_reset();
    } else if (action == "save") {
        yyjson_val * path_v = root ? yyjson_obj_get(root, "path") : nullptr;
        saved_path           = (path_v && yyjson_is_str(path_v)) ? yyjson_get_str(path_v) : "";
        if (saved_path.empty()) {
            yue2_json_error(res, 400, "save needs a \"path\"");
            return;
        }
        std::string err;
        if (!yue2_imatrix_save(saved_path, &err)) {
            yue2_json_error(res, 500, err.empty() ? "imatrix save failed" : err);
            return;
        }
        fprintf(stderr, "[YUE2-IMAT] Wrote %s (%zu tensors, %lld rows, %lld runs)\n", saved_path.c_str(),
                g_yue2_imatrix.ent.size(), (long long) yue2_imatrix_total_rows(), (long long) g_yue2_imatrix.runs);
    } else if (action != "status") {
        yue2_json_error(res, 400, "action must be start, stop, status, reset or save");
        return;
    }

    const size_t usable = yue2_imatrix_usable();
    if (g_yue2_imatrix.ent.size() && usable < g_yue2_imatrix.ent.size()) {
        char buf[200];
        snprintf(buf, sizeof(buf),
                 "%zu of %zu seen tensors have NO finite rows and will be omitted — the LM forward is producing "
                 "non-finite activations",
                 g_yue2_imatrix.ent.size() - usable, g_yue2_imatrix.ent.size());
        warnings.push_back(buf);
    }

    yyjson_mut_doc * o    = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * orot = yyjson_mut_obj(o);
    yyjson_mut_doc_set_root(o, orot);
    yyjson_mut_obj_add_bool(o, orot, "armed", g_yue2_imatrix.armed);
    yyjson_mut_obj_add_uint(o, orot, "runs", (uint64_t) g_yue2_imatrix.runs);
    yyjson_mut_obj_add_uint(o, orot, "tensors", (uint64_t) g_yue2_imatrix.ent.size());
    yyjson_mut_obj_add_uint(o, orot, "tensors_expected", (uint64_t) g_yue2_imatrix.allow.size());
    yyjson_mut_obj_add_uint(o, orot, "rows", (uint64_t) yue2_imatrix_total_rows());
    yyjson_mut_obj_add_uint(o, orot, "usable_tensors", (uint64_t) usable);
    yyjson_mut_obj_add_uint(o, orot, "bad_rows", (uint64_t) g_yue2_imatrix.bad_rows);
    yyjson_mut_obj_add_uint(o, orot, "matmuls", (uint64_t) g_yue2_imatrix.nodes);
    yyjson_mut_obj_add_uint(o, orot, "skipped_type", (uint64_t) g_yue2_imatrix.skipped_type);
    yyjson_mut_obj_add_uint(o, orot, "skipped_stride", (uint64_t) g_yue2_imatrix.skipped_stride);
    yyjson_mut_obj_add_uint(o, orot, "skipped_shape", (uint64_t) g_yue2_imatrix.skipped_shape);
    yyjson_mut_obj_add_strcpy(o, orot, "lm", g_yue2.lm_file.found ? g_yue2.lm_file.name.c_str() : "");
    if (!saved_path.empty()) {
        yyjson_mut_obj_add_strcpy(o, orot, "path", saved_path.c_str());
    }
    if (!warnings.empty()) {
        yyjson_mut_val * w = yyjson_mut_arr(o);
        for (const auto & s : warnings) {
            yyjson_mut_arr_add_strcpy(o, w, s.c_str());
        }
        yyjson_mut_obj_add_val(o, orot, "warnings", w);
    }
    char * json = yyjson_mut_write(o, 0, NULL);
    yyjson_mut_doc_free(o);
    res.set_content(json ? json : "{}", "application/json");
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
    svr.Post("/yue2/imatrix", yue2_handle_imatrix);
    fprintf(stderr, "[Server] YuE2 routes registered (models_dir=%s, available=%s)\n", models_dir,
            yue2_available(g_yue2) ? "yes" : "no");
}
