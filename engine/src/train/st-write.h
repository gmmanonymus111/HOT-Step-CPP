#pragma once
// st-write.h: minimal safetensors WRITER (the engine's safetensors.h is
// read-only). Emits [8B LE header_len][JSON header][raw data], data_offsets
// relative to the data section, no inter-tensor padding, header space-padded
// so (8 + header_len) % 8 == 0.
//
// Source data is ALWAYS f32; storage dtype is chosen per file.
//
// docs/plans/2026-07-27-preprocess-implementation.md §3.2

#include "yyjson.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

enum STWDType { STW_F32, STW_BF16 };

struct STWTensor {
    std::string          name;
    std::vector<int64_t> shape;  // row-major; shape[0] varies slowest
    const float *        data;   // prod(shape) f32 values, row-major
};

// float -> bfloat16 with round-to-nearest-even.
static inline uint16_t stw_f32_to_bf16(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    if (((b >> 23) & 0xFFu) == 0xFFu && (b & 0x7FFFFFu)) {
        return (uint16_t) ((b >> 16) | 0x40u);  // keep NaN
    }
    b += 0x7FFFu + ((b >> 16) & 1u);
    return (uint16_t) (b >> 16);
}

static inline int64_t stw_numel(const std::vector<int64_t> & shape) {
    int64_t n = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        n *= shape[i];
    }
    return shape.empty() ? 0 : n;
}

// Replace `path` with `tmp`, never leaving the destination missing.
//
// A bare remove(path) + rename(tmp, path) (needed because Windows rename()
// fails onto an existing file) destroys a previously VALID cache whenever the
// rename then loses -- an AV/indexer handle on the just-closed temp, a
// read-only directory, a cross-volume temp. The caller reports song_fail and
// the user ends up with fewer cached songs than before the run. So: move the
// old file aside first, rename into place, and only then drop the aside copy;
// on failure the old file is put back.
static bool stw_replace_file(const char * tmp, const char * path) {
    std::string aside;
    bool        have_aside = false;

    FILE * probe = fopen(path, "rb");
    if (probe) {
        fclose(probe);
        aside = std::string(path) + ".__old__";
        remove(aside.c_str());
        have_aside = (rename(path, aside.c_str()) == 0);
        if (!have_aside) {
            remove(path);  // could not move it aside; fall back to plain replace
        }
    }

    if (rename(tmp, path) != 0) {
        if (have_aside) {
            rename(aside.c_str(), path);  // put the good file back
        }
        return false;
    }
    if (have_aside) {
        remove(aside.c_str());
    }
    return true;
}

// Write `tensors` (in order) + `__metadata__` to `path`. Returns false and
// leaves no partial file on failure. Writes to `path + ".__writing__"` then
// renames (removing an existing destination first -- Windows rename fails
// otherwise).
static bool st_write_file(const char *                                             path,
                          const std::vector<STWTensor> &                           tensors,
                          const std::vector<std::pair<std::string, std::string>> & metadata,
                          STWDType                                                 dtype) {
    const size_t esz = (dtype == STW_F32) ? 4u : 2u;
    const char * dts = (dtype == STW_F32) ? "F32" : "BF16";

    // 1. Offsets (relative to the data section, no padding between tensors).
    std::vector<uint64_t> starts(tensors.size()), ends(tensors.size());
    uint64_t              cursor = 0;
    for (size_t i = 0; i < tensors.size(); i++) {
        int64_t n = stw_numel(tensors[i].shape);
        if (n < 0 || tensors[i].data == NULL) {
            fprintf(stderr, "[STWrite] invalid tensor '%s'\n", tensors[i].name.c_str());
            return false;
        }
        starts[i] = cursor;
        cursor += (uint64_t) n * (uint64_t) esz;
        ends[i] = cursor;
    }

    // 2. JSON header (yyjson -- captions in __metadata__ carry quotes/newlines).
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    if (!metadata.empty()) {
        yyjson_mut_val * md = yyjson_mut_obj(doc);
        for (size_t i = 0; i < metadata.size(); i++) {
            yyjson_mut_obj_add(md, yyjson_mut_strncpy(doc, metadata[i].first.c_str(), metadata[i].first.size()),
                               yyjson_mut_strncpy(doc, metadata[i].second.c_str(), metadata[i].second.size()));
        }
        yyjson_mut_obj_add_val(doc, root, "__metadata__", md);
    }

    for (size_t i = 0; i < tensors.size(); i++) {
        yyjson_mut_val * e = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_str(doc, e, "dtype", dts);
        yyjson_mut_val * sh = yyjson_mut_obj_add_arr(doc, e, "shape");
        for (size_t d = 0; d < tensors[i].shape.size(); d++) {
            yyjson_mut_arr_add_sint(doc, sh, tensors[i].shape[d]);
        }
        yyjson_mut_val * off = yyjson_mut_obj_add_arr(doc, e, "data_offsets");
        yyjson_mut_arr_add_uint(doc, off, starts[i]);
        yyjson_mut_arr_add_uint(doc, off, ends[i]);
        yyjson_mut_obj_add(root, yyjson_mut_strncpy(doc, tensors[i].name.c_str(), tensors[i].name.size()), e);
    }

    size_t hdr_len = 0;
    char * hdr     = yyjson_mut_write(doc, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &hdr_len);
    yyjson_mut_doc_free(doc);
    if (!hdr) {
        fprintf(stderr, "[STWrite] header serialization failed for %s\n", path);
        return false;
    }

    // 3. Space-pad so (8 + header_len) % 8 == 0.
    std::string header(hdr, hdr_len);
    free(hdr);
    while ((8 + header.size()) % 8 != 0) {
        header.push_back(' ');
    }

    // 4. Write to a temp file, then rename.
    std::string tmp = std::string(path) + ".__writing__";
    FILE *      f   = fopen(tmp.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[STWrite] cannot open %s for writing\n", tmp.c_str());
        return false;
    }

    uint64_t hl = (uint64_t) header.size();
    uint8_t  hl_le[8];
    for (int i = 0; i < 8; i++) {
        hl_le[i] = (uint8_t) ((hl >> (8 * i)) & 0xFFu);  // explicit little-endian
    }
    bool ok = (fwrite(hl_le, 1, 8, f) == 8);
    ok      = ok && (fwrite(header.data(), 1, header.size(), f) == header.size());

    // 5. Stream tensor data in offset order.
    std::vector<uint16_t> scratch;
    for (size_t i = 0; ok && i < tensors.size(); i++) {
        const int64_t n = stw_numel(tensors[i].shape);
        const float * p = tensors[i].data;
        if (dtype == STW_F32) {
            size_t left = (size_t) n;
            while (ok && left > 0) {
                size_t chunk = left > 65536 ? 65536 : left;
                ok           = (fwrite(p, sizeof(float), chunk, f) == chunk);
                p += chunk;
                left -= chunk;
            }
        } else {
            if (scratch.size() < 65536) {
                scratch.resize(65536);
            }
            size_t left = (size_t) n;
            while (ok && left > 0) {
                size_t chunk = left > 65536 ? 65536 : left;
                for (size_t k = 0; k < chunk; k++) {
                    scratch[k] = stw_f32_to_bf16(p[k]);
                }
                ok = (fwrite(scratch.data(), sizeof(uint16_t), chunk, f) == chunk);
                p += chunk;
                left -= chunk;
            }
        }
    }

    if (ok) {
        ok = (fflush(f) == 0) && (ferror(f) == 0);
    }
    fclose(f);

    if (!ok) {
        remove(tmp.c_str());
        fprintf(stderr, "[STWrite] write failed for %s\n", path);
        return false;
    }

    if (!stw_replace_file(tmp.c_str(), path)) {
        fprintf(stderr, "[STWrite] rename %s -> %s failed (previous file left intact)\n", tmp.c_str(), path);
        remove(tmp.c_str());
        return false;
    }
    return true;
}
