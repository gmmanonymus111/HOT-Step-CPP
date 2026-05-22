#pragma once
// silence-latent.h: read silence_latent.pt from safetensors model directories
//
// PyTorch .pt files are ZIP archives with a data entry containing raw f32.
// The silence_latent tensor is [64, 15000] f32 in PyTorch layout.
// We transpose to [15000, 64] for ggml (64 contiguous per frame).
//
// Reference: convert.py read_silence_latent()

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Minimal ZIP reader for PyTorch .pt files.
// We only need to find and read the "*/data/0" entry.
//
// ZIP local file header: [PK\x03\x04][2B ver][2B flags][2B method][4B datetime]
//   [4B crc32][4B compressed_size][4B uncompressed_size][2B name_len][2B extra_len]
//   [name][extra][data]
// We look for method=0 (stored, not compressed).

static bool sl_read_silence_latent(const char * pt_path,
                                   std::vector<float> & out,
                                   int expected_dim0 = 64,
                                   int expected_dim1 = 15000) {
    FILE * f = fopen(pt_path, "rb");
    if (!f) {
        fprintf(stderr, "[SilenceLatent] Cannot open %s\n", pt_path);
        return false;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    // Read entire file (silence_latent.pt is ~3.7 MB)
    std::vector<uint8_t> buf(file_size);
    if ((long) fread(buf.data(), 1, file_size, f) != file_size) {
        fclose(f);
        fprintf(stderr, "[SilenceLatent] Read error %s\n", pt_path);
        return false;
    }
    fclose(f);

    // Scan ZIP local file headers looking for "*/data/0"
    size_t pos = 0;
    while (pos + 30 <= (size_t) file_size) {
        // Check PK\x03\x04 signature
        if (buf[pos] != 0x50 || buf[pos + 1] != 0x4B ||
            buf[pos + 2] != 0x03 || buf[pos + 3] != 0x04) {
            break;  // No more local headers
        }

        uint16_t method = *(uint16_t *) &buf[pos + 8];
        uint32_t compressed_size = *(uint32_t *) &buf[pos + 18];
        uint32_t uncompressed_size = *(uint32_t *) &buf[pos + 22];
        uint16_t name_len = *(uint16_t *) &buf[pos + 26];
        uint16_t extra_len = *(uint16_t *) &buf[pos + 28];

        size_t name_start = pos + 30;
        size_t data_start = name_start + name_len + extra_len;

        if (name_start + name_len > (size_t) file_size) break;

        // Check if name ends with "/data/0"
        std::string entry_name((char *) &buf[name_start], name_len);
        bool is_data0 = (entry_name.size() >= 7 &&
                         entry_name.compare(entry_name.size() - 7, 7, "/data/0") == 0);

        if (is_data0 && method == 0) {
            // Stored (uncompressed) — read raw f32 data
            size_t nbytes = uncompressed_size;
            size_t expected_bytes = (size_t) expected_dim0 * expected_dim1 * sizeof(float);
            if (nbytes != expected_bytes) {
                fprintf(stderr, "[SilenceLatent] WARNING: expected %zu bytes, got %u in %s\n",
                        expected_bytes, uncompressed_size, pt_path);
                // Try to use what we have
                if (nbytes < expected_bytes) {
                    return false;
                }
            }
            if (data_start + nbytes > (size_t) file_size) {
                fprintf(stderr, "[SilenceLatent] Truncated data in %s\n", pt_path);
                return false;
            }

            // Source: [64, 15000] f32 (PyTorch row-major: 15000 contiguous per row)
            // Target: [15000, 64] f32 (ggml: 64 contiguous per frame)
            const float * src = (const float *) &buf[data_start];
            out.resize(expected_dim0 * expected_dim1);
            for (int i = 0; i < expected_dim0; i++) {
                for (int j = 0; j < expected_dim1; j++) {
                    out[j * expected_dim0 + i] = src[i * expected_dim1 + j];
                }
            }

            fprintf(stderr, "[SilenceLatent] Loaded [%d, %d] f32 from %s (%.1f MB)\n",
                    expected_dim1, expected_dim0, pt_path,
                    (float) out.size() * sizeof(float) / (1024 * 1024));
            return true;
        }

        // Skip to next local header
        pos = data_start + compressed_size;
    }

    fprintf(stderr, "[SilenceLatent] No data/0 entry found in %s\n", pt_path);
    return false;
}
