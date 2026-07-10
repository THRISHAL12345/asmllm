/*
 * src/loader/gguf_loader.c
 *
 * Full C implementation for parsing GGUF v2/v3 model files.
 * Architectural Note: This C file only handles file I/O, binary header validation,
 * metadata parsing, tensor descriptor table parsing, and pointer offsets.
 * It does not contain any tensor math, activation processing, or SIMD code.
 */

#include "gguf_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int gguf_validate_header(const gguf_header_t* header) {
    if (!header) return 0;
    if (header->magic != GGUF_MAGIC) {
        return 0;
    }
    if (header->version < 2 || header->version > 3) {
        return 0;
    }
    return 1;
}

static int read_string(const uint8_t* buf, size_t size, size_t* pos, char* out_str, size_t max_len) {
    if (*pos + sizeof(uint64_t) > size) return 0;
    uint64_t len = 0;
    memcpy(&len, buf + *pos, sizeof(uint64_t));
    *pos += sizeof(uint64_t);

    if (*pos + len > size) return 0;
    size_t copy_len = len < (max_len - 1) ? (size_t)len : (max_len - 1);
    memcpy(out_str, buf + *pos, copy_len);
    out_str[copy_len] = '\0';
    *pos += (size_t)len;
    return 1;
}

static void skip_value(const uint8_t* buf, size_t size, size_t* pos, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            *pos += 1;
            break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            *pos += 2;
            break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            *pos += 4;
            break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            *pos += 8;
            break;
        case GGUF_TYPE_STRING: {
            char dummy[2];
            read_string(buf, size, pos, dummy, sizeof(dummy));
            break;
        }
        case GGUF_TYPE_ARRAY: {
            if (*pos + sizeof(uint32_t) + sizeof(uint64_t) > size) return;
            uint32_t elem_type = 0;
            uint64_t count = 0;
            memcpy(&elem_type, buf + *pos, sizeof(uint32_t));
            *pos += sizeof(uint32_t);
            memcpy(&count, buf + *pos, sizeof(uint64_t));
            *pos += sizeof(uint64_t);
            for (uint64_t i = 0; i < count && *pos < size; ++i) {
                skip_value(buf, size, pos, elem_type);
            }
            break;
        }
        default:
            break;
    }
}

int gguf_load_from_file(const char* filepath, gguf_context_t** out_ctx) {
    if (!filepath || !out_ctx) return 0;

    FILE* f = fopen(filepath, "rb");
    if (!f) return 0;

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fsize < (long)sizeof(gguf_header_t)) {
        fclose(f);
        return 0;
    }

    uint8_t* buf = (uint8_t*)malloc(fsize);
    if (!buf) {
        fclose(f);
        return 0;
    }

    if (fread(buf, 1, fsize, f) != (size_t)fsize) {
        free(buf);
        fclose(f);
        return 0;
    }
    fclose(f);

    gguf_context_t* ctx = (gguf_context_t*)calloc(1, sizeof(gguf_context_t));
    if (!ctx) {
        free(buf);
        return 0;
    }
    ctx->file_data = buf;
    ctx->file_size = (size_t)fsize;

    size_t pos = 0;
    memcpy(&ctx->header, buf + pos, sizeof(gguf_header_t));
    pos += sizeof(gguf_header_t);

    if (!gguf_validate_header(&ctx->header)) {
        free(buf);
        free(ctx);
        return 0;
    }

    uint64_t alignment = GGUF_DEFAULT_ALIGNMENT;

    // Parse KV pairs
    if (ctx->header.kv_count > 0) {
        ctx->kv_pairs = (gguf_kv_t*)calloc(ctx->header.kv_count, sizeof(gguf_kv_t));
        for (uint64_t i = 0; i < ctx->header.kv_count && pos < ctx->file_size; ++i) {
            char key[128];
            if (!read_string(buf, ctx->file_size, &pos, key, sizeof(key))) break;
            if (ctx->kv_pairs) {
                strncpy(ctx->kv_pairs[i].key, key, sizeof(ctx->kv_pairs[i].key) - 1);
            }

            if (pos + sizeof(uint32_t) > ctx->file_size) break;
            uint32_t val_type = 0;
            memcpy(&val_type, buf + pos, sizeof(uint32_t));
            pos += sizeof(uint32_t);

            if (ctx->kv_pairs) {
                ctx->kv_pairs[i].type = (gguf_value_type_t)val_type;
            }

            if (strcmp(key, "general.alignment") == 0 && val_type == GGUF_TYPE_UINT32) {
                uint32_t align_val = GGUF_DEFAULT_ALIGNMENT;
                memcpy(&align_val, buf + pos, sizeof(uint32_t));
                alignment = align_val;
            }

            skip_value(buf, ctx->file_size, &pos, val_type);
        }
    }

    // Parse Tensor Descriptors
    if (ctx->header.tensor_count > 0) {
        ctx->tensors = (gguf_tensor_info_t*)calloc(ctx->header.tensor_count, sizeof(gguf_tensor_info_t));
        for (uint64_t i = 0; i < ctx->header.tensor_count && pos < ctx->file_size; ++i) {
            gguf_tensor_info_t* ti = &ctx->tensors[i];
            if (!read_string(buf, ctx->file_size, &pos, ti->name, sizeof(ti->name))) break;

            if (pos + sizeof(uint32_t) > ctx->file_size) break;
            memcpy(&ti->n_dims, buf + pos, sizeof(uint32_t));
            pos += sizeof(uint32_t);

            for (uint32_t d = 0; d < ti->n_dims && d < 4; ++d) {
                if (pos + sizeof(uint64_t) > ctx->file_size) break;
                memcpy(&ti->dims[d], buf + pos, sizeof(uint64_t));
                pos += sizeof(uint64_t);
            }

            if (pos + sizeof(uint32_t) + sizeof(uint64_t) > ctx->file_size) break;
            memcpy(&ti->type, buf + pos, sizeof(uint32_t));
            pos += sizeof(uint32_t);
            memcpy(&ti->offset, buf + pos, sizeof(uint64_t));
            pos += sizeof(uint64_t);
        }
    }

    // Compute aligned data offset
    size_t rem = pos % alignment;
    if (rem != 0) {
        pos += (alignment - rem);
    }
    ctx->data_offset = pos;

    *out_ctx = ctx;
    return 1;
}

int gguf_find_tensor(const gguf_context_t* ctx, const char* name, const gguf_tensor_info_t** out_tensor) {
    if (!ctx || !name || !out_tensor) return 0;
    for (uint64_t i = 0; i < ctx->header.tensor_count; ++i) {
        if (strcmp(ctx->tensors[i].name, name) == 0) {
            *out_tensor = &ctx->tensors[i];
            return 1;
        }
    }
    return 0;
}

const uint8_t* gguf_get_tensor_data(const gguf_context_t* ctx, const gguf_tensor_info_t* tensor) {
    if (!ctx || !tensor) return NULL;
    size_t abs_offset = ctx->data_offset + tensor->offset;
    if (abs_offset >= ctx->file_size) return NULL;
    return ctx->file_data + abs_offset;
}

void gguf_free(gguf_context_t* ctx) {
    if (!ctx) return;
    if (ctx->kv_pairs) free(ctx->kv_pairs);
    if (ctx->tensors) free(ctx->tensors);
    if (ctx->file_data) free(ctx->file_data);
    free(ctx);
}
