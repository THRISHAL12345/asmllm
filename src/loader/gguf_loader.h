/*
 * src/loader/gguf_loader.h
 *
 * Minimal C glue header for parsing GGUF model files.
 * Architectural Note: This file and its implementation are thin C scaffolding
 * responsible solely for file I/O, binary header validation, and reading tensor
 * byte offsets. No tensor computation, forward pass math, or buffer mutability
 * is performed here.
 */

#ifndef GGUF_LOADER_H
#define GGUF_LOADER_H

#include <stdint.h>
#include <stddef.h>

#define GGUF_MAGIC 0x46554747 // "GGUF"

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_count;
} gguf_header_t;

typedef struct {
    char name[128];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
} gguf_tensor_info_t;

#ifdef __cplusplus
extern "C" {
#endif

int gguf_validate_header(const gguf_header_t* header);

#ifdef __cplusplus
}
#endif

#endif // GGUF_LOADER_H
