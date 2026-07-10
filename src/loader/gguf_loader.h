/*
 * src/loader/gguf_loader.h
 *
 * Full C implementation header for parsing GGUF v2/v3 model files.
 * Architectural Note: This file and its implementation are C scaffolding
 * responsible solely for file I/O, binary header validation, key-value metadata
 * parsing, tensor descriptor parsing, and reading weight data buffers.
 * No tensor computation, forward pass math, or SIMD kernel code is performed here.
 */

#ifndef GGUF_LOADER_H
#define GGUF_LOADER_H

#include <stdint.h>
#include <stddef.h>

#define GGUF_MAGIC 0x46554747 // "GGUF"
#define GGUF_DEFAULT_ALIGNMENT 32

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_count;
} gguf_header_t;

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12
} gguf_value_type_t;

typedef struct {
    char name[128];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
} gguf_tensor_info_t;

typedef struct {
    char key[128];
    gguf_value_type_t type;
    union {
        uint64_t u64;
        int64_t i64;
        double f64;
        char* str;
    } value;
} gguf_kv_t;

typedef struct {
    gguf_header_t header;
    gguf_kv_t* kv_pairs;
    gguf_tensor_info_t* tensors;
    uint8_t* file_data;
    size_t file_size;
    size_t data_offset;
} gguf_context_t;

#if defined(_WIN32) || defined(__CYGWIN__)
#  define GGUF_API __declspec(dllexport)
#else
#  define GGUF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

GGUF_API int gguf_validate_header(const gguf_header_t* header);

GGUF_API int gguf_load_from_file(const char* filepath, gguf_context_t** out_ctx);

GGUF_API int gguf_find_tensor(const gguf_context_t* ctx, const char* name, const gguf_tensor_info_t** out_tensor);

GGUF_API const uint8_t* gguf_get_tensor_data(const gguf_context_t* ctx, const gguf_tensor_info_t* tensor);

GGUF_API void gguf_free(gguf_context_t* ctx);

#ifdef __cplusplus
}
#endif

#endif // GGUF_LOADER_H
