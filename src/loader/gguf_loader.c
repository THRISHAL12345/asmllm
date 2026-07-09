/*
 * src/loader/gguf_loader.c
 *
 * Minimal C glue implementation for parsing GGUF model files.
 * Architectural Note: This C file only handles file I/O and header parsing.
 * It does not contain any tensor math, activation processing, or SIMD code.
 */

#include "gguf_loader.h"

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
