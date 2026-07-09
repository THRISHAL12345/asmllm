/*
 * src/runtime/threadpool.h
 *
 * Minimal C glue header for multi-threaded work dispatch and core affinity.
 * Architectural Compliance:
 *   This file and its implementation contain ONLY thread pool management,
 *   synchronization primitives, and disjoint row-tiling logic. It contains
 *   zero tensor arithmetic or compiler intrinsics. Every worker thread
 *   invokes the hand-written assembly kernel `asm_matmul_q4` exclusively.
 */

#ifndef ASMLLM_THREADPOOL_H
#define ASMLLM_THREADPOOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(_WIN64)
  #ifdef ASMLLM_BUILD_DLL
    #define ASMLLM_API __declspec(dllexport)
  #else
    #define ASMLLM_API __declspec(dllimport)
  #endif
#else
  #define ASMLLM_API __attribute__((visibility("default")))
#endif

/*
 * Initialize the thread pool with `num_threads` worker threads.
 * Worker threads are explicitly pinned to CPU cores [0 .. num_threads-1]
 * via OS affinity APIs to eliminate thread migration overhead.
 */
ASMLLM_API void asm_threadpool_init(int num_threads);

/*
 * Shutdown the thread pool and join worker threads.
 */
ASMLLM_API void asm_threadpool_shutdown(void);

/*
 * Multi-threaded Q4_0 matrix-vector multiply dispatcher.
 * Partitions output rows M across active worker threads in cache-line-aligned
 * blocks (multiples of 16 rows / 64 bytes) to guarantee zero false sharing.
 */
ASMLLM_API void asm_matmul_q4_mt(
    const uint8_t* qweights,
    const float*   scales,
    const float*   x,
    float*         y,
    int64_t        M,
    int64_t        K,
    int            num_threads
);

ASMLLM_API void asm_matmul_q8_mt(
    const int8_t*  qweights,
    const float*   scales,
    const float*   x,
    float*         y,
    int64_t        M,
    int64_t        K,
    int            num_threads
);

ASMLLM_API void asm_matmul_q5_mt(
    const uint8_t* ql,
    const uint8_t* qh,
    const float*   scales,
    const float*   x,
    float*         y,
    int64_t        M,
    int64_t        K,
    int            num_threads
);

ASMLLM_API void asm_threadpool_dispatch_q8(
    const int8_t*  qweights,
    const float*   scales,
    const float*   x,
    float*         y,
    int64_t        M,
    int64_t        K
);

ASMLLM_API void asm_threadpool_dispatch_q5(
    const uint8_t* ql,
    const uint8_t* qh,
    const float*   scales,
    const float*   x,
    float*         y,
    int64_t        M,
    int64_t        K
);

/*
 * Returns the currently initialized thread pool size.
 */
ASMLLM_API int asm_threadpool_get_num_threads(void);

#ifdef __cplusplus
}
#endif

#endif // ASMLLM_THREADPOOL_H
