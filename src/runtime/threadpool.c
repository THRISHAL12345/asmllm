/*
 * src/runtime/threadpool.c
 *
 * Minimal C glue implementation for multi-threaded work dispatch and core affinity.
 * Architectural Compliance:
 *   This file contains ONLY thread pool management, synchronization primitives,
 *   OS core affinity setting, and disjoint row-tiling logic. It contains
 *   zero tensor arithmetic or compiler intrinsics. Every worker thread
 *   invokes the hand-written assembly kernel `asm_matmul_q4` exclusively.
 */

#define ASMLLM_BUILD_DLL
#include "threadpool.h"

#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32) || defined(_WIN64)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #define _GNU_SOURCE
  #include <pthread.h>
  #include <unistd.h>
  #include <sched.h>
#endif

extern void asm_matmul_q4(
    const uint8_t* qweights,
    const float*   scales,
    const float*   x,
    float*         y,
    int64_t        M,
    int64_t        K
);

extern void asm_matmul_q8(
    const int8_t*  qweights,
    const float*   scales,
    const float*   x,
    float*         y,
    int64_t        M,
    int64_t        K
);

extern void asm_matmul_q5(
    const uint8_t* ql,
    const uint8_t* qh,
    const float*   scales,
    const float*   x,
    float*         y,
    int64_t        M,
    int64_t        K
);

#define MAX_WORKERS 64

typedef struct {
    int thread_id;
    int core_id;
    volatile int task_ready;
    int quant_type; // 0 = Q4_0, 1 = Q8_0, 2 = Q5_0

#if defined(_WIN32) || defined(_WIN64)
    HANDLE start_event;
    HANDLE done_event;
#endif

    const uint8_t* qweights;
    const uint8_t* qh;
    const float*   scales;
    const float*   x;
    float*         y;
    int64_t        M;
    int64_t        K;
    int64_t        row_start;
    int64_t        row_end;
} worker_task_t;

static worker_task_t g_tasks[MAX_WORKERS];
static volatile int g_pool_size = 0;
static volatile int g_shutdown = 0;
static volatile int g_ready_workers = 0;

#if defined(_WIN32) || defined(_WIN64)

static HANDLE g_threads[MAX_WORKERS];
static HANDLE g_ready_event;
static int g_events_initialized = 0;

static DWORD WINAPI worker_main_win32(LPVOID arg) {
    worker_task_t* task = (worker_task_t*)arg;
    InterlockedIncrement((LONG*)&g_ready_workers);
    SetEvent(g_ready_event);

    while (1) {
        WaitForSingleObject(task->start_event, INFINITE);
        if (g_shutdown) break;

        int64_t row_count = task->row_end - task->row_start;
        if (row_count > 0) {
            int64_t s_floats_per_row = task->K / 32;
            const float* s_tile = task->scales + task->row_start * s_floats_per_row;
            float*       y_tile = task->y + task->row_start;

            if (task->quant_type == 0) {
                int64_t q_bytes_per_row = task->K / 2;
                const uint8_t* q_tile = task->qweights + task->row_start * q_bytes_per_row;
                asm_matmul_q4(q_tile, s_tile, task->x, y_tile, row_count, task->K);
            } else if (task->quant_type == 1) {
                int64_t q_bytes_per_row = task->K;
                const int8_t* q_tile = (const int8_t*)task->qweights + task->row_start * q_bytes_per_row;
                asm_matmul_q8(q_tile, s_tile, task->x, y_tile, row_count, task->K);
            } else if (task->quant_type == 2) {
                int64_t ql_bytes_per_row = task->K / 2;
                int64_t qh_bytes_per_row = task->K / 8;
                const uint8_t* ql_tile = task->qweights + task->row_start * ql_bytes_per_row;
                const uint8_t* qh_tile = task->qh + task->row_start * qh_bytes_per_row;
                asm_matmul_q5(ql_tile, qh_tile, s_tile, task->x, y_tile, row_count, task->K);
            }
        }

        SetEvent(task->done_event);
    }
    return 0;
}

ASMLLM_API void asm_threadpool_init(int num_threads) {
    if (g_pool_size == num_threads && g_pool_size > 0) return;
    if (g_pool_size > 0) {
        asm_threadpool_shutdown();
    }
    if (num_threads < 1) num_threads = 1;
    if (num_threads > MAX_WORKERS) num_threads = MAX_WORKERS;

    if (!g_events_initialized) {
        g_ready_event = CreateEvent(NULL, FALSE, FALSE, NULL);
        for (int i = 0; i < MAX_WORKERS; i++) {
            g_tasks[i].start_event = CreateEvent(NULL, FALSE, FALSE, NULL);
            g_tasks[i].done_event  = CreateEvent(NULL, FALSE, FALSE, NULL);
        }
        g_events_initialized = 1;
    }

    g_shutdown = 0;
    g_ready_workers = 0;
    g_pool_size = num_threads;

    for (int i = 0; i < num_threads; i++) {
        g_tasks[i].thread_id = i;
        g_tasks[i].core_id = i;
        g_threads[i] = CreateThread(NULL, 0, worker_main_win32, &g_tasks[i], 0, NULL);
    }

    while (g_ready_workers < num_threads) {
        WaitForSingleObject(g_ready_event, 10);
    }
}

ASMLLM_API void asm_threadpool_shutdown(void) {
    if (g_pool_size <= 0) return;
    g_shutdown = 1;
    for (int i = 0; i < g_pool_size; i++) {
        SetEvent(g_tasks[i].start_event);
    }

    WaitForMultipleObjects((DWORD)g_pool_size, g_threads, TRUE, INFINITE);
    for (int i = 0; i < g_pool_size; i++) {
        CloseHandle(g_threads[i]);
    }
    g_pool_size = 0;
}

#else

// POSIX implementation
static pthread_t g_threads[MAX_WORKERS];
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_start_cond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_done_cond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_ready_cond = PTHREAD_COND_INITIALIZER;

static void* worker_main_posix(void* arg) {
    worker_task_t* task = (worker_task_t*)arg;

#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(task->core_id % CPU_SETSIZE, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif

    pthread_mutex_lock(&g_mutex);
    g_ready_workers++;
    pthread_cond_broadcast(&g_ready_cond);
    pthread_mutex_unlock(&g_mutex);

    while (1) {
        pthread_mutex_lock(&g_mutex);
        while (!task->task_ready && !g_shutdown) {
            pthread_cond_wait(&g_start_cond, &g_mutex);
        }
        if (g_shutdown) {
            pthread_mutex_unlock(&g_mutex);
            break;
        }
        pthread_mutex_unlock(&g_mutex);

        int64_t row_count = task->row_end - task->row_start;
        if (row_count > 0) {
            int64_t s_floats_per_row = task->K / 32;
            const float* s_tile = task->scales + task->row_start * s_floats_per_row;
            float*       y_tile = task->y + task->row_start;

            if (task->quant_type == 0) {
                int64_t q_bytes_per_row = task->K / 2;
                const uint8_t* q_tile = task->qweights + task->row_start * q_bytes_per_row;
                asm_matmul_q4(q_tile, s_tile, task->x, y_tile, row_count, task->K);
            } else if (task->quant_type == 1) {
                int64_t q_bytes_per_row = task->K;
                const int8_t* q_tile = (const int8_t*)task->qweights + task->row_start * q_bytes_per_row;
                asm_matmul_q8(q_tile, s_tile, task->x, y_tile, row_count, task->K);
            } else if (task->quant_type == 2) {
                int64_t ql_bytes_per_row = task->K / 2;
                int64_t qh_bytes_per_row = task->K / 8;
                const uint8_t* ql_tile = task->qweights + task->row_start * ql_bytes_per_row;
                const uint8_t* qh_tile = task->qh + task->row_start * qh_bytes_per_row;
                asm_matmul_q5(ql_tile, qh_tile, s_tile, task->x, y_tile, row_count, task->K);
            }
        }

        pthread_mutex_lock(&g_mutex);
        task->task_ready = 0;
        pthread_cond_broadcast(&g_done_cond);
        pthread_mutex_unlock(&g_mutex);
    }
    return NULL;
}

ASMLLM_API void asm_threadpool_init(int num_threads) {
    if (g_pool_size == num_threads && g_pool_size > 0) return;
    if (g_pool_size > 0) {
        asm_threadpool_shutdown();
    }
    if (num_threads < 1) num_threads = 1;
    if (num_threads > MAX_WORKERS) num_threads = MAX_WORKERS;

    pthread_mutex_lock(&g_mutex);
    g_shutdown = 0;
    g_ready_workers = 0;
    g_pool_size = num_threads;
    for (int i = 0; i < num_threads; i++) {
        g_tasks[i].task_ready = 0;
    }
    pthread_mutex_unlock(&g_mutex);

    for (int i = 0; i < num_threads; i++) {
        g_tasks[i].thread_id = i;
        g_tasks[i].core_id = i;
        pthread_create(&g_threads[i], NULL, worker_main_posix, &g_tasks[i]);
    }

    pthread_mutex_lock(&g_mutex);
    while (g_ready_workers < num_threads) {
        pthread_cond_wait(&g_ready_cond, &g_mutex);
    }
    pthread_mutex_unlock(&g_mutex);
}

ASMLLM_API void asm_threadpool_shutdown(void) {
    if (g_pool_size <= 0) return;
    pthread_mutex_lock(&g_mutex);
    g_shutdown = 1;
    pthread_cond_broadcast(&g_start_cond);
    pthread_mutex_unlock(&g_mutex);

    for (int i = 0; i < g_pool_size; i++) {
        pthread_join(g_threads[i], NULL);
    }
    g_pool_size = 0;
}

#endif

ASMLLM_API int asm_threadpool_get_num_threads(void) {
    return g_pool_size;
}

ASMLLM_API void asm_matmul_q4_mt(
    const uint8_t* qweights,
    const float*   scales,
    const float*   x,
    float*         y,
    int64_t        M,
    int64_t        K,
    int            num_threads
) {
    if (num_threads <= 1 || M < 32) {
        asm_matmul_q4(qweights, scales, x, y, M, K);
        return;
    }

    if (g_pool_size != num_threads) {
        asm_threadpool_init(num_threads);
    }

    int64_t align_rows = 16;
    int64_t tile_rows = (M + num_threads - 1) / num_threads;
    tile_rows = ((tile_rows + align_rows - 1) / align_rows) * align_rows;

#if defined(_WIN32) || defined(_WIN64)
    HANDLE done_events[MAX_WORKERS];
    for (int i = 0; i < num_threads; i++) {
        int64_t r_start = (int64_t)i * tile_rows;
        int64_t r_end   = r_start + tile_rows;
        if (r_start > M) r_start = M;
        if (r_end > M)   r_end = M;

        g_tasks[i].qweights  = qweights;
        g_tasks[i].scales    = scales;
        g_tasks[i].x         = x;
        g_tasks[i].y         = y;
        g_tasks[i].M         = M;
        g_tasks[i].K         = K;
        g_tasks[i].row_start = r_start;
        g_tasks[i].row_end   = r_end;
        g_tasks[i].quant_type = 0; // Q4_0
        done_events[i]       = g_tasks[i].done_event;
        SetEvent(g_tasks[i].start_event);
    }

    WaitForMultipleObjects((DWORD)num_threads, done_events, TRUE, INFINITE);
#else
    pthread_mutex_lock(&g_mutex);
    for (int i = 0; i < num_threads; i++) {
        int64_t r_start = (int64_t)i * tile_rows;
        int64_t r_end   = r_start + tile_rows;
        if (r_start > M) r_start = M;
        if (r_end > M)   r_end = M;

        g_tasks[i].qweights  = qweights;
        g_tasks[i].scales    = scales;
        g_tasks[i].x         = x;
        g_tasks[i].y         = y;
        g_tasks[i].M         = M;
        g_tasks[i].K         = K;
        g_tasks[i].row_start = r_start;
        g_tasks[i].row_end   = r_end;
        g_tasks[i].quant_type = 0;
        g_tasks[i].task_ready = 1;
    }

    pthread_cond_broadcast(&g_start_cond);

    for (int i = 0; i < num_threads; i++) {
        while (g_tasks[i].task_ready) {
            pthread_cond_wait(&g_done_cond, &g_mutex);
        }
    }
    pthread_mutex_unlock(&g_mutex);
#endif
}

ASMLLM_API void asm_threadpool_dispatch_q8(
    const int8_t*  qweights,
    const float*   scales,
    const float*   x,
    float*         y,
    int64_t        M,
    int64_t        K
) {
    int num_threads = g_pool_size;
    if (num_threads <= 1 || M < 32) {
        asm_matmul_q8(qweights, scales, x, y, M, K);
        return;
    }

    int64_t align_rows = 16;
    int64_t tile_rows = (M + num_threads - 1) / num_threads;
    tile_rows = ((tile_rows + align_rows - 1) / align_rows) * align_rows;

#if defined(_WIN32) || defined(_WIN64)
    HANDLE done_events[MAX_WORKERS];
    for (int i = 0; i < num_threads; i++) {
        int64_t r_start = (int64_t)i * tile_rows;
        int64_t r_end   = r_start + tile_rows;
        if (r_start > M) r_start = M;
        if (r_end > M)   r_end = M;

        g_tasks[i].qweights   = (const uint8_t*)qweights;
        g_tasks[i].scales     = scales;
        g_tasks[i].x          = x;
        g_tasks[i].y          = y;
        g_tasks[i].M          = M;
        g_tasks[i].K          = K;
        g_tasks[i].row_start  = r_start;
        g_tasks[i].row_end    = r_end;
        g_tasks[i].quant_type = 1; // Q8_0
        done_events[i]        = g_tasks[i].done_event;
        SetEvent(g_tasks[i].start_event);
    }

    WaitForMultipleObjects((DWORD)num_threads, done_events, TRUE, INFINITE);
#else
    pthread_mutex_lock(&g_mutex);
    for (int i = 0; i < num_threads; i++) {
        int64_t r_start = (int64_t)i * tile_rows;
        int64_t r_end   = r_start + tile_rows;
        if (r_start > M) r_start = M;
        if (r_end > M)   r_end = M;

        g_tasks[i].qweights   = (const uint8_t*)qweights;
        g_tasks[i].scales     = scales;
        g_tasks[i].x          = x;
        g_tasks[i].y          = y;
        g_tasks[i].M          = M;
        g_tasks[i].K          = K;
        g_tasks[i].row_start  = r_start;
        g_tasks[i].row_end    = r_end;
        g_tasks[i].quant_type = 1;
        g_tasks[i].task_ready = 1;
    }

    pthread_cond_broadcast(&g_start_cond);

    for (int i = 0; i < num_threads; i++) {
        while (g_tasks[i].task_ready) {
            pthread_cond_wait(&g_done_cond, &g_mutex);
        }
    }
    pthread_mutex_unlock(&g_mutex);
#endif
}

ASMLLM_API void asm_threadpool_dispatch_q5(
    const uint8_t* ql,
    const uint8_t* qh,
    const float*   scales,
    const float*   x,
    float*         y,
    int64_t        M,
    int64_t        K
) {
    int num_threads = g_pool_size;
    if (num_threads <= 1 || M < 32) {
        asm_matmul_q5(ql, qh, scales, x, y, M, K);
        return;
    }

    int64_t align_rows = 16;
    int64_t tile_rows = (M + num_threads - 1) / num_threads;
    tile_rows = ((tile_rows + align_rows - 1) / align_rows) * align_rows;

#if defined(_WIN32) || defined(_WIN64)
    HANDLE done_events[MAX_WORKERS];
    for (int i = 0; i < num_threads; i++) {
        int64_t r_start = (int64_t)i * tile_rows;
        int64_t r_end   = r_start + tile_rows;
        if (r_start > M) r_start = M;
        if (r_end > M)   r_end = M;

        g_tasks[i].qweights   = ql;
        g_tasks[i].qh         = qh;
        g_tasks[i].scales     = scales;
        g_tasks[i].x          = x;
        g_tasks[i].y          = y;
        g_tasks[i].M          = M;
        g_tasks[i].K          = K;
        g_tasks[i].row_start  = r_start;
        g_tasks[i].row_end    = r_end;
        g_tasks[i].quant_type = 2; // Q5_0
        done_events[i]        = g_tasks[i].done_event;
        SetEvent(g_tasks[i].start_event);
    }

    WaitForMultipleObjects((DWORD)num_threads, done_events, TRUE, INFINITE);
#else
    pthread_mutex_lock(&g_mutex);
    for (int i = 0; i < num_threads; i++) {
        int64_t r_start = (int64_t)i * tile_rows;
        int64_t r_end   = r_start + tile_rows;
        if (r_start > M) r_start = M;
        if (r_end > M)   r_end = M;

        g_tasks[i].qweights   = ql;
        g_tasks[i].qh         = qh;
        g_tasks[i].scales     = scales;
        g_tasks[i].x          = x;
        g_tasks[i].y          = y;
        g_tasks[i].M          = M;
        g_tasks[i].K          = K;
        g_tasks[i].row_start  = r_start;
        g_tasks[i].row_end    = r_end;
        g_tasks[i].quant_type = 2;
        g_tasks[i].task_ready = 1;
    }

    pthread_cond_broadcast(&g_start_cond);

    for (int i = 0; i < num_threads; i++) {
        while (g_tasks[i].task_ready) {
            pthread_cond_wait(&g_done_cond, &g_mutex);
        }
    }
    pthread_mutex_unlock(&g_mutex);
#endif
}

#if !defined(_WIN32) && !defined(_WIN64)
ASMLLM_API void asm_matmul_q8_mt(
    const int8_t*  qweights,
    const float*   scales,
    const float*   x,
    float*         y,
    int64_t        M,
    int64_t        K,
    int            num_threads
) {
    if (num_threads <= 1 || M < 32) {
        asm_matmul_q8(qweights, scales, x, y, M, K);
        return;
    }
    if (g_pool_size != num_threads) {
        asm_threadpool_init(num_threads);
    }
    asm_threadpool_dispatch_q8(qweights, scales, x, y, M, K);
}

ASMLLM_API void asm_matmul_q5_mt(
    const uint8_t* ql,
    const uint8_t* qh,
    const float*   scales,
    const float*   x,
    float*         y,
    int64_t        M,
    int64_t        K,
    int            num_threads
) {
    if (num_threads <= 1 || M < 32) {
        asm_matmul_q5(ql, qh, scales, x, y, M, K);
        return;
    }
    if (g_pool_size != num_threads) {
        asm_threadpool_init(num_threads);
    }
    asm_threadpool_dispatch_q5(ql, qh, scales, x, y, M, K);
}
#endif

