import sys
import re

with open('src/runtime/threadpool.c', 'r', encoding='utf-8') as f:
    code = f.read()

# Add includes and ATOMIC macros
includes = """
#if defined(_WIN32) || defined(_WIN64)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <intrin.h> // for YieldProcessor
  #define ATOMIC_INC(ptr) InterlockedIncrement((volatile LONG*)(ptr))
  #define ATOMIC_ADD(ptr, val) InterlockedExchangeAdd((volatile LONG*)(ptr), (val))
  #define ATOMIC_SET(ptr, val) InterlockedExchange((volatile LONG*)(ptr), (val))
  #define ATOMIC_GET(ptr) InterlockedCompareExchange((volatile LONG*)(ptr), 0, 0)
  #define THREAD_YIELD() YieldProcessor()
#else
  #define _GNU_SOURCE
  #include <pthread.h>
  #include <unistd.h>
  #include <sched.h>
  #include <stdatomic.h>
  #if defined(__x86_64__)
    #include <immintrin.h> // for _mm_pause
    #define THREAD_YIELD() _mm_pause()
  #elif defined(__aarch64__)
    #define THREAD_YIELD() asm volatile("yield" ::: "memory")
  #else
    #define THREAD_YIELD() sched_yield()
  #endif
  #define ATOMIC_INC(ptr) atomic_fetch_add_explicit((volatile _Atomic long*)(ptr), 1, memory_order_acq_rel)
  #define ATOMIC_ADD(ptr, val) atomic_fetch_add_explicit((volatile _Atomic long*)(ptr), (val), memory_order_acq_rel)
  #define ATOMIC_SET(ptr, val) atomic_store_explicit((volatile _Atomic long*)(ptr), (val), memory_order_release)
  #define ATOMIC_GET(ptr) atomic_load_explicit((volatile _Atomic long*)(ptr), memory_order_acquire)
#endif
"""

# Replace the original includes block (lines 18-26)
old_includes = """#if defined(_WIN32) || defined(_WIN64)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #define _GNU_SOURCE
  #include <pthread.h>
  #include <unistd.h>
  #include <sched.h>
#endif"""

code = code.replace(old_includes, includes)

# Modify worker_task_t
code = code.replace("int quant_type; // 0 = Q4_0, 1 = Q8_0, 2 = Q5_0", "int quant_type; // 0 = Q4_0, 1 = Q8_0, 2 = Q5_0\n    long last_gen;")

# Add globals
globals_block = """
static volatile long g_task_generation = 0;
static volatile long g_workers_completed = 0;
static volatile long g_sleeping_workers = 0;
"""
code = code.replace("static volatile int g_ready_workers = 0;", "static volatile int g_ready_workers = 0;\n" + globals_block)

# Replace worker_main_win32 logic
win_worker_old = """    while (1) {
        WaitForSingleObject(task->start_event, INFINITE);
        if (g_shutdown) break;

        int64_t row_count = task->row_end - task->row_start;
        if (row_count > 0) {"""
win_worker_new = """    while (1) {
        long current_gen;
        int spin_count = 0;
        while ((current_gen = ATOMIC_GET(&g_task_generation)) == task->last_gen) {
            if (g_shutdown) return 0;
            if (spin_count < 5000) {
                THREAD_YIELD();
                spin_count++;
            } else {
                ATOMIC_INC(&g_sleeping_workers);
                if (ATOMIC_GET(&g_task_generation) != task->last_gen) {
                    ATOMIC_ADD(&g_sleeping_workers, -1);
                    continue;
                }
                WaitForSingleObject(task->start_event, INFINITE);
                ATOMIC_ADD(&g_sleeping_workers, -1);
                spin_count = 0;
            }
        }
        if (g_shutdown) return 0;
        task->last_gen = current_gen;

        int64_t row_count = task->row_end - task->row_start;
        if (row_count > 0) {"""
code = code.replace(win_worker_old, win_worker_new)
code = code.replace("SetEvent(task->done_event);\n    }", "ATOMIC_INC(&g_workers_completed);\n    }", 1)

# Modify threadpool init for Win32
code = code.replace("g_tasks[i].core_id = i;", "g_tasks[i].core_id = i;\n        g_tasks[i].last_gen = ATOMIC_GET(&g_task_generation);", 1)


# Replace worker_main_posix logic
posix_worker_old = """    while (1) {
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
        if (row_count > 0) {"""
posix_worker_new = """    while (1) {
        long current_gen;
        int spin_count = 0;
        while ((current_gen = ATOMIC_GET(&g_task_generation)) == task->last_gen) {
            if (g_shutdown) return NULL;
            if (spin_count < 5000) {
                THREAD_YIELD();
                spin_count++;
            } else {
                pthread_mutex_lock(&g_mutex);
                ATOMIC_INC(&g_sleeping_workers);
                while (ATOMIC_GET(&g_task_generation) == task->last_gen && !g_shutdown) {
                    pthread_cond_wait(&g_start_cond, &g_mutex);
                }
                ATOMIC_ADD(&g_sleeping_workers, -1);
                pthread_mutex_unlock(&g_mutex);
                spin_count = 0;
            }
        }
        if (g_shutdown) return NULL;
        task->last_gen = current_gen;

        int64_t row_count = task->row_end - task->row_start;
        if (row_count > 0) {"""
code = code.replace(posix_worker_old, posix_worker_new)
posix_end_old = """        pthread_mutex_lock(&g_mutex);
        task->task_ready = 0;
        pthread_cond_broadcast(&g_done_cond);
        pthread_mutex_unlock(&g_mutex);
    }"""
posix_end_new = """        ATOMIC_INC(&g_workers_completed);
    }"""
code = code.replace(posix_end_old, posix_end_new)

# Modify threadpool init for POSIX
code = code.replace("g_tasks[i].task_ready = 0;", "g_tasks[i].task_ready = 0;\n        g_tasks[i].last_gen = ATOMIC_GET(&g_task_generation);", 1)


# Rewrite the wait loops in all dispatch functions
# Windows wait:
win_dispatch_old = """        done_events[i]       = g_tasks[i].done_event;
        SetEvent(g_tasks[i].start_event);
    }

    WaitForMultipleObjects((DWORD)num_threads, done_events, TRUE, INFINITE);"""
win_dispatch_new = """        done_events[i]       = g_tasks[i].done_event;
    }

    ATOMIC_SET(&g_workers_completed, 0);
    ATOMIC_INC(&g_task_generation);
    if (ATOMIC_GET(&g_sleeping_workers) > 0) {
        for (int i = 0; i < num_threads; i++) {
            SetEvent(g_tasks[i].start_event);
        }
    }

    int spin_count = 0;
    while (ATOMIC_GET(&g_workers_completed) < num_threads) {
        if (spin_count < 5000) {
            THREAD_YIELD();
            spin_count++;
        } else {
            Sleep(0);
        }
    }"""
code = code.replace(win_dispatch_old, win_dispatch_new)


# POSIX wait:
posix_dispatch_old = """        g_tasks[i].task_ready = 1;
    }

    pthread_cond_broadcast(&g_start_cond);

    for (int i = 0; i < num_threads; i++) {
        while (g_tasks[i].task_ready) {
            pthread_cond_wait(&g_done_cond, &g_mutex);
        }
    }
    pthread_mutex_unlock(&g_mutex);"""
posix_dispatch_new = """        g_tasks[i].task_ready = 1;
    }

    ATOMIC_SET(&g_workers_completed, 0);
    ATOMIC_INC(&g_task_generation);
    if (ATOMIC_GET(&g_sleeping_workers) > 0) {
        pthread_cond_broadcast(&g_start_cond);
    }
    pthread_mutex_unlock(&g_mutex);

    int spin_count = 0;
    while (ATOMIC_GET(&g_workers_completed) < num_threads) {
        if (spin_count < 5000) {
            THREAD_YIELD();
            spin_count++;
        } else {
            sched_yield();
        }
    }"""
code = code.replace(posix_dispatch_old, posix_dispatch_new)

with open('src/runtime/threadpool.c', 'w', encoding='utf-8') as f:
    f.write(code)

print("Threadpool rewrite complete.")
