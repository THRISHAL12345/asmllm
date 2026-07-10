#!/usr/bin/env python3
"""
bench/harness/bench_mt_throughput.py

Multi-threaded throughput benchmark harness for asmllm.
Evaluates multi-threaded scaling on:
  1) Q4_0 Matrix-Vector Multiply (M=4096, K=4096) across 1, 2, 4, 8 threads.
  2) End-to-end Token Generation Throughput (tokens/sec) across 1, 2, 4, 8 threads.

Adheres to rigorous benchmark protocol:
  - 5 warmups, 10 recorded trials per configuration.
  - Reports median latency, spread (IQR), GFLOPS, and tokens/sec.
  - Saves raw receipts to bench/results/<date>-mt_throughput/raw_bench.log.
"""

import ctypes
import datetime
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.correctness.test_runner import try_load_native_library
from src.runtime.generate import SmallLlamaModel, AsmEngineForward


def get_cpu_info() -> str:
    processor = platform.processor() or platform.machine()
    return f"{platform.system()} {platform.release()} ({processor})"


def percentile(data: list[float], p: float) -> float:
    s = sorted(data)
    idx = (len(s) - 1) * p
    lower = math.floor(idx)
    upper = math.ceil(idx)
    weight = idx - lower
    return s[lower] * (1.0 - weight) + s[upper] * weight


def run_mt_matmul_benchmark(native_lib, M=4096, K=4096, threads_list=[1, 2, 4, 8]):
    print("--------------------------------------------------------------------------------")
    print(f" 1. Multi-Threaded Q4_0 Matvec Throughput (M={M}, K={K})")
    print("--------------------------------------------------------------------------------")

    rng = np.random.RandomState(42)
    num_blocks = K // 32
    qweights = rng.randint(0, 256, size=(M, num_blocks, 16), dtype=np.uint8)
    scales = rng.uniform(0.05, 0.25, size=(M, num_blocks)).astype(np.float32)
    x = rng.randn(K).astype(np.float32)
    y = np.zeros(M, dtype=np.float32)

    func = native_lib.asm_matmul_q4_mt
    func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    func.restype = None
    flops_per_call = 2.0 * M * K

    results = {}
    baseline_median = None

    print(f"{'Threads':<10} {'Median (ms)':<14} {'IQR (ms)':<12} {'GFLOPS':<14} {'Speedup'}")
    print("-" * 65)

    for threads in threads_list:
        # Warmup
        for _ in range(5):
            func(qweights.ctypes.data, scales.ctypes.data, x.ctypes.data, y.ctypes.data, M, K, threads)

        trials = []
        for _ in range(10):
            t0 = time.perf_counter()
            func(qweights.ctypes.data, scales.ctypes.data, x.ctypes.data, y.ctypes.data, M, K, threads)
            t1 = time.perf_counter()
            trials.append((t1 - t0) * 1000.0)

        med_ms = percentile(trials, 0.50)
        q25 = percentile(trials, 0.25)
        q75 = percentile(trials, 0.75)
        iqr_ms = q75 - q25
        gflops = (flops_per_call / (med_ms / 1000.0)) / 1e9

        if baseline_median is None:
            baseline_median = med_ms
            speedup = 1.00
        else:
            speedup = baseline_median / med_ms

        print(f"{threads:<10} {med_ms:<14.3f} {iqr_ms:<12.3f} {gflops:<14.2f} {speedup:.2f}x")
        results[threads] = {
            "median_ms": med_ms,
            "iqr_ms": iqr_ms,
            "gflops": gflops,
            "speedup": speedup
        }

    return results


def run_mt_generation_benchmark(native_lib, threads_list=[1, 2, 4, 8], n_tokens=128):
    print("\n--------------------------------------------------------------------------------")
    print(f" 2. End-to-End Token Generation Throughput ({n_tokens} tokens)")
    print("--------------------------------------------------------------------------------")

    model = SmallLlamaModel(seed=1337)
    engine = AsmEngineForward(model, native_lib)

    print(f"{'Threads':<10} {'Median (ms)':<14} {'IQR (ms)':<12} {'Throughput (tok/sec)':<22} {'Speedup'}")
    print("-" * 75)

    init_pool = native_lib.asm_threadpool_init
    init_pool.argtypes = [ctypes.c_int]
    init_pool.restype = None

    shutdown_pool = native_lib.asm_threadpool_shutdown
    shutdown_pool.argtypes = []
    shutdown_pool.restype = None

    results = {}
    baseline_tps = None

    for threads in threads_list:
        init_pool(threads)
        engine = AsmEngineForward(model, native_lib, num_threads=threads)

        # Warmup
        for step in range(5):
            engine.forward_token(1, pos=step)

        trials = []
        for _ in range(5):
            t0 = time.perf_counter()
            for step in range(n_tokens):
                engine.forward_token(1, pos=step)
            t1 = time.perf_counter()
            trials.append((t1 - t0) * 1000.0)

        med_ms = percentile(trials, 0.50)
        q25 = percentile(trials, 0.25)
        q75 = percentile(trials, 0.75)
        iqr_ms = q75 - q25
        tps = n_tokens / (med_ms / 1000.0)

        if baseline_tps is None:
            baseline_tps = tps
            speedup = 1.00
        else:
            speedup = tps / baseline_tps

        print(f"{threads:<10} {med_ms:<14.3f} {iqr_ms:<12.3f} {tps:<22.2f} {speedup:.2f}x")
        results[threads] = {
            "median_ms": med_ms,
            "iqr_ms": iqr_ms,
            "tokens_per_sec": tps,
            "speedup": speedup
        }

    shutdown_pool()
    return results


def main():
    date_str = datetime.date.today().isoformat()
    out_dir = PROJECT_ROOT / "bench" / "results" / f"{date_str}-mt_throughput"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_log_path = out_dir / "raw_bench.log"
    json_path = out_dir / "summary.json"

    print("================================================================================")
    print(" asmllm Multi-Threaded Throughput & Scaling Benchmark Harness")
    print("================================================================================")
    print(f"Date: {date_str}")
    print(f"System Hardware: {get_cpu_info()}\n")

    native_lib = try_load_native_library()
    if not native_lib:
        print("[ERROR] Native library asmllm.dll not found.")
        sys.exit(1)

    matmul_results = run_mt_matmul_benchmark(native_lib)
    generation_results = run_mt_generation_benchmark(native_lib)

    summary = {
        "date": date_str,
        "system": get_cpu_info(),
        "matmul_q4_mt": matmul_results,
        "token_generation_mt": generation_results
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(raw_log_path, "w", encoding="utf-8") as f:
        f.write("asmllm Multi-Threaded Benchmark Summary\n")
        f.write(json.dumps(summary, indent=2) + "\n")

    print("\n================================================================================")
    print(f"[SUCCESS] Receipts saved to:\n  - {raw_log_path}\n  - {json_path}")
    print("================================================================================")


if __name__ == "__main__":
    main()
