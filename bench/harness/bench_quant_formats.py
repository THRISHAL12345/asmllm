#!/usr/bin/env python3
"""
bench/harness/bench_quant_formats.py

Reproducible benchmark driver comparing Q4_0, Q5_0, and Q8_0 Matrix-Vector Multiply
kernels (single-threaded and multi-threaded) per project benchmark protocol.
"""

import ctypes
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.correctness.test_runner import try_load_native_library
from tests.reference.ops import dequantize_q4_0_numpy, dequantize_q8_0_numpy, dequantize_q5_0_numpy


def get_cpu_info() -> str:
    try:
        processor = platform.processor()
        if sys.platform == "win32":
            import subprocess
            cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                return f"{res.stdout.strip()} ({os.cpu_count()} logical cores)"
        return f"{processor} ({os.cpu_count()} logical cores)"
    except Exception:
        return f"Unknown CPU ({os.cpu_count()} logical cores)"


def benchmark_quant_formats(M: int = 32, K: int = 4096, num_trials: int = 15):
    cpu_info = get_cpu_info()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = PROJECT_ROOT / "bench" / "results" / f"{timestamp}-quant-formats"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "benchmark.log"

    def log_and_print(msg: str):
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log_and_print("================================================================================")
    log_and_print(" asmllm Multi-Format Quantized Matrix-Vector Multiply Benchmark")
    log_and_print("================================================================================")
    log_and_print(f"Hardware:        {cpu_info}")
    log_and_print(f"OS/Platform:     {platform.platform()}")
    log_and_print(f"Python Version:  {sys.version.split()[0]}")
    log_and_print(f"Dimensions:      M={M}, K={K}")
    log_and_print(f"Trials:          {num_trials} runs per kernel (median & spread reported)")
    log_and_print("--------------------------------------------------------------------------------")

    native_lib = try_load_native_library()
    if not native_lib:
        log_and_print("[ERROR] Could not load native assembly library.")
        sys.exit(1)

    rng = np.random.RandomState(42)
    num_blocks = K // 32
    x_vec = np.ascontiguousarray(rng.randn(K).astype(np.float32))

    # Generate Q4_0
    q4_weights = np.ascontiguousarray(rng.randint(0, 256, size=(M, num_blocks, 16), dtype=np.uint8))
    q4_scales = np.ascontiguousarray(rng.uniform(0.5, 1.5, size=(M, num_blocks)).astype(np.float32))
    w4_fp32 = dequantize_q4_0_numpy(q4_weights, q4_scales)

    # Generate Q8_0
    q8_weights = np.ascontiguousarray(rng.randint(-127, 128, size=(M, num_blocks, 32), dtype=np.int8))
    q8_scales = np.ascontiguousarray(rng.uniform(0.1, 0.9, size=(M, num_blocks)).astype(np.float32))
    w8_fp32 = dequantize_q8_0_numpy(q8_weights, q8_scales)

    # Generate Q5_0
    q5_ql = np.ascontiguousarray(rng.randint(0, 256, size=(M, num_blocks, 16), dtype=np.uint8))
    q5_qh = np.ascontiguousarray(rng.randint(0, 256, size=(M, num_blocks, 4), dtype=np.uint8))
    q5_scales = np.ascontiguousarray(rng.uniform(0.5, 1.5, size=(M, num_blocks)).astype(np.float32))
    w5_fp32 = dequantize_q5_0_numpy(q5_ql, q5_qh, q5_scales)

    y_asm = np.zeros(M, dtype=np.float32)

    # Setup function prototypes
    q4_func = native_lib.asm_matmul_q4
    q4_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
    q4_mt_func = native_lib.asm_matmul_q4_mt
    q4_mt_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]

    q8_func = native_lib.asm_matmul_q8
    q8_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
    q8_mt_func = native_lib.asm_matmul_q8_mt
    q8_mt_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]

    q5_func = native_lib.asm_matmul_q5
    q5_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
    q5_mt_func = native_lib.asm_matmul_q5_mt
    q5_mt_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]

    # Baseline FP32 scalar timings
    t0 = time.perf_counter()
    for _ in range(num_trials):
        _ = w4_fp32 @ x_vec
    fp32_time = (time.perf_counter() - t0) / num_trials * 1000.0

    def run_bench(name: str, runner_fn):
        for _ in range(3):
            runner_fn()
        times = []
        for _ in range(num_trials):
            t0 = time.perf_counter()
            runner_fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        med = statistics.median(times)
        stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        speedup = fp32_time / med if med > 0 else 0.0
        log_and_print(f"{name:<22} {med:>8.3f} ms (±{stdev:>5.3f} ms) | Speedup vs FP32 NumPy: {speedup:>5.2f}x")
        return med, stdev

    log_and_print(f"FP32 NumPy Baseline   : {fp32_time:>8.3f} ms")
    log_and_print("-" * 80)

    run_bench("Q4_0 (Single-Thread)", lambda: q4_func(q4_weights.ctypes.data, q4_scales.ctypes.data, x_vec.ctypes.data, y_asm.ctypes.data, M, K))
    run_bench("Q4_0 (4-Thread AVX2)", lambda: q4_mt_func(q4_weights.ctypes.data, q4_scales.ctypes.data, x_vec.ctypes.data, y_asm.ctypes.data, M, K, 4))
    run_bench("Q8_0 (Single-Thread)", lambda: q8_func(q8_weights.ctypes.data, q8_scales.ctypes.data, x_vec.ctypes.data, y_asm.ctypes.data, M, K))
    run_bench("Q8_0 (4-Thread AVX2)", lambda: q8_mt_func(q8_weights.ctypes.data, q8_scales.ctypes.data, x_vec.ctypes.data, y_asm.ctypes.data, M, K, 4))
    run_bench("Q5_0 (Single-Thread)", lambda: q5_func(q5_ql.ctypes.data, q5_qh.ctypes.data, q5_scales.ctypes.data, x_vec.ctypes.data, y_asm.ctypes.data, M, K))
    run_bench("Q5_0 (4-Thread AVX2)", lambda: q5_mt_func(q5_ql.ctypes.data, q5_qh.ctypes.data, q5_scales.ctypes.data, x_vec.ctypes.data, y_asm.ctypes.data, M, K, 4))

    log_and_print("================================================================================")
    log_and_print(f"[SUCCESS] Benchmark complete. Logged to: {log_file}")


if __name__ == "__main__":
    benchmark_quant_formats()
