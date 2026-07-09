#!/usr/bin/env python3
"""
bench/harness/bench_matmul_q4.py

Reproducible isolated microbenchmark driver for Q4_0 Matrix-Vector Multiply.
Complies with project benchmark protocol standards.
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
from tests.reference.ops import dequantize_q4_0_numpy


def get_cpu_info() -> str:
    try:
        import platform
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


def benchmark_q4_matvec(M: int = 4096, K: int = 4096, num_trials: int = 10):
    print("================================================================================")
    print(" asmllm Q4_0 Matrix-Vector Multiply Microbenchmark")
    print("================================================================================")

    cpu_info = get_cpu_info()
    print(f"Hardware:        {cpu_info}")
    print(f"OS/Platform:     {platform.platform()}")
    print(f"Python Version:  {sys.version.split()[0]}")
    print(f"Matrix Size:     M={M}, K={K} ({K // 32} Q4_0 blocks/row)")
    print(f"Trials:          {num_trials} runs per kernel (median & spread reported)")
    print(f"Frequency Lock:  Not explicitly locked (standard host runtime environment)")
    print("--------------------------------------------------------------------------------")

    native_lib = try_load_native_library()
    if not native_lib or not hasattr(native_lib, "asm_matmul_q4"):
        print("[ERROR] Could not load asm_matmul_q4 symbol from native library.")
        sys.exit(1)

    func = native_lib.asm_matmul_q4
    func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    func.restype = None

    rng = np.random.RandomState(42)
    num_blocks = K // 32
    qweights = np.ascontiguousarray(rng.randint(0, 256, size=(M, num_blocks, 16), dtype=np.uint8))
    scales = np.ascontiguousarray(rng.uniform(0.5, 1.5, size=(M, num_blocks)).astype(np.float32))
    x_vec = np.ascontiguousarray(rng.randn(K).astype(np.float32))
    y_asm = np.zeros(M, dtype=np.float32)

    # Pre-dequantize weights for scalar/NumPy reference baseline comparison
    w_fp32 = dequantize_q4_0_numpy(qweights, scales)

    # Warm-up runs
    for _ in range(3):
        func(
            qweights.ctypes.data,
            scales.ctypes.data,
            x_vec.ctypes.data,
            y_asm.ctypes.data,
            int(M),
            int(K),
        )
        _ = w_fp32 @ x_vec

    # 1. Benchmark Hand-written AVX2 Assembly Kernel
    asm_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        func(
            qweights.ctypes.data,
            scales.ctypes.data,
            x_vec.ctypes.data,
            y_asm.ctypes.data,
            int(M),
            int(K),
        )
        t1 = time.perf_counter()
        asm_times.append((t1 - t0) * 1000.0)  # ms

    # 2. Benchmark Reference NumPy BLAS baseline
    ref_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        y_ref = w_fp32 @ x_vec
        t1 = time.perf_counter()
        ref_times.append((t1 - t0) * 1000.0)  # ms

    # Verify output match
    err = float(np.max(np.abs(y_asm - y_ref)))
    assert err < 1e-2, f"Benchmark correctness failure! err={err}"

    flops = 2.0 * M * K

    def compute_stats(times_ms):
        med = statistics.median(times_ms)
        min_v = min(times_ms)
        max_v = max(times_ms)
        iqr = statistics.quantiles(times_ms, n=4)[2] - statistics.quantiles(times_ms, n=4)[0] if len(times_ms) >= 4 else (max_v - min_v)
        gflops = (flops / (med / 1000.0)) / 1e9
        return med, min_v, max_v, iqr, gflops

    asm_med, asm_min, asm_max, asm_iqr, asm_gflops = compute_stats(asm_times)
    ref_med, ref_min, ref_max, ref_iqr, ref_gflops = compute_stats(ref_times)

    print(f"Kernel Implementation                 Median (ms)    Spread (IQR)    Throughput (GFLOPS)")
    print("--------------------------------------------------------------------------------")
    print(f"asm_matmul_q4 (Hand-Written AVX2)    {asm_med:9.3f} ms   ±{asm_iqr:6.3f} ms   {asm_gflops:8.2f} GFLOPS")
    print(f"NumPy FP32 BLAS Reference Baseline   {ref_med:9.3f} ms   ±{ref_iqr:6.3f} ms   {ref_gflops:8.2f} GFLOPS")
    print("--------------------------------------------------------------------------------")

    speedup = ref_med / asm_med
    print(f"Result: Hand-written AVX2 Q4 kernel is {speedup:.2f}x vs NumPy FP32 BLAS reference baseline.")
    print(f"Numerical verification during bench: Max Error = {err:.2e} (PASS <= 1e-2)\n")

    return {
        "cpu_info": cpu_info,
        "M": M,
        "K": K,
        "asm_med": asm_med,
        "asm_iqr": asm_iqr,
        "asm_gflops": asm_gflops,
        "ref_med": ref_med,
        "ref_iqr": ref_iqr,
        "ref_gflops": ref_gflops,
        "speedup": speedup,
        "err": err,
    }


def main():
    res = benchmark_q4_matvec(M=4096, K=4096, num_trials=10)

    # Save log to bench/results/<date>-matmul_q4/
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_dir = PROJECT_ROOT / "bench" / "results" / f"{date_str}-matmul_q4"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "raw_bench.log"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"CPU:  {res['cpu_info']}\n")
        f.write(f"Dimensions: M={res['M']}, K={res['K']}\n")
        f.write(f"asm_matmul_q4 (AVX2): {res['asm_med']:.3f} ms (IQR ±{res['asm_iqr']:.3f} ms), {res['asm_gflops']:.2f} GFLOPS\n")
        f.write(f"Reference Baseline:   {res['ref_med']:.3f} ms (IQR ±{res['ref_iqr']:.3f} ms), {res['ref_gflops']:.2f} GFLOPS\n")
        f.write(f"Speedup vs reference: {res['speedup']:.2f}x\n")
        f.write(f"Max Error: {res['err']:.2e}\n")

    print(f"[BENCH LOG SAVED] Raw log checked into: {log_path}")


if __name__ == "__main__":
    main()
