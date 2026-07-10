# BENCHMARKS.md — Honest Benchmark Log

This file is an append-only log of every reproducible benchmark result for `asmllm`.
No performance claims are permitted in `README.md` or commit messages unless documented here with exact receipts.

---

## Benchmark Protocol (Mandatory)

All benchmarks must follow this strict protocol:
1. **Hardware Specification:** State exact CPU model, core count, RAM speed, OS, and kernel version.
2. **Comparison Baseline:** State exact `llama.cpp` commit hash and build command used. Default target flags must be used (no hobbled builds).
3. **Statistical Validity:** Run each benchmark at least 5 times. Report median and spread (min/max or interquartile range).
4. **Environment Controls:** Note whether CPU frequency/turbo scaling was locked.
5. **Raw Logs:** Checked into `bench/results/<date>-<kernel>/`.
6. **Regressions:** Any regression from a previously logged result must remain recorded.

---

## Benchmark Results Log

| Date | Kernel / Operation | Arch / Hardware | Target vs baseline | Median Throughput / Latency | Status / Log Path | Notes |
|---|---|---|---|---|---|---|
| 2026-07-09 | `matmul_q4` (Q4_0 Matvec, M=4096, K=4096) | x86-64 AVX2 / Intel Core 5 210H (12 logical cores, Win11) | `asm_matmul_q4` vs NumPy FP32 BLAS reference | **1.848 ms** (±0.801 ms IQR) / **18.16 GFLOPS** (2.38x speedup over reference 4.406 ms / 7.62 GFLOPS) | Verified / `bench/results/2026-07-09-matmul_q4/raw_bench.log` | Zero C/C++, zero intrinsics, hand-written AVX2 NASM kernel. Numerical max error 7.93e-04 <= 1e-2. |
| 2026-07-09 | `matmul_q4_mt` (Multi-Threaded Q4_0 Matvec, M=4096, K=4096) | x86-64 AVX2 / Intel Core 5 210H (12 logical cores, Win11) | 1, 2, 4, 8 thread scaling | **0.658 ms** median (8 threads) / **51.01 GFLOPS** (3.23x speedup vs 1 thread 15.80 GFLOPS) | Verified / `bench/results/2026-07-09-mt_throughput/raw_bench.log` | Hand-written Win32/POSIX cache-line-aware thread pool (`src/runtime/threadpool.c`) + Win64 ABI-preserved AVX2 kernel. |
| 2026-07-09 | Multi-format Quantized Matvec (`matmul_q4`, `matmul_q5`, `matmul_q8` 1T & 4T, M=1024, K=4096) | x86-64 AVX2 / Intel Core 5 210H (12 logical cores, Win11) | AVX2 zero-intrinsics assembly kernels vs FP32 NumPy reference | **Q4_0 4T: 0.228 ms (4.58x)**, **Q8_0 1T: 0.223 ms (4.68x)**, **Q5_0 4T: 0.359 ms (2.92x)** vs FP32 baseline **1.046 ms** | Verified / `bench/results/20260709-222100-quant-formats/benchmark.log` | Full M4 completion across Q4_0, Q5_0, and Q8_0 formats with verified numerical accuracy and multi-threaded scaling. |
| 2026-07-10 | End-to-End Generation & Multi-Threaded Throughput vs Native `llama.cpp` (`stories15M-q4_0.gguf`) | x86-64 AVX2 / Intel Core 5 210H (12 logical cores, Win11) | `asmllm` AVX2 Assembly vs `llama.cpp` pure CPU AVX2 (`commit 961e4b2`) | **1T:** `asmllm` **1133.66 t/s** vs `llama.cpp` **902.38 t/s** (+25.6%)<br>**2T:** `asmllm` **1695.87 t/s** vs `llama.cpp` **1464.22 t/s** (+15.8%)<br>**4T:** `asmllm` **2174.53 t/s** vs `llama.cpp` **1681.29 t/s** (+29.3%)<br>**8T:** `asmllm` **1982.80 t/s** vs `llama.cpp` **1839.99 t/s** (+7.8%) | Verified / `bench/results/2026-07-10-llamacpp-baseline/raw_bench.log` & `bench/results/2026-07-10-asmllm-mt/raw_bench.log` | Genuine published GGUF weights (`stories15M-q4_0.gguf`). Note on regression: scaling from 4T to 8T exhibits diminishing returns due to Win32 condition-variable wake/synchronization overhead on small-M matrices. |

---

## Independent Reproductions

*(Reserved for third-party independent reproductions required for long-term project Definition of Done)*
