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
| 2026-07-09 | Multi-Format Quantized Matvec (`matmul_q4`, `matmul_q5`, `matmul_q8` 1T & 4T, M=1024, K=4096) | x86-64 AVX2 / Intel Core 5 210H (12 logical cores, Win11) | AVX2 zero-intrinsics assembly kernels vs FP32 NumPy reference | **Q4_0 4T: 0.228 ms (4.58x)**, **Q8_0 1T: 0.223 ms (4.68x)**, **Q5_0 4T: 0.359 ms (2.92x)** vs FP32 baseline **1.046 ms** | Verified / `bench/results/20260709-222100-quant-formats/benchmark.log` | Evaluates M4 multi-format AVX2 assembly kernels across Q4_0, Q5_0, and Q8_0 formats with verified numerical accuracy and multi-threaded scaling. Full 7B model evaluation awaits 7B GGUF weights. |
| 2026-07-10 | `matmul_q4_mt` (Multi-Threaded Q4_0 Matvec, M=4096, K=4096) | x86-64 AVX2 / Intel Core 5 210H (12 logical cores, Win11) | 1, 2, 4, 8 thread scaling | **1T:** 1.539 ms (**21.81 GFLOPS**)<br>**4T:** 0.324 ms (**103.60 GFLOPS**)<br>**8T:** 0.378 ms (**88.72 GFLOPS**) | Verified / `bench/results/2026-07-10-asmllm-mt/raw_bench.log` | ~2x throughput jump over 2026-07-09 row (103.6 GFLOPS vs 51.0 GFLOPS) due to cache-line-aware row alignment (`K % 32 == 0`) and optimized background worker distribution. |
| 2026-07-10 | End-to-End 128-Token Generation (`tg128`) & Multi-Threaded Throughput vs Native `llama.cpp` (`stories15M-q4_0.gguf`) | x86-64 AVX2 / Intel Core 5 210H (12 logical cores, Win11) | `asmllm` AVX2 Assembly vs `llama.cpp` pure CPU AVX2 (`commit 961e4b2`) | **1T:** `asmllm` **1011.82 t/s** vs `llama.cpp` **902.38 t/s** (+12.1%)<br>**2T:** `asmllm` **1170.75 t/s** vs `llama.cpp` **1464.22 t/s** (-20.0%)<br>**4T:** `asmllm` **1300.25 t/s** vs `llama.cpp` **1681.29 t/s** (-22.6%)<br>**8T:** `asmllm` **1671.13 t/s** vs `llama.cpp` **1839.99 t/s** (-9.2%) | Verified / `bench/results/2026-07-10-llamacpp-baseline/raw_bench.log` & `bench/results/2026-07-10-asmllm-mt/raw_bench.log` | Aligned 128-token generation sweep (`tg128`). Note on regression: scaling at small-$M$ exhibits synchronization overhead from Win32 condition variables compared to llama.cpp's mature work-stealing scheduler. |

| 2026-07-27 | `matmul_q4_avx512` (Q4_0 Matvec AVX-512) | x86-64 AVX-512 / Intel Core 5 210H (12 logical cores, Win11) | AVX-512 vs NumPy FP32 BLAS | **N/A** | Blocked / Hardware Unavailable | M6 AVX-512 kernel implemented and compiled successfully via `--avx512` opt-in flag. Execution/benchmarking fails with SIGILL because the current local CPU lacks native AVX-512 support. Blocked pending AVX-512 capable hardware as per AGENTS.md rules. |
| 2026-07-27 | `matmul_q4_mt` (Multi-Threaded Q4_0 Matvec, M=4096, K=4096) | ARM64 NEON / Apple M1 (Virtual), macOS 14 | 1, 2, 4, 8 thread scaling | **1T:** 6.034 ms (**5.56 GFLOPS**)<br>**2T:** 3.222 ms (**10.42 GFLOPS**, 1.87x)<br>**4T:** 3.083 ms (**10.88 GFLOPS**, 1.96x)<br>**8T:** 2.559 ms (**13.11 GFLOPS**, 2.36x) | Verified / `bench/results/2026-07-27-mt_throughput/raw_bench.log` | Scaling evaluated on ARM64 using lock-free atomic adaptive spin-waits. Virtualized M1 runner shows diminishing returns past 2 threads due to limited logical cores (`hw.ncpu` = 3) on the CI runner, peaking at 13.11 GFLOPS on 8 threads. |
| 2026-07-27 | ARM64 NEON 128-Token Generation (`tg128`) vs `llama.cpp` CPU NEON (`stories15M-q4_0.gguf`) | ARM64 NEON / Apple M1 (Virtual), macOS 14, GitHub Actions `macos-14` runner | `asmllm` hand-written NEON assembly vs `llama.cpp` pure CPU NEON (`commit 961e4b2`) | **1T:** `asmllm` **233.75 t/s** vs `llama.cpp` **1558.41 t/s** (asmllm 6.7x slower)<br>**2T:** `asmllm` **221.27 t/s** vs `llama.cpp` **2008.92 t/s** (asmllm 9.1x slower)<br>**4T/8T:** unreliable — llama.cpp collapses to 4.43/0.89 t/s | Verified / `bench/results/2026-07-27-arm64-neon/raw_bench.log` | Fresh CI run for M5/M6. Still an honest loss against llama.cpp's mature NEON intrinsics. Console log confirmed `sysctl -n hw.ncpu` = 3 logical cores, definitively explaining why the 4T and 8T tests catastrophically thrash on this runner. |

---

## Independent Reproductions

*(Reserved for third-party independent reproductions required for long-term project Definition of Done)*
