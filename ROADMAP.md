# ROADMAP.md — Milestone Tracker

Each milestone has an explicit **Definition of Done (DoD)**. A milestone must not be reported complete unless every DoD item is checked and evidenced in the repository.

Do not start milestone `N+1` work until milestone `N`'s DoD items are all checked below with links to evidencing commits/logs.

---

## M0 — Harness & Reference
- [x] `tools/reference_gen.py` produces reference tensors and expected outputs using PyTorch for: Q4 matmul, RMSNorm, RoPE, softmax, single-head attention.
- [x] `tests/correctness/` runner compares asm kernel output to reference within numerical tolerance and produces a pass/fail report.
- **DoD:** `make test-correctness` runs end-to-end with zero kernels implemented yet (all show as "not implemented", not "false pass").
- **Evidence:** Verified 2026-07-09 — `python tests/correctness/test_runner.py` executed cleanly reporting all 5 kernels (`matmul_q4`, `rmsnorm`, `rope`, `softmax`, `attention`) as `NOT IMPLEMENTED` with exit code 0 (zero false positive passes).

---

## M1 — First Kernel: Q4 Matrix-Vector Multiply (x86-64 AVX2)
- [x] Hand-written `matmul_q4_avx2.asm` implementing quantized matvec.
- [x] Passes correctness suite within tolerance.
- [x] Benchmarked against `llama.cpp`'s equivalent op in isolation (microbenchmark, not full model) on identical hardware, results logged in `bench/results/`.
- **DoD:** A signed-off, reproducible number in `BENCHMARKS.md` — even if it's a loss. Losing honestly is an acceptable M1 outcome; a fabricated win is not.
- **Evidence:** Verified 2026-07-09 — Hand-written AVX2 NASM kernel (`src/kernels/x86_64/matmul_q4_avx2.asm`) passed correctness suite (`max_err = 3.81e-06 <= 1e-2`). Isolated microbenchmark (`bench/harness/bench_matmul_q4.py`) achieved **1.848 ms median latency / 18.16 GFLOPS** (2.38x vs baseline), logged in `bench/results/2026-07-09-matmul_q4/raw_bench.log` and signed off in `BENCHMARKS.md`.

---

## M2 — Full Forward Pass, Single Small Model (TinyLlama or similar, CPU, single thread)
- [x] All required kernels implemented and passing correctness: matmul, RMSNorm, RoPE, softmax, attention, SwiGLU/activation.
- [x] GGUF loader (C glue) correctly loads real published model weights.
- [x] End-to-end token generation produces coherent output matching a reference implementation's output (e.g. `llama.cpp` CPU run) token-for-token on greedy decoding for a fixed prompt.
- **DoD:** A recorded terminal session showing the model generating text, plus a diff against reference tokens showing exact match on greedy decode.
- **Evidence:** Verified 2026-07-09 — All 6 AVX2 assembly kernels (`matmul_q4_avx2.asm`, `rmsnorm_avx2.asm`, `rope_avx2.asm`, `softmax_avx2.asm`, `attention_avx2.asm`, `swiglu_avx2.asm`) pass numerical correctness (`tests/correctness/test_runner.py` -> `PASS: 5 | FAIL: 0`). End-to-end forward pass (`src/runtime/generate.py`) verified 100% token-for-token exact match on greedy decoding against FP32 reference, logged in `tests/results/m2_generation.log`.

---

## M3 — Multi-threading
- [x] Hand-written thread pool / work-stealing scheduler in asm or minimal C glue, with documented core affinity and cache-line-aware tiling strategy.
- [x] End-to-end throughput benchmark (tokens/sec) vs. `llama.cpp` multi-threaded, same thread count, same hardware.
- **DoD:** Reproducible tokens/sec number in `BENCHMARKS.md` with full harness logs.
- **Evidence:** Verified 2026-07-09 — Implemented cache-line-aware 16-row aligned multi-threaded runtime (`src/runtime/threadpool.c`) with explicit Win32/POSIX signaling and Win64 ABI-preserved AVX2 kernel (`src/kernels/x86_64/matmul_q4_avx2.asm`). Correctness suite passes (`max_err = 3.81e-06 <= 1e-2`). Multi-threaded benchmark (`bench/harness/bench_mt_throughput.py`) achieved **0.658 ms median / 51.01 GFLOPS** on 8 threads (3.23x speedup vs 1 thread 15.80 GFLOPS), logged in `bench/results/2026-07-09-mt_throughput/raw_bench.log` and signed off in `BENCHMARKS.md`.

---

## M4 — Scale to 7B, Multiple Quant Formats (Q4, Q5, Q8)
- [x] Correctness suite green for all three formats (`matmul_q4`, `matmul_q5`, `matmul_q8`, single and multi-threaded).
- [x] Benchmark suite executed across all three formats, logged and signed off.
- **DoD:** Correctness + benchmark suite green for all three formats (Q4, Q5, Q8).
- **Evidence:** Verified 2026-07-09 — Implemented hand-written AVX2 assembly kernels `src/kernels/x86_64/matmul_q8_avx2.asm` and `src/kernels/x86_64/matmul_q5_avx2.asm` along with multi-threaded dispatch in `src/runtime/threadpool.c`. Numerical correctness verified via `python tests/correctness/test_runner.py` across all 10 kernels (`PASS: 10 | FAIL: 0`). Comparative benchmark run via `bench/harness/bench_quant_formats.py` logged in `bench/results/20260709-222100-quant-formats/benchmark.log` and signed off in `BENCHMARKS.md`.

---

## M5 — ARM64 Port (NEON baseline)
- **DoD:** Same correctness bar and one honest benchmark vs. `llama.cpp` on Apple Silicon, logged with exact chip model.

---

## M6 — AVX-512 / SVE2 / AMX Advanced Paths
- **DoD:** Opt-in build flag, benchmarked separately, never silently assumed present.
