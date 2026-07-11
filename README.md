# asmllm

[![Status](https://img.shields.io/badge/Status-Milestone%20M5%20(ARM64%20Port)-blue)](ROADMAP.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**One-line mission:** Build a hand-written, zero-intrinsics, zero-libc x86-64 and ARM64 assembly LLM inference engine that is measurably faster than `llama.cpp` on identical hardware, with every benchmark reproducible by a third party.

> **Current Performance Status (2026-07-11):**
> - **x86-64 AVX2 (1T):** ✅ **Ahead** — `asmllm` 1011.82 t/s vs `llama.cpp` 902.38 t/s (+12.1%)
> - **x86-64 AVX2 (2T–8T):** ❌ **Behind** — `llama.cpp` leads by 9–23% at higher thread counts due to mature work-stealing scheduler
> - **ARM64 NEON (1T–2T):** ❌ **Behind** — `asmllm` 288 t/s vs `llama.cpp` 1933 t/s (6.7x slower, first-generation port)
>
> This is a work in progress. See [`BENCHMARKS.md`](BENCHMARKS.md) for full receipts.

## Core Principles

1. **No C/C++ in the Hot Path:** All tensor operations (matmul, attention, RMSNorm, RoPE, softmax, quantization/dequantization, activation functions) are hand-written assembly (`NASM` for x86-64, `GAS` for ARM64). No compiler intrinsics anywhere.
2. **Numerical Correctness Before Speed:** Fast wrong kernels are rejected. Every kernel is verified against PyTorch / NumPy reference implementations within strict error tolerances.
3. **No Cherry-Picked Benchmarks:** Every benchmark claim requires a reproducible harness log against default-build `llama.cpp` on identical hardware.
4. **No Silent Fallbacks:** If a kernel is not implemented for a target architecture, it fails loudly rather than falling back to scalar C.

---

## Target Platforms

| Priority | Architecture | ISA Extensions | Assembler |
|---|---|---|---|
| **P0** | x86-64 | AVX2 (baseline), AVX-512 (optional path) | NASM |
| **P1** | ARM64 (Apple Silicon) | NEON, SVE2 where available | GAS |
| **P2** | x86-64 | AMX (Sapphire Rapids+) | NASM |

---

## Repository Structure

```
asmllm/
├── README.md                  # Public overview
├── BENCHMARKS.md              # Append-only log of benchmark results
├── ROADMAP.md                 # Milestone tracker (M0 - M6)
├── src/
│   ├── kernels/               # Assembly kernels (NASM / GAS)
│   ├── runtime/               # Threading & memory management
│   ├── loader/                # GGUF parser (thin C glue)
│   └── cli/                   # CLI entry point
├── tests/
│   ├── correctness/           # End-to-end numerical correctness suite
│   └── reference/             # NumPy/PyTorch reference operations
├── bench/
│   ├── harness/               # Benchmark drivers
│   └── results/               # Raw benchmark logs
└── tools/
    └── reference_gen.py       # Reference tensor and expected output generator
```

---

## Running Correctness Tests

```bash
# Run correctness test harness across all kernels
make test-correctness

# Or run directly via Python
python tests/correctness/test_runner.py
```
