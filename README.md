# asmllm

[![Status](https://img.shields.io/badge/Status-Milestone%20M0%20(Harness%20%26%20Reference)-blue)](ROADMAP.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**One-line mission:** Build a hand-written, zero-intrinsics, zero-libc x86-64 and ARM64 assembly LLM inference engine that is measurably faster than `llama.cpp` on identical hardware, with every benchmark reproducible by a third party.

---

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
