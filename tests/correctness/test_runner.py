#!/usr/bin/env python3
"""
tests/correctness/test_runner.py

End-to-end numerical correctness test runner for asmllm kernels.
Compares native assembly kernel outputs against reference implementations within
the project numerical correctness tolerances.

Per M0 Definition of Done:
Runs end-to-end with zero kernels implemented yet (all show as "NOT IMPLEMENTED", not "false pass").
"""

import ctypes
import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.reference_gen import generate_reference_data

# Correctness Tolerances
TOLERANCE_Q4 = 1e-2      # Quantized Q4 error bound tolerance
TOLERANCE_Q8 = 1e-2      # Quantized Q8 error bound tolerance
TOLERANCE_Q5 = 1e-2      # Quantized Q5 error bound tolerance
TOLERANCE_FP32 = 1e-5    # Full-precision relative/absolute tolerance

KERNELS = [
    ("matmul_q4",    TOLERANCE_Q4,   "Q4_0 Quantized Matrix-Vector Multiplication"),
    ("matmul_q4_mt", TOLERANCE_Q4,   "Multi-Threaded Q4_0 Matvec (4 threads, cache-line tiled)"),
    ("matmul_q8",    TOLERANCE_Q8,   "Q8_0 Quantized Matrix-Vector Multiplication"),
    ("matmul_q8_mt", TOLERANCE_Q8,   "Multi-Threaded Q8_0 Matvec (4 threads, cache-line tiled)"),
    ("matmul_q5",    TOLERANCE_Q5,   "Q5_0 Quantized Matrix-Vector Multiplication"),
    ("matmul_q5_mt", TOLERANCE_Q5,   "Multi-Threaded Q5_0 Matvec (4 threads, cache-line tiled)"),
    ("rmsnorm",      TOLERANCE_FP32, "Root Mean Square Layer Normalization"),
    ("rope",         TOLERANCE_FP32, "Rotary Position Embedding"),
    ("softmax",      TOLERANCE_FP32, "Numerically Stable Softmax"),
    ("attention",    TOLERANCE_FP32, "Single-Head Scaled Dot-Product Attention"),
]


def try_load_native_library():
    """
    Attempts to load the compiled asmllm runtime/kernel shared library.
    Returns None if no library has been built yet.
    """
    candidates = [
        PROJECT_ROOT / "build" / "libasmllm.so",
        PROJECT_ROOT / "build" / "asmllm.dll",
        PROJECT_ROOT / "build" / "libasmllm.dylib",
    ]
    for lib_path in candidates:
        if lib_path.exists():
            try:
                return ctypes.CDLL(str(lib_path))
            except Exception as e:
                return None
    return None


def run_kernel_test(kernel_name: str, tolerance: float, ref_data: dict, native_lib) -> tuple[str, float | None, str]:
    """
    Runs correctness verification for a single kernel.
    Returns (status, max_error, note) where status is one of:
      - 'NOT IMPLEMENTED'
      - 'PASS'
      - 'FAIL'
    """
    if native_lib is None or not hasattr(native_lib, f"asm_{kernel_name}"):
        return "NOT IMPLEMENTED", None, "No native assembly kernel built/symbol found"

    func = getattr(native_lib, f"asm_{kernel_name}")

    if kernel_name == "matmul_q4":
        qweights = np.ascontiguousarray(ref_data["q4_matmul_qweights"], dtype=np.uint8)
        scales = np.ascontiguousarray(ref_data["q4_matmul_scales"], dtype=np.float32)
        x_vec = np.ascontiguousarray(ref_data["q4_matmul_x"], dtype=np.float32)
        expected = ref_data["q4_matmul_expected"]

        M = qweights.shape[0]
        K = x_vec.shape[0]
        y_asm = np.zeros(M, dtype=np.float32)

        func.argtypes = [
            ctypes.c_void_p,  # qweights
            ctypes.c_void_p,  # scales
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # y
            ctypes.c_int64,   # M
            ctypes.c_int64,   # K
        ]
        func.restype = None

        func(
            qweights.ctypes.data,
            scales.ctypes.data,
            x_vec.ctypes.data,
            y_asm.ctypes.data,
            int(M),
            int(K),
        )

        max_err = float(np.max(np.abs(y_asm - expected)))
        if max_err <= tolerance:
            return "PASS", max_err, f"AVX2 assembly kernel verified (max err {max_err:.2e} <= {tolerance:.1e})"
        else:
            return "FAIL", max_err, f"Max error {max_err:.2e} exceeded tolerance {tolerance:.1e}"

    elif kernel_name == "matmul_q4_mt":
        qweights = np.ascontiguousarray(ref_data["q4_matmul_qweights"], dtype=np.uint8)
        scales = np.ascontiguousarray(ref_data["q4_matmul_scales"], dtype=np.float32)
        x_vec = np.ascontiguousarray(ref_data["q4_matmul_x"], dtype=np.float32)
        expected = ref_data["q4_matmul_expected"]

        M = qweights.shape[0]
        K = x_vec.shape[0]
        y_asm = np.zeros(M, dtype=np.float32)

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

        func(
            qweights.ctypes.data,
            scales.ctypes.data,
            x_vec.ctypes.data,
            y_asm.ctypes.data,
            int(M),
            int(K),
            ctypes.c_int(4),
        )

        max_err = float(np.max(np.abs(y_asm - expected)))
        if max_err <= tolerance:
            return "PASS", max_err, f"Multi-threaded AVX2 kernel verified (4 threads, max err {max_err:.2e} <= {tolerance:.1e})"
        else:
            return "FAIL", max_err, f"Max error {max_err:.2e} exceeded tolerance {tolerance:.1e}"

    elif kernel_name == "matmul_q8":
        qweights = np.ascontiguousarray(ref_data["q8_matmul_qweights"], dtype=np.int8)
        scales = np.ascontiguousarray(ref_data["q8_matmul_scales"], dtype=np.float32)
        x_vec = np.ascontiguousarray(ref_data["q8_matmul_x"], dtype=np.float32)
        expected = ref_data["q8_matmul_expected"]

        M = qweights.shape[0]
        K = x_vec.shape[0]
        y_asm = np.zeros(M, dtype=np.float32)

        func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        func.restype = None

        func(
            qweights.ctypes.data,
            scales.ctypes.data,
            x_vec.ctypes.data,
            y_asm.ctypes.data,
            int(M),
            int(K),
        )

        max_err = float(np.max(np.abs(y_asm - expected)))
        if max_err <= tolerance:
            return "PASS", max_err, f"AVX2 Q8_0 kernel verified (max err {max_err:.2e} <= {tolerance:.1e})"
        else:
            return "FAIL", max_err, f"Max error {max_err:.2e} exceeded tolerance {tolerance:.1e}"

    elif kernel_name == "matmul_q8_mt":
        qweights = np.ascontiguousarray(ref_data["q8_matmul_qweights"], dtype=np.int8)
        scales = np.ascontiguousarray(ref_data["q8_matmul_scales"], dtype=np.float32)
        x_vec = np.ascontiguousarray(ref_data["q8_matmul_x"], dtype=np.float32)
        expected = ref_data["q8_matmul_expected"]

        M = qweights.shape[0]
        K = x_vec.shape[0]
        y_asm = np.zeros(M, dtype=np.float32)

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

        func(
            qweights.ctypes.data,
            scales.ctypes.data,
            x_vec.ctypes.data,
            y_asm.ctypes.data,
            int(M),
            int(K),
            ctypes.c_int(4),
        )

        max_err = float(np.max(np.abs(y_asm - expected)))
        if max_err <= tolerance:
            return "PASS", max_err, f"Multi-threaded Q8_0 kernel verified (4 threads, max err {max_err:.2e} <= {tolerance:.1e})"
        else:
            return "FAIL", max_err, f"Max error {max_err:.2e} exceeded tolerance {tolerance:.1e}"

    elif kernel_name == "matmul_q5":
        ql = np.ascontiguousarray(ref_data["q5_matmul_ql"], dtype=np.uint8)
        qh = np.ascontiguousarray(ref_data["q5_matmul_qh"], dtype=np.uint8)
        scales = np.ascontiguousarray(ref_data["q5_matmul_scales"], dtype=np.float32)
        x_vec = np.ascontiguousarray(ref_data["q5_matmul_x"], dtype=np.float32)
        expected = ref_data["q5_matmul_expected"]

        M = ql.shape[0]
        K = x_vec.shape[0]
        y_asm = np.zeros(M, dtype=np.float32)

        func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        func.restype = None

        func(
            ql.ctypes.data,
            qh.ctypes.data,
            scales.ctypes.data,
            x_vec.ctypes.data,
            y_asm.ctypes.data,
            int(M),
            int(K),
        )

        max_err = float(np.max(np.abs(y_asm - expected)))
        if max_err <= tolerance:
            return "PASS", max_err, f"AVX2 Q5_0 kernel verified (max err {max_err:.2e} <= {tolerance:.1e})"
        else:
            return "FAIL", max_err, f"Max error {max_err:.2e} exceeded tolerance {tolerance:.1e}"

    elif kernel_name == "matmul_q5_mt":
        ql = np.ascontiguousarray(ref_data["q5_matmul_ql"], dtype=np.uint8)
        qh = np.ascontiguousarray(ref_data["q5_matmul_qh"], dtype=np.uint8)
        scales = np.ascontiguousarray(ref_data["q5_matmul_scales"], dtype=np.float32)
        x_vec = np.ascontiguousarray(ref_data["q5_matmul_x"], dtype=np.float32)
        expected = ref_data["q5_matmul_expected"]

        M = ql.shape[0]
        K = x_vec.shape[0]
        y_asm = np.zeros(M, dtype=np.float32)

        func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        func.restype = None

        func(
            ql.ctypes.data,
            qh.ctypes.data,
            scales.ctypes.data,
            x_vec.ctypes.data,
            y_asm.ctypes.data,
            int(M),
            int(K),
            ctypes.c_int(4),
        )

        max_err = float(np.max(np.abs(y_asm - expected)))
        if max_err <= tolerance:
            return "PASS", max_err, f"Multi-threaded Q5_0 kernel verified (4 threads, max err {max_err:.2e} <= {tolerance:.1e})"
        else:
            return "FAIL", max_err, f"Max error {max_err:.2e} exceeded tolerance {tolerance:.1e}"

    elif kernel_name == "rmsnorm":
        x = np.ascontiguousarray(ref_data["rmsnorm_x"], dtype=np.float32)
        weight = np.ascontiguousarray(ref_data["rmsnorm_weight"], dtype=np.float32)
        eps = float(ref_data["rmsnorm_eps"])
        expected = ref_data["rmsnorm_expected"]
        dim = x.shape[0]
        y_asm = np.zeros(dim, dtype=np.float32)

        func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_float,
        ]
        func.restype = None
        func(x.ctypes.data, weight.ctypes.data, y_asm.ctypes.data, int(dim), ctypes.c_float(eps))

        max_err = float(np.max(np.abs(y_asm - expected)))
        if max_err <= tolerance:
            return "PASS", max_err, f"AVX2 assembly kernel verified (max err {max_err:.2e} <= {tolerance:.1e})"
        else:
            return "FAIL", max_err, f"Max error {max_err:.2e} exceeded tolerance {tolerance:.1e}"

    elif kernel_name == "rope":
        q = np.ascontiguousarray(ref_data["rope_q"], dtype=np.float32)
        k = np.ascontiguousarray(ref_data["rope_k"], dtype=np.float32)
        pos = int(ref_data["rope_pos"])
        head_dim = q.shape[-1]
        exp_q = ref_data["rope_q_expected"]
        exp_k = ref_data["rope_k_expected"]

        func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
        ]
        func.restype = None
        func(q.ctypes.data, k.ctypes.data, int(head_dim), int(pos), ctypes.c_float(10000.0))

        max_err = float(max(np.max(np.abs(q - exp_q)), np.max(np.abs(k - exp_k))))
        if max_err <= tolerance:
            return "PASS", max_err, f"AVX2 assembly kernel verified (max err {max_err:.2e} <= {tolerance:.1e})"
        else:
            return "FAIL", max_err, f"Max error {max_err:.2e} exceeded tolerance {tolerance:.1e}"

    elif kernel_name == "softmax":
        x = np.ascontiguousarray(ref_data["softmax_x"], dtype=np.float32)
        expected = ref_data["softmax_expected"]
        dim = x.shape[0]
        y_asm = np.zeros(dim, dtype=np.float32)

        func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        func.restype = None
        func(x.ctypes.data, y_asm.ctypes.data, int(dim))

        max_err = float(np.max(np.abs(y_asm - expected)))
        if max_err <= tolerance:
            return "PASS", max_err, f"AVX2 assembly kernel verified (max err {max_err:.2e} <= {tolerance:.1e})"
        else:
            return "FAIL", max_err, f"Max error {max_err:.2e} exceeded tolerance {tolerance:.1e}"

    elif kernel_name == "attention":
        q = np.ascontiguousarray(ref_data["attention_q"], dtype=np.float32)
        k = np.ascontiguousarray(ref_data["attention_k"], dtype=np.float32)
        v = np.ascontiguousarray(ref_data["attention_v"], dtype=np.float32)
        expected = ref_data["attention_expected"]
        seq_len, head_dim = q.shape
        y_asm = np.zeros_like(q)

        func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
        ]
        func.restype = None
        scale = 1.0 / np.sqrt(head_dim)
        func(
            q.ctypes.data,
            k.ctypes.data,
            v.ctypes.data,
            y_asm.ctypes.data,
            int(seq_len),
            int(head_dim),
            ctypes.c_float(scale),
        )

        max_err = float(np.max(np.abs(y_asm - expected)))
        if max_err <= tolerance:
            return "PASS", max_err, f"AVX2 assembly kernel verified (max err {max_err:.2e} <= {tolerance:.1e})"
        else:
            return "FAIL", max_err, f"Max error {max_err:.2e} exceeded tolerance {tolerance:.1e}"

    return "NOT IMPLEMENTED", None, "Symbol found but dispatcher not yet wired"


def main():
    print("================================================================================")
    print(" asmllm Correctness Test Runner (Numerical Compliance)")
    print("================================================================================\n")

    ref_data = generate_reference_data(seed=42)
    native_lib = try_load_native_library()

    print(f"Loaded reference test suite ({len(ref_data)} tensors).")
    if native_lib is None:
        print("Native assembly library not found (no kernels built yet).\n")

    results = []
    counts = {"PASS": 0, "FAIL": 0, "NOT IMPLEMENTED": 0}

    print(f"{'Kernel':<15} {'Status':<17} {'Max Error':<12} {'Tolerance':<12} {'Notes'}")
    print("-" * 80)

    for k_name, tol, desc in KERNELS:
        status, max_err, notes = run_kernel_test(k_name, tol, ref_data, native_lib)
        counts[status] += 1

        err_str = f"{max_err:.2e}" if max_err is not None else "N/A"
        print(f"{k_name:<15} {status:<17} {err_str:<12} {tol:<12.1e} {notes}", flush=True)
        results.append((k_name, status, max_err, notes))

    print("-" * 80)
    print(f"Summary: PASS: {counts['PASS']} | FAIL: {counts['FAIL']} | NOT IMPLEMENTED: {counts['NOT IMPLEMENTED']}")
    print("================================================================================\n")

    if counts["FAIL"] > 0:
        print("[FAIL] One or more kernels failed numerical correctness verification vs reference.")
        sys.exit(1)
    elif counts["PASS"] == 0 and counts["NOT IMPLEMENTED"] == len(KERNELS):
        print("[M0 VERIFIED] Zero kernels implemented yet. All kernels correctly reported as NOT IMPLEMENTED (zero false passes).")
        sys.exit(0)
    else:
        print(f"[SUCCESS] All implemented kernels passed correctness checks within tolerance.")
        sys.exit(0)


if __name__ == "__main__":
    main()
