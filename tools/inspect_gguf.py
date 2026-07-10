#!/usr/bin/env python3
"""
tools/inspect_gguf.py

Loads a GGUF model file using src/loader/gguf_loader.c (via build/gguf_loader.dll)
and displays model metadata and tensor information.
"""

import ctypes
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DLL_PATH = PROJECT_ROOT / "build" / "gguf_loader.dll"

class GGUFHeader(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("tensor_count", ctypes.c_uint64),
        ("kv_count", ctypes.c_uint64),
    ]

class GGUFTensorInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 128),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint64 * 4),
        ("type", ctypes.c_uint32),
        ("offset", ctypes.c_uint64),
    ]

class GGUFContext(ctypes.Structure):
    _fields_ = [
        ("header", GGUFHeader),
        ("kv_pairs", ctypes.c_void_p),
        ("tensors", ctypes.POINTER(GGUFTensorInfo)),
        ("file_data", ctypes.c_void_p),
        ("file_size", ctypes.c_size_t),
        ("data_offset", ctypes.c_size_t),
    ]

def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "models/stories15M-q4_0.gguf"
    if not Path(filepath).exists():
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    lib = ctypes.CDLL(str(DLL_PATH))
    lib.gguf_load_from_file.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.POINTER(GGUFContext))]
    lib.gguf_load_from_file.restype = ctypes.c_int

    ctx_ptr = ctypes.POINTER(GGUFContext)()
    res = lib.gguf_load_from_file(filepath.encode("utf-8"), ctypes.byref(ctx_ptr))
    if res != 1 or not ctx_ptr:
        print("[ERROR] Failed to load GGUF file via gguf_loader.dll")
        sys.exit(1)

    ctx = ctx_ptr.contents
    print(f"=== GGUF Model Inspection: {filepath} ===")
    print(f"Magic: 0x{ctx.header.magic:08x} | Version: {ctx.header.version}")
    print(f"Tensors: {ctx.header.tensor_count} | KV Pairs: {ctx.header.kv_count}")
    print(f"Data Offset: {ctx.data_offset} bytes | Total Size: {ctx.file_size} bytes")
    print("\n--- Top Tensors ---")
    for i in range(min(15, int(ctx.header.tensor_count))):
        ti = ctx.tensors[i]
        dims = list(ti.dims[:ti.n_dims])
        print(f"[{i:2d}] {ti.name.decode('utf-8', errors='ignore'):<35} | Shape: {dims} | Type: {ti.type} | Offset: {ti.offset}")

    lib.gguf_free(ctx_ptr)

if __name__ == "__main__":
    main()
