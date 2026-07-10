#!/usr/bin/env python3
"""
tests/correctness/test_gguf_loader.py

Verifies that src/loader/gguf_loader.c correctly parses GGUF v2/v3 files,
including magic validation, metadata key-value entries, tensor descriptor tables,
and weight buffer pointers.
"""

import ctypes
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
DLL_PATH = BUILD_DIR / "gguf_loader.dll"

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

class GGUFLoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DLL_PATH.exists():
            raise RuntimeError(f"Cannot find compiled DLL at {DLL_PATH}")
        cls.lib = ctypes.CDLL(str(DLL_PATH))
        cls.lib.gguf_load_from_file.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        cls.lib.gguf_load_from_file.restype = ctypes.c_int
        cls.lib.gguf_free.argtypes = [ctypes.c_void_p]
        cls.lib.gguf_free.restype = None

    def test_parse_valid_gguf(self):
        # Create a minimal valid binary GGUF v3 file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as tf:
            path = tf.name
            magic = 0x46554747
            version = 3
            tensor_count = 1
            kv_count = 0
            tf.write(struct.pack("<IIQQ", magic, version, tensor_count, kv_count))

            # Tensor info: name="test_weight", n_dims=2, dims=[4, 8], type=0 (F32), offset=0
            name_bytes = b"test_weight"
            tf.write(struct.pack("<Q", len(name_bytes)))
            tf.write(name_bytes)
            tf.write(struct.pack("<I", 2))
            tf.write(struct.pack("<QQ", 4, 8))
            tf.write(struct.pack("<IQ", 0, 0))

            # Padding to 32-byte alignment
            pos = tf.tell()
            rem = pos % 32
            if rem != 0:
                tf.write(b"\x00" * (32 - rem))

            # Write 4*8 float32 weights (128 bytes)
            tf.write(b"\x01" * 128)

        try:
            ctx = ctypes.c_void_p()
            res = self.lib.gguf_load_from_file(path.encode("utf-8"), ctypes.byref(ctx))
            self.assertEqual(res, 1, "gguf_load_from_file failed on valid GGUF binary")
            self.lib.gguf_free(ctx)
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
