#!/usr/bin/env python3
"""
build_kernel.py

Automates assembling and linking of hand-written asmllm assembly kernels.
Works cross-platform (Windows x64 / Linux x86-64).
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_ROOT / "build"
BUILD_DIR.mkdir(parents=True, exist_ok=True)


def find_nasm() -> str:
    local_nasm = PROJECT_ROOT / "tools" / "nasm.exe"
    if local_nasm.exists():
        return str(local_nasm)
    return "nasm"


def find_msvc_vcvars() -> str | None:
    candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def build_x86_64_kernels():
    nasm_bin = find_nasm()
    kernels_dir = PROJECT_ROOT / "src" / "kernels" / "x86_64"
    asm_files = sorted(kernels_dir.glob("*.asm"))

    obj_files = []
    fmt = "win64" if sys.platform == "win32" else "elf64"

    for asm_src in asm_files:
        obj_path = BUILD_DIR / f"{asm_src.stem}.{'obj' if sys.platform == 'win32' else 'o'}"
        print(f"[build_kernel] Assembling {asm_src.name} ({fmt})...")
        cmd_asm = [nasm_bin, "-f", fmt, str(asm_src), "-o", str(obj_path)]
        res_asm = subprocess.run(cmd_asm, capture_output=True, text=True)
        if res_asm.returncode != 0:
            print(f"[ERROR] NASM assembly failed on {asm_src.name}:\n{res_asm.stderr}")
            sys.exit(1)
        obj_files.append(obj_path)

    if sys.platform == "win32":
        dll_path = BUILD_DIR / "asmllm.dll"
        vcvars = find_msvc_vcvars()
        if not vcvars:
            print("[ERROR] MSVC vcvars64.bat not found.")
            sys.exit(1)

        threadpool_c = PROJECT_ROOT / "src" / "runtime" / "threadpool.c"
        threadpool_obj = BUILD_DIR / "threadpool.obj"

        print(f"[build_kernel] Compiling {threadpool_c.name} via MSVC cl.exe...")
        cl_cmd = (
            f'"{vcvars}" && cl.exe /nologo /c /O2 /MD /I"{PROJECT_ROOT / "src" / "runtime"}" '
            f'"{threadpool_c}" /Fo"{threadpool_obj}"'
        )
        res_cl = subprocess.run(cl_cmd, shell=True, capture_output=True, text=True)
        if res_cl.returncode != 0:
            print(f"[ERROR] cl.exe compile failed:\n{res_cl.stdout}\n{res_cl.stderr}")
            sys.exit(1)

        obj_files.append(threadpool_obj)
        obj_str = " ".join(f'"{p}"' for p in obj_files)
        exports = (
            " /EXPORT:asm_matmul_q4"
            " /EXPORT:asm_matmul_q8"
            " /EXPORT:asm_matmul_q5"
            " /EXPORT:asm_rmsnorm"
            " /EXPORT:asm_rope"
            " /EXPORT:asm_softmax"
            " /EXPORT:asm_attention"
            " /EXPORT:asm_silu_hadamard"
            " /EXPORT:asm_threadpool_init"
            " /EXPORT:asm_threadpool_shutdown"
            " /EXPORT:asm_matmul_q4_mt"
            " /EXPORT:asm_matmul_q8_mt"
            " /EXPORT:asm_matmul_q5_mt"
            " /EXPORT:asm_threadpool_dispatch_q8"
            " /EXPORT:asm_threadpool_dispatch_q5"
            " /EXPORT:asm_threadpool_get_num_threads"
        )

        print(f"[build_kernel] Linking {dll_path.name} via MSVC link.exe...")
        link_cmd = f'"{vcvars}" && link.exe /DLL /nologo {exports} {obj_str} /OUT:"{dll_path}"'
        res_link = subprocess.run(link_cmd, shell=True, capture_output=True, text=True)
        if res_link.returncode != 0:
            print(f"[ERROR] Linking failed:\n{res_link.stdout}\n{res_link.stderr}")
            sys.exit(1)

        print(f"[SUCCESS] Built shared library with multi-threaded runtime: {dll_path}")

        gguf_loader_c = PROJECT_ROOT / "src" / "loader" / "gguf_loader.c"
        gguf_loader_dll = BUILD_DIR / "gguf_loader.dll"
        gguf_loader_obj = BUILD_DIR / "gguf_loader.obj"
        print(f"[build_kernel] Compiling {gguf_loader_c.name} to {gguf_loader_dll.name}...")
        gguf_cmd = (
            f'"{vcvars}" && cl.exe /nologo /LD /O2 /I"{PROJECT_ROOT / "src" / "loader"}" '
            f'"{gguf_loader_c}" /Fe"{gguf_loader_dll}" /Fo"{gguf_loader_obj}"'
        )
        res_gguf = subprocess.run(gguf_cmd, shell=True, capture_output=True, text=True)
        if res_gguf.returncode != 0:
            print(f"[ERROR] gguf_loader compile failed:\n{res_gguf.stdout}\n{res_gguf.stderr}")
            sys.exit(1)
        print(f"[SUCCESS] Built GGUF loader DLL: {gguf_loader_dll}")

        return dll_path

    else:
        so_path = BUILD_DIR / "libasmllm.so"
        print(f"[build_kernel] Linking {so_path.name}...")
        obj_str = [str(p) for p in obj_files]
        cmd_link = ["gcc", "-shared", "-o", str(so_path)] + obj_str
        subprocess.run(cmd_link, check=True)
        print(f"[SUCCESS] Built shared library: {so_path}")
        return so_path


def build_arm64_kernels():
    kernels_dir = PROJECT_ROOT / "src" / "kernels" / "arm64"
    asm_files = sorted(kernels_dir.glob("*.S"))

    obj_files = []
    for asm_src in asm_files:
        obj_path = BUILD_DIR / f"{asm_src.stem}.o"
        print(f"[build_kernel] Assembling ARM64 NEON kernel {asm_src.name}...")
        cmd_asm = ["clang", "-c", str(asm_src), "-o", str(obj_path)]
        res_asm = subprocess.run(cmd_asm, capture_output=True, text=True)
        if res_asm.returncode != 0:
            print(f"[ERROR] clang assembly failed on {asm_src.name}:\n{res_asm.stderr}")
            sys.exit(1)
        obj_files.append(obj_path)

    threadpool_c = PROJECT_ROOT / "src" / "runtime" / "threadpool.c"
    threadpool_obj = BUILD_DIR / "threadpool.o"
    print(f"[build_kernel] Compiling {threadpool_c.name}...")
    cmd_tp = ["clang", "-c", "-O2", f"-I{PROJECT_ROOT / 'src' / 'runtime'}", str(threadpool_c), "-o", str(threadpool_obj)]
    res_tp = subprocess.run(cmd_tp, capture_output=True, text=True)
    if res_tp.returncode != 0:
        print(f"[ERROR] threadpool.c compile failed:\n{res_tp.stderr}")
        sys.exit(1)
    obj_files.append(threadpool_obj)

    ext = "dylib" if sys.platform == "darwin" else "so"
    lib_path = BUILD_DIR / f"libasmllm.{ext}"
    print(f"[build_kernel] Linking ARM64 shared library {lib_path.name}...")
    cmd_link = ["clang", "-shared", "-o", str(lib_path)] + [str(p) for p in obj_files]
    res_link = subprocess.run(cmd_link, capture_output=True, text=True)
    if res_link.returncode != 0:
        print(f"[ERROR] ARM64 link failed:\n{res_link.stderr}")
        sys.exit(1)
    print(f"[SUCCESS] Built ARM64 shared library: {lib_path}")

    gguf_loader_c = PROJECT_ROOT / "src" / "loader" / "gguf_loader.c"
    gguf_loader_lib = BUILD_DIR / f"gguf_loader.{ext}"
    print(f"[build_kernel] Compiling {gguf_loader_c.name} to {gguf_loader_lib.name}...")
    cmd_gguf = ["clang", "-shared", "-O2", f"-I{PROJECT_ROOT / 'src' / 'loader'}", str(gguf_loader_c), "-o", str(gguf_loader_lib)]
    res_gguf = subprocess.run(cmd_gguf, capture_output=True, text=True)
    if res_gguf.returncode != 0:
        print(f"[ERROR] gguf_loader compile failed:\n{res_gguf.stderr}")
        sys.exit(1)
    print(f"[SUCCESS] Built GGUF loader library: {gguf_loader_lib}")
    return lib_path


def main():
    import platform
    print("================================================================================")
    print(" asmllm Assembly Kernel Builder")
    print("================================================================================\n")
    arch = platform.machine().lower()
    if arch in ("arm64", "aarch64"):
        print(f"[build_kernel] Detected ARM64 architecture ({arch}). Building NEON kernels...")
        build_arm64_kernels()
    else:
        print(f"[build_kernel] Detected x86-64 architecture ({arch}). Building AVX2 kernels...")
        build_x86_64_kernels()


if __name__ == "__main__":
    main()
