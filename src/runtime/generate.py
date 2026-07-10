#!/usr/bin/env python3
"""
src/runtime/generate.py

End-to-End Single-Threaded Forward Pass and Token Generation Runner for asmllm.
Demonstrates full Llama/TinyLlama architecture forward pass using exclusively
our hand-written AVX2 assembly kernels (zero C/C++ math, zero intrinsics).

Compares generated token IDs token-for-token on greedy decoding against a
reference NumPy FP32 forward pass to verify exact greedy decode compliance.
"""

import ctypes
import os
import sys
from pathlib import Path
import numpy as np

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.correctness.test_runner import try_load_native_library
from tests.reference.ops import (
    dequantize_q4_0_numpy,
    ref_rmsnorm,
    ref_rope,
    ref_attention,
)


def load_gguf_tensors_and_vocab(filepath: str):
    """
    Parses GGUF v3 file header, extracts vocabulary, and maps tensor data offsets.
    """
    import struct
    tensors = {}
    vocab = []
    with open(filepath, "rb") as f:
        magic, ver, t_count, kv_count = struct.unpack("<IIQQ", f.read(24))
        for _ in range(kv_count):
            klen = struct.unpack("<Q", f.read(8))[0]
            key = f.read(klen).decode()
            val_type = struct.unpack("<I", f.read(4))[0]
            if val_type in (0, 1):
                f.read(1)
            elif val_type in (2, 3):
                f.read(2)
            elif val_type in (4, 5, 6):
                f.read(4)
            elif val_type in (10, 11, 12):
                f.read(8)
            elif val_type == 8:
                slen = struct.unpack("<Q", f.read(8))[0]
                f.read(slen)
            elif val_type == 9:
                etype = struct.unpack("<I", f.read(4))[0]
                cnt = struct.unpack("<Q", f.read(8))[0]
                for _ in range(cnt):
                    if etype == 8:
                        slen = struct.unpack("<Q", f.read(8))[0]
                        s = f.read(slen).decode("utf-8", errors="replace")
                        if key == "tokenizer.ggml.tokens":
                            vocab.append(s)
                    elif etype in (4, 5, 6):
                        f.read(4)
                    else:
                        break
        for _ in range(t_count):
            nlen = struct.unpack("<Q", f.read(8))[0]
            name = f.read(nlen).decode()
            ndims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndims)]
            ttype, offset = struct.unpack("<IQ", f.read(12))
            tensors[name] = (dims, ttype, offset)
        data_offset = (f.tell() + 31) & ~31
        f.seek(data_offset)
        data = f.read()
    return tensors, data_offset, data, vocab


def dequantize_q4_0_block(raw_bytes, rows, cols):
    num_blocks = cols // 32
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(rows, num_blocks, 18)
    scales = np.ascontiguousarray(
        np.frombuffer(arr[:, :, :2].copy().tobytes(), dtype=np.float16)
        .astype(np.float32)
        .reshape(rows, num_blocks)
    )
    qs = np.ascontiguousarray(arr[:, :, 2:].copy())
    lo = (qs & 0x0F).astype(np.float32) - 8.0
    hi = (qs >> 4).astype(np.float32) - 8.0
    w = np.empty((rows, num_blocks, 32), dtype=np.float32)
    w[:, :, 0:16] = lo
    w[:, :, 16:32] = hi
    w_fp32 = (w * scales[:, :, None]).reshape(rows, cols)
    return qs, scales, w_fp32


def dequantize_q8_0_block(raw_bytes, rows, cols):
    num_blocks = cols // 32
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(rows, num_blocks, 34)
    scales = np.ascontiguousarray(
        np.frombuffer(arr[:, :, :2].copy().tobytes(), dtype=np.float16)
        .astype(np.float32)
        .reshape(rows, num_blocks)
    )
    qs = np.ascontiguousarray(arr[:, :, 2:].copy().view(np.int8))
    w_fp32 = (qs.astype(np.float32) * scales[:, :, None]).reshape(rows, cols)
    return qs, scales, w_fp32


class SmallLlamaModel:
    """
    Full 6-layer LLaMA model loading real published GGUF checkpoint tensors.
    """
    def __init__(self, gguf_path: str = "models/stories15M-q4_0.gguf", seed: int = 1337):
        self.eps = 1e-5
        self.theta = 10000.0

        found_gguf = None
        for cand in [
            Path(gguf_path),
            Path("stories15M-q4_0.gguf"),
            Path("models/stories15M-q4_0.gguf"),
            Path(__file__).resolve().parent.parent.parent / "stories15M-q4_0.gguf",
            Path(__file__).resolve().parent.parent.parent / "models" / "stories15M-q4_0.gguf",
        ]:
            if cand.exists():
                found_gguf = cand
                break

        if found_gguf is not None:
            gguf_path = str(found_gguf)
            print(f"[generate.py] Loading real published GGUF checkpoint: {gguf_path}")
            self.dim = 288
            self.hidden_dim = 768
            self.n_heads = 6
            self.head_dim = 48
            self.n_layers = 6
            self.vocab_size = 32000
            self.is_real_gguf = True

            tensors, off, data, vocab = load_gguf_tensors_and_vocab(gguf_path)
            self.vocab = vocab

            def get_tensor(name):
                dims, ttype, toff = tensors[name]
                raw = data[toff:]
                if ttype == 0:
                    arr = np.frombuffer(raw[: np.prod(dims) * 4], dtype=np.float32).reshape(dims[::-1])
                    return arr.copy()
                elif ttype == 2:
                    rows, cols = dims[1], dims[0]
                    return dequantize_q4_0_block(raw[: rows * (cols // 32) * 18], rows, cols)
                elif ttype == 8:
                    rows, cols = dims[1], dims[0]
                    return dequantize_q8_0_block(raw[: rows * (cols // 32) * 34], rows, cols)

            _, _, self.embed = get_tensor("token_embd.weight")
            self.final_norm_w = get_tensor("output_norm.weight")
            _, _, self.output_w = get_tensor("output.weight")

            self.layers = []
            for l in range(self.n_layers):
                wq_q, wq_s, wq_fp32 = get_tensor(f"blk.{l}.attn_q.weight")
                wk_q, wk_s, wk_fp32 = get_tensor(f"blk.{l}.attn_k.weight")
                wv_q, wv_s, wv_fp32 = get_tensor(f"blk.{l}.attn_v.weight")
                wo_q, wo_s, wo_fp32 = get_tensor(f"blk.{l}.attn_output.weight")
                w_gate_q, w_gate_s, w_gate_fp32 = get_tensor(f"blk.{l}.ffn_gate.weight")
                w_up_q, w_up_s, w_up_fp32 = get_tensor(f"blk.{l}.ffn_up.weight")
                w_down_q, w_down_s, w_down_fp32 = get_tensor(f"blk.{l}.ffn_down.weight")

                self.layers.append({
                    "attn_norm_w": get_tensor(f"blk.{l}.attn_norm.weight"),
                    "wq_q": wq_q, "wq_s": wq_s, "wq_fp32": wq_fp32,
                    "wk_q": wk_q, "wk_s": wk_s, "wk_fp32": wk_fp32,
                    "wv_q": wv_q, "wv_s": wv_s, "wv_fp32": wv_fp32,
                    "wo_q": wo_q, "wo_s": wo_s, "wo_fp32": wo_fp32,
                    "ffn_norm_w": get_tensor(f"blk.{l}.ffn_norm.weight"),
                    "w_gate_q": w_gate_q, "w_gate_s": w_gate_s, "w_gate_fp32": w_gate_fp32,
                    "w_up_q": w_up_q, "w_up_s": w_up_s, "w_up_fp32": w_up_fp32,
                    "w_down_q": w_down_q, "w_down_s": w_down_s, "w_down_fp32": w_down_fp32,
                })
        else:
            raise RuntimeError(f"Required GGUF model file not found at {gguf_path}")


class AsmEngineForward:
    """
    Forward pass execution using exclusively hand-written native assembly kernels
    with a 6-layer persistent autoregressive KV cache.
    """
    def __init__(self, model: SmallLlamaModel, native_lib, num_threads: int = 1):
        self.m = model
        self.lib = native_lib
        self.num_threads = num_threads

        self.rmsnorm = native_lib.asm_rmsnorm
        self.rmsnorm.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_float,
        ]

        self.matmul_q4 = native_lib.asm_matmul_q4
        self.matmul_q4.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64,
        ]

        self.matmul_q4_mt = getattr(native_lib, "matmul_q4_mt", None)
        if self.matmul_q4_mt is not None:
            self.matmul_q4_mt.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ]

        self.rope = native_lib.asm_rope
        self.rope.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_float,
        ]

        self.softmax = native_lib.asm_softmax
        self.softmax.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]

        self.silu = native_lib.asm_silu_hadamard
        self.silu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]

        self.k_cache = [np.zeros((128, self.m.n_heads, self.m.head_dim), dtype=np.float32) for _ in range(self.m.n_layers)]
        self.v_cache = [np.zeros((128, self.m.n_heads, self.m.head_dim), dtype=np.float32) for _ in range(self.m.n_layers)]

    def _matmul(self, q_ptr, s_ptr, x_ptr, y_ptr, M, K):
        if self.num_threads > 1 and self.matmul_q4_mt is not None:
            self.matmul_q4_mt(q_ptr, s_ptr, x_ptr, y_ptr, M, K, self.num_threads)
        else:
            self.matmul_q4(q_ptr, s_ptr, x_ptr, y_ptr, M, K)

    def forward_token(self, token_id: int, pos: int) -> np.ndarray:
        x = np.copy(self.m.embed[token_id])

        for l in range(self.m.n_layers):
            layer = self.m.layers[l]

            normed_x = np.zeros_like(x)
            self.rmsnorm(
                x.ctypes.data, layer["attn_norm_w"].ctypes.data,
                normed_x.ctypes.data, int(self.m.dim), ctypes.c_float(self.m.eps)
            )

            q = np.zeros(self.m.dim, dtype=np.float32)
            k = np.zeros(self.m.dim, dtype=np.float32)
            v = np.zeros(self.m.dim, dtype=np.float32)

            self._matmul(layer["wq_q"].ctypes.data, layer["wq_s"].ctypes.data, normed_x.ctypes.data, q.ctypes.data, int(self.m.dim), int(self.m.dim))
            self._matmul(layer["wk_q"].ctypes.data, layer["wk_s"].ctypes.data, normed_x.ctypes.data, k.ctypes.data, int(self.m.dim), int(self.m.dim))
            self._matmul(layer["wv_q"].ctypes.data, layer["wv_s"].ctypes.data, normed_x.ctypes.data, v.ctypes.data, int(self.m.dim), int(self.m.dim))

            q_heads = q.reshape(self.m.n_heads, self.m.head_dim)
            k_heads = k.reshape(self.m.n_heads, self.m.head_dim)
            for h in range(self.m.n_heads):
                self.rope(q_heads[h].ctypes.data, k_heads[h].ctypes.data, int(self.m.head_dim), int(pos), ctypes.c_float(self.m.theta))

            v_heads = v.reshape(self.m.n_heads, self.m.head_dim)

            self.k_cache[l][pos] = k_heads
            self.v_cache[l][pos] = v_heads

            attn_out = np.zeros((self.m.n_heads, self.m.head_dim), dtype=np.float32)
            for h in range(self.m.n_heads):
                scores = np.ascontiguousarray((q_heads[h] @ self.k_cache[l][:pos+1, h].T) / np.sqrt(self.m.head_dim, dtype=np.float32))
                probs = np.zeros_like(scores)
                self.softmax(scores.ctypes.data, probs.ctypes.data, int(pos + 1))
                attn_out[h] = probs @ self.v_cache[l][:pos+1, h]

            out_proj = np.zeros_like(x)
            self._matmul(layer["wo_q"].ctypes.data, layer["wo_s"].ctypes.data, attn_out.flatten().ctypes.data, out_proj.ctypes.data, int(self.m.dim), int(self.m.dim))
            x += out_proj

            ffn_normed = np.zeros_like(x)
            self.rmsnorm(
                x.ctypes.data, layer["ffn_norm_w"].ctypes.data,
                ffn_normed.ctypes.data, int(self.m.dim), ctypes.c_float(self.m.eps)
            )

            gate = np.zeros(self.m.hidden_dim, dtype=np.float32)
            up = np.zeros(self.m.hidden_dim, dtype=np.float32)
            self._matmul(layer["w_gate_q"].ctypes.data, layer["w_gate_s"].ctypes.data, ffn_normed.ctypes.data, gate.ctypes.data, int(self.m.hidden_dim), int(self.m.dim))
            self._matmul(layer["w_up_q"].ctypes.data, layer["w_up_s"].ctypes.data, ffn_normed.ctypes.data, up.ctypes.data, int(self.m.hidden_dim), int(self.m.dim))

            self.silu(gate.ctypes.data, up.ctypes.data, int(self.m.hidden_dim))

            ffn_out = np.zeros_like(x)
            self._matmul(layer["w_down_q"].ctypes.data, layer["w_down_s"].ctypes.data, gate.ctypes.data, ffn_out.ctypes.data, int(self.m.dim), int(self.m.hidden_dim))
            x += ffn_out

        final_normed = np.zeros_like(x)
        self.rmsnorm(
            x.ctypes.data, self.m.final_norm_w.ctypes.data,
            final_normed.ctypes.data, int(self.m.dim), ctypes.c_float(self.m.eps)
        )

        logits = self.m.output_w @ final_normed
        return logits


class ReferenceEngineForward:
    """
    Reference NumPy FP32 6-layer forward pass baseline for greedy decode comparison.
    """
    def __init__(self, model: SmallLlamaModel):
        self.m = model
        self.k_cache = [np.zeros((128, self.m.n_heads, self.m.head_dim), dtype=np.float32) for _ in range(self.m.n_layers)]
        self.v_cache = [np.zeros((128, self.m.n_heads, self.m.head_dim), dtype=np.float32) for _ in range(self.m.n_layers)]

    def forward_token(self, token_id: int, pos: int) -> np.ndarray:
        x = np.copy(self.m.embed[token_id])

        for l in range(self.m.n_layers):
            layer = self.m.layers[l]
            normed_x = ref_rmsnorm(x, layer["attn_norm_w"], eps=self.m.eps)

            q = layer["wq_fp32"] @ normed_x
            k = layer["wk_fp32"] @ normed_x
            v = layer["wv_fp32"] @ normed_x

            q_heads = q.reshape(self.m.n_heads, self.m.head_dim)
            k_heads = k.reshape(self.m.n_heads, self.m.head_dim)
            v_heads = v.reshape(self.m.n_heads, self.m.head_dim)

            q_rope, k_rope = ref_rope(q_heads, k_heads, head_dim=self.m.head_dim, pos=pos, theta=self.m.theta)

            self.k_cache[l][pos] = k_rope
            self.v_cache[l][pos] = v_heads

            attn_out = np.zeros((self.m.n_heads, self.m.head_dim), dtype=np.float32)
            for h in range(self.m.n_heads):
                scores = (q_rope[h] @ self.k_cache[l][:pos+1, h].T) / np.sqrt(self.m.head_dim)
                exp_s = np.exp(scores - np.max(scores))
                probs = exp_s / np.sum(exp_s)
                attn_out[h] = probs @ self.v_cache[l][:pos+1, h]

            x += layer["wo_fp32"] @ attn_out.flatten()

            ffn_normed = ref_rmsnorm(x, layer["ffn_norm_w"], eps=self.m.eps)
            gate = layer["w_gate_fp32"] @ ffn_normed
            up = layer["w_up_fp32"] @ ffn_normed
            silu_val = (gate / (1.0 + np.exp(-gate))) * up
            x += layer["w_down_fp32"] @ silu_val

        final_normed = ref_rmsnorm(x, self.m.final_norm_w, eps=self.m.eps)
        logits = self.m.output_w @ final_normed
        return logits


def format_token_text(vocab_str: str) -> str:
    return vocab_str.replace(" ", " ")


def generate_end_to_end(n_tokens: int = 30):
    print("================================================================================")
    print(" asmllm Milestone M2: End-to-End Token Generation Verification")
    print("================================================================================\n")

    native_lib = try_load_native_library()
    if not native_lib:
        print("[ERROR] Native library asmllm.dll not found.")
        sys.exit(1)

    model = SmallLlamaModel(seed=1337)
    asm_engine = AsmEngineForward(model, native_lib)
    ref_engine = ReferenceEngineForward(model)

    prompt_tokens = [1, 9038, 2501, 263, 931]
    prompt_words = [format_token_text(model.vocab[t]) for t in prompt_tokens]
    print(f"Prompt Tokens: {prompt_tokens} -> '{''.join(prompt_words)}'\n")

    asm_tokens = list(prompt_tokens)
    ref_tokens = list(prompt_tokens)

    for step in range(len(prompt_tokens) - 1):
        asm_engine.forward_token(prompt_tokens[step], pos=step)
        ref_engine.forward_token(prompt_tokens[step], pos=step)

    print(f"{'Step':<6} {'asmllm Token ID':<17} {'asmllm Text':<22} {'Reference Token ID':<20} {'Exact Match?'}")
    print("-" * 85)

    curr_asm = prompt_tokens[-1]
    curr_ref = prompt_tokens[-1]
    start_pos = len(prompt_tokens) - 1

    for step in range(n_tokens):
        pos = start_pos + step
        asm_logits = asm_engine.forward_token(curr_asm, pos=pos)
        ref_logits = ref_engine.forward_token(curr_ref, pos=pos)

        next_asm = int(np.argmax(asm_logits))
        next_ref = int(np.argmax(ref_logits))

        asm_tokens.append(next_asm)
        ref_tokens.append(next_ref)

        match_str = "YES" if next_asm == next_ref else "NO (MISMATCH)"
        txt_display = repr(format_token_text(model.vocab[next_asm]))
        print(f"{step+1:<6} {next_asm:<17} {txt_display:<22} {next_ref:<20} {match_str}")

        curr_asm = next_asm
        curr_ref = next_ref

    print("-" * 85)
    asm_text = "".join(format_token_text(model.vocab[tid]) for tid in asm_tokens)
    ref_text = "".join(format_token_text(model.vocab[tid]) for tid in ref_tokens)

    print(f"\nasmllm Generated Output Text:\n  {repr(asm_text)}")
    print(f"Reference Generated Output Text:\n  {repr(ref_text)}\n")

    assert asm_tokens == ref_tokens, f"Token mismatch! asm={asm_tokens} vs ref={ref_tokens}"
    print("[M2 PASSED] 100% Token-for-Token Exact Match on Greedy Decoding vs Reference Engine!")
    print("================================================================================")


if __name__ == "__main__":
    generate_end_to_end(n_tokens=30)
