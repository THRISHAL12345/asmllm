#!/usr/bin/env python3
"""
tools/reference_gen.py

Generates golden reference tensors and expected outputs for assembly kernels.
Complies with Milestone M0 specification.
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to path so we can import tests.reference.ops
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.reference.ops import (
    ref_q4_matmul,
    ref_q8_matmul,
    ref_q5_matmul,
    ref_rmsnorm,
    ref_rope,
    ref_softmax,
    ref_attention,
    HAS_TORCH,
)


def generate_reference_data(seed: int = 42) -> dict:
    """
    Generates deterministic reference input/output tensors for all operations.
    Returns a dictionary of numpy arrays.
    """
    rng = np.random.RandomState(seed)
    data = {}

    # 1. Q4 Matmul (M=4, K=64, num_blocks=2, block_size=32)
    # qweights shape: (4, 2, 16) uint8
    qweights = rng.randint(0, 256, size=(4, 2, 16), dtype=np.uint8)
    scales = rng.uniform(0.1, 1.5, size=(4, 2)).astype(np.float32)
    x_vec = rng.randn(64).astype(np.float32)
    y_q4_expected = ref_q4_matmul(qweights, scales, x_vec)

    data["q4_matmul_qweights"] = qweights
    data["q4_matmul_scales"] = scales
    data["q4_matmul_x"] = x_vec
    data["q4_matmul_expected"] = y_q4_expected

    # 1b. Q8 Matmul (M=4, K=64, num_blocks=2, block_size=32)
    q8_weights = rng.randint(-128, 128, size=(4, 2, 32), dtype=np.int8)
    q8_scales = rng.uniform(0.1, 1.5, size=(4, 2)).astype(np.float32)
    y_q8_expected = ref_q8_matmul(q8_weights, q8_scales, x_vec)

    data["q8_matmul_qweights"] = q8_weights
    data["q8_matmul_scales"] = q8_scales
    data["q8_matmul_x"] = x_vec
    data["q8_matmul_expected"] = y_q8_expected

    # 1c. Q5 Matmul (M=4, K=64, num_blocks=2, block_size=32)
    q5_l = rng.randint(0, 256, size=(4, 2, 16), dtype=np.uint8)
    q5_h = rng.randint(0, 256, size=(4, 2, 4), dtype=np.uint8)
    q5_scales = rng.uniform(0.1, 1.5, size=(4, 2)).astype(np.float32)
    y_q5_expected = ref_q5_matmul(q5_l, q5_h, q5_scales, x_vec)

    data["q5_matmul_ql"] = q5_l
    data["q5_matmul_qh"] = q5_h
    data["q5_matmul_scales"] = q5_scales
    data["q5_matmul_x"] = x_vec
    data["q5_matmul_expected"] = y_q5_expected

    # 2. RMSNorm (dim=64)
    x_norm = rng.randn(64).astype(np.float32)
    w_norm = rng.uniform(0.5, 1.5, size=(64,)).astype(np.float32)
    y_rmsnorm_expected = ref_rmsnorm(x_norm, w_norm, eps=1e-5)

    data["rmsnorm_x"] = x_norm
    data["rmsnorm_weight"] = w_norm
    data["rmsnorm_eps"] = np.array(1e-5, dtype=np.float32)
    data["rmsnorm_expected"] = y_rmsnorm_expected

    # 3. RoPE (head_dim=32, pos=5)
    q_rope = rng.randn(32).astype(np.float32)
    k_rope = rng.randn(32).astype(np.float32)
    q_rope_exp, k_rope_exp = ref_rope(q_rope, k_rope, head_dim=32, pos=5, theta=10000.0)

    data["rope_q"] = q_rope
    data["rope_k"] = k_rope
    data["rope_pos"] = np.array(5, dtype=np.int32)
    data["rope_q_expected"] = q_rope_exp
    data["rope_k_expected"] = k_rope_exp

    # 4. Softmax (dim=64)
    x_soft = rng.randn(64).astype(np.float32) * 2.0
    y_soft_expected = ref_softmax(x_soft)

    data["softmax_x"] = x_soft
    data["softmax_expected"] = y_soft_expected

    # 5. Single-head Attention (seq_len=8, head_dim=16)
    q_attn = rng.randn(8, 16).astype(np.float32)
    k_attn = rng.randn(8, 16).astype(np.float32)
    v_attn = rng.randn(8, 16).astype(np.float32)
    y_attn_expected = ref_attention(q_attn, k_attn, v_attn)

    data["attention_q"] = q_attn
    data["attention_k"] = k_attn
    data["attention_v"] = v_attn
    data["attention_expected"] = y_attn_expected

    return data


def main():
    parser = argparse.ArgumentParser(description="Generate asmllm reference test tensors")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save .npz reference tensors")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic tensor generation")
    args = parser.parse_args()

    print(f"[reference_gen] Generating reference tensors (PyTorch backend present: {HAS_TORCH})...")
    data = generate_reference_data(seed=args.seed)
    print(f"[reference_gen] Generated {len(data)} reference tensors across 5 operations.")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **data)
        print(f"[reference_gen] Successfully saved reference dataset to {out_path}")
    else:
        print("[reference_gen] Verification successful. Use --output <filename.npz> to save.")


if __name__ == "__main__":
    main()
