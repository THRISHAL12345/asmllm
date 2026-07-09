"""
Reference implementations of target LLM operations for numerical verification.
Supports PyTorch (when available) and NumPy reference implementations.
"""

import math
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def dequantize_q4_0_numpy(qblocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Dequantize Q4_0 blocks to FP32 array.
    Each block is 32 weights packed into 16 uint8 bytes (low 4 bits, high 4 bits).
    weight = (nibble - 8) * scale
    """
    rows, num_blocks, block_bytes = qblocks.shape
    assert block_bytes == 16, "Q4_0 block data must be 16 bytes (32 nibbles)"
    low = (qblocks & 0x0F).astype(np.float32) - 8.0
    high = ((qblocks >> 4) & 0x0F).astype(np.float32) - 8.0
    # Interleave or concatenate nibbles: standard Q4_0 stores low nibble first, then high nibble
    # block weights [0..15] = low, [16..31] = high
    unscaled = np.concatenate([low, high], axis=-1)  # shape (rows, num_blocks, 32)
    dequant = unscaled * scales[:, :, None]
    return dequant.reshape(rows, num_blocks * 32)


def ref_q4_matmul(qweights: np.ndarray, scales: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Reference Q4_0 matrix-vector / matrix-matrix multiplication: y = W @ x
    qweights: shape (M, num_blocks, 16) uint8
    scales: shape (M, num_blocks) float32
    x: shape (K,) or (K, N) float32, where K = num_blocks * 32
    """
    w_fp32 = dequantize_q4_0_numpy(qweights, scales)
    return w_fp32 @ x


def dequantize_q8_0_numpy(qweights: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Dequantize Q8_0 blocks to FP32 array.
    Each block is 32 signed int8 weights.
    weight = q * scale
    """
    rows, num_blocks, block_bytes = qweights.shape
    assert block_bytes == 32, "Q8_0 block data must be 32 bytes (32 int8 weights)"
    w_fp32 = qweights.astype(np.float32) * scales[:, :, None]
    return w_fp32.reshape(rows, num_blocks * 32)


def ref_q8_matmul(qweights: np.ndarray, scales: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Reference Q8_0 matrix-vector multiplication: y = W @ x
    qweights: shape (M, num_blocks, 32) int8
    scales: shape (M, num_blocks) float32
    x: shape (K,) float32, where K = num_blocks * 32
    """
    w_fp32 = dequantize_q8_0_numpy(qweights, scales)
    return w_fp32 @ x


def dequantize_q5_0_numpy(ql: np.ndarray, qh: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Dequantize Q5_0 blocks to FP32 array.
    ql: shape (M, num_blocks, 16) uint8 containing low 4 bits of 32 weights.
    qh: shape (M, num_blocks, 4) uint8 containing 5th bit (bit 4) of 32 weights packed (8 bits per byte).
    weight[i] = ((low_nibble[i] | (bit4[i] << 4)) - 16.0) * scale
    """
    rows, num_blocks, _ = ql.shape
    low = (ql & 0x0F).astype(np.int32)
    high_low = ((ql >> 4) & 0x0F).astype(np.int32)
    low_nibbles = np.concatenate([low, high_low], axis=-1)  # shape (rows, num_blocks, 32)

    # Unpack 32 bits from qh (4 bytes per block)
    # bit i of weight j is bit (j % 8) of byte (j // 8)
    # Or in our layout: weight j (0..31) bit 4 is bit (j % 8) of byte (j // 8)
    qh_bits = np.zeros((rows, num_blocks, 32), dtype=np.int32)
    for b_idx in range(4):
        byte_val = qh[:, :, b_idx]
        for bit_idx in range(8):
            w_idx = b_idx * 8 + bit_idx
            qh_bits[:, :, w_idx] = ((byte_val >> bit_idx) & 1) << 4

    unscaled = (low_nibbles | qh_bits).astype(np.float32) - 16.0
    dequant = unscaled * scales[:, :, None]
    return dequant.reshape(rows, num_blocks * 32)


def ref_q5_matmul(ql: np.ndarray, qh: np.ndarray, scales: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Reference Q5_0 matrix-vector multiplication: y = W @ x
    """
    w_fp32 = dequantize_q5_0_numpy(ql, qh, scales)
    return w_fp32 @ x



def ref_rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Reference Root Mean Square Layer Normalization:
    output = (x / sqrt(mean(x**2) + eps)) * weight
    """
    if HAS_TORCH and isinstance(x, torch.Tensor):
        x_fp32 = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + eps)
        normed = (x_fp32 / rms) * weight
        return normed.to(x.dtype)

    x_fp32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_fp32 ** 2, axis=-1, keepdims=True) + eps)
    return ((x_fp32 / rms) * weight).astype(x.dtype)


def ref_rope(q: np.ndarray, k: np.ndarray, head_dim: int, pos: int, theta: float = 10000.0):
    """
    Reference Rotary Position Embedding applied to query and key vectors.
    q, k: 1D or 2D arrays where last dim is head_dim
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = pos * freqs
    cos_theta = np.cos(angles)
    sin_theta = np.sin(angles)

    def apply_rotary(x_arr):
        x_fp32 = x_arr.astype(np.float32)
        x0 = x_fp32[..., 0::2]
        x1 = x_fp32[..., 1::2]
        out0 = x0 * cos_theta - x1 * sin_theta
        out1 = x0 * sin_theta + x1 * cos_theta
        out = np.empty_like(x_fp32)
        out[..., 0::2] = out0
        out[..., 1::2] = out1
        return out.astype(x_arr.dtype)

    return apply_rotary(q), apply_rotary(k)


def ref_softmax(x: np.ndarray) -> np.ndarray:
    """
    Reference numerically stable softmax along the last dimension.
    """
    if HAS_TORCH and isinstance(x, torch.Tensor):
        return torch.softmax(x, dim=-1)

    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def ref_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float = None) -> np.ndarray:
    """
    Reference single-head scaled dot-product attention:
    attn = softmax(Q @ K.T * scale) @ V
    q: (seq_len_q, head_dim)
    k: (seq_len_k, head_dim)
    v: (seq_len_k, head_dim)
    """
    seq_len_q, head_dim = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    scores = (q @ k.T) * scale
    attn_weights = ref_softmax(scores)
    return attn_weights @ v
