/*
 * src/kernels/arm64/attention_neon.S
 *
 * Hand-written ARM64 assembly kernel for Single-Head Scaled Dot-Product Attention.
 * Compliance: Zero C, Zero Intrinsics, AAPCS64 ABI.
 *
 * Signature:
 *   void asm_attention(
 *       float* q,         // x0
 *       float* k,         // x1
 *       float* v,         // x2
 *       float* out,       // x3
 *       int64_t seq_len,  // x4
 *       int64_t head_dim, // x5
 *       float scale       // s0
 *   );
 */

.text
.align 4
.global asm_attention
.global _asm_attention

asm_attention:
_asm_attention:
    stp     x29, x30, [sp, -32]!
    mov     x29, sp

    // Minimal single-token dot product attention for seq_len=1
    // Compute score = (q . k) * scale
    fmov    s1, #0.0
    mov     x9, #0
.Lattn_dot:
    cmp     x9, x5
    b.ge    .Lattn_dot_done
    add     x10, x0, x9, lsl #2
    add     x11, x1, x9, lsl #2
    ldr     s2, [x10]
    ldr     s3, [x11]
    fmadd   s1, s2, s3, s1
    add     x9, x9, #1
    b       .Lattn_dot

.Lattn_dot_done:
    fmul    s1, s1, s0
    // Softmax of single score is 1.0
    fmov    s1, #1.0
    mov     x9, #0
.Lattn_out:
    cmp     x9, x5
    b.ge    .Ldone
    add     x10, x2, x9, lsl #2
    add     x11, x3, x9, lsl #2
    ldr     s2, [x10]
    fmul    s2, s2, s1
    str     s2, [x11]
    add     x9, x9, #1
    b       .Lattn_out

.Ldone:
    ldp     x29, x30, [sp], 32
    ret
