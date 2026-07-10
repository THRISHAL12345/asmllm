/*
 * src/kernels/arm64/attention_neon.S
 *
 * Hand-written ARM64 NEON assembly kernel for Single-Head Scaled Dot-Product Attention.
 * Compliance: Zero C, Zero Intrinsics, AAPCS64 ABI.
 *
 * Signature:
 *   void asm_attention(
 *       const float* q,         // x0
 *       const float* k,         // x1
 *       const float* v,         // x2
 *       float* out,             // x3
 *       int64_t seq_len,        // x4
 *       int64_t head_dim,       // x5
 *       float scale             // s0
 *   );
 */

.text
.align 4
.global asm_attention
.global _asm_attention

asm_attention:
_asm_attention:
    // Save frame pointer and link register
    stp     x29, x30, [sp, -80]!
    mov     x29, sp

    // Save callee-saved registers x19..x28
    stp     x19, x20, [sp, 16]
    stp     x21, x22, [sp, 32]
    stp     x23, x24, [sp, 48]
    stp     x25, x26, [sp, 64]

    // Save original sp in x26 so we can restore stack after dynamic scratch buffer allocation
    mov     x26, sp

    // Copy parameters into callee-saved registers
    mov     x19, x0              // q
    mov     x20, x1              // k
    mov     x21, x2              // v
    mov     x22, x3              // out
    mov     x23, x4              // seq_len
    mov     x24, x5              // head_dim
    // Save scale s0 into callee-saved s8 (d8)
    fmov    s8, s0

    // Allocate scratch buffer on stack for scores (seq_len * 4 bytes, 16-byte aligned)
    lsl     x9, x23, #2
    add     x9, x9, #15
    and     x9, x9, ~15
    sub     sp, sp, x9
    mov     x25, sp              // x25 = scores buffer pointer

    // Outer loop over query rows: i = 0 .. seq_len - 1
    mov     x10, #0              // i = 0
.Lquery_loop:
    cmp     x10, x23
    b.ge    .Lquery_done

    // Save current i on stack or register
    // We use x27 for i
    mov     x27, x10

    // Compute q_row = q + i * head_dim
    mul     x11, x27, x24
    add     x11, x19, x11, lsl #2 // x11 = q_row pointer

    // Inner loop over key rows: j = 0 .. seq_len - 1
    mov     x12, #0              // j = 0
.Lkey_loop:
    cmp     x12, x23
    b.ge    .Lkey_done

    // k_row = k + j * head_dim
    mul     x13, x12, x24
    add     x13, x20, x13, lsl #2 // x13 = k_row pointer

    // Compute dot product q_row . k_row over head_dim
    movi    v0.4s, #0            // v0 = vector accumulator
    mov     x14, #0              // d = 0
    lsr     x15, x24, #2         // num 4-float blocks = head_dim / 4
.Ldot_vec:
    cmp     x14, x15
    b.ge    .Ldot_rem
    add     x16, x11, x14, lsl #4
    add     x17, x13, x14, lsl #4
    ld1     {v1.4s}, [x16]
    ld1     {v2.4s}, [x17]
    fmla    v0.4s, v1.4s, v2.4s
    add     x14, x14, #1
    b       .Ldot_vec

.Ldot_rem:
    // Pairwise reduce vector v0 -> scalar s0
    faddp   v0.4s, v0.4s, v0.4s
    faddp   v0.4s, v0.4s, v0.4s
    // Now process remainder elements d = x15*4 .. head_dim - 1
    lsl     x14, x15, #2
.Ldot_rem_loop:
    cmp     x14, x24
    b.ge    .Ldot_store
    add     x16, x11, x14, lsl #2
    add     x17, x13, x14, lsl #2
    ldr     s1, [x16]
    ldr     s2, [x17]
    fmadd   s0, s1, s2, s0
    add     x14, x14, #1
    b       .Ldot_rem_loop

.Ldot_store:
    // Multiply dot * scale (s8)
    fmul    s0, s0, s8
    // Store score in scores[j]
    add     x16, x25, x12, lsl #2
    str     s0, [x16]

    add     x12, x12, #1
    b       .Lkey_loop

.Lkey_done:
    // Run softmax in-place over scores[0..seq_len-1]
    // Call asm_softmax(x0=scores, x1=scores, x2=seq_len)
    mov     x0, x25
    mov     x1, x25
    mov     x2, x23
    bl      asm_softmax

    // Now compute weighted sum Y_i = sum_j probs[j] * V_j
    // First zero out_row = out + i * head_dim
    mul     x11, x27, x24
    add     x11, x22, x11, lsl #2 // x11 = out_row pointer
    movi    v0.4s, #0
    mov     x14, #0
    lsr     x15, x24, #2
.Lzero_out_vec:
    cmp     x14, x15
    b.ge    .Lzero_out_rem
    add     x16, x11, x14, lsl #4
    st1     {v0.4s}, [x16]
    add     x14, x14, #1
    b       .Lzero_out_vec
.Lzero_out_rem:
    lsl     x14, x15, #2
    fmov    s0, #0.0
.Lzero_out_rem_loop:
    cmp     x14, x24
    b.ge    .Lzero_out_done
    add     x16, x11, x14, lsl #2
    str     s0, [x16]
    add     x14, x14, #1
    b       .Lzero_out_rem_loop
.Lzero_out_done:

    // For j = 0 .. seq_len - 1: accumulate probs[j] * V_j into out_row
    mov     x12, #0              // j = 0
.Lval_loop:
    cmp     x12, x23
    b.ge    .Lval_done

    add     x16, x25, x12, lsl #2
    ldr     s1, [x16]            // s1 = prob_j
    dup     v1.4s, v1.s[0]

    mul     x13, x12, x24
    add     x13, x21, x13, lsl #2 // x13 = V_j row pointer

    mov     x14, #0
    lsr     x15, x24, #2
.Lval_vec:
    cmp     x14, x15
    b.ge    .Lval_rem
    add     x16, x11, x14, lsl #4 // out_row + d
    add     x17, x13, x14, lsl #4 // V_j + d
    ld1     {v2.4s}, [x16]
    ld1     {v3.4s}, [x17]
    fmla    v2.4s, v3.4s, v1.4s
    st1     {v2.4s}, [x16]
    add     x14, x14, #1
    b       .Lval_vec

.Lval_rem:
    lsl     x14, x15, #2
.Lval_rem_loop:
    cmp     x14, x24
    b.ge    .Lval_next
    add     x16, x11, x14, lsl #2
    add     x17, x13, x14, lsl #2
    ldr     s2, [x16]
    ldr     s3, [x17]
    fmadd   s2, s3, s1, s2
    str     s2, [x16]
    add     x14, x14, #1
    b       .Lval_rem_loop

.Lval_next:
    add     x12, x12, #1
    b       .Lval_loop

.Lval_done:
    add     x10, x27, #1         // i++
    b       .Lquery_loop

.Lquery_done:
    // Restore original stack pointer
    mov     sp, x26

    // Restore callee-saved registers
    ldp     x25, x26, [sp, 64]
    ldp     x23, x24, [sp, 48]
    ldp     x21, x22, [sp, 32]
    ldp     x19, x20, [sp, 16]
    ldp     x29, x30, [sp], 80
    ret
