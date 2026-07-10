/*
 * src/kernels/arm64/softmax_neon.S
 *
 * Hand-written ARM64 assembly kernel for Numerically Stable Softmax.
 * Compliance: Zero C, Zero Intrinsics, AAPCS64 ABI.
 *
 * Signature:
 *   void asm_softmax(
 *       const float* x,   // x0: input vector pointer
 *       float*       y,   // x1: output vector pointer
 *       int64_t      dim  // x2: vector dimension
 *   );
 */

.text
.align 4
.global asm_softmax
.global _asm_softmax

asm_softmax:
_asm_softmax:
    stp     x29, x30, [sp, -32]!
    mov     x29, sp
    stp     x19, x20, [sp, 16]

    // Pass 1: Find max value in vector x
    ldr     s0, [x0]             // max_val = x[0]
    mov     x19, #1
.Lmax_loop:
    cmp     x19, x2
    b.ge    .Lmax_done
    add     x9, x0, x19, lsl #2
    ldr     s1, [x9]
    fmax    s0, s0, s1
    add     x19, x19, #1
    b       .Lmax_loop

.Lmax_done:
    // s0 = max_val
    // Pass 2: Compute exp(x_i - max_val) and sum
    fmov    s1, #0.0             // sum = 0.0
    mov     x19, #0

.Lexp_loop:
    cmp     x19, x2
    b.ge    .Lexp_done

    add     x9, x0, x19, lsl #2
    ldr     s2, [x9]
    fsub    s2, s2, s0           // z = x_i - max_val

    // Compute exp(z) using Taylor polynomial (1 + z + z^2/2! + ... + z^8/8!)
    fmov    s3, #1.0             // term = 1
    fmov    s4, #1.0             // exp_z = 1
    mov     x9, #1
.Lpoly_loop:
    cmp     x9, #10
    b.gt    .Lpoly_done
    fmul    s3, s3, s2
    ucvtf   s5, x9
    fdiv    s3, s3, s5
    fadd    s4, s4, s3
    add     x9, x9, #1
    b       .Lpoly_loop
.Lpoly_done:
    // Ensure non-negative
    fmov    s5, #0.0
    fmax    s4, s4, s5

    add     x10, x1, x19, lsl #2
    str     s4, [x10]            // y[i] = exp_z
    fadd    s1, s1, s4           // sum += exp_z

    add     x19, x19, #1
    b       .Lexp_loop

.Lexp_done:
    // Pass 3: Normalize y[i] /= sum
    fmov    s5, #1.0
    fdiv    s5, s5, s1           // inv_sum = 1 / sum
    dup     v1.4s, v5

    lsr     x20, x2, #2
    mov     x19, #0
.Lnorm_loop:
    cmp     x19, x20
    b.ge    .Lnorm_rem
    add     x10, x1, x19, lsl #4
    ld1     {v2.4s}, [x10]
    fmul    v2.4s, v2.4s, v1.4s
    st1     {v2.4s}, [x10]
    add     x19, x19, #1
    b       .Lnorm_loop

.Lnorm_rem:
    lsl     x19, x20, #2
.Lrem_loop:
    cmp     x19, x2
    b.ge    .Ldone
    add     x10, x1, x19, lsl #2
    ldr     s2, [x10]
    fmul    s2, s2, s5
    str     s2, [x10]
    add     x19, x19, #1
    b       .Lrem_loop

.Ldone:
    ldp     x19, x20, [sp, 16]
    ldp     x29, x30, [sp], 32
    ret
