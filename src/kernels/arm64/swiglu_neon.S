/*
 * src/kernels/arm64/swiglu_neon.S
 *
 * Hand-written ARM64 assembly kernel for SiLU Hadamard product.
 * Compliance: Zero C, Zero Intrinsics, AAPCS64 ABI.
 *
 * Signature:
 *   void asm_silu_hadamard(
 *       float*       x,   // x0: in/out vector pointer
 *       const float* y,   // x1: gate vector pointer
 *       int64_t      dim  // x2: vector dimension
 *   );
 */

.text
.align 4
.global asm_silu_hadamard
.global _asm_silu_hadamard

asm_silu_hadamard:
_asm_silu_hadamard:
    stp     x29, x30, [sp, -32]!
    mov     x29, sp
    stp     x19, x20, [sp, 16]

    mov     x19, #0
.Lsilu_loop:
    cmp     x19, x2
    b.ge    .Ldone

    add     x9, x0, x19, lsl #2
    add     x10, x1, x19, lsl #2
    ldr     s0, [x9]             // s0 = x_i
    ldr     s1, [x10]            // s1 = y_i

    // Compute sigmoid(x_i) = 1 / (1 + exp(-x_i))
    fneg    s2, s0               // s2 = -x_i

    // Compute exp(s2) via Taylor polynomial
    fmov    s3, #1.0
    fmov    s4, #1.0
    mov     x11, #1
.Lexp_loop:
    cmp     x11, #10
    b.gt    .Lexp_done
    fmul    s3, s3, s2
    ucvtf   s5, x11
    fdiv    s3, s3, s5
    fadd    s4, s4, s3
    add     x11, x11, #1
    b       .Lexp_loop
.Lexp_done:
    fmov    s5, #0.0
    fmax    s4, s4, s5           // exp(-x_i) >= 0

    fmov    s6, #1.0
    fadd    s4, s4, s6           // 1 + exp(-x_i)
    fdiv    s4, s6, s4           // sigmoid(x_i)

    fmul    s0, s0, s4           // silu(x_i) = x_i * sigmoid(x_i)
    fmul    s0, s0, s1           // out = silu(x_i) * y_i
    str     s0, [x9]

    add     x19, x19, #1
    b       .Lsilu_loop

.Ldone:
    ldp     x19, x20, [sp, 16]
    ldp     x29, x30, [sp], 32
    ret
