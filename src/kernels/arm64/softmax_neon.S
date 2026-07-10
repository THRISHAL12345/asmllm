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

    // Range-reduced exp(z): n = round(z / ln2), r = z - n * ln2
    ldr     s10, .Linv_ln2
    fmul    s10, s2, s10         // s10 = z / ln2
    fcvtas  w10, s10             // w10 = n = round(z / ln2)
    scvtf   s11, w10             // s11 = float(n)
    ldr     s12, .Lln2
    fmsub   s13, s11, s12, s2    // s13 = r = z - n * ln2

    // Evaluate Horner polynomial for exp(r) on [-0.3465, 0.3465]
    // P(r) = 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r*(1/120)))))
    ldr     s4, .Lc5             // 1/120
    ldr     s12, .Lc4            // 1/24
    fmadd   s4, s4, s13, s12
    ldr     s12, .Lc3            // 1/6
    fmadd   s4, s4, s13, s12
    ldr     s12, .Lc2            // 1/2
    fmadd   s4, s4, s13, s12
    fmov    s12, #1.0
    fmadd   s4, s4, s13, s12
    fmadd   s4, s4, s13, s12     // s4 = P(r)

    // Compute 2^n via IEEE-754 exponent bit manipulation
    mov     w11, #-126
    cmp     w10, w11
    b.lt    .Lexp_underflow
    add     w10, w10, #127
    lsl     w10, w10, #23
    fmov    s14, w10
    fmul    s4, s4, s14          // s4 = exp(z)
    b       .Lexp_store
.Lexp_underflow:
    fmov    s4, #0.0
.Lexp_store:

    add     x10, x1, x19, lsl #2
    str     s4, [x10]            // y[i] = exp_z
    fadd    s1, s1, s4           // sum += exp_z

    add     x19, x19, #1
    b       .Lexp_loop

.Lexp_done:
    // Pass 3: Normalize y[i] /= sum
    fmov    s5, #1.0
    fdiv    s5, s5, s1           // inv_sum = 1 / sum
    dup     v1.4s, v5.s[0]

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

.section .rodata
.align 4
.Linv_ln2:  .float 1.4426950408889634
.Lln2:      .float 0.6931471805599453
.Lc5:       .float 0.008333333333333333
.Lc4:       .float 0.041666666666666664
.Lc3:       .float 0.16666666666666666
.Lc2:       .float 0.5
