/*
 * src/kernels/arm64/rmsnorm_neon.S
 *
 * Hand-written ARM64 NEON assembly kernel for Root Mean Square Layer Normalization.
 * Compliance: Zero C, Zero Intrinsics, AAPCS64 ABI.
 *
 * Signature:
 *   void asm_rmsnorm(
 *       const float* x,   // x0: input vector pointer
 *       const float* w,   // x1: weight vector pointer
 *       float*       y,   // x2: output vector pointer
 *       int64_t      dim, // x3: vector dimension
 *       float        eps  // s0: epsilon for stability
 *   );
 */

.text
.align 4
.global asm_rmsnorm
.global _asm_rmsnorm

asm_rmsnorm:
_asm_rmsnorm:
    stp     x29, x30, [sp, -32]!
    mov     x29, sp
    stp     x19, x20, [sp, 16]

    // 1. Compute sum of squares: sum(x[i] * x[i]) across dim
    fmov    v1.4s, #0.0          // v1 = vector accumulator
    mov     x19, #0              // i = 0

    // Number of 4-float blocks
    lsr     x20, x3, #2

.Lsum_loop:
    cmp     x19, x20
    b.ge    .Lsum_done
    add     x9, x0, x19, lsl #4
    ld1     {v2.4s}, [x9]
    fmla    v1.4s, v2.4s, v2.4s
    add     x19, x19, #1
    b       .Lsum_loop

.Lsum_done:
    // Pairwise add vector v1 -> scalar s1
    faddp   v1.4s, v1.4s, v1.4s
    faddp   v1.4s, v1.4s, v1.4s
    // s1 now holds sum(x_i^2)

    // Convert dim (x3) to float in s2
    ucvtf   s2, x3
    fdiv    s1, s1, s2           // s1 = mean_sq = sum / dim
    fadd    s1, s1, s0           // s1 = mean_sq + eps
    fsqrt   s1, s1               // s1 = sqrt(mean_sq + eps)
    fmov    s2, #1.0
    fdiv    s1, s2, s1           // s1 = rsqrt_val = 1.0 / sqrt(mean_sq + eps)
    dup     v1.4s, v1.s[0]       // v1 = broadcast rsqrt_val across 4 lanes

    // 2. Compute y[i] = x[i] * rsqrt_val * w[i]
    mov     x19, #0
.Lnorm_loop:
    cmp     x19, x20
    b.ge    .Lnorm_done
    add     x9, x0, x19, lsl #4
    add     x10, x1, x19, lsl #4
    add     x11, x2, x19, lsl #4

    ld1     {v2.4s}, [x9]
    ld1     {v3.4s}, [x10]
    fmul    v2.4s, v2.4s, v1.4s
    fmul    v2.4s, v2.4s, v3.4s
    st1     {v2.4s}, [x11]

    add     x19, x19, #1
    b       .Lnorm_loop

.Lnorm_done:
    ldp     x19, x20, [sp, 16]
    ldp     x29, x30, [sp], 32
    ret
