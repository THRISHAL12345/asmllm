/*
 * src/kernels/arm64/matmul_q8_neon.S
 *
 * Hand-written ARM64 NEON assembly kernel for Q8_0 Matrix-Vector Multiply.
 * Compliance: Zero C, Zero Intrinsics, AAPCS64 ABI.
 *
 * Signature:
 *   void asm_matmul_q8(
 *       const int8_t*  qweights, // x0: Q8_0 weight blocks (34 bytes per 32 weights)
 *       const float*   scales,   // x1: FP32 scales
 *       const float*   x,        // x2: FP32 input vector
 *       float*         y,        // x3: FP32 output vector
 *       int64_t        M,        // x4: rows
 *       int64_t        K         // x5: cols
 *   );
 */

.text
.align 4
.global asm_matmul_q8
.global _asm_matmul_q8

asm_matmul_q8:
_asm_matmul_q8:
    stp     x29, x30, [sp, -48]!
    mov     x29, sp
    stp     x19, x20, [sp, 16]
    stp     x21, x22, [sp, 32]

    lsr     x19, x5, #5          // num_blocks = K / 32

    mov     x20, #0
.Lrow_loop:
    cmp     x20, x4
    b.ge    .Ldone

    fmov    v0.4s, #1.0
    fsub    v0.4s, v0.4s, v0.4s  // v0 accumulator = 0

    mov     x21, #0
.Lblock_loop:
    cmp     x21, x19
    b.ge    .Lblock_done

    ldr     s1, [x1]
    add     x1, x1, #4
    dup     v1.4s, v1.s[0]

    // Process 32 int8 weights
    mov     x22, #0
.Lq8_inner:
    cmp     x22, #32
    b.ge    .Lq8_inner_done
    ldrsb   w9, [x0, x22]
    scvtf   s2, w9
    ldr     s3, [x2, x22, lsl #2]
    fmul    s2, s2, s1
    fmadd   s0, s2, s3, s0
    add     x22, x22, #1
    b       .Lq8_inner

.Lq8_inner_done:
    add     x0, x0, #32
    add     x2, x2, #128
    add     x21, x21, #1
    b       .Lblock_loop

.Lblock_done:
    str     s0, [x3, x20, lsl #2]
    sub     x2, x2, x5, lsl #2
    add     x20, x20, #1
    b       .Lrow_loop

.Ldone:
    ldp     x21, x22, [sp, 32]
    ldp     x19, x20, [sp, 16]
    ldp     x29, x30, [sp], 48
    ret
