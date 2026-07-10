/*
 * src/kernels/arm64/matmul_q5_neon.S
 *
 * Hand-written ARM64 NEON assembly kernel for Q5_0 Matrix-Vector Multiply.
 * Compliance: Zero C, Zero Intrinsics, AAPCS64 ABI.
 *
 * Signature:
 *   void asm_matmul_q5(
 *       const uint8_t* qweights, // x0: Q5_0 weight blocks (22 bytes per 32 weights)
 *       const float*   scales,   // x1: FP32 scales
 *       const float*   x,        // x2: FP32 input vector
 *       float*         y,        // x3: FP32 output vector
 *       int64_t        M,        // x4: rows
 *       int64_t        K         // x5: cols
 *   );
 */

.text
.align 4
.global asm_matmul_q5
.global _asm_matmul_q5

asm_matmul_q5:
_asm_matmul_q5:
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
    fsub    v0.4s, v0.4s, v0.4s

    mov     x21, #0
.Lblock_loop:
    cmp     x21, x19
    b.ge    .Lblock_done

    ldr     s1, [x1]
    add     x1, x1, #4
    dup     v1.4s, v1.s[0]

    add     x0, x0, #2           // Skip 2-byte fp16 scale
    ldr     w10, [x0]            // 4 bytes of high bit flags qh
    add     x0, x0, #4           // Now at 16 bytes of low nibbles qs

    mov     x22, #0
.Lq5_inner:
    cmp     x22, #32
    b.ge    .Lq5_inner_done

    // extract low 4 bits from qs[x22/2]
    lsr     x11, x22, #1
    ldrb    w12, [x0, x11]
    tst     x22, #1
    csel    w13, w12, w12, eq
    b.eq    .Leven_nibble
    lsr     w13, w12, #4
.Leven_nibble:
    and     w13, w13, #0x0f

    // extract 5th bit from qh
    lsr     w14, w10, w22
    and     w14, w14, #1
    lsl     w14, w14, #4
    orr     w13, w13, w14

    sub     w13, w13, #16        // bias 16 for Q5_0
    scvtf   s2, w13
    ldr     s3, [x2, x22, lsl #2]
    fmul    s2, s2, s1
    fmla    s0, s2, s3

    add     x22, x22, #1
    b       .Lq5_inner

.Lq5_inner_done:
    add     x0, x0, #16
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
