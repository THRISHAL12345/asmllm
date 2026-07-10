/*
 * src/kernels/arm64/matmul_q5_neon.S
 *
 * Hand-written ARM64 NEON assembly kernel for Q5_0 Matrix-Vector Multiply.
 * Compliance: Zero C, Zero Intrinsics, AAPCS64 ABI.
 *
 * Signature:
 *   void asm_matmul_q5(
 *       const uint8_t* ql,     // x0: Q5_0 low nibbles (16 bytes per block of 32 weights)
 *       const uint8_t* qh,     // x1: Q5_0 high bit flags (4 bytes per block of 32 weights)
 *       const float*   scales, // x2: FP32 scales
 *       const float*   x,      // x3: FP32 input vector
 *       float*         y,      // x4: FP32 output vector
 *       int64_t        M,      // x5: rows
 *       int64_t        K       // x6: cols
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

    lsr     x19, x6, #5          // num_blocks = K / 32

    mov     x20, #0
.Lrow_loop:
    cmp     x20, x5              // cmp row index to M
    b.ge    .Ldone

    fmov    v0.4s, #1.0
    fsub    v0.4s, v0.4s, v0.4s

    mov     x21, #0
.Lblock_loop:
    cmp     x21, x19
    b.ge    .Lblock_done

    // Load 4 bytes of high bit flags from qh (x1)
    ldr     w10, [x1]
    add     x1, x1, #4

    // Load FP32 scale from scales (x2)
    ldr     s1, [x2]
    add     x2, x2, #4
    dup     v1.4s, v1.s[0]

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
    ldr     s3, [x3, x22, lsl #2] // input vector element from x (x3)
    fmul    s2, s2, s1
    fmadd   s0, s2, s3, s0

    add     x22, x22, #1
    b       .Lq5_inner

.Lq5_inner_done:
    add     x0, x0, #16
    add     x3, x3, #128         // advance input vector x by 32 elements (128 bytes)
    add     x21, x21, #1
    b       .Lblock_loop

.Lblock_done:
    str     s0, [x4, x20, lsl #2] // store to output y (x4)
    sub     x3, x3, x6, lsl #2   // rewind input vector pointer x3 by K elements
    add     x20, x20, #1
    b       .Lrow_loop

.Ldone:
    ldp     x21, x22, [sp, 32]
    ldp     x19, x20, [sp, 16]
    ldp     x29, x30, [sp], 48
    ret
