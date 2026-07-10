/*
 * src/kernels/arm64/matmul_q4_neon.S
 *
 * Hand-written ARM64 NEON assembly kernel for Q4_0 Matrix-Vector Multiply.
 * Compliance: Zero C, Zero Intrinsics, AAPCS64 ABI.
 *
 * Signature:
 *   void asm_matmul_q4(
 *       const uint8_t* qweights, // x0: Q4_0 weight blocks (M x (K/32) x 16 bytes)
 *       const float*   scales,   // x1: FP32 scales (M x (K/32) x 4 bytes)
 *       const float*   x,        // x2: FP32 input vector (K elements)
 *       float*         y,        // x3: FP32 output vector (M elements)
 *       int64_t        M,        // x4: number of output rows
 *       int64_t        K         // x5: number of input columns
 *   );
 */

.text
.align 4
.global asm_matmul_q4
.global _asm_matmul_q4

asm_matmul_q4:
_asm_matmul_q4:
    stp     x29, x30, [sp, -48]!
    mov     x29, sp
    stp     x19, x20, [sp, 16]
    stp     x21, x22, [sp, 32]

    lsr     x19, x5, #5          // x19 = K / 32

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

    ldr     q2, [x0]
    add     x0, x0, #16

    movi    v3.16b, #0x0f
    and     v4.16b, v2.16b, v3.16b
    movi    v5.16b, #8
    sub     v4.16b, v4.16b, v5.16b

    sxtl    v6.8h, v4.8b
    sxtl2   v7.8h, v4.16b

    sxtl    v16.4s, v6.4h
    sxtl2   v17.4s, v6.8h
    scvtf   v16.4s, v16.4s
    scvtf   v17.4s, v17.4s
    fmul    v16.4s, v16.4s, v1.4s
    fmul    v17.4s, v17.4s, v1.4s

    ldp     q24, q25, [x2]
    fmla    v0.4s, v16.4s, v24.4s
    fmla    v0.4s, v17.4s, v25.4s

    sxtl    v18.4s, v7.4h
    sxtl2   v19.4s, v7.8h
    scvtf   v18.4s, v18.4s
    scvtf   v19.4s, v19.4s
    fmul    v18.4s, v18.4s, v1.4s
    fmul    v19.4s, v19.4s, v1.4s

    ldp     q26, q27, [x2, #32]
    fmla    v0.4s, v18.4s, v26.4s
    fmla    v0.4s, v19.4s, v27.4s

    ushr    v4.16b, v2.16b, #4
    sub     v4.16b, v4.16b, v5.16b

    sxtl    v6.8h, v4.8b
    sxtl2   v7.8h, v4.16b

    sxtl    v16.4s, v6.4h
    sxtl2   v17.4s, v6.8h
    scvtf   v16.4s, v16.4s
    scvtf   v17.4s, v17.4s
    fmul    v16.4s, v16.4s, v1.4s
    fmul    v17.4s, v17.4s, v1.4s

    ldp     q24, q25, [x2, #64]
    fmla    v0.4s, v16.4s, v24.4s
    fmla    v0.4s, v17.4s, v25.4s

    sxtl    v18.4s, v7.4h
    sxtl2   v19.4s, v7.8h
    scvtf   v18.4s, v18.4s
    scvtf   v19.4s, v19.4s
    fmul    v18.4s, v18.4s, v1.4s
    fmul    v19.4s, v19.4s, v1.4s

    ldp     q26, q27, [x2, #96]
    fmla    v0.4s, v18.4s, v26.4s
    fmla    v0.4s, v19.4s, v27.4s

    add     x2, x2, #128
    add     x21, x21, #1
    b       .Lblock_loop

.Lblock_done:
    faddp   v0.4s, v0.4s, v0.4s
    faddp   v0.4s, v0.4s, v0.4s
    str     s0, [x3, x20, lsl #2]

    sub     x2, x2, x5, lsl #2
    add     x20, x20, #1
    b       .Lrow_loop

.Ldone:
    ldp     x21, x22, [sp, 32]
    ldp     x19, x20, [sp, 16]
    ldp     x29, x30, [sp], 48
    ret
