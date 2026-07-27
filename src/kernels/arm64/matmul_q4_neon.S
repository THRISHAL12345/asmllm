/*
 * src/kernels/arm64/matmul_q4_neon.S
 *
 * Hand-written ARM64 NEON assembly kernel for Q4_0 Matrix-Vector Multiply.
 * Compliance: Zero C, Zero Intrinsics, AAPCS64 ABI.
 * Architecture: ARMv8.2-A (SDOT) with 4x1 micro-kernel tiling and dynamic Q8_0 activation quantization.
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
.arch armv8.2-a+dotprod
.global asm_matmul_q4
.global _asm_matmul_q4

asm_matmul_q4:
_asm_matmul_q4:
    stp     x29, x30, [sp, -96]!
    mov     x29, sp
    stp     x19, x20, [sp, 16]
    stp     x21, x22, [sp, 32]
    stp     x23, x24, [sp, 48]
    stp     x25, x26, [sp, 64]
    stp     x27, x28, [sp, 80]

    // x19 = K / 32 (number of blocks per row)
    lsr     x19, x5, #5

    // --- Dynamic Quantization of x (FP32 -> Q8_0) ---
    // We allocate stack space to store the Q8_0 version of x.
    // We need K bytes for int8 values, and (K/32)*4 bytes for scales.
    add     x20, x5, #15
    and     x20, x20, #~15      // x20 = K aligned to 16 bytes
    lsl     x21, x19, #2
    add     x21, x21, #15
    and     x21, x21, #~15      // x21 = K/32 * 4 aligned to 16 bytes
    add     x22, x20, x21       // x22 = total allocation size

    sub     sp, sp, x22         // allocate dynamic stack
    mov     x23, sp             // x23 = ptr to x_qs (int8s)
    add     x24, x23, x20       // x24 = ptr to x_scales (floats)

    mov     x8, #0              // block index
    mov     x9, x2              // ptr to x (floats)
    mov     x10, x23            // ptr to x_qs
    mov     x11, x24            // ptr to x_scales

.Lquant_loop:
    cmp     x8, x19
    b.ge    .Lquant_done

    // Load 32 floats
    ldp     q0, q1, [x9]
    ldp     q2, q3, [x9, #32]
    ldp     q4, q5, [x9, #64]
    ldp     q6, q7, [x9, #96]
    add     x9, x9, #128

    // Find absolute max
    fabs    v16.4s, v0.4s
    fabs    v17.4s, v1.4s
    fmax    v16.4s, v16.4s, v17.4s
    fabs    v17.4s, v2.4s
    fmax    v16.4s, v16.4s, v17.4s
    fabs    v17.4s, v3.4s
    fmax    v16.4s, v16.4s, v17.4s
    fabs    v17.4s, v4.4s
    fmax    v16.4s, v16.4s, v17.4s
    fabs    v17.4s, v5.4s
    fmax    v16.4s, v16.4s, v17.4s
    fabs    v17.4s, v6.4s
    fmax    v16.4s, v16.4s, v17.4s
    fabs    v17.4s, v7.4s
    fmax    v16.4s, v16.4s, v17.4s
    
    // Max of 4 elements in v16
    fmaxv   s16, v16.4s         // s16 = max_abs_val

    // Compute scale = max_abs_val / 127.0
    mov     w17, #127
    scvtf   s17, w17
    fdiv    s18, s16, s17       // s18 = scale
    str     s18, [x11], #4      // store scale

    // Compute inv_scale = 127.0 / max_abs_val
    fcmp    s16, #0.0
    beq     .Lquant_zero
    fdiv    s17, s17, s16       // s17 = inv_scale
    dup     v17.4s, v17.s[0]
    b       .Lquant_mul
.Lquant_zero:
    movi    v17.4s, #0
.Lquant_mul:
    // Multiply and convert to integer
    fmul    v0.4s, v0.4s, v17.4s
    fcvtns  v0.4s, v0.4s
    fmul    v1.4s, v1.4s, v17.4s
    fcvtns  v1.4s, v1.4s
    fmul    v2.4s, v2.4s, v17.4s
    fcvtns  v2.4s, v2.4s
    fmul    v3.4s, v3.4s, v17.4s
    fcvtns  v3.4s, v3.4s
    fmul    v4.4s, v4.4s, v17.4s
    fcvtns  v4.4s, v4.4s
    fmul    v5.4s, v5.4s, v17.4s
    fcvtns  v5.4s, v5.4s
    fmul    v6.4s, v6.4s, v17.4s
    fcvtns  v6.4s, v6.4s
    fmul    v7.4s, v7.4s, v17.4s
    fcvtns  v7.4s, v7.4s

    // Narrow 32-bit to 16-bit
    sqxtn   v0.4h, v0.4s
    sqxtn2  v0.8h, v1.4s
    sqxtn   v1.4h, v2.4s
    sqxtn2  v1.8h, v3.4s
    sqxtn   v2.4h, v4.4s
    sqxtn2  v2.8h, v5.4s
    sqxtn   v3.4h, v6.4s
    sqxtn2  v3.8h, v7.4s

    // Narrow 16-bit to 8-bit
    sqxtn   v0.8b, v0.8h
    sqxtn2  v0.16b, v1.8h
    sqxtn   v1.8b, v2.8h
    sqxtn2  v1.16b, v3.8h

    // Store 32 bytes of int8s
    stp     q0, q1, [x10]
    add     x10, x10, #32

    add     x8, x8, #1
    b       .Lquant_loop
.Lquant_done:

    // --- 4x1 Micro-Kernel Tiling ---
    mov     x8, #0              // row index (i)
    
    lsr     x10, x5, #1         // x10 = K/2 bytes per row (qweights)
    lsr     x11, x5, #5
    lsl     x11, x11, #2        // x11 = K/32*4 bytes per row (scales)

.Lrow_loop_4x:
    mov     x9, x4
    sub     x9, x9, x8
    cmp     x9, #4
    blt     .Lrow_loop_1x

    // Initialize 4 accumulators (FP32)
    movi    v20.4s, #0
    movi    v21.4s, #0
    movi    v22.4s, #0
    movi    v23.4s, #0

    // Pointers for 4 rows
    madd    x12, x8, x10, x0    // w0
    add     x13, x12, x10       // w1
    add     x14, x13, x10       // w2
    add     x15, x14, x10       // w3

    madd    x25, x8, x11, x1    // s0
    add     x26, x25, x11       // s1
    add     x27, x26, x11       // s2
    add     x28, x27, x11       // s3

    mov     x9, x23             // ptr to x_qs (int8)
    mov     x16, x24            // ptr to x_scales (float)

    mov     x17, #0             // block index
.Lblock_loop_4x:
    cmp     x17, x19
    b.ge    .Lblock_done_4x

    // Load x_qs (32 bytes = 2 NEON vectors)
    ldp     q0, q1, [x9]
    add     x9, x9, #32

    movi    v6.16b, #0x0f
    movi    v7.16b, #8

    // Row 0
    ldr     q2, [x12], #16
    and     v3.16b, v2.16b, v6.16b
    ushr    v4.16b, v2.16b, #4
    sub     v3.16b, v3.16b, v7.16b
    sub     v4.16b, v4.16b, v7.16b
    movi    v24.4s, #0
    sdot    v24.4s, v3.16b, v0.16b
    sdot    v24.4s, v4.16b, v1.16b
    addv    s24, v24.4s
    scvtf   s24, s24

    // Row 1
    ldr     q2, [x13], #16
    and     v3.16b, v2.16b, v6.16b
    ushr    v4.16b, v2.16b, #4
    sub     v3.16b, v3.16b, v7.16b
    sub     v4.16b, v4.16b, v7.16b
    movi    v25.4s, #0
    sdot    v25.4s, v3.16b, v0.16b
    sdot    v25.4s, v4.16b, v1.16b
    addv    s25, v25.4s
    scvtf   s25, s25

    // Row 2
    ldr     q2, [x14], #16
    and     v3.16b, v2.16b, v6.16b
    ushr    v4.16b, v2.16b, #4
    sub     v3.16b, v3.16b, v7.16b
    sub     v4.16b, v4.16b, v7.16b
    movi    v26.4s, #0
    sdot    v26.4s, v3.16b, v0.16b
    sdot    v26.4s, v4.16b, v1.16b
    addv    s26, v26.4s
    scvtf   s26, s26

    // Row 3
    ldr     q2, [x15], #16
    and     v3.16b, v2.16b, v6.16b
    ushr    v4.16b, v2.16b, #4
    sub     v3.16b, v3.16b, v7.16b
    sub     v4.16b, v4.16b, v7.16b
    movi    v27.4s, #0
    sdot    v27.4s, v3.16b, v0.16b
    sdot    v27.4s, v4.16b, v1.16b
    addv    s27, v27.4s
    scvtf   s27, s27

    // Scale and accumulate
    ldr     s18, [x16], #4      // x_scale
    
    ldr     s5, [x25], #4
    fmul    s5, s5, s18
    fmadd   s20, s24, s5, s20

    ldr     s5, [x26], #4
    fmul    s5, s5, s18
    fmadd   s21, s25, s5, s21

    ldr     s5, [x27], #4
    fmul    s5, s5, s18
    fmadd   s22, s26, s5, s22

    ldr     s5, [x28], #4
    fmul    s5, s5, s18
    fmadd   s23, s27, s5, s23

    add     x17, x17, #1
    b       .Lblock_loop_4x

.Lblock_done_4x:
    str     s20, [x3, x8, lsl #2]
    add     x8, x8, #1
    str     s21, [x3, x8, lsl #2]
    add     x8, x8, #1
    str     s22, [x3, x8, lsl #2]
    add     x8, x8, #1
    str     s23, [x3, x8, lsl #2]
    add     x8, x8, #1
    
    b       .Lrow_loop_4x

.Lrow_loop_1x:
    cmp     x8, x4
    b.ge    .Ldone

    movi    v20.4s, #0

    madd    x12, x8, x10, x0    // w0
    madd    x25, x8, x11, x1    // s0

    mov     x9, x23             // ptr to x_qs (int8)
    mov     x16, x24            // ptr to x_scales (float)

    mov     x17, #0
.Lblock_loop_1x:
    cmp     x17, x19
    b.ge    .Lblock_done_1x

    ldp     q0, q1, [x9]
    add     x9, x9, #32

    movi    v6.16b, #0x0f
    movi    v7.16b, #8

    ldr     q2, [x12], #16
    and     v3.16b, v2.16b, v6.16b
    ushr    v4.16b, v2.16b, #4
    sub     v3.16b, v3.16b, v7.16b
    sub     v4.16b, v4.16b, v7.16b
    movi    v24.4s, #0
    sdot    v24.4s, v3.16b, v0.16b
    sdot    v24.4s, v4.16b, v1.16b
    addv    s24, v24.4s
    scvtf   s24, s24

    ldr     s18, [x16], #4
    ldr     s5, [x25], #4
    fmul    s5, s5, s18
    fmadd   s20, s24, s5, s20

    add     x17, x17, #1
    b       .Lblock_loop_1x

.Lblock_done_1x:
    str     s20, [x3, x8, lsl #2]
    add     x8, x8, #1
    b       .Lrow_loop_1x

.Ldone:
    add     sp, sp, x22         // deallocate dynamic stack
    ldp     x27, x28, [sp, 80]
    ldp     x25, x26, [sp, 64]
    ldp     x23, x24, [sp, 48]
    ldp     x21, x22, [sp, 32]
    ldp     x19, x20, [sp, 16]
    ldp     x29, x30, [sp], 96
    ret
