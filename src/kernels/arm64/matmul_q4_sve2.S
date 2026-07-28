// src/kernels/arm64/matmul_q4_sve2.S
// SVE2 optimized Q4_0 Matrix-Vector Multiplication

.text
.align 4
.arch armv9-a+sve2
.global asm_matmul_q4_sve2
.global _asm_matmul_q4_sve2

// void asm_matmul_q4_sve2(
//     const uint8_t* qweights, // x0: Q4_0 weight blocks (M x (K/32) x 16 bytes)
//     const float*   scales,   // x1: FP32 scales (M x (K/32) x 4 bytes)
//     const float*   x,        // x2: FP32 input vector (K elements)
//     float*         y,        // x3: FP32 output vector (M elements)
//     int64_t        M,        // x4: number of output rows
//     int64_t        K         // x5: number of input columns
// );

asm_matmul_q4_sve2:
_asm_matmul_q4_sve2:
    // Prologue: save callee-saved registers (x19-x28, x29, x30)
    stp     x29, x30, [sp, -96]!
    mov     x29, sp
    stp     x19, x20, [sp, 16]
    stp     x21, x22, [sp, 32]
    stp     x23, x24, [sp, 48]
    stp     x25, x26, [sp, 64]
    stp     x27, x28, [sp, 80]

    // Note: SVE callee-saved registers are z8-z15 (bottom 64 bits only) and p4-p15.
    // We will only use z0-z7 and p0-p3 to avoid saving them.

    // x19 = K / 32 (number of blocks per row)
    lsr     x19, x5, #5

    // Allocate stack space for Q8_0 dynamic quantization of x
    // K bytes for int8 values, plus alignment
    add     x20, x5, #15
    and     x20, x20, #~15
    add     x22, x20, #16
    sub     sp, sp, x22

    mov     x23, sp             // x23 = ptr to Q8_0 x

    // --- Dynamic Quantization (FP32 -> Q8_0) ---
    // We will just do a simple scalar/NEON quantization for x to keep the prototype simple,
    // or we can use SVE to quantize it!
    mov     x8, #0              // x_idx = 0
    ptrue   p0.s                // p0 = all active for 32-bit float

.Lquant_loop_sve2:
    cmp     x8, x5
    b.ge    .Lquant_done_sve2

    // For a strict prototype, we use standard NEON quantization since x is guaranteed to be K % 32 == 0
    // Load 32 floats (4x 128-bit NEON vectors)
    add     x9, x2, x8, lsl #2
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x9]
    add     x9, x9, #64
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x9]

    // Find absolute maximums (NEON)
    fabs    v24.4s, v0.4s
    fabs    v25.4s, v1.4s
    fabs    v26.4s, v2.4s
    fabs    v27.4s, v3.4s
    fmax    v24.4s, v24.4s, v25.4s
    fmax    v26.4s, v26.4s, v27.4s
    fmax    v24.4s, v24.4s, v26.4s
    
    fabs    v25.4s, v4.4s
    fabs    v26.4s, v5.4s
    fabs    v27.4s, v6.4s
    fabs    v28.4s, v7.4s
    fmax    v25.4s, v25.4s, v26.4s
    fmax    v27.4s, v27.4s, v28.4s
    fmax    v25.4s, v25.4s, v27.4s
    
    fmax    v24.4s, v24.4s, v25.4s
    fmaxv   s24, v24.4s

    // Calculate scale = max_val / 127.0
    fmov    s25, #127.0
    fdiv    s25, s24, s25
    fmov    s26, #0.0
    fcmp    s25, s26
    fcsel   s25, s25, s26, gt
    
    // Calculate inverse scale
    fmov    s26, #1.0
    fdiv    s26, s26, s25
    dup     v26.4s, v26.s[0]

    // Scale and convert
    fmul    v0.4s, v0.4s, v26.4s
    fmul    v1.4s, v1.4s, v26.4s
    fmul    v2.4s, v2.4s, v26.4s
    fmul    v3.4s, v3.4s, v26.4s
    fmul    v4.4s, v4.4s, v26.4s
    fmul    v5.4s, v5.4s, v26.4s
    fmul    v6.4s, v6.4s, v26.4s
    fmul    v7.4s, v7.4s, v26.4s

    fcvtzs  v0.4s, v0.4s
    fcvtzs  v1.4s, v1.4s
    fcvtzs  v2.4s, v2.4s
    fcvtzs  v3.4s, v3.4s
    fcvtzs  v4.4s, v4.4s
    fcvtzs  v5.4s, v5.4s
    fcvtzs  v6.4s, v6.4s
    fcvtzs  v7.4s, v7.4s

    sqxtn   v0.4h, v0.4s
    sqxtn2  v0.8h, v1.4s
    sqxtn   v1.4h, v2.4s
    sqxtn2  v1.8h, v3.4s
    sqxtn   v2.4h, v4.4s
    sqxtn2  v2.8h, v5.4s
    sqxtn   v3.4h, v6.4s
    sqxtn2  v3.8h, v7.4s

    sqxtn   v0.8b, v0.8h
    sqxtn2  v0.16b, v1.8h
    sqxtn   v1.8b, v2.8h
    sqxtn2  v1.16b, v3.8h

    // Store 32 bytes to x23
    add     x9, x23, x8
    stp     q0, q1, [x9]

    add     x8, x8, #32
    b       .Lquant_loop_sve2

.Lquant_done_sve2:

    // --- SVE2 Matmul ---
    mov     x8, #0              // i = 0 (row index)
    mov     x10, x19            // stride for weights (in blocks)
    lsl     x10, x10, #4        // x10 = K/32 * 16 (bytes per row of weights)
    mov     x11, x19            
    lsl     x11, x11, #2        // x11 = K/32 * 4 (bytes per row of scales)

    // Predicate for 16 bytes (vl16) to process 1 block per iteration exactly
    ptrue   p0.b, vl16

.Lrow_loop_4x_sve:
    mov     x9, x4
    sub     x9, x9, x8
    cmp     x9, #4
    blt     .Lrow_loop_1x_sve

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
    mov     x17, #0             // block index
.Lblock_loop_4x_sve:
    cmp     x17, x19
    b.ge    .Lblock_done_4x_sve

    // Load x_qs (32 bytes = 2 SVE vectors of 16 bytes)
    ld1b    z0.b, p0/z, [x9]
    add     x9, x9, #16
    ld1b    z1.b, p0/z, [x9]
    add     x9, x9, #16

    // Pre-load constants
    mov     w16, #0x0f
    dup     z6.b, w16
    mov     w16, #8
    dup     z7.b, w16

    // Row 0
    ld1b    z2.b, p0/z, [x12]
    add     x12, x12, #16
    and     z3.b, z2.b, z6.b
    lsr     z4.b, z2.b, #4
    sub     z3.b, z3.b, z7.b
    sub     z4.b, z4.b, z7.b
    mov     z24.s, #0
    sdot    z24.s, z3.b, z0.b
    sdot    z24.s, z4.b, z1.b
    uaddv   d24, p0, z24.s
    scvtf   s24, s24

    // Row 1
    ld1b    z2.b, p0/z, [x13]
    add     x13, x13, #16
    and     z3.b, z2.b, z6.b
    lsr     z4.b, z2.b, #4
    sub     z3.b, z3.b, z7.b
    sub     z4.b, z4.b, z7.b
    mov     z25.s, #0
    sdot    z25.s, z3.b, z0.b
    sdot    z25.s, z4.b, z1.b
    uaddv   d25, p0, z25.s
    scvtf   s25, s25

    // Row 2
    ld1b    z2.b, p0/z, [x14]
    add     x14, x14, #16
    and     z3.b, z2.b, z6.b
    lsr     z4.b, z2.b, #4
    sub     z3.b, z3.b, z7.b
    sub     z4.b, z4.b, z7.b
    mov     z26.s, #0
    sdot    z26.s, z3.b, z0.b
    sdot    z26.s, z4.b, z1.b
    uaddv   d26, p0, z26.s
    scvtf   s26, s26

    // Row 3
    ld1b    z2.b, p0/z, [x15]
    add     x15, x15, #16
    and     z3.b, z2.b, z6.b
    lsr     z4.b, z2.b, #4
    sub     z3.b, z3.b, z7.b
    sub     z4.b, z4.b, z7.b
    mov     z27.s, #0
    sdot    z27.s, z3.b, z0.b
    sdot    z27.s, z4.b, z1.b
    uaddv   d27, p0, z27.s
    scvtf   s27, s27

    // Scale and accumulate
    ldr     s5, [x25], #4
    fmadd   s20, s24, s5, s20

    ldr     s5, [x26], #4
    fmadd   s21, s25, s5, s21

    ldr     s5, [x27], #4
    fmadd   s22, s26, s5, s22

    ldr     s5, [x28], #4
    fmadd   s23, s27, s5, s23

    add     x17, x17, #1
    b       .Lblock_loop_4x_sve

.Lblock_done_4x_sve:
    str     s20, [x3, x8, lsl #2]
    add     x8, x8, #1
    str     s21, [x3, x8, lsl #2]
    add     x8, x8, #1
    str     s22, [x3, x8, lsl #2]
    add     x8, x8, #1
    str     s23, [x3, x8, lsl #2]
    add     x8, x8, #1
    
    b       .Lrow_loop_4x_sve

.Lrow_loop_1x_sve:
    cmp     x8, x4
    b.ge    .Ldone_sve

    movi    v20.4s, #0
    madd    x12, x8, x10, x0    // w0
    madd    x25, x8, x11, x1    // s0

    mov     x9, x23             // ptr to x_qs (int8)
    mov     x17, #0
.Lblock_loop_1x_sve:
    cmp     x17, x19
    b.ge    .Lblock_done_1x_sve

    ld1b    z0.b, p0/z, [x9]
    add     x9, x9, #16
    ld1b    z1.b, p0/z, [x9]
    add     x9, x9, #16

    mov     w16, #0x0f
    dup     z6.b, w16
    mov     w16, #8
    dup     z7.b, w16

    ld1b    z2.b, p0/z, [x12]
    add     x12, x12, #16
    and     z3.b, z2.b, z6.b
    lsr     z4.b, z2.b, #4
    sub     z3.b, z3.b, z7.b
    sub     z4.b, z4.b, z7.b
    mov     z24.s, #0
    sdot    z24.s, z3.b, z0.b
    sdot    z24.s, z4.b, z1.b
    uaddv   d24, p0, z24.s
    scvtf   s24, s24

    ldr     s5, [x25], #4
    fmadd   s20, s24, s5, s20

    add     x17, x17, #1
    b       .Lblock_loop_1x_sve

.Lblock_done_1x_sve:
    str     s20, [x3, x8, lsl #2]
    add     x8, x8, #1
    b       .Lrow_loop_1x_sve

.Ldone_sve:
    add     sp, sp, x22         // deallocate dynamic stack
    ldp     x27, x28, [sp, 80]
    ldp     x25, x26, [sp, 64]
    ldp     x23, x24, [sp, 48]
    ldp     x21, x22, [sp, 32]
    ldp     x19, x20, [sp, 16]
    ldp     x29, x30, [sp], 96
    ret
