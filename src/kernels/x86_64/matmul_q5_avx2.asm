;===============================================================================
; asmllm - Hand-written x86-64 AVX2 assembly kernel for Q5_0 Matrix-Vector Mult
; Compliance: Zero C, Zero Intrinsics, AVX2
;===============================================================================
;
; Function Header Block:
; Function: asm_matmul_q5
; One-line description:
;   Computes quantized Q5_0 matrix-vector multiply y = W @ x using AVX2 SIMD.
;
; Signature:
;   void asm_matmul_q5(
;       const uint8_t* ql,        // arg 0: Q5_0 low nibbles (M x num_blocks x 16 bytes)
;       const uint8_t* qh,        // arg 1: Q5_0 high bits (M x num_blocks x 4 bytes)
;       const float*   scales,    // arg 2: FP32 scales (M x num_blocks x 4 bytes)
;       const float*   x,         // arg 3: FP32 input vector (K elements)
;       float*         y,         // arg 4: FP32 output vector (M elements)
;       int64_t        M,         // arg 5: number of output rows
;       int64_t        K          // arg 6: number of input columns
;   );
;
; Inputs (Registers after normalization):
;   RDI = ql pointer
;   RSI = qh pointer
;   RDX = scales pointer
;   RCX = x input vector pointer
;   R8  = y output vector pointer
;   R9  = M (number of rows)
;   [RBP+64] = K (number of columns)
;
; Outputs:
;   Writes FP32 dot-product results to y[0..M-1].
;
; Clobbered Registers:
;   RAX, R10, R11, XMM0-XMM15, YMM0-YMM15 (callee-saved registers preserved).
;===============================================================================

%define WIN64 1

section .data
align 32
mask_0f:         times 16 db 0x0F
align 32
sixteen_f32:     dd 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0
align 32
bit_mask_1_128:  dd 1, 2, 4, 8, 16, 32, 64, 128

section .text
global asm_matmul_q5
export asm_matmul_q5

asm_matmul_q5:
    push rbp
    mov  rbp, rsp
    push rbx
    push rdi
    push rsi
    push r12
    push r13
    push r14
    push r15
    sub  rsp, 168       ; 160 bytes for XMM6-15 + 8 bytes alignment

%ifdef WIN64
    vmovups [rsp + 0],   xmm6
    vmovups [rsp + 16],  xmm7
    vmovups [rsp + 32],  xmm8
    vmovups [rsp + 48],  xmm9
    vmovups [rsp + 64],  xmm10
    vmovups [rsp + 80],  xmm11
    vmovups [rsp + 96],  xmm12
    vmovups [rsp + 112], xmm13
    vmovups [rsp + 128], xmm14
    vmovups [rsp + 144], xmm15

    ; Microsoft x64 ABI argument normalization (7 args):
    ; RCX=ql, RDX=qh, R8=scales, R9=x, [RBP+48]=y, [RBP+56]=M, [RBP+64]=K
    mov  rdi, rcx       ; RDI = ql
    mov  rsi, rdx       ; RSI = qh
    mov  rdx, r8        ; RDX = scales
    mov  rcx, r9        ; RCX = x
    mov  r8,  [rbp+48]  ; R8  = y
    mov  r9,  [rbp+56]  ; R9  = M
    mov  r10, [rbp+64]  ; R10 = K
%endif

    ; Compute num_blocks = K / 32
    shr  r10, 5

    ; R11 = current row index m (0 to M-1)
    xor  r11, r11

.matmul_q5.row_loop:
    cmp  r11, r9
    jge  .matmul_q5.done

    vxorps ymm0, ymm0, ymm0     ; YMM0 = row accumulator
    xor    r12, r12             ; R12 = block index b

.matmul_q5.block_loop:
    cmp  r12, r10
    jge  .matmul_q5.store_row

    ; r14 = flat block index = m * num_blocks + b
    mov  r14, r11
    imul r14, r10
    add  r14, r12

    ; ql byte offset = r14 * 16
    mov  rax, r14
    shl  rax, 4
    vmovdqu xmm1, [rdi + rax]

    ; qh byte offset = r14 * 4
    mov  rbx, r14
    shl  rbx, 2

    ; Unpack 16 low nibbles into XMM2 (weights 0..15) and XMM4 (weights 16..31)
    vmovdqu xmm2, [rel mask_0f]
    vpand   xmm2, xmm1, xmm2

    vpsrlw  xmm4, xmm1, 4
    vmovdqu xmm5, [rel mask_0f]
    vpand   xmm4, xmm4, xmm5

    ; YMM7 accumulates unscaled dot product for this block
    vxorps ymm7, ymm7, ymm7

    ; Input vector byte offset: r13 = b * 32 floats = b * 128 bytes
    mov  r13, r12
    shl  r13, 7

    ; Slice 0 (weights 0..7) - qh byte 0
    movzx eax, byte [rsi + rbx + 0]
    vmovd xmm8, eax
    vpbroadcastd ymm8, xmm8
    vmovdqu ymm9, [rel bit_mask_1_128]
    vpand   ymm8, ymm8, ymm9
    vpcmpeqd ymm8, ymm8, ymm9
    vpand   ymm8, ymm8, [rel sixteen_f32]

    vpmovzxbd ymm3, xmm2
    vcvtdq2ps ymm3, ymm3
    vaddps    ymm3, ymm3, ymm8
    vsubps    ymm3, ymm3, [rel sixteen_f32]
    vfmadd231ps ymm7, ymm3, [rcx + r13 + 0]

    ; Slice 1 (weights 8..15) - qh byte 1
    movzx eax, byte [rsi + rbx + 1]
    vmovd xmm8, eax
    vpbroadcastd ymm8, xmm8
    vmovdqu ymm9, [rel bit_mask_1_128]
    vpand   ymm8, ymm8, ymm9
    vpcmpeqd ymm8, ymm8, ymm9
    vpand   ymm8, ymm8, [rel sixteen_f32]

    vpsrldq   xmm6, xmm2, 8
    vpmovzxbd ymm6, xmm6
    vcvtdq2ps ymm6, ymm6
    vaddps    ymm6, ymm6, ymm8
    vsubps    ymm6, ymm6, [rel sixteen_f32]
    vfmadd231ps ymm7, ymm6, [rcx + r13 + 32]

    ; Slice 2 (weights 16..23) - qh byte 2
    movzx eax, byte [rsi + rbx + 2]
    vmovd xmm8, eax
    vpbroadcastd ymm8, xmm8
    vmovdqu ymm9, [rel bit_mask_1_128]
    vpand   ymm8, ymm8, ymm9
    vpcmpeqd ymm8, ymm8, ymm9
    vpand   ymm8, ymm8, [rel sixteen_f32]

    vpmovzxbd ymm3, xmm4
    vcvtdq2ps ymm3, ymm3
    vaddps    ymm3, ymm3, ymm8
    vsubps    ymm3, ymm3, [rel sixteen_f32]
    vfmadd231ps ymm7, ymm3, [rcx + r13 + 64]

    ; Slice 3 (weights 24..31) - qh byte 3
    movzx eax, byte [rsi + rbx + 3]
    vmovd xmm8, eax
    vpbroadcastd ymm8, xmm8
    vmovdqu ymm9, [rel bit_mask_1_128]
    vpand   ymm8, ymm8, ymm9
    vpcmpeqd ymm8, ymm8, ymm9
    vpand   ymm8, ymm8, [rel sixteen_f32]

    vpsrldq   xmm6, xmm4, 8
    vpmovzxbd ymm6, xmm6
    vcvtdq2ps ymm6, ymm6
    vaddps    ymm6, ymm6, ymm8
    vsubps    ymm6, ymm6, [rel sixteen_f32]
    vfmadd231ps ymm7, ymm6, [rcx + r13 + 96]

    ; Multiply accumulated block dot product by block scale and add to row sum YMM0
    vbroadcastss ymm15, [rdx + r14*4]
    vfmadd231ps  ymm0, ymm7, ymm15

    inc  r12
    jmp  .matmul_q5.block_loop

.matmul_q5.store_row:
    vextractf128 xmm1, ymm0, 1
    vaddps       xmm0, xmm0, xmm1
    vpermilps    xmm1, xmm0, 0b01001110
    vaddps       xmm0, xmm0, xmm1
    vpermilps    xmm1, xmm0, 0b10110001
    vaddps       xmm0, xmm0, xmm1

    vmovss [r8 + r11*4], xmm0

    inc  r11
    jmp  .matmul_q5.row_loop

.matmul_q5.done:
    vzeroupper
%ifdef WIN64
    vmovups xmm6,  [rsp + 0]
    vmovups xmm7,  [rsp + 16]
    vmovups xmm8,  [rsp + 32]
    vmovups xmm9,  [rsp + 48]
    vmovups xmm10, [rsp + 64]
    vmovups xmm11, [rsp + 80]
    vmovups xmm12, [rsp + 96]
    vmovups xmm13, [rsp + 112]
    vmovups xmm14, [rsp + 128]
    vmovups xmm15, [rsp + 144]
%endif
    add  rsp, 168
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rsi
    pop  rdi
    pop  rbx
    pop  rbp
    ret

;===============================================================================
; Multi-Threaded Entry Point: asm_matmul_q5_mt
;===============================================================================
extern asm_threadpool_init
extern asm_threadpool_dispatch_q5

global asm_matmul_q5_mt
export asm_matmul_q5_mt

asm_matmul_q5_mt:
    push rbp
    mov  rbp, rsp
    push rbx
    push rdi
    push rsi
    push r12
    push r13
    push r14
    push r15
    sub  rsp, 40

%ifdef WIN64
    mov  rdi, rcx       ; ql
    mov  rsi, rdx       ; qh
    mov  rdx, r8        ; scales
    mov  rcx, r9        ; x
    mov  r8,  [rbp+48]  ; y
    mov  r9,  [rbp+56]  ; M
    mov  r10, [rbp+64]  ; K
    mov  r11, [rbp+72]  ; num_threads
%endif

    cmp  r11, 1
    jle  .matmul_q5_mt.single_thread

    push r11
    push r10
    push r9
    push r8
    push rcx
    push rdx
    push rsi
    push rdi

    mov  rcx, r11
    sub  rsp, 32
    call asm_threadpool_init
    add  rsp, 32

    pop  rdi
    pop  rsi
    pop  rdx
    pop  rcx
    pop  r8
    pop  r9
    pop  r10
    pop  r11

    sub  rsp, 56
    mov  [rsp+32], r8   ; y
    mov  [rsp+40], r9   ; M
    mov  [rsp+48], r10  ; K
    mov  r8,  rdx       ; scales
    mov  r9,  rcx       ; x
    mov  rdx, rsi       ; qh
    mov  rcx, rdi       ; ql
    call asm_threadpool_dispatch_q5
    add  rsp, 56
    jmp  .matmul_q5_mt.exit

.matmul_q5_mt.single_thread:
    sub  rsp, 56
    mov  [rsp+32], r8   ; y
    mov  [rsp+40], r9   ; M
    mov  [rsp+48], r10  ; K
    mov  r8,  rdx       ; scales
    mov  r9,  rcx       ; x
    mov  rdx, rsi       ; qh
    mov  rcx, rdi       ; ql
    call asm_matmul_q5
    add  rsp, 56

.matmul_q5_mt.exit:
    add  rsp, 40
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rsi
    pop  rdi
    pop  rbx
    pop  rbp
    ret
