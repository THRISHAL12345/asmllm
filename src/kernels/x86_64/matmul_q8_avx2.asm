;===============================================================================
; asmllm - Hand-written x86-64 AVX2 assembly kernel for Q8_0 Matrix-Vector Mult
; Compliance: Zero C, Zero Intrinsics, AVX2
;===============================================================================
;
; Function Header Block:
; Function: asm_matmul_q8
; One-line description:
;   Computes quantized Q8_0 matrix-vector multiply y = W @ x using AVX2 SIMD.
;
; Signature:
;   void asm_matmul_q8(
;       const int8_t* qweights,   // arg 0: Q8_0 weight blocks (M x num_blocks x 32 bytes)
;       const float*  scales,     // arg 1: FP32 scales (M x num_blocks x 4 bytes)
;       const float*  x,          // arg 2: FP32 input vector (K elements)
;       float*        y,          // arg 3: FP32 output vector (M elements)
;       int64_t       M,          // arg 4: number of output rows
;       int64_t       K           // arg 5: number of input columns
;   );
;
; Inputs (Registers after normalization):
;   RDI = qweights pointer
;   RSI = scales pointer
;   RDX = x input vector pointer
;   RCX = y output vector pointer
;   R8  = M (number of rows)
;   R9  = K (number of columns)
;
; Outputs:
;   Writes FP32 dot-product results to y[0..M-1].
;
; Clobbered Registers:
;   RAX, R10, R11, XMM0-XMM15, YMM0-YMM15 (callee-saved registers preserved).
;
; Preconditions:
;   - AVX2 and FMA3 instruction set support.
;   - Q8_0 block layout: 32 signed int8 weights (32 bytes per block) + 1 FP32 scale.
;===============================================================================

%define WIN64 1

; Named Constants
BLOCK_SIZE  equ 32      ; Number of weights per Q8_0 block
BLOCK_BYTES equ 32      ; Number of bytes per Q8_0 block (32 int8 weights)
FLOAT_BYTES equ 4       ; Bytes per FP32 scalar

section .text
global asm_matmul_q8
export asm_matmul_q8

asm_matmul_q8:
    ; Cross-platform ABI prologue
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

    ; Microsoft x64 ABI argument normalization:
    ; RCX=qweights, RDX=scales, R8=x, R9=y, [RBP+48]=M, [RBP+56]=K
    mov  rdi, rcx       ; RDI = qweights
    mov  rsi, rdx       ; RSI = scales
    mov  rdx, r8        ; RDX = x
    mov  rcx, r9        ; RCX = y
    mov  r8,  [rbp+48]  ; R8  = M
    mov  r9,  [rbp+56]  ; R9  = K
%endif

    ; Compute number of full 32-element blocks: num_blocks = K / 32
    mov  r10, r9
    shr  r10, 5

    ; R11 = current row index m (0 to M-1)
    xor  r11, r11

.matmul_q8.row_loop:
    cmp  r11, r8
    jge  .matmul_q8.done

    ; YMM0 holds the running FP32 dot-product sum for row m
    vxorps ymm0, ymm0, ymm0

    ; R12 = current block index b (0 to num_blocks - 1)
    xor  r12, r12

.matmul_q8.block_loop:
    cmp  r12, r10
    jge  .matmul_q8.store_row

    ; Calculate block index flat: r14 = m * num_blocks + b
    mov  r14, r11
    imul r14, r10
    add  r14, r12

    ; Calculate byte offset for current block's weights:
    ; weight_offset = r14 * 32
    mov  rax, r14
    shl  rax, 5

    ; Load 32 signed int8 weights into XMM1 and XMM2 (16 bytes each)
    vmovdqu xmm1, [rdi + rax + 0]
    vmovdqu xmm2, [rdi + rax + 16]

    ; YMM7 accumulates unscaled block dot product
    vxorps ymm7, ymm7, ymm7

    ; Input vector offset: r13 = b * 32 floats = b * 128 bytes
    mov  r13, r12
    shl  r13, 7

    ; Slice 0 (weights 0..7)
    vpmovsxbd ymm3, xmm1
    vcvtdq2ps ymm3, ymm3
    vfmadd231ps ymm7, ymm3, [rdx + r13 + 0]

    ; Slice 1 (weights 8..15)
    vpsrldq   xmm4, xmm1, 8
    vpmovsxbd ymm4, xmm4
    vcvtdq2ps ymm4, ymm4
    vfmadd231ps ymm7, ymm4, [rdx + r13 + 32]

    ; Slice 2 (weights 16..23)
    vpmovsxbd ymm5, xmm2
    vcvtdq2ps ymm5, ymm5
    vfmadd231ps ymm7, ymm5, [rdx + r13 + 64]

    ; Slice 3 (weights 24..31)
    vpsrldq   xmm6, xmm2, 8
    vpmovsxbd ymm6, xmm6
    vcvtdq2ps ymm6, ymm6
    vfmadd231ps ymm7, ymm6, [rdx + r13 + 96]

    ; Multiply accumulated block dot product by block scale and add to row sum YMM0
    vbroadcastss ymm15, [rsi + r14*4]
    vfmadd231ps  ymm0, ymm7, ymm15

    inc  r12
    jmp  .matmul_q8.block_loop

.matmul_q8.store_row:
    ; Horizontal sum of 8 FP32 elements in YMM0 -> scalar XMM0
    vextractf128 xmm1, ymm0, 1
    vaddps       xmm0, xmm0, xmm1
    vpermilps    xmm1, xmm0, 0b01001110
    vaddps       xmm0, xmm0, xmm1
    vpermilps    xmm1, xmm0, 0b10110001
    vaddps       xmm0, xmm0, xmm1

    vmovss [rcx + r11*4], xmm0

    inc  r11
    jmp  .matmul_q8.row_loop

.matmul_q8.done:
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
; Multi-Threaded Entry Point: asm_matmul_q8_mt
; Dispatches row tiles to worker thread pool
;===============================================================================
extern asm_threadpool_init
extern asm_threadpool_dispatch_q8

global asm_matmul_q8_mt
export asm_matmul_q8_mt

asm_matmul_q8_mt:
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
    ; Normalization: RCX=qweights, RDX=scales, R8=x, R9=y, [RBP+48]=M, [RBP+56]=K, [RBP+64]=num_threads
    mov  rdi, rcx
    mov  rsi, rdx
    mov  rdx, r8
    mov  rcx, r9
    mov  r8,  [rbp+48]
    mov  r9,  [rbp+56]
    mov  r10, [rbp+64]
%else
    mov  r10, [rbp+16]
%endif

    cmp  r10, 1
    jle  .matmul_q8_mt.single_thread

    push r10
    push r9
    push r8
    push rcx
    push rdx
    push rsi
    push rdi

%ifdef WIN64
    mov  rcx, r10
    sub  rsp, 32
    call asm_threadpool_init
    add  rsp, 32
%else
    mov  rdi, r10
    call asm_threadpool_init
%endif

    pop  rdi
    pop  rsi
    pop  rdx
    pop  rcx
    pop  r8
    pop  r9
    pop  r10

%ifdef WIN64
    sub  rsp, 48
    mov  [rsp+32], r8
    mov  [rsp+40], r9
    mov  r8,  rdx
    mov  r9,  rcx
    mov  rdx, rsi
    mov  rcx, rdi
    call asm_threadpool_dispatch_q8
    add  rsp, 48
%else
    call asm_threadpool_dispatch_q8
%endif
    jmp  .matmul_q8_mt.exit

.matmul_q8_mt.single_thread:
%ifdef WIN64
    sub  rsp, 48
    mov  [rsp+32], r8
    mov  [rsp+40], r9
    mov  r8,  rdx
    mov  r9,  rcx
    mov  rdx, rsi
    mov  rcx, rdi
    call asm_matmul_q8
    add  rsp, 48
%else
    call asm_matmul_q8
%endif

.matmul_q8_mt.exit:
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
