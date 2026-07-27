;===============================================================================
; asmllm - Hand-written x86-64 AVX-512 assembly kernel for Q4_0 Matrix-Vector Mult
; Compliance: Zero C, Zero Intrinsics, AVX-512
;===============================================================================
;
; Function Header Block:
; Function: asm_matmul_q4_avx512
; One-line description:
;   Computes quantized Q4_0 matrix-vector multiply y = W @ x using AVX-512 SIMD.
;
; Signature:
;   void asm_matmul_q4_avx512(
;       const uint8_t* qweights,  // arg 0: Q4_0 weight blocks (M x num_blocks x 16 bytes)
;       const float*   scales,    // arg 1: FP32 scales (M x num_blocks x 4 bytes)
;       const float*   x,         // arg 2: FP32 input vector (K elements)
;       float*         y,         // arg 3: FP32 output vector (M elements)
;       int64_t        M,         // arg 4: number of output rows
;       int64_t        K          // arg 5: number of input columns
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
;   RAX, R10, R11, XMM0-XMM31, YMM0-YMM31, ZMM0-ZMM31
;
; Preconditions:
;   - AVX-512F instruction set support.
;   - Q4_0 block layout: 32 weights packed into 16 bytes (low nibble first).
;===============================================================================

%define WIN64 1

; Named Constants
BLOCK_SIZE  equ 32      ; Number of weights per Q4_0 block
BLOCK_BYTES equ 16      ; Number of bytes per Q4_0 block (32 nibbles)
FLOAT_BYTES equ 4       ; Bytes per FP32 scalar

section .rdata align=64
mask_0f:
    times 16 db 0x0F    ; 16-byte mask to isolate lower 4 bits (nibble)
eight_f32:
    times 16 dd 8.0     ; Broadcasted FP32 bias value 8.0 for Q4 dequantization (16 floats)

section .text
global asm_matmul_q4_avx512
export asm_matmul_q4_avx512

asm_matmul_q4_avx512:
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
    mov  rax, r9
    shr  rax, 5         ; RAX = K / 32 (num_blocks per row)
    mov  r10, rax       ; R10 = num_blocks

    ; R11 = current row index m (0 to M-1)
    xor  r11, r11

.matmul_q4.row_loop:
    cmp  r11, r8
    jge  .matmul_q4.done

    ; Zero out row accumulator ZMM15 (holds 16 partial FP32 dot-product sums)
    vxorps zmm15, zmm15, zmm15

    ; R12 = current block index b (0 to num_blocks - 1)
    xor  r12, r12

.matmul_q4.block_loop:
    cmp  r12, r10
    jge  .matmul_q4.remainder

    ; Calculate byte offset for current block's weights:
    ; weight_offset = (m * num_blocks + b) * 16
    mov  rax, r11
    imul rax, r10
    add  rax, r12
    shl  rax, 4         ; * 16 bytes per block
    vmovdqu xmm0, [rdi + rax]

    ; Unpack 16 low nibbles (weights 0..15) into XMM1
    vmovdqa xmm1, [rel mask_0f]
    vpand   xmm1, xmm0, xmm1

    ; Unpack 16 high nibbles (weights 16..31) into XMM2
    ; Isolates upper nibble of each byte
    vpsrlw  xmm2, xmm0, 4
    vmovdqa xmm3, [rel mask_0f]
    vpand   xmm2, xmm2, xmm3

    ; Convert low nibbles 0..15 to FP32 in ZMM3 (zero extend 16 bytes to 16 dwords)
    vpmovzxbd zmm3, xmm1
    vcvtdq2ps zmm3, zmm3
    vsubps    zmm3, zmm3, [rel eight_f32]

    ; Convert high nibbles 16..31 to FP32 in ZMM4 (zero extend 16 bytes to 16 dwords)
    vpmovzxbd zmm4, xmm2
    vcvtdq2ps zmm4, zmm4
    vsubps    zmm4, zmm4, [rel eight_f32]

    ; Calculate input offset into vector x: b * 32 * 4 bytes = b * 128
    mov  rax, r12
    shl  rax, 7

    ; Accumulate into ZMM7: keeps inner loop entirely in SIMD vector execution
    vxorps zmm7, zmm7, zmm7
    ; 16 floats (0..15)
    vmovups zmm5, [rdx + rax]
    vfmadd231ps zmm7, zmm3, zmm5
    ; 16 floats (16..31)
    vmovups zmm6, [rdx + rax + 64]
    vfmadd231ps zmm7, zmm4, zmm6

    ; Load block scale from scales array:
    ; scale_offset = (m * num_blocks + b) * 4
    mov  rax, r11
    imul rax, r10
    add  rax, r12
    shl  rax, 2         ; * 4 bytes per float scale
    vbroadcastss zmm8, [rsi + rax]

    ; Multiply block dot product by scale and add to row accumulator ZMM15
    vfmadd231ps zmm15, zmm7, zmm8

    inc  r12
    jmp  .matmul_q4.block_loop

.matmul_q4.remainder:
    ; Check if K is an exact multiple of 32
    mov  rax, r10
    shl  rax, 5         ; processed elements = num_blocks * 32
    cmp  rax, r9
    jge  .matmul_q4.store_row

    ; Horizontal sum of full blocks in ZMM15 first into XMM0
    vextractf32x8 ymm0, zmm15, 1
    vaddps        ymm0, ymm0, ymm15
    vextractf128  xmm1, ymm0, 1
    vaddps        xmm0, xmm0, xmm1
    vpermilps     xmm1, xmm0, 0x0E
    vaddps        xmm0, xmm0, xmm1
    vpermilps     xmm1, xmm0, 0x01
    vaddss        xmm0, xmm0, xmm1

    ; Process remaining scalar elements if K % 32 != 0 (tail remainder handling)
    mov  r14, rax       ; R14 = num_blocks * 32
    mov  r13, rax       ; R13 = current tail element index k (from num_blocks*32 to K-1)
.matmul_q4.tail_loop:
    cmp  r13, r9
    jge  .matmul_q4.store_scalar

    ; Calculate weight byte inside partial block:
    ; block b = r10, byte_idx = (r13 - num_blocks*32) >> 1
    mov  rax, r13
    sub  rax, r14       ; rem_idx = k - num_blocks*32
    mov  rbx, rax
    shr  rbx, 1         ; byte_idx = rem_idx / 2

    ; Load byte from qweights[m, r10, byte_idx]
    mov  rax, r11
    imul rax, r10
    add  rax, r10
    shl  rax, 4
    add  rax, rbx
    movzx ebx, byte [rdi + rax]

    ; Check if rem_idx is even or odd
    test r13, 1
    jnz  .matmul_q4.tail_odd
    and  ebx, 0x0F
    jmp  .matmul_q4.tail_fma
.matmul_q4.tail_odd:
    shr  ebx, 4
    and  ebx, 0x0F
.matmul_q4.tail_fma:
    sub  ebx, 8
    vcvtsi2ss xmm2, xmm2, ebx

    ; Multiply by block scale
    mov  rax, r11
    imul rax, r10
    add  rax, r10
    vmovss xmm3, [rsi + rax*4]
    vmulss xmm2, xmm2, xmm3

    ; Multiply by x[k] and accumulate into xmm0
    vmovss xmm3, [rdx + r13*4]
    vfmadd231ss xmm0, xmm2, xmm3

    inc  r13
    jmp  .matmul_q4.tail_loop

.matmul_q4.store_scalar:
    vmovss [rcx + r11*4], xmm0
    inc  r11
    jmp  .matmul_q4.row_loop

.matmul_q4.store_row:
    ; Horizontal sum of 16 float32 values in ZMM15 -> scalar float in XMM0[0]
    vextractf32x8 ymm0, zmm15, 1
    vaddps        ymm0, ymm0, ymm15
    vextractf128  xmm1, ymm0, 1
    vaddps        xmm0, xmm0, xmm1
    vpermilps     xmm1, xmm0, 0x0E
    vaddps        xmm0, xmm0, xmm1
    vpermilps     xmm1, xmm0, 0x01
    vaddss        xmm0, xmm0, xmm1

    ; Store scalar sum into y[m]
    vmovss [rcx + r11*4], xmm0

    inc  r11
    jmp  .matmul_q4.row_loop

.matmul_q4.done:
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
