;===============================================================================
; asmllm - Hand-written x86-64 AVX2 assembly kernel for RMSNorm
; Compliance: Zero C, Zero Intrinsics, AVX2
;===============================================================================
;
; Function Header Block:
; Function: asm_rmsnorm
; One-line description:
;   Computes Root Mean Square Layer Normalization y = (x / sqrt(mean(x^2)+eps)) * w
;
; Signature:
;   void asm_rmsnorm(
;       const float* x,       // arg 0 (RCX): input vector (dim floats)
;       const float* weight,  // arg 1 (RDX): scale weights (dim floats)
;       float*       y,       // arg 2 (R8):  output vector (dim floats)
;       int64_t      dim,     // arg 3 (R9):  vector dimension
;       float        eps      // arg 4 ([RBP+48]): epsilon stabilization
;   );
;
; Inputs (Registers after normalization):
;   RDI = x pointer
;   RSI = weight pointer
;   RDX = y pointer
;   RCX = dim
;   XMM14 = eps float
;
; Outputs:
;   Writes normalized FP32 values to y[0..dim-1].
;
; Clobbered Registers:
;   RAX, R10, R11, XMM0-XMM15, YMM0-YMM15.
;===============================================================================

%define WIN64 1

FLOAT_BYTES equ 4
VEC_LANES   equ 8

section .rdata align=16
one_f32:
    dd 1.0

section .text
global asm_rmsnorm
export asm_rmsnorm

asm_rmsnorm:
    push rbp
    mov  rbp, rsp
    push rbx
    push r12
    push r13
    sub  rsp, 40

%ifdef WIN64
    mov  rdi, rcx       ; RDI = x
    mov  rsi, rdx       ; RSI = weight
    mov  rdx, r8        ; RDX = y
    mov  rcx, r9        ; RCX = dim
    vmovss xmm14, dword [rbp+48] ; XMM14 = eps
%endif

    ; --- Pass 1: Compute sum of squares ---
    vxorps ymm0, ymm0, ymm0     ; YMM0 holds 8-lane sum of squares
    mov    rax, rcx
    shr    rax, 3               ; RAX = dim / 8 full vectors
    mov    r10, rax             ; R10 = num_blocks
    xor    r11, r11             ; R11 = current block index b

.rmsnorm.sum_loop:
    cmp    r11, r10
    jge    .rmsnorm.sum_remainder
    mov    rax, r11
    shl    rax, 5               ; byte offset = b * 32
    vmovups ymm1, [rdi + rax]
    vfmadd231ps ymm0, ymm1, ymm1 ; accumulate x_i^2
    inc    r11
    jmp    .rmsnorm.sum_loop

.rmsnorm.sum_remainder:
    ; Horizontal reduction of YMM0 -> scalar XMM0[0]
    vextractf128 xmm1, ymm0, 1
    vaddps       xmm0, xmm0, xmm1
    vpermilps    xmm1, xmm0, 0x0E
    vaddps       xmm0, xmm0, xmm1
    vpermilps    xmm1, xmm0, 0x01
    vaddss       xmm0, xmm0, xmm1

    ; Process remaining scalar elements if dim % 8 != 0 (tail remainder handling)
    mov    rax, r10
    shl    rax, 3               ; processed elements
    mov    r12, rax             ; R12 = current element index i

.rmsnorm.tail_sum_loop:
    cmp    r12, rcx
    jge    .rmsnorm.compute_scale
    vmovss xmm1, [rdi + r12*4]
    vfmadd231ss xmm0, xmm1, xmm1
    inc    r12
    jmp    .rmsnorm.tail_sum_loop

.rmsnorm.compute_scale:
    ; Compute mean = sum / dim
    vxorps  xmm1, xmm1, xmm1
    vcvtsi2ss xmm1, xmm1, rcx   ; convert dim integer to float
    vdivss  xmm0, xmm0, xmm1    ; mean = sum / dim

    ; Add eps
    vaddss  xmm0, xmm0, xmm14   ; mean + eps

    ; Compute exact square root
    vsqrtss xmm0, xmm0, xmm0    ; sqrt(mean + eps)

    ; Compute exact reciprocal: 1.0 / sqrt(mean + eps)
    vmovss  xmm1, [rel one_f32]
    vdivss  xmm0, xmm1, xmm0    ; inv_rms scalar float

    ; Broadcast inv_rms across all 8 lanes of YMM2
    vbroadcastss ymm2, xmm0

    ; --- Pass 2: Normalize and scale x_i * inv_rms * w_i ---
    xor    r11, r11

.rmsnorm.norm_loop:
    cmp    r11, r10
    jge    .rmsnorm.norm_remainder
    mov    rax, r11
    shl    rax, 5               ; byte offset = b * 32
    vmovups ymm3, [rdi + rax]   ; load x[i..i+7]
    vmovups ymm4, [rsi + rax]   ; load weight[i..i+7]
    vmulps  ymm3, ymm3, ymm2    ; x * inv_rms
    vmulps  ymm3, ymm3, ymm4    ; * weight
    vmovups [rdx + rax], ymm3   ; store to y
    inc    r11
    jmp    .rmsnorm.norm_loop

.rmsnorm.norm_remainder:
    mov    rax, r10
    shl    rax, 3
    mov    r12, rax

.rmsnorm.tail_norm_loop:
    cmp    r12, rcx
    jge    .rmsnorm.done
    vmovss xmm3, [rdi + r12*4]
    vmulss xmm3, xmm3, xmm0     ; x * inv_rms scalar
    vmovss xmm4, [rsi + r12*4]
    vmulss xmm3, xmm3, xmm4     ; * weight
    vmovss [rdx + r12*4], xmm3
    inc    r12
    jmp    .rmsnorm.tail_norm_loop

.rmsnorm.done:
    vzeroupper
    add  rsp, 40
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret
