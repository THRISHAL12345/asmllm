;===============================================================================
; asmllm - Hand-written x86-64 assembly kernel for Numerically Stable Softmax
; Compliance: Zero C, Zero Intrinsics, AVX2 + x87 hardware exp
;===============================================================================
;
; Function Header Block:
; Function: asm_softmax
; One-line description:
;   Computes numerically stable softmax y = exp(x - max(x)) / sum(exp(x - max(x)))
;
; Signature:
;   void asm_softmax(
;       const float* x,   // arg 0 (RCX): input vector (dim floats)
;       float*       y,   // arg 1 (RDX): output vector (dim floats)
;       int64_t      dim  // arg 2 (R8):  vector dimension
;   );
;
; Inputs:
;   RDI = x pointer
;   RSI = y pointer
;   RCX = dim
;===============================================================================

%define WIN64 1

section .rdata align=16
log2_e:
    dd 1.4426950408889634  ; log2(e) float32

section .text
global asm_softmax
export asm_softmax

asm_softmax:
    push rbp
    mov  rbp, rsp
    push rbx
    push r12
    push r13
    sub  rsp, 32

%ifdef WIN64
    mov  rdi, rcx       ; RDI = x
    mov  rsi, rdx       ; RSI = y
    mov  rcx, r8        ; RCX = dim
%endif

    ; --- Pass 1: Find max value in vector x ---
    vmovss xmm0, [rdi]  ; Initialize max with x[0]
    xor    r10, r10     ; i = 0

.softmax.max_loop:
    cmp    r10, rcx
    jge    .softmax.max_done
    vmovss xmm1, [rdi + r10*4]
    vmaxss xmm0, xmm0, xmm1
    inc    r10
    jmp    .softmax.max_loop

.softmax.max_done:
    movss  [rbp-4], xmm0        ; [rbp-4] = max_val

    ; --- Pass 2: Compute exp(x_i - max_val) and accumulate sum ---
    vxorps xmm15, xmm15, xmm15  ; XMM15 holds sum of exponentials
    xor    r10, r10             ; i = 0

.softmax.exp_loop:
    cmp    r10, rcx
    jge    .softmax.exp_done

    vmovss xmm1, [rdi + r10*4]
    vsubss xmm1, xmm1, [rbp-4]  ; z = x_i - max_val
    movss  [rbp-8], xmm1

    ; Compute exp(z) = 2^(z * log2(e)) using x87 hardware:
    fld  dword [rbp-8]          ; ST(0) = z
    fldl2e                      ; ST(0) = log2(e), ST(1) = z
    fmulp st1, st0              ; ST(0) = z * log2(e)

    fld  st0
    frndint                     ; n = round(z * log2(e))
    fsub st1, st0               ; ST(0) = n, ST(1) = f
    fxch st1                    ; ST(0) = f, ST(1) = n
    f2xm1                       ; ST(0) = 2^f - 1
    fld1
    faddp st1, st0              ; ST(0) = 2^f
    fscale                      ; ST(0) = 2^f * 2^n = exp(z)
    fstp st1                    ; Pop n

    fstp dword [rbp-12]         ; store exp(z)
    movss xmm2, [rbp-12]
    vmovss [rsi + r10*4], xmm2  ; y[i] = exp(z)
    vaddss xmm15, xmm15, xmm2   ; sum += exp(z)

    inc  r10
    jmp  .softmax.exp_loop

.softmax.exp_done:
    ; --- Pass 3: Normalize y[i] /= sum ---
    vbroadcastss ymm14, xmm15   ; broadcast sum across 8 lanes

    mov  rax, rcx
    shr  rax, 3
    mov  r11, rax               ; R11 = dim / 8 blocks
    xor  r10, r10               ; block index b = 0

.softmax.div_loop:
    cmp  r10, r11
    jge  .softmax.div_remainder
    mov  rax, r10
    shl  rax, 5                 ; byte offset = b * 32
    vmovups ymm0, [rsi + rax]
    vdivps  ymm0, ymm0, ymm14   ; y_block /= sum
    vmovups [rsi + rax], ymm0
    inc  r10
    jmp  .softmax.div_loop

.softmax.div_remainder:
    mov  rax, r11
    shl  rax, 3
    mov  r12, rax

.softmax.tail_div_loop:
    cmp  r12, rcx
    jge  .softmax.done
    vmovss xmm0, [rsi + r12*4]
    vdivss xmm0, xmm0, xmm15    ; y_i /= sum scalar
    vmovss [rsi + r12*4], xmm0
    inc  r12
    jmp  .softmax.tail_div_loop

.softmax.done:
    vzeroupper
    add  rsp, 32
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret
