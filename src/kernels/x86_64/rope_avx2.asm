;===============================================================================
; asmllm - Hand-written x86-64 assembly kernel for Rotary Position Embedding
; Compliance: Zero C, Zero Intrinsics, AVX2 + x87 hardware trig/pow
;===============================================================================
;
; Function Header Block:
; Function: asm_rope
; One-line description:
;   Applies Rotary Position Embedding (RoPE) to query and key vectors in-place.
;
; Signature:
;   void asm_rope(
;       float*  q,         // arg 0 (RCX): query vector (head_dim floats)
;       float*  k,         // arg 1 (RDX): key vector (head_dim floats)
;       int64_t head_dim,  // arg 2 (R8):  dimension per attention head (even)
;       int64_t pos,       // arg 3 (R9):  sequence position index
;       float   theta      // arg 4 ([RBP+48]): RoPE base theta (e.g. 10000.0)
;   );
;
; Inputs:
;   RDI = q pointer
;   RSI = k pointer
;   RBX = head_dim
;   R12 = pos
;   [rbp-4] = theta float
;===============================================================================

%define WIN64 1

section .text
global asm_rope
export asm_rope

asm_rope:
    push rbp
    mov  rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    sub  rsp, 56        ; Local stack space for x87 float temporaries

%ifdef WIN64
    mov  rdi, rcx       ; RDI = q
    mov  rsi, rdx       ; RSI = k
    mov  rbx, r8        ; RBX = head_dim
    mov  r12, r9        ; R12 = pos
    mov  eax, [rbp+48]
    mov  [rbp-4], eax   ; [rbp-4] = theta
%endif

    ; Convert pos integer to float at [rbp-8]
    cvtsi2ss xmm0, r12
    movss    [rbp-8], xmm0

    ; Convert head_dim integer to float at [rbp-12]
    cvtsi2ss xmm0, rbx
    movss    [rbp-12], xmm0

    ; R13 = pair index p = 0 .. (head_dim/2 - 1)
    xor  r13, r13
    mov  r14, rbx
    shr  r14, 1         ; R14 = head_dim / 2 pairs

.rope.pair_loop:
    cmp  r13, r14
    jge  .rope.done

    ; Compute exponent e = -2.0 * p / head_dim
    mov  rax, r13
    shl  rax, 1         ; 2*p
    neg  rax            ; -2*p
    cvtsi2ss xmm0, rax
    divss    xmm0, [rbp-12]
    movss    [rbp-16], xmm0  ; [rbp-16] = e = -2p / head_dim

    ; Compute freq = theta ^ e using x87 hardware log2/exp2:
    ; ST(0) = e * log2(theta)
    fld  dword [rbp-16] ; ST(0) = e
    fld  dword [rbp-4]  ; ST(0) = theta, ST(1) = e
    fyl2x               ; ST(0) = e * log2(theta) = z

    ; Compute 2^z where z = ST(0):
    fld  st0            ; ST(0) = z, ST(1) = z
    frndint             ; ST(0) = round(z) = n, ST(1) = z
    fsub st1, st0       ; ST(0) = n, ST(1) = f = z - n
    fxch st1            ; ST(0) = f, ST(1) = n
    f2xm1               ; ST(0) = 2^f - 1
    fld1
    faddp st1, st0      ; ST(0) = 2^f
    fscale              ; ST(0) = 2^f * 2^n = 2^z = freq
    fstp st1            ; Pop n, leaving ST(0) = freq

    ; Compute angle = pos * freq
    fmul dword [rbp-8]  ; ST(0) = angle

    ; Compute sin(angle) and cos(angle) simultaneously
    fsincos             ; ST(0) = cos(angle), ST(1) = sin(angle)

    fstp dword [rbp-20] ; [rbp-20] = cos_val, ST(0) = sin(angle)
    fstp dword [rbp-24] ; [rbp-24] = sin_val, ST stack empty

    movss xmm1, [rbp-20] ; XMM1 = cos_val
    movss xmm2, [rbp-24] ; XMM2 = sin_val

    ; Apply rotation to query vector q[2*p], q[2*p+1]
    mov   rax, r13
    shl   rax, 3        ; byte offset = p * 8 (two float32s)
    movss xmm3, [rdi + rax]     ; q0 = q[2p]
    movss xmm4, [rdi + rax + 4] ; q1 = q[2p+1]

    ; out0 = q0 * cos - q1 * sin
    movss xmm5, xmm3
    mulss xmm5, xmm1
    movss xmm6, xmm4
    mulss xmm6, xmm2
    subss xmm5, xmm6
    movss [rdi + rax], xmm5

    ; out1 = q0 * sin + q1 * cos
    movss xmm5, xmm3
    mulss xmm5, xmm2
    movss xmm6, xmm4
    mulss xmm6, xmm1
    addss xmm5, xmm6
    movss [rdi + rax + 4], xmm5

    ; Apply rotation to key vector k[2*p], k[2*p+1]
    movss xmm3, [rsi + rax]     ; k0 = k[2p]
    movss xmm4, [rsi + rax + 4] ; k1 = k[2p+1]

    ; out0 = k0 * cos - k1 * sin
    movss xmm5, xmm3
    mulss xmm5, xmm1
    movss xmm6, xmm4
    mulss xmm6, xmm2
    subss xmm5, xmm6
    movss [rsi + rax], xmm5

    ; out1 = k0 * sin + k1 * cos
    movss xmm5, xmm3
    mulss xmm5, xmm2
    movss xmm6, xmm4
    mulss xmm6, xmm1
    addss xmm5, xmm6
    movss [rsi + rax + 4], xmm5

    inc  r13
    jmp  .rope.pair_loop

.rope.done:
    add  rsp, 56
    pop  r14
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret
