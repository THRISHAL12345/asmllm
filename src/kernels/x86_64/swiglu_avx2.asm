;===============================================================================
; asmllm - Hand-written x86-64 assembly kernel for SiLU / SwiGLU Hadamard
; Compliance: Zero C, Zero Intrinsics, AVX2 + x87 hardware exp
;===============================================================================
;
; Function Header Block:
; Function: asm_silu_hadamard
; One-line description:
;   Computes SiLU(gate) * up in-place: gate[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
;
; Signature:
;   void asm_silu_hadamard(
;       float*       gate, // arg 0 (RCX): gate vector in-place (dim floats)
;       const float* up,   // arg 1 (RDX): up projection vector (dim floats)
;       int64_t      dim   // arg 2 (R8):  vector dimension
;   );
;===============================================================================

%define WIN64 1

section .rdata align=16
one_f32:
    dd 1.0

section .text
global asm_silu_hadamard
export asm_silu_hadamard

asm_silu_hadamard:
    push rbp
    mov  rbp, rsp
    push rbx
    push r12
    push r13
    sub  rsp, 32

%ifdef WIN64
    mov  rdi, rcx       ; RDI = gate
    mov  rsi, rdx       ; RSI = up
    mov  rcx, r8        ; RCX = dim
%endif

    xor  r10, r10       ; i = 0

.silu.loop:
    cmp  r10, rcx
    jge  .silu.done

    movss xmm0, [rdi + r10*4]   ; g_i
    vxorps xmm1, xmm1, xmm1
    vsubss xmm1, xmm1, xmm0     ; z = -g_i
    movss [rbp-4], xmm1

    ; Compute exp(-g_i) via x87 hardware:
    fld  dword [rbp-4]          ; ST(0) = z
    fldl2e                      ; ST(0) = log2(e), ST(1) = z
    fmulp st1, st0              ; ST(0) = z * log2(e)

    fld  st0
    frndint
    fsub st1, st0
    fxch st1
    f2xm1
    fld1
    faddp st1, st0
    fscale
    fstp st1                    ; ST(0) = exp(-g_i)

    fld1                        ; ST(0) = 1.0, ST(1) = exp(-g_i)
    faddp st1, st0              ; ST(0) = 1.0 + exp(-g_i)
    fstp dword [rbp-8]

    movss xmm2, [rbp-8]         ; XMM2 = 1 + exp(-g_i)
    vdivss xmm3, xmm0, xmm2     ; XMM3 = g_i / (1 + exp(-g_i))

    movss xmm4, [rsi + r10*4]   ; load u_i
    vmulss xmm3, xmm3, xmm4     ; silu(g_i) * u_i
    movss [rdi + r10*4], xmm3   ; store in-place

    inc  r10
    jmp  .silu.loop

.silu.done:
    add  rsp, 32
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret
