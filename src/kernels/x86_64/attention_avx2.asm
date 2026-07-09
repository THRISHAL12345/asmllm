;===============================================================================
; asmllm - Hand-written x86-64 assembly kernel for Single-Head Attention
; Compliance: Zero C, Zero Intrinsics, AVX2
;===============================================================================
;
; Function Header Block:
; Function: asm_attention
; One-line description:
;   Computes single-head scaled dot-product attention y = softmax(Q@K^T * scale)@V
;
; Signature:
;   void asm_attention(
;       const float* q,        // arg 0 (RCX): query matrix (seq_len x head_dim)
;       const float* k,        // arg 1 (RDX): key matrix (seq_len x head_dim)
;       const float* v,        // arg 2 (R8):  value matrix (seq_len x head_dim)
;       float*       y,        // arg 3 (R9):  output matrix (seq_len x head_dim)
;       int64_t      seq_len,  // arg 4 ([RBP+48]): sequence length N
;       int64_t      head_dim, // arg 5 ([RBP+56]): head dimension D
;       float        scale     // arg 6 ([RBP+64]): 1/sqrt(D) scale float
;   );
;===============================================================================

%define WIN64 1

extern asm_softmax

section .text
global asm_attention
export asm_attention

asm_attention:
    push rbp
    mov  rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    ; Allocate dynamic scratch buffer on stack for attention scores (seq_len floats)
    ; Stack must remain 16-byte aligned
    mov  rax, [rbp+48]          ; RAX = seq_len
    shl  rax, 2                 ; seq_len * 4 bytes
    add  rax, 31
    and  rax, ~31               ; Align to 32 bytes
    sub  rsp, rax
    mov  r15, rsp               ; R15 = pointer to scratch scores buffer

%ifdef WIN64
    mov  rdi, rcx               ; RDI = q
    mov  rsi, rdx               ; RSI = k
    mov  rdx, r8                ; RDX = v
    mov  rcx, r9                ; RCX = y
    mov  r8,  [rbp+48]          ; R8  = seq_len
    mov  r9,  [rbp+56]          ; R9  = head_dim
    vmovss xmm14, dword [rbp+64] ; XMM14 = scale
%endif

    ; R10 = outer loop index i (query row 0 .. seq_len-1)
    xor  r10, r10

.attn.row_loop:
    mov  r8,  [rbp+48]          ; R8  = seq_len
    mov  r9,  [rbp+56]          ; R9  = head_dim
    vmovss xmm14, dword [rbp+64] ; XMM14 = scale
    cmp  r10, r8
    jge  .attn.done

    ; --- Step 1: Compute scores[j] = (q[i] . k[j]) * scale for j = 0..seq_len-1 ---
    xor  r11, r11               ; R11 = key row j

.attn.score_loop:
    cmp  r11, r8
    jge  .attn.score_done

    ; Compute dot product of q[i] and k[j] across head_dim elements
    vxorps ymm0, ymm0, ymm0     ; dot accumulator

    mov  rax, r10
    imul rax, r9                ; i * head_dim
    shl  rax, 2                 ; * 4 bytes
    lea  rbx, [rdi + rax]       ; RBX = pointer to q[i]

    mov  rax, r11
    imul rax, r9
    shl  rax, 2
    lea  r12, [rsi + rax]       ; R12 = pointer to k[j]

    xor  r13, r13               ; d = 0
    mov  r14, r9
    shr  r14, 3                 ; head_dim / 8 blocks

.attn.dot_loop:
    cmp  r13, r14
    jge  .attn.dot_remainder
    mov  rax, r13
    shl  rax, 5                 ; d * 32 bytes
    vmovups ymm1, [rbx + rax]
    vfmadd231ps ymm0, ymm1, [r12 + rax]
    inc  r13
    jmp  .attn.dot_loop

.attn.dot_remainder:
    ; Horizontal sum of YMM0 -> scalar XMM0
    vextractf128 xmm1, ymm0, 1
    vaddps       xmm0, xmm0, xmm1
    vpermilps    xmm1, xmm0, 0x0E
    vaddps       xmm0, xmm0, xmm1
    vpermilps    xmm1, xmm0, 0x01
    vaddss       xmm0, xmm0, xmm1

    mov  rax, r14
    shl  rax, 3                 ; processed d
.attn.tail_dot_loop:
    cmp  rax, r9
    jge  .attn.store_score
    vmovss xmm1, [rbx + rax*4]
    vfmadd231ss xmm0, xmm1, [r12 + rax*4]
    inc  rax
    jmp  .attn.tail_dot_loop

.attn.store_score:
    vmulss xmm0, xmm0, xmm14    ; multiply dot product by scale
    vmovss [r15 + r11*4], xmm0  ; scores[j] = dot * scale

    inc  r11
    jmp  .attn.score_loop

.attn.score_done:
    ; --- Step 2: Apply Softmax to scores[0..seq_len-1] in scratch buffer R15 ---
    ; Save caller-saved/kernel registers around asm_softmax call
    push rdi
    push rsi
    push rdx
    push rcx
    push r8
    push r9
    push r10
    sub  rsp, 32                ; MSVC 32-byte shadow space

    mov  rcx, r15               ; arg 0: x = scores
    mov  rdx, r15               ; arg 1: y = scores (in-place)
    mov  r8,  [rbp+48]          ; arg 2: dim = seq_len
    call asm_softmax

    add  rsp, 32
    pop  r10
    pop  r9
    pop  r8
    pop  rcx
    pop  rdx
    pop  rsi
    pop  rdi

    ; --- Step 3: Compute output row y[i] = sum_j (scores[j] * v[j]) ---
    ; Zero initialize y[i, 0..head_dim-1]
    mov  rax, r10
    imul rax, r9
    shl  rax, 2
    lea  rbx, [rcx + rax]       ; RBX = pointer to y[i]

    xor  rax, rax
.attn.zero_y:
    cmp  rax, r9
    jge  .attn.acc_v
    mov  dword [rbx + rax*4], 0
    inc  rax
    jmp  .attn.zero_y

.attn.acc_v:
    xor  r11, r11               ; j = 0 .. seq_len-1

.attn.v_loop:
    cmp  r11, r8
    jge  .attn.next_row

    vbroadcastss ymm1, [r15 + r11*4] ; YMM1 = broadcasted score[j]

    mov  rax, r11
    imul rax, r9
    shl  rax, 2
    lea  r12, [rdx + rax]       ; R12 = pointer to v[j]

    xor  r13, r13               ; d = 0
    mov  r14, r9
    shr  r14, 3                 ; head_dim / 8 blocks

.attn.v_block_loop:
    cmp  r13, r14
    jge  .attn.v_remainder
    mov  rax, r13
    shl  rax, 5
    vmovups ymm2, [rbx + rax]
    vfmadd231ps ymm2, ymm1, [r12 + rax] ; y[i] += score[j] * v[j]
    vmovups [rbx + rax], ymm2
    inc  r13
    jmp  .attn.v_block_loop

.attn.v_remainder:
    mov  rax, r14
    shl  rax, 3
.attn.tail_v_loop:
    cmp  rax, r9
    jge  .attn.v_next
    vmovss xmm2, [rbx + rax*4]
    vmovss xmm3, [r12 + rax*4]
    vfmadd231ss xmm2, xmm1, xmm3
    vmovss [rbx + rax*4], xmm2
    inc  rax
    jmp  .attn.tail_v_loop

.attn.v_next:
    inc  r11
    jmp  .attn.v_loop

.attn.next_row:
    inc  r10
    jmp  .attn.row_loop

.attn.done:
    vzeroupper
    lea  rsp, [rbp - 40]
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret
