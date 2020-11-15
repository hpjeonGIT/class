; power x y or x^y
section .text
extern  printf
extern  puts
extern  atoi
global  main

main:
  push r12
  push r13
  push r14

  cmp  rdi,3
  jne  error1

  mov  r12,rsi       ; argv

  mov  rdi,[r12+16]  ; argv[2]
  call atoi          ; y in eax
  cmp  eax,0         ;
  jl   error2
  mov  r13d, eax     ; y exponent in r13d

  mov  rdi,[r12+8]   ; argv[1]
  call atoi          ; x in eax
  mov  r14d, eax     ; x in r14d
  mov  eax, 1        ; base of multiplication

top:
  test r13d,r13d
  jz   found
  imul eax, r14d     ; multiply in another x
  dec  r13d
  jmp  top

found:
  mov    rdi, answer
  movsxd rsi, eax
  xor    rax, rax
  call   printf
  jmp    done

error1:
  mov  edi, wrongArgumentCount
  call puts
  jmp  done

error2:
  mov  edi, badExponentType
  call puts
  jmp  done

done:
  pop   r14
  pop   r13
  pop   r12
  ret

wrongArgumentCount: db "Program requires 2 arguments", 10, 0
badExponentType   : db "Only positive exponent", 10, 0

answer: db "%d", 10, 0
