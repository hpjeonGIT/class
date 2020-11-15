section .text
global main
extern puts

main:
  push rdi
  push rsi
  sub rsp,8  ; align stack before call

  mov rdi,[rsi]
  call puts
  add rsp, 8
  pop rsi
  pop rdi

  add rsi, 8  ; point to next argument
  dec rdi
  jnz main

  ret
