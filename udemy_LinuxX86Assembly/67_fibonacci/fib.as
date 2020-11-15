section .text
global main
extern printf

main:
  push rbx
  mov  ecx,50
  xor  rax, rax ; will hold the current number
  xor  rbx, rbx ; will hold the next number
  inc  rbx
top:
  push rax
  push rcx
  mov  rdi,format ; 1st parameter
  mov  rsi, rax   ; 2nd parameter
  xor  rax, rax

  call printf
  pop  rcx
  pop  rax

  mov  rdx, rax ; save the current number
  mov  rax, rbx ; next number = current number
  add  rbx, rdx ; get the new number
  dec  ecx
  jnz  top

format: db "%20ld",10,0
