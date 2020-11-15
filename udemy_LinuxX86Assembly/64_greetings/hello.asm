; Calling C libraries using nasm
; nasm -felf64 hello.asm
; gcc hello.o -no-pie
; ./a.out

section .text
global main
extern puts

main:
  mov rdi, message
  call puts
  ret

message: db "Good day!",0
