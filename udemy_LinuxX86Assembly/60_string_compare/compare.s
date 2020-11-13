.section .data
value1: .ascii "Test"
value2: .ascii "Tesx"

.section .text
.globl _start
_start:
  nop
  movl $1, %eax
  leal value1, %esi
  leal value2, %edi
  cld
  cmpsl
  je equal
  movl $1, %ebx
  int $0x80

equal:
    movl $0, %ebx
    int $0x80
    nop # check if %ebx is zero or one
