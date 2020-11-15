.section .data
value1: .ascii "This is a test of CMPS instructions"
value2: .ascii "That is a test of CMPS instructions"

.section .text
.globl _start
_start:
  nop
  leal value1, %esi
  leal value2, %edi
  movl $39, %ecx
  cld
  repe cmpsb
  je equal
  movl $1, %ebx
  int $0x80

equal:
  movl $0, %ebx
hr2: nop
  nop
  int $0x80
