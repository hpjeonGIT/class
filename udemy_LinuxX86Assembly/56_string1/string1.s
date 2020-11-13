#movsb: moving a single byte
#movsw: moving 2 bytes or 16 bits (word)
#movsl: moving 4 bytes or longword (doubleword)

#lea: Loading effective address

#movl $myval, %edi
#leal myval, %edi


.section .data
value1: .ascii "This is a test string.\n"

.section .bss
   .lcomm output,23

.section .text
.globl _start

_start:
  nop
  leal value1, %esi
  leal output, %edi
  movsb
  movsw
  movsl

  movl $1, %eax
  movl $0, %ebx
  int $0x80
