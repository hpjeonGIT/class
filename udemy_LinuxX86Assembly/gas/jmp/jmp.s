.section .text
.globl _start
_start:
     nop
     movl $34, %eax
     jmp _bottom

     nop
     nop

_bottom:
    movl $56, %ecx
