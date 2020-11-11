.section .text
.globl _start
_start:
    nop
    movl $15, %eax
    movl $130, %ebx
    movl $0,  %ecx

    cmp  %eax, %ebx
    jge  greater

tp1:
    nop
    movl $1, %eax
    int  $0x80

greater:
   nop
   addl $40, %ecx
   nop
   nop
