.section .data
int1: .int 43
int2: .int 4

.section .text
.globl _start

_start:
    nop
    movl $0, %eax
    movl $10, %ebx
    cmp %eax, %ebx
    cmovl %eax, %ebx
    ret
