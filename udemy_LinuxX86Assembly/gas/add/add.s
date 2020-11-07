
.section .data
data:
    .int 40

.section .text
.globl _start


_start:
      nop
      movl $0,%eax
      movl $0,%ebx
      movl $0,%ecx
      movb $20,%al
      addb $10,%al
      addb $30,%al
      movl $122,%ebx
      addl data,%ebx
      nop
      nop
