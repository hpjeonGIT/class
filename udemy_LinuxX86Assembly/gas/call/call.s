.section .text
.globl _start
_start:
      nop
      movl  $65, %eax
      movl  $89, %ebx
      call  _bottom
      movl  $77, %ecx
      ret

_bottom:
      addl   $100, %eax
      addl   $50,  %ebx
      ret
