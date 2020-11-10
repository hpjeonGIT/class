.section .data
data: .int 40
.section .text
.globl _start
_start:
      nop
      movl $0, %eax
      movl $0, %ebx
      movb $25, %al
      subb $10, %al # subtract by byte. Will yield 15
      movsx %al, %eax

      movw  $100, %cx
      subw  %cx, %bx
      movsx %bx, %ebx # p $bx will be -100
      movl  $100,%edx
      subl  %eax,%edx # p $edx will be 85
      subl  data, %eax
      subl  %eax, data

      movl  $1, %eax
      movl  $0, %ebx
      int   $0x80
