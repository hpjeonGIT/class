.section .data
string1: .ascii "This is a test - a long test string to scan"
length: .int 44
string2: .ascii "-"

.section .text
.globl _start
_start:
  nop
  leal string1, %edi
  leal string2, %esi
  movl length, %ecx
  lodsb
  cld
  repne scasb
  jne notfound

  movl $0, %ebx
hr1: nop
     nop

notfound:
  movl $1,%eax
  nop
  nop
