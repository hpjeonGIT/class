.section .data
string1: .ascii "This is a Test, of the conversion program!\n"
length: .int 43

.section .text
.globl _start

_start:
    nop
    leal string1, %esi
    movl %esi, %edi
    movl length, %ecx
    CLD
top:
    LODSB
    cmpb $'a', %al
    jl skip
    cmpb $'z', %al
    jg skip
    subb $0x20,%al

skip:
  stosb
  loop top

exit:  nop
  nop
