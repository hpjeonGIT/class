.section .data
values:
  .int 10, 12, 6, 9, 13, 67, 5, 3, 45, 87

.section .text
.globl _start

_start:
    nop
    movl $values, %esi
    movl $9, %ecx
    movl $9, %ebx

loop:
    movl (%esi), %eax
    cmp   %eax, 4(%esi) # 4 for 4bytes of int
    jge   skip
    xchg  %eax, 4(%esi)
    movl  %eax, (%esi)

skip:
  add   $4, %esi
  dec   %ebx
  jnz   loop
  dec   %ecx
  jz    end
  movl  $values, %esi
  movl  %ecx, %ebx
  jmp loop

end:
  nop
  nop
