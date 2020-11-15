.text
.globl main

main:
  push %rbx
  mov  $50,  %ecx  # will countdown
  xor  %rax, %rax # will hold current number
  xor  %rbx, %rbx # will hold next number
  inc  %rbx
top:
  push %rax
  push %rcx
  mov  $format, %rdi # 1st argument
  mov  %rax, %rsi    # 2nd argument
  xor  %rax, %rax
  call printf        # printf(format, current_number)

  pop %rcx
  pop %rax

  mov %rax, %rdx  # save current number
  mov %rbx, %rax  # next number = current number
  add %rdx, %rbx  # get new number
  dec %ecx        #
  jnz top

  pop %rbx
  ret

format:
  .asciz "%20ld\n"
