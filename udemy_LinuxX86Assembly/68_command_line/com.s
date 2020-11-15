# in main(int argc, char** argv)
# argc = %rdi, argv= %rsi
.text
.globl main

main:
  push %rdi       # save rdi
  push %rsi       # save rsi
  sub $8, %rsp    # align stack before call

  mov (%rsi),%rdi
  call puts

  add $8, %rsp   # restore rsp to pre-aligned value
  pop %rsi       # restore rsi
  pop %rdi       # restore rdi

  add $8, %rsi   # point to next argument
  dec %rdi       # decrement the number of arguments
  jnz  main
  
  ret
