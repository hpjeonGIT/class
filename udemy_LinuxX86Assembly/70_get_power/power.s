# argc = rdi
# x^y

.text
.globl main

main:
  # push 3 registers to align stack

  push %r12
  push %r13
  push %r14

  cmp $3, %rdi  # number of arguments must be 2 + 1 (name of executable)
  jne error1

  mov %rsi,%r12      # argv

  mov 16(%r12), %rdi # argv[2]
  call atoi          # y in eax
  cmp $0, %eax
  jl error2
  mov %eax, %r13d    # y in r13d

  mov 8(%r12),%rdi   # argv[1]
  call atoi          # x in eax
  mov %eax,%r14d     # x in r14d

  mov $1, %eax       # base of the multiplication
tpp:
  test %r13d, %r13d
  jz   found
  imul %r14d, %eax
  dec  %r13d
  jmp  tpp

found:
  mov    $answer, %rdi
  movslq %eax, %rsi
  xor    %rax, %rax
  call   printf
  jmp    done


error1:
  mov $wrongArgumentCount, %edi
  call puts
  jmp done

error2:
  mov $negativeExponent, %edi
  call puts
  jmp done

done:
  pop %r14
  pop %r13
  pop %r12
  ret

answer: .asciz "%d\n"

wrongArgumentCount: .asciz "Program needs 2 arguments\n"

negativeExponent: .asciz "Only positive number\n"
