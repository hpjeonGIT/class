.text
.globl factorial

factorial:
  cmp    $1, %rdi     # n<=1 ?
  jnbe   bottom       # if not, do recursive call at bottom
  mov    $1, %rax     #
  ret

bottom:
  push %rdi           # save n onto the stack
  dec  %rdi           # n--
  call factorial      # factorial (n-1)
  pop  %rdi           # restore n
  imul %rdi,%rax      # n*factorial(n-1)
  ret
