.text
.globl _start
_start:
       mov $1, %rax  # System call number in rax
       mov $1, %rdi  # file handle 1 i.e stdout
       mov $message,%rsi
       mov $13, %rdx
       syscall
       mov $60, %rax  # syscall60 for exit
       xor %rdi,%rdi
       syscall

message:
    .ascii "Hello World\n"
