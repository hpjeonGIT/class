.text

message:
      .asciz "Hello World"

.globl main

main:
    mov $message, %rdi
    call puts
    ret
