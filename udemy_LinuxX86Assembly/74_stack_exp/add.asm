; sys_write (int fd, char * buf, length)
; 0=stdin, 1=stdout, 2=stderr

section .data
SYS_WRITE  equ  1
STD_OUT    equ  1
SYS_EXIT   equ  60
EXIT_CODE  equ  0

NEW_LINE   db 0xa
BAD_ARG_COUNT db "Program accepts 2 arguments", 0xa

section .text
global _start

; int main(argc, argv)
;[rsp] - top of stack = argc
;[rsp + 8]  argv[0]
;[rsp +16]  argv[1]

_start:
  pop   rcx       ; rcx = argc

  cmp   rcx,3     ; 0: name of exe 1: first arg. 2: second arg.
  jne   argcError

  add   rsp, 8    ; skip argv[0], which is the name of the exe.

  pop   rsi       ; rsi = argv[1]

  call str_to_int
  mov  r10, rax   ; arg1 in r10
  pop  rsi        ; rsi = argv[2]
  call str_to_int
  mov  r11,rax    ; arg2 in r11

  add  r10, r11
  ; converts to string
  mov  rax, r10
  xor  r12, r12
  jmp  int_to_str

argcError:
  mov   rax, SYS_WRITE
  mov   rdi, STD_OUT
  mov   rsi, BAD_ARG_COUNT
  mov   rdx, 45   ; message length
  syscall
  jmp   exitt


int_to_str:
  mov   rdx,0
  mov   rbx,10
  div   rbx
  add   rdx,48
  push  rdx
  inc   r12
  cmp   rax,0x0
  jne   int_to_str
  jmp   print

str_to_int:
  xor   rax,rax
  mov   rcx,10

top:
  cmp   [rsi], byte 0 ; checks the end of string
  je    done1

  mov   bl,[rsi]      ; moves the current char to bl
  sub   bl,48         ; convert to number
  mul   rcx           ; rax = rax * rcx
  add   rax, rbx      ; rax = rax + number in rbx (from bl)
  inc   rsi           ; gets next

  jmp   top

print:
  mov    rax,1
  mul    r12
  mov    r12,8
  mul    r12
  mov    rdx,rax
  mov    rax, SYS_WRITE
  mov    RDI, STD_OUT
  mov    rsi,rsp
  syscall
  mov    rax, SYS_WRITE
  mov    rdi, STD_OUT
  mov    rsi, NEW_LINE
  mov    rdx, 1
  syscall

exitt:
  mov    rax, SYS_EXIT
  mov    rdi, EXIT_CODE
  syscall


done1:
  ret
