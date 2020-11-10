# Title
- Linux x86 Assembly Language Programming from Ground Up

#Tools
- sudo apt install nasm
- sudo snap install atom --classic
  - Atom package: gpp-compiler, language-nasmx86

# Section 2
## steps
- Assembler
- Linker
## Command
- nasm -felf64 hello.asm
- ld hello.o
- ./a.out

# Section 5
- Instruction execution sequence: fetch -> decode -> execute
- X86 operating mode
  - Protected mode
  - Real-Address mode
  - System management mode
- 8086 Registers  
  - General Purpose Registers (GPR): AH/AL, BH/BL, CH/CL, DH/DL
  - Index Registers: SI(source index), DI, DP, SP(stack pointer)
  - Instruction Pointer: IP
  - Segment Registers: CS(code seg), DS(data seg), ES(extra seg), SS(stack seg)
  - Flags Register: Flags
- 80386
  - GPR: EAX, EBX, ECX, EDX, EBP, ESP, ESI, EDI
- x64
  - GPR: RAX, RBX, RCX, RDX, RBP, RSP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15
  - RIP
  - RFLAGS
- X86 Flags
  - Overflow (OF)
  - Sign (SF)
  - Zero (ZF)
  - Carry (CF)
  - Parity (PF)
  - Auxiliary Carry (AC)
- MMX Registers: 8 of 64 bit registers
- XMM Registers: 8 of 128 bit Registers

## Section 6.17
- Make gas/add/add.s
- Assemble: as -gstabs -o add.o add.s
- Link: ld -o add add.o
- Debugging: gdb -q add
  - b *_start
  - r
  - n # x5 repetition
  - p $eax # if n x4, 0 is printed as movb $20,%al is not executed yet

## Section 6.19
- Directives
  - .CODE: start of a code Segment
  - .DATA: start of a data Segment
  - .STACK: start of a stack Segment
  - .END: end of a module
  - .DD: allocates a double storage
  - .DWORD: same as .DD
- [label:] mnemonic [operands] [;comment]
  - Ex) start : mov eax,10000h ; EAX=10000h

## Section 6.20
- Big Endian: offset by significant digits
  - First digit to the first offset
- Little Endian: opposite of Big Endian
  - Last digit to the first offset

## Section 6.21
```
.section .text
.globl  _start
_start:
      nop
      movl $0x12345678, %ebx
      bswap %ebx
      nop
      nop
```
- as -gstabs -o endian.o endian.s
- ld -o endian endian.o
- gdb -q endian
  - b *_start
  - r
  - n # x3
  - info Registers
```
rbx            0x78563412	2018915346
```
  - ebx=> rbx register. 78 is stored in the first offset then 56 is on the second. 34 for third and 12 on the last offset.

## Section 6.22
- Call conventions
- x86_32, 1st-6th arguments: ebc, ecx, edx, esi ,edi, epb
- x86_64, 1st-6th arguments: rdi, rsi, rdx, rcx, r8, r9
- x87 (FP), 1st-6th arguments: xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6

## Section 6.23
- GAS Directives
  - .section data
  - .section text
  - .lcomm: local common section
  - .bss: block start by symbol
  - .globl: make label accessible by other files

## Section 6.24
- NASM Directives
  - .section .data
  - .section .text
  - BITS
  - SECTION/SEGMENT
  - EXTERN
  - COMMON
  - GLOBAL
  - STATIC
  - EQU

## Section 6.25
- Linux syscalls
```
section .data
   msg db  "hello, world!"
section .text
   global _start
_start:
  mov rax, 1  ; calls sys_write function
  mov rdi, 1  ; first argument, 1 for standard output
  mov rsi, msg ; 2nd argument
  mov rdx, 13  ; 3rd argument. count of bytes
  syscall
  mov rax, 60  ; calls sys_exit
  mov rdi, 0
  syscall
```
- a function size_t sys_write( unsigned int fd, const char * buf, size_t count)
  - fd -> rdi, 0 for standard input/1 for standard output/2 for standard error
  - buf -> rsi, points to a character array
  - count -> rdx, specifies the number of bytes to written
- Google linux syscall table for the lists

## Section 6.26
- Check gas/sys_call/hello.s
- gcc -c hello.s -no-pie
- ld -o a.out hello.o

## Section 7.27
- Types of operand types
  - immediate: imm, imm8, imm16, imm32
  - registers: reg8, reg16, reg32, reg, sreg
  - memory: mem, mem8, mem16, mem32

## Section 7.28
- MOV instruction
  - MOV destination, source
```
.data
count WORD 1
.code
mov ecx, 0
mov cx, count
```
  - MOVZX, mov with zero-extend
```
.data
byteVal BYTE 10001111b
.code
movzx ax,byteVal ; AX=0000000010001111b
```
  - MOVSX, mov with sign-extend
```
.data
byteVal BYTE 10001111b
.code
movsx ax,byteVal ; AX=1111111110001111b
```

## Section 7.30
- Memory addressing modes
  - Effective Address = BaseReg + IndexReg * ScaleFactor + Disp
  - Ex) mov eax, [MyArray+esi*4]
  - Ex) mov eax, [ebx+esi+12]

## Section 8.31
- Arithmetic Instruction
  - ADD, SUB, MUL, IMUL, DIV, IDIV, INC, DEC, IndexReg

## Section 8.32
- Increment/decrement: INC/DEC reg/mem
```
.data
myWord WORD 1000h
.code
inc myWord ; myWord = 1001h
mov bx, myWord
dec bx ; bx = 1000h
```

## Section 8.33
- ADD/SUB dest, source
```
.data
var1 DWORD 3000h
var2 DWORD 1000h
.code
mov eax, var1 ; eax = 3000h
sub eax, var2 ; eax = 2000h
```

## Section 8.34
- as -gstabs sub.o sub.s
- ld -o sub sub.o
- gdb -q sub
  - b *_start
  - run

## Section 8.35
- OFFSET
- PTR
- TYPE
- LENGTHOF
- SIZEOF
- LABEL
- ALIGN

## Section 8.36
- sal: Shift Arithmetic Left
- gdb command:
  - x : displays memory contents
  - x/d: displays memory contents in decimal
  - Ex) x/d &value1

## Section 8.37
- Index addressing in AT&T syntax
- base_address(offset_address, index, size)
```
movl $2, %edi
movl values(, %edi, 4), %eax
```
