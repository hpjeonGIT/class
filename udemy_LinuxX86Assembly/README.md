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

## Section 9.37
- Index addressing in AT&T syntax
- base_address(offset_address, index, size)
```
movl $2, %edi
movl values(, %edi, 4), %eax
```

## Section 10.38
- JE : if equal
- JZ : if zero
- JNE : if not equal
- JNZ : if not zero
- JG : if the first operand is greater than second
- JGE : if the first operand is greater or equal to second

## Section 10.39
- Conditional branching
- JMP destination

## Section 10.40
- Logic instructions
- AND destination, source
- OR destination, source
- CMP destination, source

## Section 10.42
- cd ConditionalJump
- as -gstabs -o jump.o jump.s
- ld -o jump jump.o
- As ebx is greater than eax, tp1: is skipped and goes to greater:
- In gdb, use `info register` to see the current value of registers such as rax, rbx, rcx, ...

## Section 10.43
- jmp _bottom will go to _bottom: section, without running the left-over in the _start: section
- cd jmp
- as -gstabs -o jmp.o jmp.s
- ld -o jmp jmp.o

## Section 10.44
- CALL jump
- jmp just go to _bottom: and no return to start. CALL will return to _start to continue the left-over (below of call _bottom)
- gdb doesn't show the section of _bottom: while results are reflected in the register
  - _bottom: section is executed as one line

## Section 10.45
- Instruction operands
| Type    |   Example            | C/C++              |
|---------|----------------------|--------------------|
|Immediate| mov eax,42           | eax=42             |
|         | imul ebx,11h         | ebx *=0x11         |
|         | xor dl,55h           | dl ^=0x55          |
|         | add esi,8            | esi += 8           |
|---------|----------------------|--------------------|
|Register | mov eax,ebx          | eax = ebx          |
|         | inc ecx              | ecx +=1            |
|         | add ebx,esi          | ebx +=esi          |
|         | mul ebx              | ebx:eax = eax*ebx  |
|---------|----------------------|--------------------|
| Memory  | mov eax,[ebx]        | eax =*ebx          |
|         | add eax,[val1]       | eax +=*val1        |
|         | or ecx,[ebx+esi]     | ecx |= *(ebx+esi)  |
|         | sub word ptr[edi],12 | *(short*) edi -=12 |


## Section 11.46
- Inline assembly with C/C++
- Using __asm__() function in the C code
- gcc -no-pie main.c

## Section 11.47
- Single% for operands. %% for registers.
- Extended assembly
  - a: eax, rax, ax, al
  - b: ebx, rbx, bx, bl
  - c: ecx, rcx, cx, cl
  - __asm__("assembly code": "=a"(result): "d"(data1), "c"(data2))
    - This means : edx = data1, ecx=data2, result=eax

## Section 11.48
- Inline assembly extended format with placeholders
```
__asm__("assembly code... using %1,%2,%0"
      : "=r"(result)
      : "r"(data1), "r"(data2));
```
  - %0 will represent reg for result
  - %1 will represent reg for data1
  - %2 will represent reg for data2

## Section 11.49-51
- Referencing placeholders
- %1 => %[value1] as alternate placeholders

## Section 11.52
- Using memory location
```
__asm__(
  "divb %2\n\t"
  "movl %%eax, %0"
  :"=m"(result)
  :"a"(dividend),"m"(divisor)
);
```

## Section 11.53
- Using FPU
  - f: any floating point reg
  - t: top floating point reg
  - u: second floating point reg
```
__asm__(
  "fsincos"
  :"=t"(cosine),"=u"(sine)
  :"0"(radian)
);
```
  - fsincos: computes sine/cosine of the source operand in register ST(0), which is a stack, then stores the sine in ST(0) and pushes cosine onto the top of FPU register stack. This is faster than executing FSIN and FCOS sequentially

## Section 11.54
- Jump in the Assembly
```
__asm__(
  "cmp %1, %2\n\t"
  "jge greater\n\t"
  "movl %1, %0\n\t"
  "jmp end\n"
  "greater: \n\t"
  "movl %2,%0\n"
  "end:"
  :"=r"(result)
  :"r"(a), "r"(b)
);
```

## Section 11.55
- Assembly MACRO
```
#define GREATER(a,b,result) ({\
  __asm__( \
          "cmp %1, %2\n\t" \
          "jge 0f\n\t" \
          "movl %1,%0\n\t" \
          "jmp 1f\n" \
          "0: \n\t" \
          "movl %2, %0\n\t" \
          "1: " \
          :"=r"(result) \
          :"r"(a), "r"(b) \
        ); \
})
...
   GREATER(data1,data2,result);
```

## Section 12.56
- Moving bytes of strings
- movsb: moving a single byte
- movsw: moving 2 bytes or 16 bits (word)
- movsl: moving 4 bytes or longword (doubleword)
```
25	  movsw
(gdb) x/s &output
0x6000e8 <output>:	"T"
(gdb) n
26	  movsl
(gdb) x/s &output
0x6000e8 <output>:	"Thi"
(gdb) n
28	  movl $1, %eax
(gdb) x/s &output
0x6000e8 <output>:	"This is"
```

## Section 12.57
- Moving entire strings
```
_load:
  movsb
  loop _load
```
- When entire bytes are moved, loop _load will exit by default (?)

## Section 12.58
- Load string with REP instruction
- MOVS (copy mem to mem)
- STOS (store AL/AX/EAX/RAX)
- SCAS (scan string)
- CMPS (compare string)
- LODS (load string)
- CLD: clears the direction flag as 0, and instructions work by incrementing the pointer to the data
- STD: Sets the Direction flag as 0, going backwards
- REP: Repeat String operation
  - `rep movsb`
