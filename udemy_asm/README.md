## Mastering x86-64 Real Assembly Language from Zero | ASM+2025
- Instructor: OCSALY Academy | 560.000+ Students

## Section 1: Writing our first 64BIT Assembly Program

### 1. If you know Assembly Language, Everything is Open Source

### 2. Why NASM is best and comparing it with other assemblers
- sudo apt install nasm

### 3. Hello world with makefile
- hello.asm:
```asm
section.data
  hello db 'Hello, World!',0; null terminated string
section .bss  
section .text
  global main
main:
 ;
 mov rax, 1 ; syscall number for sys_write
 mov rdi, 1 ; file descriptor 1 (stdout)
 mov rsi, hello
 mov rdx, 13 ; length of the string
 syscall
 ; exit
 mov rax,60
 mov rdi, 0
 syscall
```
 - Makefile:
```bash
hello: hello.o
  gcc -o hello hello.o -no-pie
hello.o: hello.asm
  nasm -f elf64 -g -F dwarf hello.asm -l hello.lst
```
- `make` produces hello.lst and hello executable
  - hello.lst: list file of recording the process of assembly to exe generation
- `main` is required instead of `_start` as it is compiled to an object file then linked to exe with crt0

### 4. Installing SASM

### 5. Sticking to traditions - Hello world program without makefile
- hello2.asm:
```asm
section .data
  hello db 'Hello, World!',0; null terminated string
section .bss  
section .text
  global _start ; Entry point
_start:
 ;
 mov rax, 1 ; syscall number for sys_write
 mov rdi, 1 ; file descriptor 1 (stdout)
 mov rsi, hello ; pointer to the string to write
 mov rdx, 13 ; length of the string
 syscall
 ; exit
 mov rax,60 ; syscall number for sys_exit
 mov rdi, 0 ; return code 0
 syscall    ; invoke the system call
```
- Building
  - nasm -f elf64 hello2.asm -o hello2.o
  - gcc -nostartfiles -o hello2.exe hello.o


## Section 2: Disaasembly and Disassembler

### 6. The Disassembly Theory
- First generation language
  - Machine language
- Second generation language
  - Assembly
  - Uses mnemonics
- Third generation language  
  - Pyhon, C, C#, ...

### 7. Disassembly - What
- Compilation is a many-to-many operation
- Decompilers are language and library dependent

## Section 3: Understanding Data Types

### 8. Understanding CPU architectures and Binaries

### 9. Converting Decimal to Binary with Basic Math

## Section 4: Introduction to Computer Engineering & Science

### 10. How Computer Communicates
- Modern computer components
  - CPU
  - Memory 
  - IO
    - keyboard/mouse
    - Disk
    - NIC
- Bus
  - Data bus: carries actual data
  - Control bus: carries instruction
  - Address bus: carries address

### 11. Preparing Dev Env

### 12. What happens when you use computer

## Section 5: Computer Arithmetic foir Beginners

### 13. How Addition Happens in Decimal
- Carry Flag: unsigned
- Overflow Flag: signed

### 14. Addition Arithmetic in Unsigned Integers

### 15. Substraction in Decimal

### 16. Substraction Arithmetic in Unsigned Integers
- Carry Flag when negative

### 17. Substraction Arithmetic in Signed Integers (Negative Numbers in Binary)

## Section 6: Boolean Algebra

### 18. Boolean Algebra for Low Level Computing

## Section 7: Electronics

### 19. Introduction to Electronics for Hardware Engineers
- 1 Coulomb/ 1sec = 1 Ampere

### 20. Logic Gates and Their Hardware Implementation
- Switch: 1 or zero
- Resistor: limits flow and creates heat
- Capacitor: stores energy in an electric field and resists certain voltage changes
- Inductor: stores energy in an magnetic field and resists certain current changes

### 21. Capacitors and Implementation in Circuits
- Capacitor: small energy tank. Releases energy when needed
  - Smooths out sudden change
  - $ V(t)   = {1 \over C} \int_0^t I(t) dt$

### 22. Why use Inductors on Motherboards at all
- Inductor: resists current changes
  - Smooths out current into CPU

### 23. Power Consumption and Transistors

### 24. MOSFET transistors

### 25. CMOS Switch in CPU's

## Section 8: Combinational Logic Circuits

### 26. Introduction to Logic Circuits
- LSB: Least Significant Bit 
  - Right most bit

### 27. Full Adders
| x_i | y_i |Carry_(i+1) | Sum_i |
|----|----|------|-----|
| 0  |  0 |  0   |  0  |
| 0  |  1 |  0   |  1  |
| 1  |  0 |  0   |  1  |
| 1  |  1 |  1   |  0  |

- A half adder: has two inputs (x + y)
- Full adder: has three inputs (carry + x + y)

### 28. NAND and NOR
- You can build any logic gates using NAND and NOR
- Universal gates


## Section 9: Introduction to Ghidra

### 29. Understanding how Ghidra works and other important steps to do
- Reverse engineering framework by NSA
  - Project based system
- sudo snap install ghidra
- Make a new project
  - Import a.out
```
Project File Name: 	a.out
Last Modified:	Wed Dec 31 14:10:10 EST 2025
Readonly:	false
Program Name:	a.out
Language ID:	x86:LE:64:default (4.6)
Compiler ID:	gcc
Processor:	x86
Endian:	Little
Address Size:	64
Minimum Address:	00100000
Maximum Address:	_elfSectionHeaders::000007bf
# of Bytes:	6470
# of Memory Blocks:	33
# of Instructions:	8
# of Defined Data:	109
# of Functions:	14
# of Symbols:	48
# of Data Types:	38
# of Data Type Categories:	2
Created With Ghidra Version:	12.0
Date Created:	Wed Dec 31 14:10:10 EST 2025
ELF File Type:	shared object
ELF GNU Program Prop[processor opt 0xc0000002]:	03 00 00 00
ELF GNU Program Prop[processor opt 0xc0008002]:	01 00 00 00
ELF Note[GNU BuildId]:	ce69e228b62365b698bac3bf837cb1c5668a8079
ELF Note[required kernel ABI]:	Linux 3.2.0
ELF Original Image Base:	0x0
ELF Prelinked:	false
ELF Source File [   0]:	Scrt1.o
ELF Source File [   1]:	crtstuff.c
ELF Source File [   2]:	hello.c
ELF Source File [   3]:	crtstuff.c
ELF Source File [   4]:	
Elf Comment[0]:	GCC: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Executable Format:	Executable and Linking Format (ELF)
Executable Location:	/home/hpjeon/hw/class/udemy_asm/section9/a.out
Executable MD5:	5d4af9c476d49e248af667c03a8be6e7
Executable SHA256:	b832f925aa05706c5d467a03b2f56c6a1c21aed986e6bc3d0c6c22e26cd8ad9b
FSRL:	file:///home/hpjeon/hw/class/udemy_asm/section9/a.out?MD5=5d4af9c476d49e248af667c03a8be6e7
Preferred Root Namespace Category:	
Relocatable:	true
Required Library [    0]:	libc.so.6
```

### 30. Ghidra Analyzers and User Interface

### 31. Customizing Ghidra and Graphs

### 32. Getting Familiar with Ghidra Top Menu Bar

## Section 10: Introduction to Low Level computing

### 33. Introduction to Low Level Computing

### 34. From Characters to Bits
- UTF-8: 2 hexadecimal digits or 8 bits

### 35. Creating our Instruction Set Architecture CPU
- How CPU knows which bits are instructions or data?
- ISA (Instruction Set Architecture)

| operation   | opcode  | Mnemonic |
|-------------|---------|----------|
| Addition    | 0001110 | ADD |
| Subtraction | 0001111 | SUB |
| Load        | 0001000 | LOAD|

- 7bits for opcode
- 3bits for immediate value
- 3bits for registers (Rn)
- 3bits for destination registers (Rd)

## Section 11: Introduction to Reverse Engineering

### 36. Reverse Engineering & Malware Analysis

### 37. 5 Important Steps in Reverse Engineering and Malware Analysis
1. Seeking Approval
2. Static Analysis
    - EXE, DLL, ELF, x86, ARM, API calls, ...
3. Dynamic Analysis
4. Low Level Analysis
5. Reporting

### 38. Virtualization Environment for our Work
- VMware or VBox

### 39. Using Builtin Tools for System Analysis
- Windows: registry

## Section 12: Assembly Language

### 40. Registers
- GPR (General Purpose Registers): arithmetics, logic, memory access
- Segment registers: manage segment memory addressing
- Flags registers: tracks results and CPU states after operation
- Instruction pointers: IP, EPI, RIP, shows where the current instruction is
- Original X86 used 16bit GPR
  - AX: accumulator, AH, AL
  - BX: base address, BH, BL
  - CX: Control register for loop
  - DX: Data register
  - SI: source index
  - DI: destination index
  - SP: stack pointer
  - BP: base pointer
- By IA-32, 32bit is introduced
  - EAX
  - EBX
  - ECX
  - EDX
  - ESI
  - EDI
  - ESP
  - EBP
- By x86_64
  - RAX
  - RBX
  - RCX
  - RDX
  - R8
  - ...
  - R15
  - RSP
  - RBP

### 41. Flags

| Offset | Abbreviation| Description |
|--------|-------------|-------------|
| 0      | CF          | Carry Flag  |
| 1      | -           | Reserved  |
| 2      | PF          | Parity Flag  |
| 3      | -           | Reserved  |
| 4      | AF          | Adjust Flag  |
| 6      | ZF          | Zero Flag  |
| 7      | SF          | Sign Flag  |
| 8      | TF          | Trap Flag   |
| 9      | IF          | Interrup Flag  |
| 10     | DF          | Direction Flag  |
| 11     | OF          | Overflow Flag  |

- TF enables a single step debugging

### 42. Memory Addressing and Endianess
- Big endian: most signficant bit
- Little endian: least signficant bit

### 43. Read and Understand Assembly
```
label/addr   mnemonic   operands               | comment
00A92AB7     MOV        EAX, dword ptr[0A9BFA] ; comment
```

### 44. Opcodes
- Operation codes
  - Machine code

### 45. Manipulating Memory
- MOV instruction

## Section 13: Debugging x86-64

### 46. Starting gdb and setting flavors
- Using hello.asm from Section 1
```bash
$ gdb ./hello
(gdb) list
1	section .data
2	  hello db 'Hello, World!',0; null terminated string
3	section .bss  
4	section .text
5	  global main
6	main:
7	 ;
8	 mov rax, 1
9	 mov rdi, 1
10	 mov rsi, hello
(gdb) 
```

### 47. Debugging and Finding Variables in Memory addresses
```bash
(gdb) disassemble main
Dump of assembler code for function main:
   0x0000000000401110 <+0>:	mov    $0x1,%eax
   0x0000000000401115 <+5>:	mov    $0x1,%edi
   0x000000000040111a <+10>:	movabs $0x404010,%rsi
   0x0000000000401124 <+20>:	mov    $0xd,%edx
   0x0000000000401129 <+25>:	syscall
   0x000000000040112b <+27>:	mov    $0x3c,%eax
   0x0000000000401130 <+32>:	mov    $0x0,%edi
   0x0000000000401135 <+37>:	syscall
End of assembler dump.
(gdb) set disassembly-flavor intel
(gdb) disassemble main
Dump of assembler code for function main:
   0x0000000000401110 <+0>:	mov    eax,0x1
   0x0000000000401115 <+5>:	mov    edi,0x1
   0x000000000040111a <+10>:	movabs rsi,0x404010 # "Hello, World!"
   0x0000000000401124 <+20>:	mov    edx,0xd
   0x0000000000401129 <+25>:	syscall
   0x000000000040112b <+27>:	mov    eax,0x3c
   0x0000000000401130 <+32>:	mov    edi,0x0
   0x0000000000401135 <+37>:	syscall
End of assembler dump.
(gdb) x/s 0x404010 # s for string
0x404010 <hello>:	"Hello, World!"
(gdb) x/c 0x404010 # c for character
0x404010 <hello>:	72 'H'
(gdb) x/c 0x404011
0x404011:	101 'e'
(gdb) x/c 0x404012
0x404012:	108 'l'
(gdb) x/c 0x404013
0x404013:	108 'l'
(gdb) x/c 0x404014
0x404014:	111 'o'
(gdb) x/5c 0x404010
0x404010 <hello>:	72 'H'	101 'e'	108 'l'	108 'l'	111 'o'
```

### 48. Learning more with GDB
```bash
(gdb) break main
Breakpoint 1 at 0x401110: file hello.asm, line 8.
(gdb) run
Breakpoint 1, main () at hello.asm:8
8	 mov rax, 1
(gdb) info reg
rax            0x401110            4198672
rbx            0x7fffffffd8a8      140737488345256
rcx            0x403e40            4210240
rdx            0x7fffffffd8b8      140737488345272
rsi            0x7fffffffd8a8      140737488345256
rdi            0x1                 1
rbp            0x7fffffffd820      0x7fffffffd820
rsp            0x7fffffffd788      0x7fffffffd788
r8             0x0                 0
r9             0x7ffff7fca380      140737353917312
r10            0x7fffffffd4a0      140737488344224
r11            0x203               515
r12            0x1                 1
r13            0x0                 0
r14            0x403e40            4210240
r15            0x7ffff7ffd000      140737354125312
rip            0x401110            0x401110 <main>
eflags         0x246               [ PF ZF IF ]
cs             0x33                51
ss             0x2b                43
ds             0x0                 0
es             0x0                 0
fs             0x0                 0
gs             0x0                 0
k0             0x400040            4194368
k1             0x22001             139265
k2             0x0                 0
k3             0x0                 0
k4             0x0                 0
k5             0x0                 0
k6             0x0                 0
k7             0x0                 0
fs_base        0x7ffff7fa4740      140737353762624
gs_base        0x0                 0
(gdb) step
9	 mov rdi, 1
(gdb) i r
rax            0x1                 1  # now rax is "1"
...
rip            0x401115            0x401115 <main+5> #  rip has progressed
...
(gdb) c # completes
(gdb) b main
(gdb) r
Breakpoint 2, main () at hello.asm:8
8	 mov rax, 1
(gdb) p $rax
$1 = 4198672
(gdb) s
9	 mov rdi, 1
(gdb) p $rax
$3 = 1  # now updated as "1"
```

## Section 14: Writing our second 64Bit Assembly Program

### 49. Coding ASM file
- In .data section
  - 10: Line feed
  - 0: null character
- kicking.asm:
```asm
; kicking.asm
section .data
  msg1  db "Hello, World!",10,0 ; string with NL (new line) and 0
  msg1Len equ $-msg1-1          ; measure the length of msg
1, minus the 0
  msg2  db "Kicking and Alive!",10,0 ; string with NL and 0
  msg2Len equ $-msg2-1
  radius dq 357
  pi     dq 3.14 
section .bss
section .text
  global main
main:
  push rbp        ; function prologue
  mov rbp, rsp    ; function prologue
  mov rax, 1      ; 1 = write
  mov rdi, 1      ; 1 = to stdout
  mov rsi,msg1    ; string to display
  mov rdx,msg1Len ; length of the string
  syscall
  mov rax,1       ; 1 = write
  mov rdi,1       ; 1 = to stdout 
  mov rsi,msg2    ; string to display
  mov rdx,msg2Len ; length of the string
  syscall
  mov rsp,rbp     ; function epilogue
  pop rbp         ; function epilogue
  mov rax,60      ; 60 = exit
  mov rdi, 0      ; 0 = success exit code
  syscall
```

### 50. Analyzing Output with GDB and creating makefile
- Makefile:
```bash
kicking: kicking.o
	gcc -o kicking kicking.o -no-pie
kicking.o: kicking.asm
	nasm -f elf64 -g kicking.asm
```
- gdb:

```bash
$ gdb ./kicking 
(gdb) disassemble main
Dump of assembler code for function main:
   0x0000000000401110 <+0>:	push   %rbp
   0x0000000000401111 <+1>:	mov    %rsp,%rbp
   0x0000000000401114 <+4>:	mov    $0x1,%eax
   0x0000000000401119 <+9>:	mov    $0x1,%edi
   0x000000000040111e <+14>:	movabs $0x404010,%rsi
   0x0000000000401128 <+24>:	mov    $0xe,%edx
   0x000000000040112d <+29>:	syscall
   0x000000000040112f <+31>:	mov    $0x1,%eax
   0x0000000000401134 <+36>:	mov    $0x1,%edi
   0x0000000000401139 <+41>:	movabs $0x40401f,%rsi
   0x0000000000401143 <+51>:	mov    $0x13,%edx
   0x0000000000401148 <+56>:	syscall
   0x000000000040114a <+58>:	mov    %rbp,%rsp
   0x000000000040114d <+61>:	pop    %rbp
   0x000000000040114e <+62>:	mov    $0x3c,%eax
   0x0000000000401153 <+67>:	mov    $0x0,%edi
   0x0000000000401158 <+72>:	syscall
End of assembler dump.
(gdb) x/s 0x404010
0x404010 <msg1>:	"Hello, World!\n"
(gdb) x/s &msg1
0x404010 <msg1>:	"Hello, World!\n"
(gdb) x/s &msg2
0x40401f <msg2>:	"Kicking and Alive!\n"
(gdb) x/dw &radius
0x404033 <radius>:	357
(gdb) x/dw &pi
0x40403b <pi>:	1374389535 # as pi is floating num, print mode must be changed
(gdb) x/fg &pi
0x40403b <pi>:	3.1400000000000001 # now prints correctly
```
- .lst file: Human readable file
  - Update Makefile as:
```bash
kicking: kicking.o
	gcc -o kicking kicking.o -no-pie
kicking.o: kicking.asm
	nasm -f elf64 -g kicking.asm -l kicking.lst
```
  - Delete kicking and kicking.o then run make
```bash
$ more kicking.lst
     1                                  ; kicking.asm
     2                                  section .data
     3 00000000 48656C6C6F2C20576F-       msg1  db "Hello, World!",10,0 ; string with NL (new line) and 0
     3 00000009 726C64210A00       
     4                                    msg1Len equ $-msg1-1          ; measure the length of msg1, minus the 0
     5 0000000F 4B69636B696E672061-       msg2  db "Kicking and Alive!",10,0 ; string with NL and 0
     5 00000018 6E6420416C69766521-
     5 00000021 0A00               
     6                                    msg2Len equ $-msg2-1
     7 00000023 6501000000000000          radius dq 357
     8 0000002B 1F85EB51B81E0940          pi     dq 3.14 
     9                                  section .bss
    10                                  section .text
    11                                    global main
    12                                  main:
    13 00000000 55                        push rbp        ; function prologue
    14 00000001 4889E5                    mov rbp, rsp    ; function prologue
    15 00000004 B801000000                mov rax, 1      ; 1 = write
    16 00000009 BF01000000                mov rdi, 1      ; 1 = to stdout
    17 0000000E 48BE-                     mov rsi,msg1    ; string to display
    17 00000010 [0000000000000000] 
    18 00000018 BA0E000000                mov rdx,msg1Len ; length of the string
    19 0000001D 0F05                      syscall
    20 0000001F B801000000                mov rax,1       ; 1 = write
    21 00000024 BF01000000                mov rdi,1       ; 1 = to stdout 
    22 00000029 48BE-                     mov rsi,msg2    ; string to display
    22 0000002B [0F00000000000000] 
    23 00000033 BA13000000                mov rdx,msg2Len ; length of the string
    24 00000038 0F05                      syscall
    25 0000003A 4889EC                    mov rsp,rbp     ; function epilogue
    26 0000003D 5D                        pop rbp         ; function epilogue
    27 0000003E B83C000000                mov rax,60      ; 60 = exit
    28 00000043 BF00000000                mov rdi, 0      ; 0 = success exit code
    29 00000048 0F05                      syscall
```

## Section 15: Binary Analysis

### 51. Analysis of Binary and 4 Stages of Compilation
- Preprocessing
- Compiling
- Assemblying
- Linking

### 52. Preprocessing
```c
#include <stdio.h>
#define FORMAT_STRING "%s"
#define MESSAGE "Hello world!\n"
int main(int argc, char *argv[]){
  printf(FORMAT_STRING,MESSAGE);
  return 0;
}
```
- Preprocessing:
```bash
$ gcc -E -P myapp.c 
typedef long unsigned int size_t;
typedef __builtin_va_list __gnuc_va_list;
typedef unsigned char __u_char;
typedef unsigned short int __u_short;
typedef unsigned int __u_int;
typedef unsigned long int __u_long;
typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef signed short int __int16_t;
typedef unsigned short int __uint16_t;
typedef signed int __int32_t;
typedef unsigned int __uint32_t;
typedef signed long int __int64_t;
typedef unsigned long int __uint64_t;
typedef __int8_t __int_least8_t;
typedef __uint8_t __uint_least8_t;
typedef __int16_t __int_least16_t;
typedef __uint16_t __uint_least16_t;
typedef __int32_t __int_least32_t;
typedef __uint32_t __uint_least32_t;
typedef __int64_t __int_least64_t;
typedef __uint64_t __uint_least64_t;
typedef long int __quad_t;
typedef unsigned long int __u_quad_t;
typedef long int __intmax_t;
typedef unsigned long int __uintmax_t;
typedef unsigned long int __dev_t;
typedef unsigned int __uid_t;
typedef unsigned int __gid_t;
typedef unsigned long int __ino_t;
typedef unsigned long int __ino64_t;
typedef unsigned int __mode_t;
typedef unsigned long int __nlink_t;
typedef long int __off_t;
typedef long int __off64_t;
typedef int __pid_t;
typedef struct { int __val[2]; } __fsid_t;
typedef long int __clock_t;
typedef unsigned long int __rlim_t;
typedef unsigned long int __rlim64_t;
typedef unsigned int __id_t;
typedef long int __time_t;
typedef unsigned int __useconds_t;
typedef long int __suseconds_t;
typedef long int __suseconds64_t;
typedef int __daddr_t;
typedef int __key_t;
typedef int __clockid_t;
typedef void * __timer_t;
typedef long int __blksize_t;
typedef long int __blkcnt_t;
typedef long int __blkcnt64_t;
typedef unsigned long int __fsblkcnt_t;
typedef unsigned long int __fsblkcnt64_t;
typedef unsigned long int __fsfilcnt_t;
typedef unsigned long int __fsfilcnt64_t;
typedef long int __fsword_t;
typedef long int __ssize_t;
typedef long int __syscall_slong_t;
typedef unsigned long int __syscall_ulong_t;
typedef __off64_t __loff_t;
typedef char *__caddr_t;
typedef long int __intptr_t;
typedef unsigned int __socklen_t;
typedef int __sig_atomic_t;
typedef struct
{
  int __count;
  union
  {
    unsigned int __wch;
    char __wchb[4];
  } __value;
} __mbstate_t;
typedef struct _G_fpos_t
{
  __off_t __pos;
  __mbstate_t __state;
} __fpos_t;
typedef struct _G_fpos64_t
{
  __off64_t __pos;
  __mbstate_t __state;
} __fpos64_t;
struct _IO_FILE;
typedef struct _IO_FILE __FILE;
struct _IO_FILE;
typedef struct _IO_FILE FILE;
struct _IO_FILE;
struct _IO_marker;
struct _IO_codecvt;
struct _IO_wide_data;
typedef void _IO_lock_t;
struct _IO_FILE
{
  int _flags;
  char *_IO_read_ptr;
  char *_IO_read_end;
  char *_IO_read_base;
  char *_IO_write_base;
  char *_IO_write_ptr;
  char *_IO_write_end;
  char *_IO_buf_base;
  char *_IO_buf_end;
  char *_IO_save_base;
  char *_IO_backup_base;
  char *_IO_save_end;
  struct _IO_marker *_markers;
  struct _IO_FILE *_chain;
  int _fileno;
  int _flags2;
  __off_t _old_offset;
  unsigned short _cur_column;
  signed char _vtable_offset;
  char _shortbuf[1];
  _IO_lock_t *_lock;
  __off64_t _offset;
  struct _IO_codecvt *_codecvt;
  struct _IO_wide_data *_wide_data;
  struct _IO_FILE *_freeres_list;
  void *_freeres_buf;
  size_t __pad5;
  int _mode;
  char _unused2[15 * sizeof (int) - 4 * sizeof (void *) - sizeof (size_t)];
};
typedef __ssize_t cookie_read_function_t (void *__cookie, char *__buf,
                                          size_t __nbytes);
typedef __ssize_t cookie_write_function_t (void *__cookie, const char *__buf,
                                           size_t __nbytes);
typedef int cookie_seek_function_t (void *__cookie, __off64_t *__pos, int __w);
typedef int cookie_close_function_t (void *__cookie);
typedef struct _IO_cookie_io_functions_t
{
  cookie_read_function_t *read;
  cookie_write_function_t *write;
  cookie_seek_function_t *seek;
  cookie_close_function_t *close;
} cookie_io_functions_t;
typedef __gnuc_va_list va_list;
typedef __off_t off_t;
typedef __ssize_t ssize_t;
typedef __fpos_t fpos_t;
extern FILE *stdin;
extern FILE *stdout;
extern FILE *stderr;
extern int remove (const char *__filename) __attribute__ ((__nothrow__ , __leaf__));
extern int rename (const char *__old, const char *__new) __attribute__ ((__nothrow__ , __leaf__));
extern int renameat (int __oldfd, const char *__old, int __newfd,
       const char *__new) __attribute__ ((__nothrow__ , __leaf__));
extern int fclose (FILE *__stream) __attribute__ ((__nonnull__ (1)));
extern FILE *tmpfile (void)
  __attribute__ ((__malloc__)) __attribute__ ((__malloc__ (fclose, 1))) ;
extern char *tmpnam (char[20]) __attribute__ ((__nothrow__ , __leaf__)) ;
extern char *tmpnam_r (char __s[20]) __attribute__ ((__nothrow__ , __leaf__)) ;
extern char *tempnam (const char *__dir, const char *__pfx)
   __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__malloc__)) __attribute__ ((__malloc__ (__builtin_free, 1)));
extern int fflush (FILE *__stream);
extern int fflush_unlocked (FILE *__stream);
extern FILE *fopen (const char *__restrict __filename,
      const char *__restrict __modes)
  __attribute__ ((__malloc__)) __attribute__ ((__malloc__ (fclose, 1))) ;
extern FILE *freopen (const char *__restrict __filename,
        const char *__restrict __modes,
        FILE *__restrict __stream) __attribute__ ((__nonnull__ (3)));
extern FILE *fdopen (int __fd, const char *__modes) __attribute__ ((__nothrow__ , __leaf__))
  __attribute__ ((__malloc__)) __attribute__ ((__malloc__ (fclose, 1))) ;
extern FILE *fopencookie (void *__restrict __magic_cookie,
     const char *__restrict __modes,
     cookie_io_functions_t __io_funcs) __attribute__ ((__nothrow__ , __leaf__))
  __attribute__ ((__malloc__)) __attribute__ ((__malloc__ (fclose, 1))) ;
extern FILE *fmemopen (void *__s, size_t __len, const char *__modes)
  __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__malloc__)) __attribute__ ((__malloc__ (fclose, 1))) ;
extern FILE *open_memstream (char **__bufloc, size_t *__sizeloc) __attribute__ ((__nothrow__ , __leaf__))
  __attribute__ ((__malloc__)) __attribute__ ((__malloc__ (fclose, 1))) ;
extern void setbuf (FILE *__restrict __stream, char *__restrict __buf) __attribute__ ((__nothrow__ , __leaf__))
  __attribute__ ((__nonnull__ (1)));
extern int setvbuf (FILE *__restrict __stream, char *__restrict __buf,
      int __modes, size_t __n) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern void setbuffer (FILE *__restrict __stream, char *__restrict __buf,
         size_t __size) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern void setlinebuf (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int fprintf (FILE *__restrict __stream,
      const char *__restrict __format, ...) __attribute__ ((__nonnull__ (1)));
extern int printf (const char *__restrict __format, ...);
extern int sprintf (char *__restrict __s,
      const char *__restrict __format, ...) __attribute__ ((__nothrow__));
extern int vfprintf (FILE *__restrict __s, const char *__restrict __format,
       __gnuc_va_list __arg) __attribute__ ((__nonnull__ (1)));
extern int vprintf (const char *__restrict __format, __gnuc_va_list __arg);
extern int vsprintf (char *__restrict __s, const char *__restrict __format,
       __gnuc_va_list __arg) __attribute__ ((__nothrow__));
extern int snprintf (char *__restrict __s, size_t __maxlen,
       const char *__restrict __format, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 3, 4)));
extern int vsnprintf (char *__restrict __s, size_t __maxlen,
        const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 3, 0)));
extern int vasprintf (char **__restrict __ptr, const char *__restrict __f,
        __gnuc_va_list __arg)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 2, 0))) ;
extern int __asprintf (char **__restrict __ptr,
         const char *__restrict __fmt, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 2, 3))) ;
extern int asprintf (char **__restrict __ptr,
       const char *__restrict __fmt, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 2, 3))) ;
extern int vdprintf (int __fd, const char *__restrict __fmt,
       __gnuc_va_list __arg)
     __attribute__ ((__format__ (__printf__, 2, 0)));
extern int dprintf (int __fd, const char *__restrict __fmt, ...)
     __attribute__ ((__format__ (__printf__, 2, 3)));
extern int fscanf (FILE *__restrict __stream,
     const char *__restrict __format, ...) __attribute__ ((__nonnull__ (1)));
extern int scanf (const char *__restrict __format, ...) ;
extern int sscanf (const char *__restrict __s,
     const char *__restrict __format, ...) __attribute__ ((__nothrow__ , __leaf__));
extern int fscanf (FILE *__restrict __stream, const char *__restrict __format, ...) __asm__ ("" "__isoc99_fscanf") __attribute__ ((__nonnull__ (1)));
extern int scanf (const char *__restrict __format, ...) __asm__ ("" "__isoc99_scanf") ;
extern int sscanf (const char *__restrict __s, const char *__restrict __format, ...) __asm__ ("" "__isoc99_sscanf") __attribute__ ((__nothrow__ , __leaf__));
extern int vfscanf (FILE *__restrict __s, const char *__restrict __format,
      __gnuc_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 2, 0))) __attribute__ ((__nonnull__ (1)));
extern int vscanf (const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 1, 0))) ;
extern int vsscanf (const char *__restrict __s,
      const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__format__ (__scanf__, 2, 0)));
extern int vfscanf (FILE *__restrict __s, const char *__restrict __format, __gnuc_va_list __arg) __asm__ ("" "__isoc99_vfscanf")
     __attribute__ ((__format__ (__scanf__, 2, 0))) __attribute__ ((__nonnull__ (1)));
extern int vscanf (const char *__restrict __format, __gnuc_va_list __arg) __asm__ ("" "__isoc99_vscanf")
     __attribute__ ((__format__ (__scanf__, 1, 0))) ;
extern int vsscanf (const char *__restrict __s, const char *__restrict __format, __gnuc_va_list __arg) __asm__ ("" "__isoc99_vsscanf") __attribute__ ((__nothrow__ , __leaf__))
     __attribute__ ((__format__ (__scanf__, 2, 0)));
extern int fgetc (FILE *__stream) __attribute__ ((__nonnull__ (1)));
extern int getc (FILE *__stream) __attribute__ ((__nonnull__ (1)));
extern int getchar (void);
extern int getc_unlocked (FILE *__stream) __attribute__ ((__nonnull__ (1)));
extern int getchar_unlocked (void);
extern int fgetc_unlocked (FILE *__stream) __attribute__ ((__nonnull__ (1)));
extern int fputc (int __c, FILE *__stream) __attribute__ ((__nonnull__ (2)));
extern int putc (int __c, FILE *__stream) __attribute__ ((__nonnull__ (2)));
extern int putchar (int __c);
extern int fputc_unlocked (int __c, FILE *__stream) __attribute__ ((__nonnull__ (2)));
extern int putc_unlocked (int __c, FILE *__stream) __attribute__ ((__nonnull__ (2)));
extern int putchar_unlocked (int __c);
extern int getw (FILE *__stream) __attribute__ ((__nonnull__ (1)));
extern int putw (int __w, FILE *__stream) __attribute__ ((__nonnull__ (2)));
extern char *fgets (char *__restrict __s, int __n, FILE *__restrict __stream)
     __attribute__ ((__access__ (__write_only__, 1, 2))) __attribute__ ((__nonnull__ (3)));
extern __ssize_t __getdelim (char **__restrict __lineptr,
                             size_t *__restrict __n, int __delimiter,
                             FILE *__restrict __stream) __attribute__ ((__nonnull__ (4)));
extern __ssize_t getdelim (char **__restrict __lineptr,
                           size_t *__restrict __n, int __delimiter,
                           FILE *__restrict __stream) __attribute__ ((__nonnull__ (4)));
extern __ssize_t getline (char **__restrict __lineptr,
                          size_t *__restrict __n,
                          FILE *__restrict __stream) __attribute__ ((__nonnull__ (3)));
extern int fputs (const char *__restrict __s, FILE *__restrict __stream)
  __attribute__ ((__nonnull__ (2)));
extern int puts (const char *__s);
extern int ungetc (int __c, FILE *__stream) __attribute__ ((__nonnull__ (2)));
extern size_t fread (void *__restrict __ptr, size_t __size,
       size_t __n, FILE *__restrict __stream)
  __attribute__ ((__nonnull__ (4)));
extern size_t fwrite (const void *__restrict __ptr, size_t __size,
        size_t __n, FILE *__restrict __s) __attribute__ ((__nonnull__ (4)));
extern size_t fread_unlocked (void *__restrict __ptr, size_t __size,
         size_t __n, FILE *__restrict __stream)
  __attribute__ ((__nonnull__ (4)));
extern size_t fwrite_unlocked (const void *__restrict __ptr, size_t __size,
          size_t __n, FILE *__restrict __stream)
  __attribute__ ((__nonnull__ (4)));
extern int fseek (FILE *__stream, long int __off, int __whence)
  __attribute__ ((__nonnull__ (1)));
extern long int ftell (FILE *__stream) __attribute__ ((__nonnull__ (1)));
extern void rewind (FILE *__stream) __attribute__ ((__nonnull__ (1)));
extern int fseeko (FILE *__stream, __off_t __off, int __whence)
  __attribute__ ((__nonnull__ (1)));
extern __off_t ftello (FILE *__stream) __attribute__ ((__nonnull__ (1)));
extern int fgetpos (FILE *__restrict __stream, fpos_t *__restrict __pos)
  __attribute__ ((__nonnull__ (1)));
extern int fsetpos (FILE *__stream, const fpos_t *__pos) __attribute__ ((__nonnull__ (1)));
extern void clearerr (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int feof (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int ferror (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern void clearerr_unlocked (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int feof_unlocked (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int ferror_unlocked (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern void perror (const char *__s) __attribute__ ((__cold__));
extern int fileno (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int fileno_unlocked (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pclose (FILE *__stream) __attribute__ ((__nonnull__ (1)));
extern FILE *popen (const char *__command, const char *__modes)
  __attribute__ ((__malloc__)) __attribute__ ((__malloc__ (pclose, 1))) ;
extern char *ctermid (char *__s) __attribute__ ((__nothrow__ , __leaf__))
  __attribute__ ((__access__ (__write_only__, 1)));
extern void flockfile (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int ftrylockfile (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern void funlockfile (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int __uflow (FILE *);
extern int __overflow (FILE *, int);

int main(int argc, char *argv[]){
  printf("%s","Hello world!\n");
  return 0;
}
```

### 53. Compilation Phase
```bash
$ gcc -S -masm=intel myapp.c 
$ cat myapp.s 
	.file	"myapp.c"
	.intel_syntax noprefix
	.text
	.section	.rodata
.LC0:
	.string	"Hello world!"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	sub	rsp, 16
	mov	DWORD PTR -4[rbp], edi
	mov	QWORD PTR -16[rbp], rsi
	lea	rax, .LC0[rip]
	mov	rdi, rax
	call	puts@PLT
	mov	eax, 0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
```

### 54. Assembly Phase
```bash
$ gcc -c myapp.c
$ file myapp.o
myapp.o: ELF 64-bit LSB relocatable, x86-64, version 1 (SYSV), not stripped
```
- LSB: Least Significant Bit (Little Endian)

### 55. Linking Phase
```bash
$ gcc myapp.c
$ file ./a.out
./a.out: ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, BuildID[sha1]=fe53a8b23c061f1d1b695d3300c321c5f5d8e45e, for GNU/Linux 3.2.0, not stripped
```

## Section 16: Symbols, Stripped and Not Stripped Binaries

### 56. Using READELF for Viewing Symbolic Information
```bash
$ readelf --syms ./a.out 

Symbol table '.dynsym' contains 7 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND 
     1: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND _[...]@GLIBC_2.34 (2)
     2: 0000000000000000     0 NOTYPE  WEAK   DEFAULT  UND _ITM_deregisterT[...]
     3: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND puts@GLIBC_2.2.5 (3)
     4: 0000000000000000     0 NOTYPE  WEAK   DEFAULT  UND __gmon_start__
     5: 0000000000000000     0 NOTYPE  WEAK   DEFAULT  UND _ITM_registerTMC[...]
     6: 0000000000000000     0 FUNC    WEAK   DEFAULT  UND [...]@GLIBC_2.2.5 (3)

Symbol table '.symtab' contains 36 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND 
     1: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS Scrt1.o
     2: 000000000000038c    32 OBJECT  LOCAL  DEFAULT    4 __abi_tag
     3: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS crtstuff.c
     4: 0000000000001090     0 FUNC    LOCAL  DEFAULT   16 deregister_tm_clones
     5: 00000000000010c0     0 FUNC    LOCAL  DEFAULT   16 register_tm_clones
     6: 0000000000001100     0 FUNC    LOCAL  DEFAULT   16 __do_global_dtors_aux
     7: 0000000000004010     1 OBJECT  LOCAL  DEFAULT   26 completed.0
     8: 0000000000003dc0     0 OBJECT  LOCAL  DEFAULT   22 __do_global_dtor[...]
     9: 0000000000001140     0 FUNC    LOCAL  DEFAULT   16 frame_dummy
    10: 0000000000003db8     0 OBJECT  LOCAL  DEFAULT   21 __frame_dummy_in[...]
    11: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS myapp.c
    12: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS crtstuff.c
    13: 00000000000020f0     0 OBJECT  LOCAL  DEFAULT   20 __FRAME_END__
    14: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS 
    15: 0000000000003dc8     0 OBJECT  LOCAL  DEFAULT   23 _DYNAMIC
    16: 0000000000002014     0 NOTYPE  LOCAL  DEFAULT   19 __GNU_EH_FRAME_HDR
    17: 0000000000003fb8     0 OBJECT  LOCAL  DEFAULT   24 _GLOBAL_OFFSET_TABLE_
    18: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND __libc_start_mai[...]
    19: 0000000000000000     0 NOTYPE  WEAK   DEFAULT  UND _ITM_deregisterT[...]
    20: 0000000000004000     0 NOTYPE  WEAK   DEFAULT   25 data_start
    21: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND puts@GLIBC_2.2.5
    22: 0000000000004010     0 NOTYPE  GLOBAL DEFAULT   25 _edata
    23: 0000000000001174     0 FUNC    GLOBAL HIDDEN    17 _fini
    24: 0000000000004000     0 NOTYPE  GLOBAL DEFAULT   25 __data_start
    25: 0000000000000000     0 NOTYPE  WEAK   DEFAULT  UND __gmon_start__
    26: 0000000000004008     0 OBJECT  GLOBAL HIDDEN    25 __dso_handle
    27: 0000000000002000     4 OBJECT  GLOBAL DEFAULT   18 _IO_stdin_used
    28: 0000000000004018     0 NOTYPE  GLOBAL DEFAULT   26 _end
    29: 0000000000001060    38 FUNC    GLOBAL DEFAULT   16 _start
    30: 0000000000004010     0 NOTYPE  GLOBAL DEFAULT   26 __bss_start
    31: 0000000000001149    41 FUNC    GLOBAL DEFAULT   16 main  #<--------------
    32: 0000000000004010     0 OBJECT  GLOBAL HIDDEN    25 __TMC_END__
    33: 0000000000000000     0 NOTYPE  WEAK   DEFAULT  UND _ITM_registerTMC[...]
    34: 0000000000000000     0 FUNC    WEAK   DEFAULT  UND __cxa_finalize@G[...]
    35: 0000000000001000     0 FUNC    GLOBAL HIDDEN    12 _init
```
- From `31: 0000000000001149    41 FUNC    GLOBAL DEFAULT   16 main`, find the memory address

### 57. Revealing Contents of Object File
- Using objdump:
```bash
$ objdump -sj .rodata myapp.o # readonly data

myapp.o:     file format elf64-x86-64

Contents of section .rodata:
 0000 48656c6c 6f20776f 726c6421 00        Hello world!.   
$ objdump -M intel -d myapp.o

myapp.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <main>:
   0:	f3 0f 1e fa          	endbr64
   4:	55                   	push   rbp
   5:	48 89 e5             	mov    rbp,rsp
   8:	48 83 ec 10          	sub    rsp,0x10
   c:	89 7d fc             	mov    DWORD PTR [rbp-0x4],edi
   f:	48 89 75 f0          	mov    QWORD PTR [rbp-0x10],rsi
  13:	48 8d 05 00 00 00 00 	lea    rax,[rip+0x0]        # 1a <main+0x1a>
  1a:	48 89 c7             	mov    rdi,rax
  1d:	e8 00 00 00 00       	call   22 <main+0x22>
  22:	b8 00 00 00 00       	mov    eax,0x0
  27:	c9                   	leave
  28:	c3                   	ret
$ readelf --relocs myapp.o

Relocation section '.rela.text' at offset 0x1a8 contains 2 entries:
  Offset          Info           Type           Sym. Value    Sym. Name + Addend
000000000016  000300000002 R_X86_64_PC32     0000000000000000 .rodata - 4
00000000001e  000500000004 R_X86_64_PLT32    0000000000000000 puts - 4

Relocation section '.rela.eh_frame' at offset 0x1d8 contains 1 entry:
  Offset          Info           Type           Sym. Value    Sym. Name + Addend
000000000020  000200000002 R_X86_64_PC32     0000000000000000 .text + 0
```

### 58. Trying to Analyze Binary Executable
```bash
$ objdump -M intel -d a.out

a.out:     file format elf64-x86-64


Disassembly of section .init:

0000000000001000 <_init>:
    1000:	f3 0f 1e fa          	endbr64
    1004:	48 83 ec 08          	sub    rsp,0x8
    1008:	48 8b 05 d9 2f 00 00 	mov    rax,QWORD PTR [rip+0x2fd9]        # 3fe8 <__gmon_start__@Base>
    100f:	48 85 c0             	test   rax,rax
    1012:	74 02                	je     1016 <_init+0x16>
    1014:	ff d0                	call   rax
    1016:	48 83 c4 08          	add    rsp,0x8
    101a:	c3                   	ret

Disassembly of section .plt:

0000000000001020 <.plt>:
    1020:	ff 35 9a 2f 00 00    	push   QWORD PTR [rip+0x2f9a]        # 3fc0 <_GLOBAL_OFFSET_TABLE_+0x8>
    1026:	ff 25 9c 2f 00 00    	jmp    QWORD PTR [rip+0x2f9c]        # 3fc8 <_GLOBAL_OFFSET_TABLE_+0x10>
    102c:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]
    1030:	f3 0f 1e fa          	endbr64
    1034:	68 00 00 00 00       	push   0x0
    1039:	e9 e2 ff ff ff       	jmp    1020 <_init+0x20>
    103e:	66 90                	xchg   ax,ax

Disassembly of section .plt.got:

0000000000001040 <__cxa_finalize@plt>:
    1040:	f3 0f 1e fa          	endbr64
    1044:	ff 25 ae 2f 00 00    	jmp    QWORD PTR [rip+0x2fae]        # 3ff8 <__cxa_finalize@GLIBC_2.2.5>
    104a:	66 0f 1f 44 00 00    	nop    WORD PTR [rax+rax*1+0x0]

Disassembly of section .plt.sec:

0000000000001050 <puts@plt>:
    1050:	f3 0f 1e fa          	endbr64
    1054:	ff 25 76 2f 00 00    	jmp    QWORD PTR [rip+0x2f76]        # 3fd0 <puts@GLIBC_2.2.5>
    105a:	66 0f 1f 44 00 00    	nop    WORD PTR [rax+rax*1+0x0]

Disassembly of section .text:  ## Below are source codes-----------------------

0000000000001060 <_start>:
    1060:	f3 0f 1e fa          	endbr64
    1064:	31 ed                	xor    ebp,ebp
    1066:	49 89 d1             	mov    r9,rdx
    1069:	5e                   	pop    rsi
    106a:	48 89 e2             	mov    rdx,rsp
    106d:	48 83 e4 f0          	and    rsp,0xfffffffffffffff0
    1071:	50                   	push   rax
    1072:	54                   	push   rsp
    1073:	45 31 c0             	xor    r8d,r8d
    1076:	31 c9                	xor    ecx,ecx
    1078:	48 8d 3d ca 00 00 00 	lea    rdi,[rip+0xca]        # 1149 <main>
    107f:	ff 15 53 2f 00 00    	call   QWORD PTR [rip+0x2f53]        # 3fd8 <__libc_start_main@GLIBC_2.34>
    1085:	f4                   	hlt
    1086:	66 2e 0f 1f 84 00 00 	cs nop WORD PTR [rax+rax*1+0x0]
    108d:	00 00 00 

0000000000001090 <deregister_tm_clones>:
    1090:	48 8d 3d 79 2f 00 00 	lea    rdi,[rip+0x2f79]        # 4010 <__TMC_END__>
    1097:	48 8d 05 72 2f 00 00 	lea    rax,[rip+0x2f72]        # 4010 <__TMC_END__>
    109e:	48 39 f8             	cmp    rax,rdi
    10a1:	74 15                	je     10b8 <deregister_tm_clones+0x28>
    10a3:	48 8b 05 36 2f 00 00 	mov    rax,QWORD PTR [rip+0x2f36]        # 3fe0 <_ITM_deregisterTMCloneTable@Base>
    10aa:	48 85 c0             	test   rax,rax
    10ad:	74 09                	je     10b8 <deregister_tm_clones+0x28>
    10af:	ff e0                	jmp    rax
    10b1:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]
    10b8:	c3                   	ret
    10b9:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]

00000000000010c0 <register_tm_clones>:
    10c0:	48 8d 3d 49 2f 00 00 	lea    rdi,[rip+0x2f49]        # 4010 <__TMC_END__>
    10c7:	48 8d 35 42 2f 00 00 	lea    rsi,[rip+0x2f42]        # 4010 <__TMC_END__>
    10ce:	48 29 fe             	sub    rsi,rdi
    10d1:	48 89 f0             	mov    rax,rsi
    10d4:	48 c1 ee 3f          	shr    rsi,0x3f
    10d8:	48 c1 f8 03          	sar    rax,0x3
    10dc:	48 01 c6             	add    rsi,rax
    10df:	48 d1 fe             	sar    rsi,1
    10e2:	74 14                	je     10f8 <register_tm_clones+0x38>
    10e4:	48 8b 05 05 2f 00 00 	mov    rax,QWORD PTR [rip+0x2f05]        # 3ff0 <_ITM_registerTMCloneTable@Base>
    10eb:	48 85 c0             	test   rax,rax
    10ee:	74 08                	je     10f8 <register_tm_clones+0x38>
    10f0:	ff e0                	jmp    rax
    10f2:	66 0f 1f 44 00 00    	nop    WORD PTR [rax+rax*1+0x0]
    10f8:	c3                   	ret
    10f9:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]

0000000000001100 <__do_global_dtors_aux>:
    1100:	f3 0f 1e fa          	endbr64
    1104:	80 3d 05 2f 00 00 00 	cmp    BYTE PTR [rip+0x2f05],0x0        # 4010 <__TMC_END__>
    110b:	75 2b                	jne    1138 <__do_global_dtors_aux+0x38>
    110d:	55                   	push   rbp
    110e:	48 83 3d e2 2e 00 00 	cmp    QWORD PTR [rip+0x2ee2],0x0        # 3ff8 <__cxa_finalize@GLIBC_2.2.5>
    1115:	00 
    1116:	48 89 e5             	mov    rbp,rsp
    1119:	74 0c                	je     1127 <__do_global_dtors_aux+0x27>
    111b:	48 8b 3d e6 2e 00 00 	mov    rdi,QWORD PTR [rip+0x2ee6]        # 4008 <__dso_handle>
    1122:	e8 19 ff ff ff       	call   1040 <__cxa_finalize@plt>
    1127:	e8 64 ff ff ff       	call   1090 <deregister_tm_clones>
    112c:	c6 05 dd 2e 00 00 01 	mov    BYTE PTR [rip+0x2edd],0x1        # 4010 <__TMC_END__>
    1133:	5d                   	pop    rbp
    1134:	c3                   	ret
    1135:	0f 1f 00             	nop    DWORD PTR [rax]
    1138:	c3                   	ret
    1139:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]

0000000000001140 <frame_dummy>:
    1140:	f3 0f 1e fa          	endbr64
    1144:	e9 77 ff ff ff       	jmp    10c0 <register_tm_clones>

0000000000001149 <main>:
    1149:	f3 0f 1e fa          	endbr64
    114d:	55                   	push   rbp
    114e:	48 89 e5             	mov    rbp,rsp
    1151:	48 83 ec 10          	sub    rsp,0x10
    1155:	89 7d fc             	mov    DWORD PTR [rbp-0x4],edi
    1158:	48 89 75 f0          	mov    QWORD PTR [rbp-0x10],rsi
    115c:	48 8d 05 a1 0e 00 00 	lea    rax,[rip+0xea1]        # 2004 <_IO_stdin_used+0x4>
    1163:	48 89 c7             	mov    rdi,rax
    1166:	e8 e5 fe ff ff       	call   1050 <puts@plt>
    116b:	b8 00 00 00 00       	mov    eax,0x0
    1170:	c9                   	leave
    1171:	c3                   	ret

Disassembly of section .fini:

0000000000001174 <_fini>:
    1174:	f3 0f 1e fa          	endbr64
    1178:	48 83 ec 08          	sub    rsp,0x8
    117c:	48 83 c4 08          	add    rsp,0x8
    1180:	c3                   	ret
```
- Compared to myapp.o, a.out has more details on `<main>`

### 59. How binary loads and executes in theory
```bash
$ readelf -p .interp a.out

String dump of section '.interp':
  [     0]  /lib64/ld-linux-x86-64.so.2
```

## Section 17: Linux - ELF Format

### 60. Exploring the Executable and Linkable Format (ELF) amd Executable Header
- ELF format
  - Executable header
  - Program header
  - Section
  - Section header
- /usr/include/elf.h for the datatype
```bash
$ readelf -h ./a.out
ELF Header:
  Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00 
  Class:                             ELF64
  Data:                              2's complement, little endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              DYN (Position-Independent Executable file)
  Machine:                           Advanced Micro Devices X86-64
  Version:                           0x1
  Entry point address:               0x1060
  Start of program headers:          64 (bytes into file)
  Start of section headers:          13976 (bytes into file)
  Flags:                             0x0
  Size of this header:               64 (bytes)
  Size of program headers:           56 (bytes)
  Number of program headers:         13
  Size of section headers:           64 (bytes)
  Number of section headers:         31
  Section header string table index: 30
```
- Magic number
  - 0x7f: start of the magic number
  - 0x45: E
  - 0x4c: L
  - 0x46: F

### 61. Learning ELF Fields

### 62. Learning ELF Program Header Fields
```bash
$ readelf --wide --segments ./a.out 

Elf file type is DYN (Position-Independent Executable file)
Entry point 0x1060
There are 13 program headers, starting at offset 64

Program Headers:
  Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
  PHDR           0x000040 0x0000000000000040 0x0000000000000040 0x0002d8 0x0002d8 R   0x8
  INTERP         0x000318 0x0000000000000318 0x0000000000000318 0x00001c 0x00001c R   0x1
      [Requesting program interpreter: /lib64/ld-linux-x86-64.so.2]
  LOAD           0x000000 0x0000000000000000 0x0000000000000000 0x000628 0x000628 R   0x1000
  LOAD           0x001000 0x0000000000001000 0x0000000000001000 0x000181 0x000181 R E 0x1000
  LOAD           0x002000 0x0000000000002000 0x0000000000002000 0x0000f4 0x0000f4 R   0x1000
  LOAD           0x002db8 0x0000000000003db8 0x0000000000003db8 0x000258 0x000260 RW  0x1000
  DYNAMIC        0x002dc8 0x0000000000003dc8 0x0000000000003dc8 0x0001f0 0x0001f0 RW  0x8
  NOTE           0x000338 0x0000000000000338 0x0000000000000338 0x000030 0x000030 R   0x8
  NOTE           0x000368 0x0000000000000368 0x0000000000000368 0x000044 0x000044 R   0x4
  GNU_PROPERTY   0x000338 0x0000000000000338 0x0000000000000338 0x000030 0x000030 R   0x8
  GNU_EH_FRAME   0x002014 0x0000000000002014 0x0000000000002014 0x000034 0x000034 R   0x4
  GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0x10
  GNU_RELRO      0x002db8 0x0000000000003db8 0x0000000000003db8 0x000248 0x000248 R   0x1

 Section to Segment mapping:
  Segment Sections...
   00     
   01     .interp 
   02     .interp .note.gnu.property .note.gnu.build-id .note.ABI-tag .gnu.hash .dynsym .dynstr .gnu.version .gnu.version_r .rela.dyn .rela.plt 
   03     .init .plt .plt.got .plt.sec .text .fini 
   04     .rodata .eh_frame_hdr .eh_frame 
   05     .init_array .fini_array .dynamic .got .data .bss 
   06     .dynamic 
   07     .note.gnu.property 
   08     .note.gnu.build-id .note.ABI-tag 
   09     .note.gnu.property 
   10     .eh_frame_hdr 
   11     
   12     .init_array .fini_array .dynamic .got 
```

## Section 18: Windows - PE Format

### 63. Fundamentals of Windows PE Format
- Portable Executable
  - MS-DOS Headers
  - MS-DOS stub
  - PE Signature
  - PE File Header
  - PE Optional Header
  - Section Header
  - Section

## Section 19: OR XOR AND

### 64. OR Logic

### 65. NOT Logic

### 66. XOR Logic

### 67. AND Logic

## Section 20: Data Display Debugger - DDD

### 68. Developing another Assembly Program to Analyze with DDD
- sudo apt install ddd
- Edit ~/.ddd/init for larger font size
```asm
; move.asm
section .data
  bNum db 123         ; 8bit
  wNum db 12345       ; 16bit
  dNum dd 1234567890  ; 32bit
  qNum1 dq 1234567890123456789 ; 64bit
  qNum2 dq 123456     ; 64bit
  qNum3 dq 3.14       ; 64bit
section .bss
section .text
  global main
main:
  push rbp
  mov rbp,rsp
  mov rax,-1           ; fill rax with 1s
  mov al,byte [bNum]   ; does NOT clear upper bits of rax
  xor rax,rax          ; clearing rax
  mov al,byte [bNum]   ; now rax has correct value
  mov rax,-1           ; fill rax with 1s
  mov ax,word [wNum]   ; does not clear upper bits of rax
  mov rax,rax          ; clear rax
  mov ax,word[wNum]    ; now rax has the correct value
  mov rax,-1           ; fill rax with 1s
  mov rax,qword[qNum1] ; 
  mov qword[qNum2],rax ;
  mov rax, 123456      ; source operand an immediate value
  movsd xmm0, [qNum3]    ; instruction for floaing point
  mov rsp,rbp
  pop rbp
  ret
```

### 69. Analyzing Previously Written Code
- Makefile:
```bash
move: move.o
	gcc -o move move.o -no-pie
move.o: move.asm
	nasm -f elf64 -g move.asm -o move.o -l move.lst
```

### 70. Using DDD and Analyzing RAX Values
- Couldn't change the size of fonts in the data section

## Section 21: Jump and Loop

### 71. ERRORS AND SOLUTIONS

### 72. Using Conditions and Jumping
```asm
; jumping.asm
extern printf
section .data
  number1 dq 42
  number2 dq 41
  fmt1    db "NUMBER1 >= NUMBER2", 10, 0 ; format string for com
parison result 1
  fmt2    db "NUMBER1 < NUMBER2",10,0
section .text
global main
main:
  push rbp            ; save the base pointer
  mov  rbp,rsp        ; set up the base pointer
  mov  rax,[number1]  ;
  mov  rbx,[number2]  ;
  cmp  rax,rbx
  jge  greater        ; jump to greater if rax>=rbx (greater or 
equal)
  mov  rdi, fmt2      ; loading the format string when rax < rbx
  mov  rax,0
  call printf         ; print fmt2 string
  jmp  exit           ; jump to the exit label
greater:
  mov  rdi,fmt1
  mov  rax,0
  call printf
exit: 
  mov  rsp,rbp
  pop  rbp
  ret  
```

### 73. Jump if equal
- je
```asm
  mov al,[value1]
  mov bl,[value2]
  cmp al,bl
  je equal_found
```

### 74. Jump if Not Equal
- jne

### 75. Jump if Greater
- jg

### 76. Greater than or Equal to
- jge

### 77. Jump if Less
- jl

### 78. Jump if less or equal
- jle

### 79. Jump if Above
- ja: jump if above
  - Comparison
  - Conditional assessment
  - Jump or sequential execution
  - For unsigned comparison
    - For signed comparison, jg
```asm
; ja_ex.asm
extern printf
section .data
  value1 db 25
  value2 db 20
  fmt1   db 'v1 > v2', 10, 0
  fmt2   db 'v2 <= v1', 10, 0
section .text
  global main
main:
  mov al,[value1]
  mov bl,[value2]
  cmp al,bl
  ja above
not_above:
  mov  rdi, fmt2      
  mov  rax,0
  call printf   
  jmp done
above:
  mov  rdi, fmt1 
  mov  rax,0
  call printf
done:
 mov rax,60
 mov rdi, 0
 syscall
```

### 80. Jump if Above or Equal
- jae

### 81. Jump if below
- jb

### 82. Jump if below or equal
- jbe

## Section 22: Assembly Project using Jump and Loop

### 83. Developing Loop and Calculator Project with Assembly
- calc.asm:
```asm
extern printf
section .data
  number dq 5    ; 8bytes (quadword)
  fmt db "The sum from 0 to %ld is %ld",10,0
section .bss
section .text
global main
main:
  push rbp
  mov  rbp, rsp
  mov  rcx, [number] ; rcx is a default loop counter
  mov  rax, 0
bloop:
  add rax,rcx # every loop, rcx--
  loop bloop
  mov rdi,fmt
  mov rsi,[number]
  mov rdx,rax
  mov rax,0
  call printf
  mov rsp,rbp
  pop rbp
  ret 
```

### 84. Testing our Project
- Makefile:
```bash
calc: calc.o
	gcc -o calc calc.o -no-pie
calc.o: calc.asm
	nasm -f elf64 -g  -F dwarf calc.asm -l calc.lst
```

## Section 23: Memory Manipulation

### 85. Project EXABYTE
- 16 Exabytes: memory limits in 64bit OS

### 86. Testing and Analyzing Project with Readelf and GDB
- `[bNum]`: memory address of bNum
- `bNum`; value of bNum variable
- rax can hold 8 byte data
```asm
; exabyte.asm
section .data
  bNum db 123
  wNum db 12345
  warray times 5 dw 0
  dNum dd 12345
  dNum1 dq 12345
  text1 db "abc",0
  qNum2 dq 3.141592654
  text2 db "cde",0
section .bss
  bvar  resb 1
  dvar  resd 1
  wvar  resw 10
  qvar  resq 3
section .text
  global main
main:
  mov rbp,rsp      ; for correct debugging
  push rbp
  mov  rbp,rsp
  lea  rax,[bNum] ; `x $rax` -> 0x404010 <bNum>:	0x0000397b, matching with `x &bNum`
  mov  rax,bNum
  mov  rax,[bNum]
  mov  [bvar], rax
  lea  rax,[bvar]
  lea  rax,[wNum]
  mov  rax,[wNum]
  lea  rax,[text1]  ; load the memory address of text1 into rax
  mov  rax,text1    ; load the memory address of text1 into rax
  mov  rax,text1+1  ; load 2nd char of text1 to RAX
  lea  rax,[text1+1]
  mov  rax,[text1]  ; load the stored at the memory address of text1
 into rax
  mov  rax,[text1+1] ; 
  mov  rsp,rbp
  pop  rbp
  ret
```
- `Entry point address:               0x401020`
  - From `readelf --file-header ./exa`
- From gdb:
```bash
(gdb) b _start
Breakpoint 1 at 0x401020  #<------- this matches the entry point
...
(gdb) disassemble main
Dump of assembler code for function main:
=> 0x0000000000401110 <+0>:	mov    %rsp,%rbp
   0x0000000000401113 <+3>:	push   %rbp
   0x0000000000401114 <+4>:	mov    %rsp,%rbp
   0x0000000000401117 <+7>:	lea    0x404010,%rax
```
- `main` location is some bytes after from `_start`

## Section 24: Calculator with Assembly

### 87. Defining variables

### 88. Addition and Subtraction
- In .bss section:
  - RESB 1: allocates 1 byte
  - RESW 1: allocates 2 bytes
  - RESD 1: allocates 4 bytes
  - RESQ 1: allocates 8 bytes

### 89. Last Decorations
```asm
; calculator.asm
extern printf
section .data
  number1    dq 128
  number2    dq 19
  neg_number dq  -12
  fmt        db "The number are %ld and %ld",10,0
  fmtint     db "%s %ld", 10,0
  sumi       db "The sum is ", 0
  difi       db "The diff is ", 0
  inci       db "Number 1 incremented: ", 0
  deci       db "Number 1 decremented: ", 0
  sali       db "Number 1 shift left 2 (x4) : ", 0
  sari       db "Number 1 shift right 2 (/4) : ", 0
  sariex     db "Number 1 shift right 2 (/4) with sign expression: ", 0
  multi      db "The product is ", 0
  divi       db "The integer quotient is ", 0
  remi       db "The modulo is ", 0
section .bss
  resulti resq 1
  modulo  resq 1
section .text
global main
main:
  push rbp
  mov  rbp,rsp
  ; display
  mov  rdi, fmt
  mov  rsi, [number1]
  mov  rdx, [number2]
  call printf
  ; adding
  mov  rax, [number1]
  add  rax, [number2]
  mov  [resulti], rax
  ; display results
  mov  rdi, fmtint
  mov  rsi, sumi
  mov  rdx, [resulti]
  mov  rax, 0
  call printf
  ; subtracting
  mov  rax, [number1]
  sub  rax, [number2]
  mov  [resulti], rax
  ; display
  mov  rdi, fmtint
  mov  rsi, difi
  mov  rdx, [resulti]
  mov  rax, 0
  call printf
  ; incrementing
  mov  rax, [number1]
  inc  rax
  mov  [resulti],rax
  ; display
  mov  rdi, fmtint
  mov  rsi, inci
  mov  rdx, [resulti]
  mov  rax, 0
  call printf
  ; decrementing
  mov  rax, [number1]
  dec  rax
  mov  [resulti],rax
  ; display
  mov  rdi, fmtint
  mov  rsi, deci
  mov  rdx, [resulti]
  mov  rax, 0
  call printf
  ; shift arithmetic left
  mov  rax, [number1]
  sal  rax,2
  mov  [resulti], rax
  ; display
  mov  rdi, fmtint
  mov  rsi, sali
  mov  rdx, [resulti]
  mov  rax, 0
  call printf
  ; shift arithmetic right
  mov  rax, [number1]
  sar  rax,2
  mov  [resulti], rax
  ; display
  mov  rdi, fmtint
  mov  rsi, sari
  mov  rdx, [resulti]
  mov  rax, 0
  call printf
  ; shift arithmetic right with sign extension
  mov  rax, [neg_number]
  sar  rax,2
  mov  [resulti], rax
  ; display
  mov  rdi, fmtint
  mov  rsi, sariex
  mov  rdx, [resulti]
  mov  rax, 0
  call printf
  ; multiply
  mov  rax, [number1]
  imul qword[number2]
  mov  [resulti], rax
  ; display
  mov  rdi, fmtint
  mov  rsi, multi
  mov  rdx, [resulti]
  mov  rax, 0
  call printf
  ; division
  mov  rax, [number1]
  mov  rdx, 0 ; initialize the moduleo
  idiv qword[number2]
  mov  [resulti], rax
  mov  [modulo], rdx
  ; display the quotient
  mov  rdi, fmtint
  mov  rsi, divi
  mov  rdx, [resulti]
  mov  rax, 0
  call printf
  ; display the remainder
  mov  rdi, fmtint
  mov  rsi, remi
  mov  rdx, [modulo]
  mov  rax, 0
  call printf
  ; exit
  mov rbp, rsp
  pop rbp
  ret
```

### 90. Explaining Registers in Practice
- Demo:
```bash
$ ./calc 
The number are 128 and 19
The sum is  147
The diff is  109
Number 1 incremented:  129
Number 1 decremented:  127
Number 1 shift left 2 (x4) :  512
Number 1 shift right 2 (/4) :  32
Number 1 shift right 2 (/4) with sign expression:  -3
The product is  2432
The integer quotient is  6
The modulo is  14
```

### 91. Completing Section

## Section 25: Stack

### 92. Manipulating Stack
- LIFO (Last In First Out)
- Reversing a string:
```asm
extern printf
section .data
  myString db "ABCDE",0
  myStringLen equ $-myString-1 ;
  fmt1 db "THE ORIGINAL STRING: %s",10,0
  fmt2 db "THE REVERSED STRING: %s",10,0
section .text
global main
main:
  push rbp
  mov  rbp, rsp
  ;
  mov  rdi, fmt1
  mov  rsi, myString
  mov  rax, 0
  call printf 
  xor  rax, rax
  mov  rbx, myString
  mov  rcx, myStringLen
  mov  r12, 0
pushLoop:
  mov  al, byte[rbx+r12]
  push rax
  inc  r12
  loop pushLoop
  ; 
  mov  rax, myString
  mov  rcx, myStringLen
  mov  r12, 0 
popLoop:
  pop  rax
  mov  byte[rbx+r12],al 
  inc  r12
  loop popLoop
  mov byte[rbx+r12],0
  ;
  mov  rdi, fmt2
  mov  rsi, myString
  mov  rax, 0
  call printf
  ;
  mov  rbp, rsp
  pop  rbp
  ret
```
- Result:
```bash
$ ./stack_test 
THE ORIGINAL STRING: ABCDE
THE REVERSED STRING: EDCBA
```

## Section 26: Functions

### 93. Developing our First Functional Function
- func_one.asm:
```asm
extern printf
section .data
radius dq 10.0
pi     dq 3.14
formatString db  "The area of circle %.2f",10,0
section .text
global main
main:
  push rbp
  mov rbp, rsp
  call area
  mov rdi, formatString
  movsd xmm1, [radius]
  mov rax,  1
  call printf
  leave
  ret
area:
  push rbp
  mov  rbp, rsp
  movsd  xmm0, [radius]
  mulsd  xmm0, [radius]
  mulsd  xmm0, [pi]
  leave
  ret
```
- Result:
```bash
$ ./func_one 
The area of circle 314.00
```

### 94. Gaining Deeper understanding of Functions and Local variables in assembly
```py
extern printf
section .data
  radius dq 10.0
section .text
area:
  section .data
  .pi dq 3.141592654
  section .text
  push   rbp
  mov    rbp, rsp
  movsd  xmm0, [radius]
  mulsd  xmm0, [radius]
  mulsd  xmm0, [.pi]
  leave
  ret
circumference:
  section .data
  .pi dq 3.14
  section .text
  push   rbp
  mov    rbp, rsp
  movsd  xmm0, [radius]
  addsd  xmm0, [radius]
  mulsd  xmm0, [.pi]
  leave
  ret
circle:
  section .data
  .format_area db "The area is %f", 10,0
  .format_circumference db "The circumference is %f", 10,0
  section .text
  push  rbp
  mov   rbp, rsp
  call  area
  mov   rdi, .format_area
  mov   rax, 1
  call  printf
  call  circumference
  mov   rdi, .format_circumference
  mov   rax, 1
  call  printf
  leave
  ret
global main
main:
  push  rbp
  mov   rbp, rsp
  call  circle
  leave
  ret
```
- Result
```bash
$ ./func_two 
The area is 314.159265
The circumference is 62.800000
```

## Section 27: Stack Frame and External Functions

### 95. Fundamentals of Stack frame and nested function calls
- 16 byte stack alignment for SIMD
  - The 16-byte stack alignment is a requirement of the x86-64 calling conventions (both System V and Microsoft) primarily to ensure compatibility and optimal performance of Streaming SIMD Extensions (SSE) instructions. 
- If the stack pointer (RSP) "ends in 0" (meaning its value is a multiple of 16), it means the stack is 16-byte aligned at that moment. 
- stack_align.asm 
```asm
extern printf
section .data
formatText db "2 times pi equals %.14f"
pi         dq  3.14159265358979
section .text
func3:
  push   rbp ; function prologue
  movsd  xmm0, [pi]
  addsd  xmm0, [pi]
  mov    rdi,  formatText
  mov    rax,  1
  call   printf
  pop    rbp ; function epilogue
  ret
func2:
  push   rbp ; function prologue
  call   func3
  pop    rbp ; function epilogue
  ret
func1:
  push   rbp ; function prologue
  call   func2
  pop    rbp ; function epilogue
  ret
global main
main:
  push   rbp
  call   func1
  pop    rbp
  ret
```

### 96. Developing our own External Function and Calling them
- myfunc.asm 
```asm
extern printf
extern c_area
extern c_circum
extern r_area
extern r_circum
global pi
section .data
  pi       dq   3.141592654
  radius   dq  10.0
  side1    dq   4
  side2    dq   5
  fmt_f    db   "%s %f",10,0
  fmt_i    db   "%s %d",10,0
  ca       db   "The circle area is ", 0
  cc       db   "The circle circumference is ",0
  ra       db   "The rectangle area is ",0
  rc       db   "The rectangle circumference is ", 0
section .text
global main
main:
  push   rbp
  mov    rbp, rsp
  movsd  xmm0, qword[radius]
  call   c_area
  mov    rdi, fmt_f
  mov    rsi, ca
  mov    rax, 1
  call   printf
  movsd  xmm0, qword[radius]
  call   c_circum
  mov    rdi,  fmt_f
  mov    rsi, cc
  mov    rax, 1
  call   printf
  mov    rdi,  [side1]
  mov    rsi,  [side2]
  call   r_area
  mov    rdi, fmt_i
  mov    rsi, ra
  mov    rdx, rax
  mov    rax, 0
  call   printf
  mov    rdi, [side1]
  mov    rsi, [side2]
  call   r_circum
  mov    rdi, fmt_i
  mov    rsi, rc
  mov    rdx, rax
  mov    rax, 0
  call   printf
  mov    rsp, rbp
  pop    rbp
  ret
```
- circle.asm 
```asm
extern pi
section .data
section .bss
section .text
global  c_area
c_area:
  push  rbp
  mov   rbp, rsp
  movsd xmm1, qword[pi]
  mulsd xmm0, xmm0  ; xmm0 value is from myfunc.asm
  mulsd xmm0, xmm1
  mov   rsp, rbp
  pop   rbp
  ret
global c_circum
c_circum:
  push  rbp
  mov   rbp, rsp
  movsd xmm1, qword[pi]
  addsd xmm0, xmm0
  mulsd xmm0, xmm1
  mov   rsp, rbp
  pop   rbp
  ret
```
- rect.asm 
```asm
section .data
section .bss
section .text
global r_area
r_area:
  push  rbp
  mov   rbp, rsp
  mov   rax, rsi
  imul  rax, rdi
  mov   rsp, rbp
  pop   rbp
  ret
global r_circum
r_circum:
  push  rbp
  mov   rbp, rsp
  mov   rax, rsi
  add   rax, rdi
  add   rax, rax
  mov   rsp, rbp
  pop   rbp
  ret
```
- Makefile
```bash
myfunc: myfunc.o circle.o rect.o
	gcc -g -o myfunc myfunc.o circle.o rect.o -no-pie
myfunc.o: myfunc.asm
	nasm -f elf64 -g -F dwarf myfunc.asm -l myfunc.lst
circle.o: circle.asm
	nasm -f elf64 -g -F dwarf circle.asm -l circle.lst
rect.o: rect.asm
	nasm -f elf64 -g -F dwarf rect.asm -l rect.lst
```
- Demo:
```bash
$ ./myfunc 
The circle area is  314.159265
The circle circumference is  62.831853
The rectangle area is  20
The rectangle circumference is  18
```

## Section 28: FuncArg and Preserving Registers

### 97. Running out of REGISTERS ?!?!
- calling_conv.asm 
```asm
extern printf
section .data
first   db "A",0
second  db "B",0
third   db "C",0
fourth  db "D",0
fifth   db "E",0
sixth   db "F",0
seventh db "G",0
eighth  db "H",0
nineth  db "I",0
tenth   db "J",0
format_string   db "The string is: %s%s%s%s%s%s%s%s%s%s", 10,0
format_string2  db "PI vaue = %f", 10,0
pi      dq 3.14
section .bss
section .text
global main
main:
  push rbp
  mov  rbp, rsp
  mov  rdi, format_string
  mov  rsi, first
  mov  rdx, second
  mov  rcx, third
  mov  r8, fourth
  mov  r9, fifth ; <--- now all registers are consumed
  push tenth ; <--- pushing to stack
  push nineth
  push eighth
  push seventh
  push sixth
  mov  rax,0
  call printf
  and  rsp, 0xfffffffffffffff0
  movsd xmm0, [pi]
  mov  rax , 1
  mov  rdi, format_string2
  call printf
  leave
  ret
```

## Section 29: Bits EVERYWHERE !!!

### 98. BITS BITS BITS !!!
- bit1.asm:
```asm
extern printb
extern printf
section .data
  msgn1  db "Number 1",10,0
  msgn2  db "Number 2",10,0
  msg1   db "XOR",10,0
  msg2   db "OR",10,0
  msg3   db "AND",10,0
  msg4   db "NOT Number 1",10,0
  msg5   db "SHL 2 lower byte of number 1",10,0
  msg6   db "SHR 2 lower byte of number 1",10,0
  msg7   db "SAL 2 lower byte of number 1",10,0
  msg8   db "SAR 2 lower byte of number 1",10,0
  msg9   db "ROL 2 lower byte of number 1",10,0
  msg10  db "ROL 2 lower byte of number 2",10,0
  msg11  db "ROR 2 lower byte of number 1",10,0
  msg12  db "ROR 2 lower byte of number 2",10,0
  number1 dq -72
  number2 dq 1064
section .bss
section .text
  global main
main:
  push rbp
  mov  rbp, rsp
  mov  rsi, msgn1
  call printmsg ; will be provided from C code
  mov  rdi, [number1]
  call printb
  mov  rsi, msgn2
  call printmsg
  mov  rdi,[number2]
  call printb
  mov  rsi, msg1
  call printmsg
  mov  rax, [number1]
  xor  rax, [number2]
  mov  rdi, rax
  call printb
  mov  rsi, msg2
  call printmsg
  mov  rax, [number1]
  or   rax, [number2]
  mov  rdi, rax
  call printb
  mov  rsi, msg3
  call printmsg
  mov  rax, [number1]
  and  rax, [number2]
  mov  rdi, rax
  call printb
  mov  rsi, msg4
  call printmsg
  mov  rax, [number1]
  not  rax
  mov  rdi, rax
  call printb
  mov  rsi, msg5
  call printmsg
  mov  rax, [number1]
  shl  al, 2
  mov  rdi, rax
  call printb
  mov  rsi, msg6
  call printmsg
  mov  rax, [number1]
  shr  al, 2
  mov  rdi, rax
  call printb
  mov  rsi, msg7
  call printmsg
  ;
  mov  rax, [number1]
  sal  al, 2
  mov  rdi, rax
  call printb
  ;
  mov  rsi, msg8
  call printmsg
  mov  rax, [number1]
  sar  al, 2
  mov  rdi, rax
  call printb
  ;
  mov  rsi, msg9
  call printmsg
  mov  rax, [number1]
  rol  al, 2
  mov  rdi, rax
  call printb
  ;
  mov  rsi, msg10
  call printmsg
  mov  rax, [number2]
  rol  al, 2
  mov  rdi, rax
  call printb
  ;
  mov  rsi, msg11
  call printmsg
  mov  rax, [number1]
  ror  al, 2
  mov  rdi, rax
  call printb
  ;
  mov  rsi, msg12
  call printmsg
  mov  rax, [number2]
  ror  al, 2
  mov  rdi, rax
  call printb
  ;
  leave 
  ret
printmsg:
  section .data
    .fmstr db "%s",0
  section .text
    mov rdi, .fmstr
    mov rax, 0
    call printf
    ret
```
- printb.c:
```c
#include <stdio.h>
void printb( long long n) {
   long long s,c;
   for(c=63; c >=0; c--) {
      s = n >> c;
      if ((c+1)%8 == 0)
         printf(" ");
      if (s &1 )
         printf("1");
      else
         printf("0");
   }
   printf("\n"); 
}
```
- Makefile:
```bash
bit1: bit1.o printb.o
	gcc -g -o bit1 bit1.o printb.o -no-pie
bit1.o: bit1.asm
	nasm -f elf64 -g -F dwarf bit1.asm -l bit.lst
printb: printb.o
	gcc -c printb.c
```
- Result:
```bash
$ ./bit1
Number 1
 11111111 11111111 11111111 11111111 11111111 11111111 11111111 10111000
Number 2
 00000000 00000000 00000000 00000000 00000000 00000000 00000100 00101000
XOR
 11111111 11111111 11111111 11111111 11111111 11111111 11111011 10010000
OR
 11111111 11111111 11111111 11111111 11111111 11111111 11111111 10111000
AND
 00000000 00000000 00000000 00000000 00000000 00000000 00000100 00101000
NOT Number 1
 00000000 00000000 00000000 00000000 00000000 00000000 00000000 01000111
SHL 2 lower byte of number 1
 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11100000
SHR 2 lower byte of number 1
 11111111 11111111 11111111 11111111 11111111 11111111 11111111 00101110
SAL 2 lower byte of number 1
 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11100000
SAR 2 lower byte of number 1
 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11101110
ROL 2 lower byte of number 1
 11111111 11111111 11111111 11111111 11111111 11111111 11111111 11100010
ROL 2 lower byte of number 2
 00000000 00000000 00000000 00000000 00000000 00000000 00000100 10100000
ROR 2 lower byte of number 1
 11111111 11111111 11111111 11111111 11111111 11111111 11111111 00101110
ROR 2 lower byte of number 2
 00000000 00000000 00000000 00000000 00000000 00000000 00000100 00001010
```

## Section 30: Macros

### 99. Macros in Assembly
```asm
extern printf
%define double_it(r) sal r,1
%macro prntf 2  ; reads 2 args, each will be %1 and %2 below
section .data
%%arg1   db %1,0  ; %% macro variables
%%fmtint db "%s %ld",10,0
section .text
  mov rdi, %%fmtint
  mov rsi, %%arg1
  mov rdx, [%2]
  mov rax, 0
  call printf
%endmacro
section .data
number dq 15
section .bss
section .text
global main
main:
  push rbp
  mov  rbp, rsp
  prntf  "The number is",  number
  mov  rax, [number]
  double_it(rax)
  mov [number], rax
  prntf "The number times 2 is ", number
  leave
  ret
```

## Seciton 31: Input Output - Console I/O

### 100. Console Input - BE CAREFUL with INPUT !!
- syscall with rax=1 and rdi=1
  - stdout
  - writes the data in rsi and rdx
- syscall with rax=0 and rdi=0
  - stdin
  - Reads data into rsi as many as rdx
    - What if the input is longer than rdx?
    - Exceeding components are fed into Linux CLI (?)
- read_usr_inp.asm:
```asm
section .data
 message1     db  "Hello world",10,0
 message1_len equ $-message1
 message2     db  "Enter Text: ", 0
 message2_len equ $-message2
 message3     db "Your input: ", 0
 message3_len equ $-message3
 input_length equ 10
section .bss
 input resb  input_length+1
 terminator
section .text
global main
main:
 push  rbp
 mov   rbp, rsp
 mov   rsi, message1
 mov   rdx, message1_len
  call prints
 mov   rsi, message2
 mov   rdx, message2_len
 call  prints
 mov   rsi, input
 mov   rdx, input_length
 call  reads
 mov   rsi, message3
 mov   rdx, message3_len
 call  prints
 mov   rsi, input
 mov   rdx, input_length
 call  prints
 leave
 ret
prints:
  push  rbp
  mov   rbp, rsp
  mov   rax, 1
  mov   rdi, 1
  syscall
  leave
  ret
reads:
  push rbp,
  mov  rbp, rsp
  mov  rax, 0
  mov  rdi, 0
  syscall
  leave 
  ret
```
- Demo:
```bash
$ ./read_usr_inp 
Hello world
Enter Text: Hi world
Your input: Hi world
$ ./read_usr_inp 
Hello world
Enter Text: 0123456789ABCDE
Your input: 0123456789$ ABCDE
ABCDE: command not found
```

### 101. Console Input - MORE SECURE INPUT
```asm
section .data
  message1  db "hello world",10,0
  message2  db "input (only a-z): ", 0
  message3  db "Your input: ", 0
  input_length equ 10
  NL        db 0xa
section .bss
  input resb  input_length+1
section .text
global main
main: 
  push  rbp
  mov   rbp, rsp
  mov   rdi, message1
  call  prints
  mov   rdi, message2
  call  prints
  mov   rdi, input
  mov   rsi, input_length
  call  reads
  mov   rdi, message3
  call  prints
  mov   rdi, input
  call  prints
  mov   rdi, NL
  call  prints
  leave
  ret
prints:
  push rbp
  mov  rbp, rsp
  push r12
  xor  rdx, rdx
  mov  r12, rdi
  .lengthloop:
  cmp byte [r12],0
  je   .lengthfound
  inc  rdx
  inc  r12
  jmp  .lengthloop
  .lengthfound: 
  cmp  rdx, 0
  je   .done
  mov rsi, rdi
  mov rax, 1
  mov rdi, 1
  syscall
  .done:
  pop r12
  leave
  ret
reads:
  section .data
  section .bss
  .inputchar resb 1
  section .text
  push  rbp
  mov   rbp, rsp
  push  r12
  push  r13
  push  r14
  mov   r12, rdi
  mov   r13, rsi
  xor   r14, r14
.readc:
  mov rax,0
  mov rdi, 0
  lea rsi, [.inputchar]
  mov rdx, 1
  syscall
  mov al, [.inputchar]
  cmp al, byte[NL]
  je  .done
  cmp  al, 97 ; lower case a
  jl   .readc
  cmp  al, 122 ; lower case z
  jg   .readc
  inc  r14
  cmp  r14, r13
  ja   .readc
  mov  byte[r12], al
  inc  r12
  jmp  .readc
  .done:
  inc r12
  mov byte[r12],0
  pop r14
  pop r13
  pop r12
  leave
  ret
```
- Demo:
```bash
$ ./secure_inp 
hello world
input (only a-z): Hello world 123456
Your input: elloworld
```  

## Section 32: Input Output - File I/O

### 102. Understanding File Management in Assembly
- Create a file then write data in the file
- Overwrite the file
- Append the data
- Write data at a certain position
- Read data from the file
- Read data from a certain position
- Delete the file

### 103. Creating File in Assembly

### 104. Deleting and Overwriting File in Assembly

### 105. Opening and Writing File in Assembly

### 106. Finally Manipulating files in Assembly Program
```asm
section .data
    CREATE      equ      1
    OVERWRITE   equ      1
    APPEND      equ      1
    O_WRITE     equ     1
    READ        equ     1
    O_READ      equ     1
    DELETE      equ     1
    ;SYSCALL SYMBOLS
    NR_READ     equ     0
    NR_WRITE    equ     1
    NR_OPEN     equ     2
    NR_CLOSE    equ     3
    NR_LSEEK    equ     8
    NR_CREATE   equ     85
    NR_UNLINK   equ     87    
    ;FILE FLAGS
    O_CREATE    equ     00000100q
    O_APPEND    equ     00002000q    
    ;ACCESS MODE
    O_READ_ONLY     equ     000000q
    O_WRITE_ONLY    equ     000001q
    O_READ_WRITE    equ     000002q    
    S_USER_READ     equ     00400q
    S_USER_WRITE    equ     00200q    
    NL              equ     0xa
    bufferLength    equ     64
    fileName        db      "hacked.txt",0    
    FD              dq      0    
    myText1         db  "1. HELLO EVERYONE !",NL,0
    myText1_Len     db  $-myText1-1
    myText2         db  "2. MY NAME IS TYPHON !",NL,0
    myText2_Len     db  $-myText2-1
    myText3         db  "3. OUR CODE WORKS !",NL,0
    myText3_Len     db  $-myText3-1
    myText4         db  "4. THIS CODE REALLY WORKS PERFECTLY, BYE . . .",NL,0
    myText4_Len     db  $-myText4-1    
    ;ERROR MESSAGES DEFINED HERE
    Error_Create    db  "Error Creating file",NL,0
    Error_Close     db  "Error Closing file",NL,0
    Error_Write     db  "Error Writing file",NL,0
    Error_Open      db  "Error Opening file",NL,0
    Error_Append    db  "Error Apennding to file",NL,0
    Error_Delete    db  "Error Deleting file",NL,0
    Error_Read      db  "Error Reading file",NL,0
    Error_Print     db  "Error Printing file",NL,0
    Error_Position  db  "Error Positioning file",NL,0
    Ok_Create       db  "File created and Opened : OK",NL,0
    Ok_Close        db  "File closed : OK",NL,0
    Ok_Write        db  "File Written : OK",NL,0
    Ok_Open         db  "File Opened R/W: OK",NL,0
    Ok_Append       db  "File Appended : OK",NL,0
    Ok_Delete       db  "File Deleted : OK",NL,0
    Ok_Read         db  "File Readed : OK",NL,0
    Ok_Position     db  "Positioned in File : OK",NL,0
section .bss
    buffer resb bufferLength
section .text
    global  main
main:
    push    rbp
    mov     rbp,    rsp
;CREATE FILE
%IF CREATE
    mov     rdi,    fileName
    call    createFile
    mov     qword[FD],  rax
    ;WRITE TO FILE #1
    mov     rdi,    qword[FD]
    mov     rsi,    myText1
    mov     rdx,    qword[myText1_Len]
    call    writeFile
    ;CLOSE FILE
    mov     rdi,    qword[FD]
    call    closeFile
%ENDIF
%IF OVERWRITE
;OPEN FILE
    mov     rdi,    fileName
    call    openFile
    mov     qword[FD],  rax
    ;WRITE TO FILE #2 OVERWRITE
    mov     rdi,    qword[FD]
    mov     rsi,    myText2
    mov     rdx,    myText2_Len
    call    writeFile
    ;CLOSE FILE
    mov     rdi,    [FD]
    call    closeFile
%ENDIF
%IF APPEND
;OPEN AND APPEND TO A FILE AND THEN CLOSE
;open file to append
    mov     rdi,    fileName
    call    appendFile
    mov     qword[FD],  rax
    ;write to a file #3 APPEND
    mov     rdi,    qword[FD]
    mov     rsi,    myText3
    mov     rdx,    qword[myText3_Len]
    call    writeFile
    ;close file
    mov     rdi,    qword[FD]
    call    closeFile
%ENDIF
%IF O_WRITE
;OPEN AND OVERWRITE AT AN OFFSET INA FILE THEN CLOSE IT
;open file
    mov     rdi,    fileName
    call    openFile
    mov     qword[FD]   ,rax
    ;position file at offset
    mov     rdi,    qword[FD]
    mov     rsi,    qword[myText2_Len]
    mov     rdx,    0
    call    positionFile
    ;write to file at offset
    mov     rdi,    qword[FD]
    mov     rsi,    myText4
    mov     rdx,    qword[myText4_Len]
    call    writeFile
    ;close file
    mov     rdi,    qword[FD]
    call    closeFile
    %ENDIF
%IF READ
;OPEN AND READ FROM A FILE AND THEN CLOSE IT
    ;open file to read
    mov     rdi,    fileName
    call    openFile
    mov     qword[FD],  rax
    ;read from a file
    mov     rdi,    qword[FD]
    mov     rsi,    buffer
    mov     rdx,    bufferLength
    call    readFile
    mov     rdi,    rax
    call    printString
    mov     rdi,    qword[FD]
    call    closeFile
    %ENDIF
%IF O_READ
    ;open file to read
    mov     rdi,    fileName
    call    openFile
    mov     qword[FD],  rax
    ;position file at offset
    mov     rdi,    qword[FD]
    mov     rsi,    qword[myText2_Len]
    mov     rdx,    0
    call    positionFile    
    ;read from file
    mov     rdi,    qword[FD]
    mov     rsi,    buffer
    mov     rdx,    10
    call    readFile
    mov     rdi,    rax
    call    printString
    ;close file
    mov     rdi,    qword[FD]
    call    closeFile
    %ENDIF
%IF DELETE
    ;DELETE A FILE    
    ;delete file
    mov     rdi,    fileName
    call    deleteFile
    %ENDIF
    leave
    ret
;+++++++++++++++++++FILE MANIPULATION FUNCTIONS+++++++++++++++++++
global readFile
readFile:
    mov     rax,    NR_READ
    SYSCALL
    cmp     rax,    0
    jl      readerror
    mov     byte[rsi+rax],  0
    mov     rax,    rsi
    mov     rdi,    Ok_Read
    push    rax
    call    printString
    pop     rax
    ret
readerror:
    mov     rdi,    Error_Read
    call    printString
    ret
;Delete file function
global  deleteFile
deleteFile:
    mov     rax,    NR_UNLINK
    SYSCALL
    cmp     rax,    0
    jl      deleteerror
    mov     rdi,    Ok_Delete
    call    printString
    ret
deleteerror:
    mov     rdi,    Error_Delete
    call    printString
    ret
;Append file function
global appendFile
appendFile:
    mov     rax,    NR_OPEN
    mov     rsi,    O_READ_WRITE|O_APPEND
    SYSCALL
    cmp     rax,    0
    jl      appenderror
    mov     rdi,    Ok_Append
    push    rax
    call    printString
    pop     rax
    ret
    appenderror:
    mov     rdi,    Error_Append
    call    printString
    ret
;Open File function
global openFile
openFile:
    mov     rax,    NR_OPEN
    mov     rsi,    O_READ_WRITE
    SYSCALL
    cmp     rax,    0
    jl      openerror
    mov     rdi,    Ok_Open
    push    rax
    call    printString
    pop     rax
    ret
    openerror:
    mov     rdi,    Error_Open
    call    printString
    ret
;Write file function
global writeFile
writeFile:
    mov     rax,    NR_WRITE
    SYSCALL
    cmp     rax,    0 
    jl      writeerror
    mov     rdi,    Ok_Write
    call    printString
    ret
    writeerror:
    mov     rdi,    Error_Write
    call    printString
    ret
; Position file function
global positionFile
positionFile:
    mov     rax,    NR_LSEEK
    SYSCALL
    cmp     rax,    0
    jl      positionerror
    mov     rdi,    Ok_Position
    call    printString
    ret
    positionerror:
    mov     rdi,    Error_Position
    call    printString
    ret
;Close file cuntion
global closeFile
closeFile:
    mov     rax,    NR_CLOSE
    SYSCALL
    cmp     rax,    0
    jl      closeerror
    mov     rdi,    Ok_Close
    call    printString
    ret
    closeerror:
    mov     rdi,    Error_Close
    call    printString
    ret
;Create file function
global createFile
createFile:
    mov     rax,    NR_CREATE
    mov     rsi,    S_USER_READ|S_USER_WRITE
    SYSCALL
    cmp     rax,    0
    jl      createerror
    mov     rdi,    Ok_Create
    push    rax
    call    printString
    pop     rax
    ret
    createerror:    
    mov     rdi,    Error_Create
    call    printString
    ret
;PRINT FEEDBACK
global  printString
printString:
    mov     r12,    rdi
    mov     rdx,    0
    strLoop:
    cmp     byte[r12],0
    je      strDone
    inc     rdx
    inc     r12
    jmp     strLoop
    strDone:
    cmp     rdx,    0
    je      prtDone
    mov     rsi,    rdi 
    mov     rax,    1
    mov     rdi,    1
    SYSCALL
    prtDone:
        ret
```

## Section 33: Learning C/C++ for Low Level Programming: Language Features

### 107. CPP Newlines,Tabs and Spaces

### 108. Initializers in CPP

### 109. CPP Writing Styles

### 110. Statements and Expressions, RVALUE and LVALUE

### 111. Comma
```cpp
#include<iostream>
int main() {
  int a = 9;
  int b = 4;
  int c;
  c = a,b; // c = a
  std::cout << c <<std::endl; // prints 9
  c = (a, b); // c =b
  std::cout << c <<std::endl; // prints 4
  return 0;
}
```

### 112. Types and Variables

### 113. Literals in CPP

### 114. Defining Constants

### 115. Const Expressions
- `constexpr double twopi = 2*pi;`: handled during the compile time, not runtime

### 116. Beginning to Pointers

### 117. Namespaces in CPP

### 118. Scoping Variables and Static Keyword
- In C++, a static variable is guaranteed to be zero-initialized if there is no explicit initializer
  - Static variables are allocated in .bss and initialized at load time
  - Ref: https://stackoverflow.com/questions/3373108/why-are-static-variables-auto-initialized-to-zero
```cpp
#include<iostream>
int inc(int i){
  static int value;
  value +=i;
  return value;
}
int main() {
  std::cout << inc(5) << std::endl; // prints 5
  std::cout << inc(20) << std::endl; // prints 25
  std::cout << inc(100) << std::endl; // prints 125
  return 0;
}
``` 

### 119. Conditional Statements

## Section 34: Learning C/C++ for Low Level Programming: Types

### 120. Types in CPP - Binary, Octal and Hex
- binary: prefix 0b
- octal: prefix 0 - `int b = 0123;` is 83 in decimal
- decimal: default
- hexadecimal: prefix 0x

### 121. Floating Point types in CPP - Float, Double, Long Double

### 122. Char types in CPP
- char: ASCII
- char16_t: UTF-16
- char32_t: UTF-32
- signed char
- unsigned char
- wchar_t: unicode

### 123. Enum Types

### 124. Boolean Types and Comparison Operators

### 125. Void Type and Void Function

### 126. Operators 101

### 127. Default Values in CPP
- Global variables in C++ are guaranteed to be initialized to zero by default if you do not explicitly assign them a value

## Section 35: Conditional Statements

### 128. Switch Statement with ENUM

### 129. Conditional Statements in CPP

### 130. For Loop

### 131. Continue Keyword

### 132. Do-While Loop

### 133. Goto
```cpp
goto mylabel;
...
mylabel:
...
```

## Section 36: Learning C/C++ for Low Level Programming: Pointers

### 134. Pointers in Practice

### 135. Pointers in Practice - Part 2

### 136. Pointers in Practice - Part 3

### 137. Pointers in Practice - Part 4

## Section 37: Learning C/C++ for Low Level Programming: Functions

### 138. Introduction to Functions

### 139. Functions - Part 1

### 140. Functions - Part 2

### 141. Functions - Part 3

### 142. Functions - Part 4

### 143. Functions - Part 5

### 144. Functions - Part 6

### 145. Functions - Part 7

## Section 37: Learning C/C++ for Low Level Programming: Arrays and Pointers

### 146. Understanding Arrays

### 147. Manipulating Arrays

### 148. Starting with Array Pointers

### 149. Pointer Increment vs Array Indexing

