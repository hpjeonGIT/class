## Title: Assembly Language Foundation Course for Ethical Hackers
- Instructor: Swapnil Singh

## Section 1: Introduction

1. Who can join this course and what skills needed for this course

## Section 2: Building the basics for assembly language programming

2. Computer system working and the CPU role

3. Internal components of the CPU

4. What are the registers inside the CPU?
- Registers
  - General purpose
![gen](./ch04_genRegister.png)  
  - Segment
![seg](./ch04_segRegister.png)  
    - CS: Code segment contains the pointer to the code segment in memory. Code segments is where the instruction codes are stored
    - DS/ES/FS/GS: Data segment pointer
    - SS: Stack segment
  - Instruction pointer    
![IP](./ch04_instRegister.png)

5. Flags of the CPU
- Status Flag: success or fail if the instruction succeeds or fails

6. Flag structure of the CPU
- Status flags: Collection of binary indicators that represent the current state of processor
![status_flags](./ch06_statusflag.png)
- System flags: Control the overall behavior of the CPU and its intersection with the system
![system_flags](./ch06_sysflags.png)
- Control flags: Control specific behavior in the processor. Only 1 (DF) in 32bit 
![control_flags](./ch06_controlflags.png)

7. Flags working in CPU
- Ref: https://www.ic.unicamp.br/~celio/mc404-2006/flags.html
- Carry Flag: When the number exceeds the highest possible number. Indicates the result isn't correct when interpreted as unsigned
- Parity Flag: this flag is set to 1 when there is even number of one bits in result, and to 0 when there is odd number of one bits
- Auxiliary Flag: set to 1 when there is an unsigned overflow for low nibble (4 bits)
- Zero Flag: set to 1 when result is zero. For none zero result this flag is set to 0
- Sign Flag: set to 1 when result is negative. When result is positive it is set to 0
- Overflow Flag: Indicates the result isn't correct when interpreted as signed
  - CF and OF are not same
  - Ref: https://stackoverflow.com/questions/69124873/understanding-the-difference-between-overflow-and-carry-flags
- Direction Flag: this flag is used by some instructions to process data chains, when this flag is set to 0 - the processing is done forward, when this flag is set to 1 the processing is done backward

8. Program's memory layout in the computer system

| Memory layout|
| :--: |
| command line arguments & environment variables |
| stack |
|  ... |
| heap |
| Unintialized data (.bss) |
| Initialized data  (.data) |
| .text (executable code instructions )|

- Sample.c
```c
#include <stdio.h>
int a;      // to .bss
int b=4;    // to .data
int main()  // entire function goes to stack
{ 
  return 0; // to .text
}
```
- gcc sample.c
- size ./a.out
```bash
   text	   data	    bss	    dec	    hex	filename
   1418	    548	     12	   1978	    7ba	./a.out
```
- Default bss size is 8 bytes (when there is no uninitalized data)
  - By `int a`, 4+8 = 12 bytes
- Default data has 544 bytes (when there is no initialized data)

9. How to view the stack of a program
```c
#include <stdio.h>
int a;      // to .bss
int b=4;    // to .data
int main()  // entire function goes to stack
{
  while (1) {
    sleep(1);
  }
  return 0;
}
```
- Ruinnning the sample code
```bash
- gcc ch9.c
- a.out& (infinite loop)
$ ./a.out &
[3] 88419  # find this process from /proc folder
$ cat /proc/88419/maps |grep stack
7ffce0dff000-7ffce0e20000 rw-p 00000000 00:00 0                          [stack]
```
  - This shows the range of stack memory

## Section 3: Hello World in Assembly

10. Structure of the assembly program
- Basic assembly code structure
```asm
.section .data

.section .bss

.section .text

.globl _start

_start:                         
```
- as ch10.s -o ch10.o
- ld ch10.o -o a.exe
  - a.exe will not run but just a sample demo
```bash
$ size a.exe 
   text	   data	    bss	    dec	    hex	filename
      0	      0	      0	      0	      0	a.exe
```

11. System calls Before Hello World in Assembly
```c
#include <stdio.h>
int main() {
  printf("hello world\n");
  return 0;
}
```
- system call through strace:
```bash
$ gcc ch11.c
$ strace -c ./a.out
hello world
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
  0.00    0.000000           0         1           read
  0.00    0.000000           0         1           write
  0.00    0.000000           0         2           close
  0.00    0.000000           0        16        15 stat
  0.00    0.000000           0         3           fstat
  0.00    0.000000           0         7           mmap
  0.00    0.000000           0         3           mprotect
  0.00    0.000000           0         1           munmap
  0.00    0.000000           0         3           brk
  0.00    0.000000           0         6           pread64
  0.00    0.000000           0         1         1 access
  0.00    0.000000           0         1           execve
  0.00    0.000000           0         2         1 arch_prctl
  0.00    0.000000           0        26        24 openat
------ ----------- ----------- --------- --------- ----------------
100.00    0.000000                    73        41 total
$ strace ./a.out
execve("./a.out", ["./a.out"], 0x7ffea76f5280 /* 62 vars */) = 0
brk(NULL)                               = 0x555e09bf4000
...
write(1, "hello world\n", 12hello world
)           = 12
exit_group(0)                           = ?
+++ exited with 0 +++
```
- As shown above, write() function has 3 arguments
- Let's try python now
  - hello.py
```py
print("hello world")
```
- strace for python
```bash
$ strace python3 hello.py
...
write(1, "hello world\n", 12hello world
)           = 12
...
exit_group(0)                           = ?
+++ exited with 0 +++
```
- Python3 uses the same system call of write()

12. Writing our first Hello World program in assembly
- Steps
  - Find the system call number of write
  - Send arguments for write()
- Number of system calls is found at /usr/include/asm-generic/unistd.h
  - Q: This doesn't match. Use https://blog.rchapman.org/posts/Linux_System_Call_Table_for_x86_64/
```c
#define __NR_write 64
__SYSCALL(__NR_write, sys_write)
```
  - In 32bit OS, the call number of write() is 4 
- ch12.s
```asm
.section .data
       msg: 
            .ascii "Hello world\n"
.section .text
.globl _start
_start: 
       movl $4, %eax   # syscall number
       movl $1,  %ebx  # file descriptor for write syscall
       movl $msg,%ecx  # move buffer point to write syscall
       movl $13, %edx  # 13 bytes of Hello world\n       
       int  $0x80      # interrupt - calling system call
       movl $1, %eax   # 1 is syscall number for exit
       movl $0, %ebx   # 0 argument for exit() 
       int $0x80       # activate exit()
```
- Demo:
```bash
$ as ch12.s -o ch12.o; ld ch12.o -o ch12.exe
$ ./ch12.exe 
Hello world
```
- This runs as 32bit mode as it uses `int $0x80` for interrupt
  - int $0x80: interrupt at 0x80, asking to check %eax and %ebx. When %eax is 1, it is exit(). The value of %ebx is the status code for exit()
    - Ref: https://stackoverflow.com/questions/1817577/what-does-int-0x80-mean-in-assembly-code
- Following is the 64bit version
```asm
.section .data
       msg: 
            .ascii "Hello world\n"
.section .text
.globl _start
_start: 
       mov $1,  %rax   # syscall number
       mov $1,  %rdi  # file descriptor for write syscall
       mov $msg,%rsi  # move buffer point to write syscall
       mov $13, %rdx  # 13 bytes of Hello world\n       
      syscall      # interrupt - calling system call
       mov $60, %rax  
       mov $0, %rbx   # 0 argument for exit() 
      syscall       # activate exit()
```
  - Ref: https://jameshfisher.com/2018/03/10/linux-assembly-hello-world/

13. Compiling an assembly program in gcc
- Rename `_start` with `main`
```asm
.section .data
       msg: 
            .ascii "Hello world\n"
.section .text
.globl main
main: 
       movl $4, %eax   # syscall number
       movl $1,  %ebx  # file descriptor for write syscall
       movl $msg,%ecx  # move buffer point to write syscall
       movl $13, %edx  # 13 bytes of Hello world\n       
       int  $0x80      # interrupt - calling system call
       movl $1, %eax   # 1 is syscall number for exit
       movl $0, %ebx   # 0 argument for exit() 
       int $0x80       # activate exit()
```
- Then compile using `gcc hello.s -o hello.exe -no-pie`

14. Debugging our assembly program
```bash
$ as ch12_32bit.s -o hello.o
$ ld hello.o -o hello.exe
$ gdb -q ./hello.exe
Reading symbols from ./hello.exe...
(No debugging symbols found in ./hello.exe)
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	mov    eax,0x4
   0x0000000000401005 <+5>:	mov    ebx,0x1
   0x000000000040100a <+10>:	mov    ecx,0x402000
   0x000000000040100f <+15>:	mov    edx,0xd # d in hex => 13 in decimal
   0x0000000000401014 <+20>:	int    0x80
   0x0000000000401016 <+22>:	mov    eax,0x1
   0x000000000040101b <+27>:	mov    ebx,0x0
   0x0000000000401020 <+32>:	int    0x80
gdb-peda$ b * 0x0000000000401000
Note: breakpoint 1 also set at pc 0x401000.
Breakpoint 2 at 0x401000
gdb-peda$ run
...
[-------------------------------------code-------------------------------------]
   0x400ffa:	add    BYTE PTR [rax],al
   0x400ffc:	add    BYTE PTR [rax],al
   0x400ffe:	add    BYTE PTR [rax],al
=> 0x401000 <_start>:	mov    eax,0x4
   0x401005 <_start+5>:	mov    ebx,0x1
   0x40100a <_start+10>:	mov    ecx,0x402000
   0x40100f <_start+15>:	mov    edx,0xd
   0x401014 <_start+20>:	int    0x80
...
gdb-peda$ info registers
rax            0x0                 0x0  # empty at this moment
rbx            0x0                 0x0
rcx            0x0                 0x0
rdx            0x0                 0x0
rsi            0x0                 0x0
...
gdb-peda$ ni # next instruction
...
gdb-peda$ info registers
rax            0x4                 0x4 # <--- now this is 4
...
gdb-peda$ ni 
gdb-peda$ ni 
gdb-peda$ info registers
rax            0x4                 0x4
rbx            0x1                 0x1
rcx            0x402000            0x402000
...
gdb-peda$ x/s 0x402000
0x402000:	"Hello world\n"
...
gdb-peda$  p/x $eax  # print in hexa
$4 = 0x4
gdb-peda$  p/d $eax  # print in decimal
$5 = 4
gdb-peda$ p/t $eax   # print in binary
$6 = 100

```
- No debug info at this moment
- If gdb continues to run, instead of pausing at break point, add `nop` in the top of _start section

15. Using C library functions in assembly program
```asm
.section .data
   msg: 
       .ascii "Hello World\n"
.section .text
.globl _start
_start:
   pushl $msg
   call printf # this is a C function. Needs -lc from ld command
   pushl $0    # Exit function
   call exit
```
- as hello.s -o hello.o
- ld -lc -dynamic-linker /lib/ld-linux.so.2 hello.o -o hello
  - This is for 32bit OS

## Section 4: Moving Data

16. Defined the data in data section in assembly program

| Directive | Data type |
|-----------|-----------|
| .ascii | Text string |
| .asciz | Null-terminated text string |
| .byte  | Byte value |
| .double| Double-precision floating-point number |
| .float | Single-precision floating-point number |
| .int  | 32bit integer |
| .long | 32bit integer |
| .octa | 16-bit integer |
| .quad | 8-byte integer |
| .short | 16-bit integer |
|. single| Single precision floating-point number |

- Sampe C code
```c
#include <stdio.h>
char *string = "Hello world\n";
int a = 3;
double pi = 3.14;
int main() {
  return 0;
}
```
- Sample asm code
```asm
.section .data
   string:
      .ascii "Hello world\n"
   a:
       .int 3
   pi: 
       .double 3.14
.section .text
.globl _start
_start:
   movl $1, %eax
   movl $0, %ebx
   int $0x80
```
- as asm.s -o asm.o
- ld asm.o -o asm.exe
- gdb -q ./asm.exe
```bash
gdb-peda$ b * _start
Breakpoint 1 at 0x401000
gdb-peda$ run
...
gdb-peda$ info variables
All defined variables:
Non-debugging symbols:
0x0000000000402000  string
0x000000000040200c  a
0x0000000000402010  pi
0x0000000000402018  __bss_start
0x0000000000402018  _edata
0x0000000000402018  _end
gdb-peda$ x/s 0x0000000000402000
0x402000:	"Hello world\n\003"  # the last 003 is from 0x000000000040200c
gdb-peda$ x/d &a
0x40200c:	3
gdb-peda$ x/f &pi
0x402010:	3.1400000000000001
```

17. Using static symbols in assembly programs
```asm
.section .data
       msg: 
            .ascii "Hello world\n"
      .equ str_len, 13
.section .text
.globl _start
_start: 
       movl $4, %eax   # syscall number
       movl $1,  %ebx  # file descriptor for write syscall
       movl $msg,%ecx  # move buffer point to write syscall
       movl $str_len, %edx  #  Using static symbol
       int  $0x80      # interrupt - calling system call
       movl $1, %eax   # 1 is syscall number for exit
       movl $0, %ebx   # 0 argument for exit() 
       int $0x80       # activate exit()
```
- as as17.s -o as17.o
- ld as17.o -o as17.exe

18. How to define and use data in the bss section
- man read:
```bash
EAD(2)                    Linux Programmer's Manual                   READ(2)
NAME
       read - read from a file descriptor
SYNOPSIS
       #include <unistd.h>
       ssize_t read(int fd, void *buf, size_t count);
```
- A sample assembly code with 32bit:
```asm
.section .bss
        .comm buffer,15 # common memory
.section .text
.globl _start
_start:
       # read syscall for taking user input
       movl $3, %eax
       movl $0, %ebx # file descriptor for input
       movl $buffer, %ecx # buffer pointer
       movl $15, %edx
       int $0x80
       # write syscall for printing the buffer
       movl $4, %eax
       movl $1, %ebx
       movl $buffer, %ecx
       movl $15, %edx
       int $0x80
       # exit syscall to exit the program
       movl $1, %eax
       movl $0, %ebx
       int $0x80
```
- as as18.s -o as18.o
- ld as18.o -o as18.exe
-  ./as18.exe
  - Enter: Hello Udemy
  - Shows: Hello Udemy
- A following is the 64bit version
```asm
section .bss
        .comm buffer,15 # common memory
.section .text
.globl _start
_start:
       # read syscall for taking user input
       mov $0, %rax
       mov $0, %rdi # file descriptor for input. Note %rdi
       mov $buffer, %rsi # buffer pointer. Note %rsi
       mov $15, %rdx
       syscall
       # write syscall for printing the buffer
       mov $1, %rax
       mov $1, %rdi
       mov $buffer, %rsi
       mov $15, %rdx
       syscall
       # exit syscall to exit the program
       mov $60, %rax
       mov $0, %rbx
       syscall
```

19. Moving data in Assembly programming
- Syntax
  - mov* source, destiny
  - Ex: `movl $4, %eax`
- mov data size
  - movl: 32bit long data
  - movw: 16bit word data
  - movb: 8bit byte data

20. Practical Demonstration of moving data in assembly
```asm
.section .text
.globl _start
_start:
  movl $25, %eax
  movw $4, %bx
  movb $1, %cl
  movl $1, %eax
  movl $0, %ebx
  int $0x80
```
- Demo:
```bash
$ as as20.s -o as20.o
$ ld as20.o -o as20.exe
$ gdb -q as20.exe
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
=> 0x0000000000401000 <+0>:	mov    eax,0x19
   0x0000000000401005 <+5>:	mov    bx,0x4
   0x0000000000401009 <+9>:	mov    cl,0x1
   0x000000000040100b <+11>:	mov    eax,0x1
   0x0000000000401010 <+16>:	mov    ebx,0x0
   0x0000000000401015 <+21>:	int    0x80
End of assembler dump.
gdb-peda$ b * _start
gdb-peda$ run
gdb-peda$ ni # run ni 3 times
gdb-peda$ info registers
rax            0x19                0x19
rbx            0x4                 0x4
rcx            0x1                 0x1
...
```
- 64bit [rcx] => 34bit [ecx] => 16bit [cx] = 16bit [ch][cl]
- If the `movb $1, %cl` is replaced with `movb $1, %ch`, $rcx becomes [01][00], which becomes 256 in decimal
```bash
gdb-peda$ p $rcx
$1 = 0x100
```

21. More advanced data movements in assembly
```asm
.section .bss
   .comm mydata,4  # assigns 4 bytes into mydata (address might be 0x402000)
.section .text
.globl _start
_start:
       nop
       movl $100, mydata     # mydata is now 100
       movl mydata,%ecx      # 100 into %ecx
       movl $mydata, %edx    # move the address of mydata into edx
       movl $500, %eax       # move 500 into eax
       movl %eax,(%edx)      # move eax value to the content of the address
       movl $1, %eax
       movl $0, %ebx
       int $0x80
```
- In GDB,
  - To see the variable: x/d &mydata
- Q: after 500->eax->(edx), edx becomes -12. Any overflow?
  - 200 works OK. 300 not.

22. Accessing and moving indexed values in assembly
```asm
.section .data
  Numbers:
    .int 10,20,30,40,50,60
.section .text
.globl _start
_start:
  # base_address(offset_address,index,size) index begins from 0
  # Number(,2,4) when offset is 0
  movl $2, %edi
  movl Numbers(,%edi,4), %eax
  #movl Numbers(,2,4), %eax # this doesn't work. index must be given from register
  movl $1,%eax
  movl $0,%ebx
  int $0x80
```
- In gdb, disassemble the code
- Make sure $eax has 30 when done

23. Direct and indirect addressing in assembly
- Direct memory addressing
  - `movl $5, 0xAAAA`: put value 5 into 0xAAAA
- Indirect memory addressing
  - When we cannot access the memory address
  - `movl 0xAAAA,eax`: put the address into a register
  - `movl $5,(eax)`: put 5 into the address which eax points to

24. Practical example of direct and indirect addressing in assembly
```asm
.section .data
  Number:
    .int 0
.section .text
.globl _start
_start:
  nop
  #direct addressing
  movl $5, Number
  #indirect addressing
  movl $Number, %eax # move the address of Number into eax
  movl $10,(%eax)    # move 10 into the what eax points to
  movl $1,%eax
  movl $0,%ebx
  int $0x80
```
- Demo:
```bash
$ as as24.s -o as24.o
$ ld as24.o -o as24.exe
$ gdb -q as24.exe
Reading symbols from as24.exe...
(No debugging symbols found in as24.exe)
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	nop
   0x0000000000401001 <+1>:	mov    DWORD PTR ds:0x402000,0x5
   0x000000000040100c <+12>:	mov    eax,0x402000
   0x0000000000401011 <+17>:	mov    DWORD PTR [eax],0xa
   0x0000000000401018 <+24>:	mov    eax,0x1
   0x000000000040101d <+29>:	mov    ebx,0x0
   0x0000000000401022 <+34>:	int    0x80
End of assembler dump.
gdb-peda$ b * _start
Breakpoint 1 at 0x401000
gdb-peda$ run
gdb-peda$ ni 
gdb-peda$ ni 
gdb-peda$ x/d &Number
0x402000:	5
gdb-peda$ ni 
gdb-peda$ ni 
gdb-peda$ x/d &Number
0x402000:	10
```

25. Concept of indirect address pointer
```asm
.section .data
  MyNumber:
    .int 4,8  # 8 byte memory space is made
    #  |0 0 0 8| 0 0 0 4| value
    #  |7 6 5 4| 3 2 1 0| index
.section .text
.globl _start
_start:
  nop
  nop
  movl $MyNumber, %eax
  movl $2,  (%eax)  # |0 0 0 8| 0 0 0 2| at [eax]
  movb $7, 1(%eax)  # |0 0 0 8| 0 0 7 2| at [eax +0x1]
  movb $9, 2(%eax) # |0 0 0 8| 0 9 0 2| at [eax +0x2]
  movw $3, 6(%eax) # |0 3 0 8| 0 9 0 2| at [eax +0x6]
  movl $1,%eax
  movl $0,%ebx
  int $0x80
```
- Note that **(%eax) and 1(%eax) are not same**
- Demo:
```bash
$ as as25.s -o as25.o
$ ld as25.o -o as25.exe
$ gdb -q as25.exe
gdb-peda$ b * _start
Breakpoint 1 at 0x401000
gdb-peda$ run
gdb-peda$ ni
gdb-peda$ ni
gdb-peda$ ni
gdb-peda$ x/8d $eax
0x402000:	4	0	0	0	8	0	0	0  # MyNumber, as it is
gdb-peda$ ni
gdb-peda$ x/8d $eax
0x402000:	2	0	0	0	8	0	0	0  # 2 is located in the 0th index
gdb-peda$ ni
gdb-peda$ x/8d $eax
0x402000:	2	7	0	0	8	0	0	0  # 7 is at 1st index 
gdb-peda$ ni
gdb-peda$ x/8d $eax
0x402000:	2	7	9	0	8	0	0	0  # 9 is at 2nd index
gdb-peda$ ni
gdb-peda$ x/8d $eax
0x402000:	2	7	9	0	8	0	3	0  # 3 is at 6th index
```

26. Accessing indexed memory locations in assembly
```asm
.section .data
  my_list:
    .int 11,22,33
.section .text
.globl _start
_start:
  nop
  nop
  # base_address (offset_address, index, size)
  movl $2, %edi
  movl  my_list(,%edi,4), %eax # zero-offset, 2nd index, size of 4 bytes
  movl $1,%eax
  movl $0,%ebx
  int $0x80
```
- 0 offset must be empty or use register
```asm
movl $0, %ebx
movl my_lsit(%ebx, %edi,4), %eax
```
- Results:
```bash
$gdb-peda$ info registers
rax            0x21                0x21
```
- Hexa 21 is 33 in decimal

27. How to create a stack frame in assembly
```asm
.section .text
.globl _start
_start:
   nop
   nop
   # copy the esp address to ebp
   movl %esp, %ebp
   # create some memory space in stack frame
   subl $8, %esp # from esp, subtract 8 bytes
   # exit
   movl $1,%eax
   movl $0,%ebx
   int $0x80
```
- %esp is the stack pointer
  - subtracting 8 bytes will move the esp to 8 bytes above
- At _start(), a stack frame is created
```bash
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
=> 0x0000000000401000 <+0>:	nop
   0x0000000000401001 <+1>:	nop
   0x0000000000401002 <+2>:	mov    ebp,esp
   0x0000000000401004 <+4>:	sub    esp,0x8
   0x0000000000401007 <+7>:	mov    eax,0x1
   0x000000000040100c <+12>:	mov    ebx,0x0
   0x0000000000401011 <+17>:	int    0x80
```
- Initially, ebp is empty
```bash
rbp            0x0                 0x0
rsp            0x7fffffffd630      0x7fffffffd630
```
- After esp moves to ebp:
```bash
rbp            0xffffd630          0xffffd630
rsp            0x7fffffffd630      0x7fffffffd630
gdb-peda$ p $ebp
$2 = 0xffffd630
gdb-peda$ p $esp
$4 = 0xffffd630
gdb-peda$ p/d $ebp-$esp
$6 = 0  # same address
```
  - Now ebp and esp are same
- After subtracting 8 bytes from esp:
```bash
gdb-peda$ p/d $ebp-$esp
$7 = 8
gdb-peda$ p $esp
$8 = 0xffffd628
gdb-peda$ p $ebp
$9 = 0xffffd630
```

28. Adding and removing data on stack in assembly
- Scenario
  - push A
  - push B
    - esp points the latest data pushed (B now)
  - ebp is the bottom of the stack
  - pop (remove)
  - Now esp points A
- 32bit code:
```asm
.section .text
.globl _start
_start:
  nop
  nop
  # copy the data address to ebp
  movl %esp, %ebp
  # create some memory space in stack frame
  subl $8, %esp
  # adding the data in stack frame
  movl $100, %eax
  pushl %eax
  movl $200, %ebx
  pushl %ebx
  # removing data from stack
  popl %ebx
  popl %eax
  # exit
  movl $1, %eax
  movl $0, %ebx
  int $0x80
```
- 64bit code
```asm
.section .text
.globl _start
_start:
  nop
  nop
  # copy the data address to ebp
  mov %rsp, %rbp
  # create some memory space in stack frame
  sub $8, %rsp
  # adding the data in stack frame
  mov $100, %rax
  push %rax
  mov $200, %rbx
  push %rbx
  # removing data from stack
  pop %rbx
  pop %rax
  # exit
  syscall
```
- Demo:
```bash
$ as as28.as -o as28.o
$ ld as28.o -o as28.exe
$ gdb -q as28.exe
Reading symbols from as28.exe...
(No debugging symbols found in as28.exe)
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	nop
   0x0000000000401001 <+1>:	nop
   0x0000000000401002 <+2>:	mov    rbp,rsp
   0x0000000000401005 <+5>:	sub    rsp,0x8
   0x0000000000401009 <+9>:	mov    rax,0x64
   0x0000000000401010 <+16>:	push   rax
   0x0000000000401011 <+17>:	mov    rbx,0xc8
   0x0000000000401018 <+24>:	push   rbx
   0x0000000000401019 <+25>:	pop    rbx
   0x000000000040101a <+26>:	pop    rax
   0x000000000040101b <+27>:	syscall 
End of assembler dump.
...
# running <+5>
gdb-peda$ x/3wx $rsp
0x7fffffffd628:	0x00000000	0x00000000	0x00000001
#---------------^rsp--------------------^rbp
# running <+16>
gdb-peda$ x/3wx $rsp
0x7fffffffd620:	0x00000064	0x00000000	0x00000000
#---------------^ value from rax. Note that rsp has shifted 8bytes above
# running <+24>
gdb-peda$ x/3wx $rsp
0x7fffffffd618:	0x000000c8	0x00000000	0x00000064
#---------------^ value from rbx. rsp has shifted another 8 bytes
# running <+25>
gdb-peda$ x/3wx $rsp
0x7fffffffd620:	0x00000064	0x00000000	0x00000000
#---------------^ value from rbx is gone
# running <+26>
gdb-peda$ x/3wx $rsp
0x7fffffffd628:	0x00000000	0x00000000	0x00000001
#---------------^ value from rax is gone 
```

29. Data exchange instructions in assembly
- xchg: exchanging two addresses
- bswap: byte swapping (not bit swap!)
```asm
.section .text
.globl _start
_start:
  movl $3, %eax
  movl $5, %ebx
  xchg %eax,%ebx
  #
  movl $0x12345678, %ecx
  bswap %ecx
  # exit
  movl $1, %eax
  movl $0, %ebx
  int $0x80
```  
- Demo:
```bash
$ as as29.as -o as29.o
$ ld as29.o -o as29.exe
$ gdb -q as29.exe
Reading symbols from as29.exe...
(No debugging symbols found in as29.exe)
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	mov    eax,0x3
   0x0000000000401005 <+5>:	mov    ebx,0x5
   0x000000000040100a <+10>:	xchg   ebx,eax
   0x000000000040100b <+11>:	mov    ecx,0x12345678
   0x0000000000401010 <+16>:	bswap  ecx
   0x0000000000401012 <+18>:	mov    eax,0x1
   0x0000000000401017 <+23>:	mov    ebx,0x0
   0x000000000040101c <+28>:	int    0x80
End of assembler dump.
# Running <+5>
gdb-peda$ p $eax
$3 = 0x3
gdb-peda$ p $ebx
$4 = 0x5
# Running <+10>
gdb-peda$ p $eax
$5 = 0x5
gdb-peda$ p $ebx
$6 = 0x3
# Running <+11>
gdb-peda$ p $ecx
$7 = 0x12345678
# Running <+16>
gdb-peda$ p $ecx
$8 = 0x78563412
```

30. Setting and clearing the flag bits Carry Flag
- Carry Flag: unsigned overflow
- stc: set the carry flag, CF=1
- clc: clears the carry flag, CF=0
- cmc: complements the carry flag, CF=NOT(CF)
```asm
.section .text
.globl _start
_start:
  nop
  nop
  # set the CF
  stc
  # unset the CF
  clc
  nop
  nop
  # exit
  movl $1, %eax
  movl $0, %ebx
  int $0x80
```
- Demo:
```bash
$ as as30.as -o as30.o
$ ld as30.o -o as30.exe
$ gdb -q as30.exe
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	nop
   0x0000000000401001 <+1>:	nop
   0x0000000000401002 <+2>:	stc    
   0x0000000000401003 <+3>:	clc    
   0x0000000000401004 <+4>:	nop
   0x0000000000401005 <+5>:	nop
   0x0000000000401006 <+6>:	mov    eax,0x1
   0x000000000040100b <+11>:	mov    ebx,0x0
   0x0000000000401010 <+16>:	int    0x80
End of assembler dump.
gdb-peda$ b * _start
gdb-peda$ display / $eflags
gdb-peda$ run
1: $eflags = [ IF ]
# Running <+2>
1: $eflags = [ CF IF ] # CF is found
# Running <+2>
1: $eflags = [ IF ] # CF is gone
```

31. Setting and clearing the Overflow Flag in assembly
- Overflow Flag: Signed overflow
```asm
.section .text
.globl _start
_start:
  nop
  nop
  #set the OF
  movl $0x7fffffff, %eax 
  addl $1, %eax # inducing overflow
  # clear the OF
  xorl  %eax,%eax
  # exit
  movl $1, %eax
  movl $0, %ebx
  int $0x80
```
- In gdb:
```bash
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	nop
   0x0000000000401001 <+1>:	nop
   0x0000000000401002 <+2>:	mov    eax,0x7fffffff
   0x0000000000401007 <+7>:	add    eax,0x1
   0x000000000040100a <+10>:	xor    eax,eax
=> 0x000000000040100c <+12>:	mov    eax,0x1
   0x0000000000401011 <+17>:	mov    ebx,0x0
   0x0000000000401016 <+22>:	int    0x80
# running <+2>
gdb-peda$ p/d $eax
$4 = 2147483647
# running <+7>
gdb-peda$ p/d $eax
$5 = -2147483648  # over-flow
gdb-peda$ display / $eflags
1: $eflags = [ PF AF SF IF OF ] # now OF is found
# running <+10>, OF is gone
```

32. Setting and clearing the Parity Flag in assembly
- Why PF in modern processor: https://stackoverflow.com/questions/25707130/what-is-the-purpose-of-the-parity-flag-on-a-cpu
```asm
.section .text
.globl _start
_start:
  nop
  nop
  # setting PF
  movb $0xAA, %al  # 10101010 into al register
  test %al,%al     # 10101010 AND 10101010 = 10101010
  # clearing the PF
  movb $0x57, %al  # 1010111
  test %al,%al     # 1010111 AND 1010111 = 1010111
  nop
  nop
  #exit
  movl $1, %eax
  movl $0, %ebx
  int  $0x80
```
- Demo:
```bash
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	nop
   0x0000000000401001 <+1>:	nop
   0x0000000000401002 <+2>:	mov    al,0xaa
   0x0000000000401004 <+4>:	test   al,al
   0x0000000000401006 <+6>:	mov    al,0x57
   0x0000000000401008 <+8>:	test   al,al
   0x000000000040100a <+10>:	nop
   0x000000000040100b <+11>:	nop
   0x000000000040100c <+12>:	mov    eax,0x1
   0x0000000000401011 <+17>:	mov    ebx,0x0
   0x0000000000401016 <+22>:	int    0x80
End of assembler dump.
gdb-peda$ b * _start
gdb-peda$ run
gdb-peda$ display / $eflags
1: $eflags = [ IF ]
# Running <+4>
1: $eflags = [ PF SF IF ]
# Running <+8>
1: $eflags = [ IF ]
```

33. Setting and clearing the Sign Flag in assembly
```asm
.section .text
.globl _start
_start:
  nop
  nop
  # setting the SF
  movl $-1, %eax
  test %eax,%eax
  # clearing the SF
  movl $42,  %eax
  test %eax, %eax
  nop
  nop
  #exit
  movl $1, %eax
  movl $0, %ebx
  int  $0x80
```
- Demo using gdb:
```bash
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	nop
   0x0000000000401001 <+1>:	nop
   0x0000000000401002 <+2>:	mov    eax,0xffffffff
   0x0000000000401007 <+7>:	test   eax,eax
   0x0000000000401009 <+9>:	mov    eax,0x2a
   0x000000000040100e <+14>:	test   eax,eax
=> 0x0000000000401010 <+16>:	nop
   0x0000000000401011 <+17>:	nop
   0x0000000000401012 <+18>:	mov    eax,0x1
   0x0000000000401017 <+23>:	mov    ebx,0x0
   0x000000000040101c <+28>:	int    0x80
# Running <+7>
2: $eflags = [ PF SF IF ]
# Running <+14>
2: $eflags = [ IF ]
```

34. Setting and clearing the Zero Flag in assembly
- Zero Flag: when 0 is resulted like 0 + 0 = 0
```asm
.section .text
.globl _start
_start:
  nop
  nop
  # setting the Zero Flag
  movl $0, %eax
  test %eax,%eax  # 0 AND 0 = 0
  # clearing the Zero Flag
  movl $1, %eax
  test %eax, %eax
  nop
  nop
  #exit
  movl $1, %eax
  movl $0, %eax
  int  $0x80
```
- Demo
```bash
$ as as34.s -o as34.o
$ ld as34.o -o as34.exe
$ gdb -q as34.exe
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	nop
   0x0000000000401001 <+1>:	nop
   0x0000000000401002 <+2>:	mov    eax,0x0
   0x0000000000401007 <+7>:	test   eax,eax
   0x0000000000401009 <+9>:	mov    eax,0x1
   0x000000000040100e <+14>:	test   eax,eax
   0x0000000000401010 <+16>:	nop
   0x0000000000401011 <+17>:	nop
   0x0000000000401012 <+18>:	mov    eax,0x1
   0x0000000000401017 <+23>:	mov    eax,0x0
   0x000000000040101c <+28>:	int    0x80
End of assembler dump.
gdb-peda$ display / $eflags
gdb-peda$ b * _start
gdb-peda$ run
# Running <+7>
1: $eflags = [ PF ZF IF ]
# Running <+14>
1: $eflags = [ IF ]
```

## Section 5: Controlling Execution flow in assembly

35. Understanding EIP register in assembly
- EIP: Instruction pointer. It points the current location along gdb runs
  - When we use 'ni', then EIP(or RSP in 64bit) keeps changing
```asm
.section .text
.globl _start
_start:
  nop
  nop
  movl $4, %eax
  movl $3, %ebx
  movl $5, %ecx
  movl $7, %edx
  #exit
  movl $1, %eax
  movl $0, %eax
  int  $0x80
```
- Demo:
```bash
$ as as35.s -o as35.o
$ ld as35.o -o as35.exe
$ gdb -q as35.exe
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	nop
   0x0000000000401001 <+1>:	nop
   0x0000000000401002 <+2>:	mov    eax,0x4
   0x0000000000401007 <+7>:	mov    ebx,0x3
   0x000000000040100c <+12>:	mov    ecx,0x5
   0x0000000000401011 <+17>:	mov    edx,0x7
   0x0000000000401016 <+22>:	mov    eax,0x1
   0x000000000040101b <+27>:	mov    eax,0x0
   0x0000000000401020 <+32>:	int    0x80
End of assembler dump.
gdb-peda$ b * _start
Breakpoint 1 at 0x401000
gdb-peda$ run
gdb-peda$ i r rip
rip            0x401000            0x401000 <_start>
gdb-peda$ ni # rip (or eip) moves to the next instruction
gdb-peda$ i r rip
rip            0x401001            0x401001 <_start+1>
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	nop
=> 0x0000000000401001 <+1>:	nop  # => shows the current rip (or eip) location
   0x0000000000401002 <+2>:	mov    eax,0x4
   0x0000000000401007 <+7>:	mov    ebx,0x3
   0x000000000040100c <+12>:	mov    ecx,0x5
   0x0000000000401011 <+17>:	mov    edx,0x7
   0x0000000000401016 <+22>:	mov    eax,0x1
   0x000000000040101b <+27>:	mov    eax,0x0
   0x0000000000401020 <+32>:	int    0x80
End of assembler dump.
```

36. Jump instruction in assembly
```asm
.section .text
.globl _start
_start:
  nop
  nop
  movl $1, %eax
  movl $2, %ebx
  jmp jump_here # label or function name
  #exit syscall
  movl $1, %eax
  movl $999, %ebx
  int $0x80
jump_here:
  movl $100, %eax
  movl $200, %ebx  
```  
- Demo:
```bash
$ as as36.as -o as36.o
$ ld as36.o -o as36.exe
$ gdb -q as36.exe
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	nop
   0x0000000000401001 <+1>:	nop
   0x0000000000401002 <+2>:	mov    eax,0x1
   0x0000000000401007 <+7>:	mov    ebx,0x2
   0x000000000040100c <+12>:	jmp    0x40101a <jump_here>
   0x000000000040100e <+14>:	mov    eax,0x1
   0x0000000000401013 <+19>:	mov    ebx,0x3e7
   0x0000000000401018 <+24>:	int    0x80
End of assembler dump.
gdb-peda$ disassemble jump_here
Dump of assembler code for function jump_here:
   0x000000000040101a <+0>:	mov    eax,0x64
   0x000000000040101f <+5>:	mov    ebx,0xc8
End of assembler dump.
```
- There is no automatic return in jump_here and it will hang

37. What is the work of call instruction in assembly
- How to return to _start from jump_here in Section 36?
  - jmp vs call
    - No mechanism to return in jmp
    - When using call, use 'ret' in the correponding label or function
```asm
.section .text
.globl _start
_start:
  nop
  nop
  movl $1, %eax
  movl $2, %ebx
  call loc1 # label or function name
  #exit syscall
  movl $1, %eax
  movl $999, %ebx
  int $0x80
loc1:
  movl $100, %eax
  movl $200, %ebx  
  ret
```  
- Demo:
```bash
$ as as37.s -o as37.o
$ ld as37.o -o as37.exe
$ gdb -q as37.exe
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	nop
   0x0000000000401001 <+1>:	nop
   0x0000000000401002 <+2>:	mov    eax,0x1
   0x0000000000401007 <+7>:	mov    ebx,0x2
   0x000000000040100c <+12>:	call   0x40101d <loc1>
   0x0000000000401011 <+17>:	mov    eax,0x1
   0x0000000000401016 <+22>:	mov    ebx,0x3e7
   0x000000000040101b <+27>:	int    0x80
End of assembler dump.
gdb-peda$ disassemble loc1
Dump of assembler code for function loc1:
   0x000000000040101d <+0>:	mov    eax,0x64
   0x0000000000401022 <+5>:	mov    ebx,0xc8
   0x0000000000401027 <+10>:	ret    
End of assembler dump.
gdb-peda$ b * 0x000000000040100c # call loc1
gdb-peda$ b * 0x000000000040101d # first line of loc1
gdb-peda$ ni  #stops at <+12>
gdb-peda$ ni  #stops at <+0> of loc1
gdb-peda$ x/3xw $rsp
0x7fffffffd628:	0x00401011	0x00000000	0x00000001
#---------------^ this is the address of <+17> at _start
# this is how loc1 returns to the <+17> of _start
```

38. Conditional jump in indepth in assembly
- Compare instruction
  - `cmp source, destination`
  - Ex: 
    - `cmp %eax,%ebx`
      - Executes `sub %ebx-%eax`
        - When Zero Flag is found, jump will be taken
        - No zero Flag, then no jump
  - `je`: jump when equal
    - Actually this checks if ZF is set or not@gdfdfd
```asm
.section .text
.globl _start
_start:
  movl $4, %eax
  movl $8, %ebx
  cmp %eax, %ebx
  je  jump_here  # if eax == ebx then jump
  #exit
  movl $1, %eax
  movl $0, %ebx
  int $0x80
jump_here:
  #exit
  movl $1, %eax
  movl $7, %ebx
  int $0x80
```
- Demo:
```bash
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	mov    eax,0x4
   0x0000000000401005 <+5>:	mov    ebx,0x8
=> 0x000000000040100a <+10>:	cmp    ebx,eax
   0x000000000040100c <+12>:	je     0x40101a <jump_here>
   0x000000000040100e <+14>:	mov    eax,0x1
   0x0000000000401013 <+19>:	mov    ebx,0x0
   0x0000000000401018 <+24>:	int    0x80
End of assembler dump.
gdb-peda$ p/d $ebx-$eax
$1 = 4 # Zero Flag will not be set
gdb-peda$ ni
gdb-peda$ i r $eflags
eflags         0x202               [ IF ] # no ZF
```

39. Using Zero Flag as a conditional jump in assembly
- jz: jump when zero
```asm
.section .text
.globl _start
_start:
  movl $4,%eax
  subl $4,%eax
  jz jump_here
  #exit
  movl $1,%eax
  movl $0,%ebx
  int $0x80
jump_here:
  movl $1,%eax
  movl $7,%ebx
  int $0x80
```
- Demo:
```bash
$ as as39.as -o as39.o
$ ld as39.o -o as39.exe
$ ./as39.exe
$ echo $?
7  # return value is 7 from jump_here
```
- From gdb:
```bash
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	mov    eax,0x4
   0x0000000000401005 <+5>:	sub    eax,0x4
   0x0000000000401008 <+8>:	je     0x401016 <jump_here> # jz is converted into je
   0x000000000040100a <+10>:	mov    eax,0x1
   0x000000000040100f <+15>:	mov    ebx,0x0
   0x0000000000401014 <+20>:	int    0x80
End of assembler dump.
```

40. Using Overflow Flag as conditional jump in assembly
- `jo`: jump when Overflow Flag is set
```asm
.section .text
.globl _start
_start:
  movl $0x7fffffff,%eax
  addl $1, %eax
  jo jump_here  
  #exit
  movl $1,%eax
  movl $0,%ebx
  int $0x80
jump_here:
  movl $1,%eax
  movl $7,%ebx
  int $0x80
```
- Demo:
```bash
$ as as40.s -o as40.o
$ ld as40.o -o as40.exe
$ ./as40.exe
$ echo $?
7
```

41. Using Parity Flag as a conditional jump in assembly
- `jp`: jump when Parity Flag is set
```asm
.section .text
.globl _start
_start:
  movl $4,%eax
  subl $1, %eax # 3 in decimal = 11 in binary
  jp jump_here  
  #exit
  movl $1,%eax
  movl $0,%ebx
  int $0x80
jump_here:
  movl $1,%eax
  movl $7,%ebx
  int $0x80
```

## Section 6: Using numbers in assembly language

42. Types of numbers which are used in assembly programming
- Numeric data type
  - Integers
    - Unsigned
    - Signed
  - SIMD integers
    - MMX integers
    - SSE integers
  - Binary coded decimal
  - Floating point numbers

43. What are signed and unsigned integers and how unsigned integers are stored

44. How unsigned integers are stored in memory

45. Using unsigned integers in assembly program
```asm
.section .data
  mynumber: .byte 256 # 0...255
.section .text
.globl _start
_start:
  movb mynumber,%al # moving 256 into al reg
  # exit
  movl $1,%eax
  movl $0,%ebx
  int  $0x80
```
- Demo:
```bash
$ as as45.s -o as45.o
as45.s: Assembler messages:
as45.s:3: Warning: value 0x100 truncated to 0x0
$ ld as45.o -o as45.exe
$ gdb -q as45.exe
gdb-peda$ p/d $mynumber
$1 = 0
```

46. Concept of how a CPU stores the numbers in memory
- Decimal: 12345678 => Hex 0xbc614e by CPU => 0x4e61bc in RAM as little endian

47. Practical demonstration of how CPU store numbers in memory
```asm
.section .data
  decimal_number: .int 12345678
  hexadecimal_number: .long 0x12345678
.section .text
.globl _start
_start:
  nop
  nop
  #exit syscall
  movl $1,%eax
  movl $0,%ebx
  int  $0x80
```  
- In gdb:
```bash
gdb-peda$ p 12345678
$7 = 0xbc614e
gdb-peda$ x/4bx  &decimal_number
0x402000:	0x4e	0x61	0xbc	0x00
#---------^ reverse order of 0xbc 61 4e, which is 12345678
gdb-peda$ x/4bx &hexadecimal_number
0x402004:	0x78	0x56	0x34	0x12
#---------^ reverse order of 0x12345678
```

48. How to use signed numbers in assembly programming
```asm
.section .data
  number1: .int -27
  number2: .int -3
.section .text
.globl _start
_start:
  nop
  nop
  movl number1,%eax
  movl number2,%ebx
  addl %eax,%ebx
  #exit
  movl $1,%eax
  movl $2,%ebx
  int  $0x80
```
- Demo:
```bash
$ as as48.as -o as48.o
$ ld as48.o -o as48.exe
$ gdb -q as48.exe
gdb-peda$ p/x -27
$1 = 0xffffffe5
gdb-peda$ x/4bx &number1
0x402000:	0xe5	0xff	0xff	0xff
...
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	nop
   0x0000000000401001 <+1>:	nop
   0x0000000000401002 <+2>:	mov    eax,DWORD PTR ds:0x402000
   0x0000000000401009 <+9>:	mov    ebx,DWORD PTR ds:0x402004
   0x0000000000401010 <+16>:	add    ebx,eax
=> 0x0000000000401012 <+18>:	mov    eax,0x1
   0x0000000000401017 <+23>:	mov    ebx,0x2
   0x000000000040101c <+28>:	int    0x80
End of assembler dump.
gdb-peda$ p/d $ebx
$3 = -30
```

49. SIMD registers concept in assembly
- MMX (64bit)
- SSE (128bit)

50. MMX registers in assembly
- 64bit MMX register
  - from mm0 to mm7
  - Each mm* has 8bit
  - Can store 8 of  8bit integers
  - Can store 4 of 16bit integers
  - Can store 2 of 32bit integers
- Instructions
  - `movq source,destination`

51. Using mmx registers in assembly to move integers
```asm
.section .data
  value1: .int 1,2
.section .text
.globl _start
_start:
  #mov value1 into mmx register
  movq value1,%mm0
  #exit
  movl $1,%eax
  movl $0,%ebx
  int  $0x80
```
- Demo:
```bash
$ as as51.s -o as51.o
$ ld as51.o -o as51.exe
$ gdb -q as51.exe
Reading symbols from as51.exe...
(No debugging symbols found in as51.exe)
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	movq   mm0,QWORD PTR ds:0x402000
   0x0000000000401008 <+8>:	mov    eax,0x1
   0x000000000040100d <+13>:	mov    ebx,0x0
   0x0000000000401012 <+18>:	int    0x80
End of assembler dump.
gdb-peda$ b * _start
Breakpoint 1 at 0x401000
gdb-peda$ run
gdb-peda$ x/8x &value1
0x402000:	0x01	0x00	0x00	0x00	0x02	0x00	0x00	0x00
```
- Looks like mm0 is not shown appropriately in gdb
  - Values are found at st0 register
```bash
gdb-peda$ i r float
st0            <invalid float value> (raw 0xffff0000000200000001)
```
  - `p/x $mm0` shows `0x0`, implies that it doesn't work as expected

52. Understanding SSE registers
- 128bit SSE register
- Register names: xmm0, ... xmm7
  - Each 16bit
- Sixteen of 8bit integers
- Eight of 16bit word integers
- Four of 32bit integers
- Two of 64bit integers 
- Instruction
  - `movdqa source,destination`

53. Using SSE reigsters in assembly
```asm
.section .data
  value1: .int 1,2,3,4
  value2: .byte 1,2,3,4,5,6,7,8
.section .text
.globl _start
_start:
  movdqa value1,%xmm0
  movdqa value2,%xmm1
  #exit
  movl $1,%eax
  movl $0,%ebx
  int  $0x80
```
- Demo:
```bash
$ as as53.s -o as53.o
$ ld as53.o -o as53.exe
$ gdb -q as53.exe
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	movdqa xmm0,XMMWORD PTR ds:0x402000
   0x0000000000401009 <+9>:	movdqa xmm1,XMMWORD PTR ds:0x402010
   0x0000000000401012 <+18>:	mov    eax,0x1
   0x0000000000401017 <+23>:	mov    ebx,0x0
   0x000000000040101c <+28>:	int    0x80
gdb-peda$ b * _start
Breakpoint 1 at 0x401000
gdb-peda$ run
gdb-peda$ p $xmm0
$1 = {
  v4_float = {0, 0, 0, 0},
  v2_double = {0, 0},
  v16_int8 = {0x0 <repeats 16 times>},
  v8_int16 = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
  v4_int32 = {0x0, 0x0, 0x0, 0x0},
  v2_int64 = {0x0, 0x0},
  uint128 = 0x0
}
gdb-peda$ ni # running <+0>
gdb-peda$ p $xmm0
$2 = {
  v4_float = {1.40129846e-45, 2.80259693e-45, 4.20389539e-45, 5.60519386e-45},
  v2_double = {4.2439915824246103e-314, 8.4879831653432862e-314},
  v16_int8 = {0x1, 0x0, 0x0, 0x0, 0x2, 0x0, 0x0, 0x0, 0x3, 0x0, 0x0, 0x0, 0x4, 
    0x0, 0x0, 0x0},
  v8_int16 = {0x1, 0x0, 0x2, 0x0, 0x3, 0x0, 0x4, 0x0},
  v4_int32 = {0x1, 0x2, 0x3, 0x4},
  v2_int64 = {0x200000001, 0x400000003},
  uint128 = 0x4000000030000000200000001
}
gdb-peda$ p $xmm1
$3 = {
  v4_float = {0, 0, 0, 0},
  v2_double = {0, 0},
  v16_int8 = {0x0 <repeats 16 times>},
  v8_int16 = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
  v4_int32 = {0x0, 0x0, 0x0, 0x0},
  v2_int64 = {0x0, 0x0},
  uint128 = 0x0
}
gdb-peda$ ni # running <+9>
gdb-peda$ p $xmm1
$4 = {
  v4_float = {1.53998961e-36, 4.06321607e-34, 0, 0},
  v2_double = {5.447603722011605e-270, 0},
  v16_int8 = {0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x0, 0x0, 0x0, 0x0, 0x0, 
    0x0, 0x0, 0x0},
  v8_int16 = {0x201, 0x403, 0x605, 0x807, 0x0, 0x0, 0x0, 0x0},
  v4_int32 = {0x4030201, 0x8070605, 0x0, 0x0},
  v2_int64 = {0x807060504030201, 0x0},
  uint128 = 0x807060504030201
}
```

54. What are BCD data types
- BCD: Binary Coded Decimal
  - Rule: maximum decimal digit to convert in binary is 9. Any number more than 9 cannot be converted
  - Easy conversion b/w binary and decimal but wasteful as 1111,1110,1100,1101,1010,1011 are not used

| Binary coded | Decimal |
|--------------|---------|
| 0000 | 0 |
| 0001 | 1 |
| 0010 | 2 |
| 0011 | 3 |
| 0100 | 4 |
| 0101 | 5 |
| 0110 | 6 |
| 0111 | 7 |
| 1000 | 8 |
| 1001 | 9 |

- Ex:
  - 46 -> split into 4 & 6, [0100] & [0110]
- BCD data can be stored only in FPU registers like st0, ... st7
  - 80bits are assigned but only 72bits can be used for BCD. 79th, the last bit is for sign. 72..78 are not used
  - Ref: https://en.wikipedia.org/wiki/Intel_BCD_opcodes
- Instruction
  - `fbld source`: moves source data, which is BCD, into FPU (st*) registeres
- Why BCD?
  - Exact representation of decimal number
  - Ref: https://stackoverflow.com/questions/2359527/assembler-why-bcd-exists


55. Storing BCD data types in memory
```asm
.section .data
  bcd_data: .byte 0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9
.section .text
.globl _start
_start:
  fbld bcd_data # will store bcd_data into st0 register
  #exit
  movl $1, %eax
  movl $0, %ebx
  int  $0x80
```
- Demo:
```bash
$ as  as55.as -o as55.o
$ ld as55.o -o as55.exe
$ gdb -q as55.exe
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	fbld   TBYTE PTR ds:0x402000
   0x0000000000401007 <+7>:	mov    eax,0x1
   0x000000000040100c <+12>:	mov    ebx,0x0
   0x0000000000401011 <+17>:	int    0x80
End of assembler dump.
gdb-peda$ b * _start
gdb-peda$ run
gdb-peda$ ni
gdb-peda$ info reg st0
st0            90807060504030201   (raw 0x4037a14e452c3855fc80)
gdb-peda$ x/9x &bcd_data
0x402000:	0x01	0x02	0x03	0x04	0x05	0x06	0x07	0x08
0x402008:	0x09
```
- If bcd_data has 10 data instead of 9, 10th data will not be stored in st0 (not in st1 either)

56. How floating point numbers are moved
- Instruction
  - `fldl source`: move source, which is float data, into st0 (FPU)
```asm
.section .data
  myfloat1: .float 1.23
  myfloat2: .double 1234.5432
.section .bss
.lcomm data,8  # .lcomm allocate zero-initialized storage from bss, keeping it private
.section .text
.globl _start
_start:
  flds myfloat1  # sends 1.23 into st0
  fldl myfloat2  # sends 1234.5432 into st0, 1.23 of st0 moves to st1
  fstl data      # takes the value of st0 into data
  #exit 
  movl $1, %eax
  movl $0, %ebx
  int  $0x80
```

57. Storing floating point numbers into memory
- Demo:
```bash
$ as as56.s -o as56.o
$ ld as56.o -o as56.exe
$ gdb -q as56.exe
Reading symbols from as56.exe...
(No debugging symbols found in as56.exe)
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	fld    DWORD PTR ds:0x402000
   0x0000000000401007 <+7>:	fld    QWORD PTR ds:0x402004
   0x000000000040100e <+14>:	fst    QWORD PTR ds:0x402010
   0x0000000000401015 <+21>:	mov    eax,0x1
   0x000000000040101a <+26>:	mov    ebx,0x0
   0x000000000040101f <+31>:	int    0x80
End of assembler dump.
gdb-peda$ b * _start
Breakpoint 1 at 0x401000
gdb-peda$ run
gdb-peda$ ni # running <+0>
gdb-peda$ p $st0
$1 = 1.23000001907348632812
gdb-peda$ p $st1
$2 = 0
gdb-peda$ ni # running <+7>
gdb-peda$ p $st0
$3 = 1234.54320000000006985
gdb-peda$ p $st1  # previous st0 value has moved to st1
$4 = 1.23000001907348632812
gdb-peda$ x &data
0x402010 <data>:	""
gdb-peda& ni # running <+14>
gdb-peda$ x &data
0x402010 <data>:	0xc0
gdb-peda$ i r st0 # fstl doesn't change st0...st7
st0            1234.54320000000006985 (raw 0x40099a5161e4f7660000)
gdb-peda$ i r st1
st1            1.23000001907348632812 (raw 0x3fff9d70a40000000000)
```

58. Storing multiple float values at once
- How to send 4 floats into xmm0 in a single command?
- `movups source, destination`
  - movups: packed single
  - movupd: packed double
```asm
.section .data
  myfloat: .float 1.2, 3.6, 77.45, 11.06
.section .text
.globl _start
_start:
  movups myfloat,%xmm0
  #exit
  movl $1,%eax
  movl $0,%ebx
  int  $0x80
```
- Using gdb:
```bash
gdb-peda$ p $xmm0
$1 = {
  v4_float = {1.20000005, 3.5999999, 77.4499969, 11.0600004},
  v2_double = {179.19998149871827, 1111491.2601760863},
  v16_int8 = {0x9a, 0x99, 0x99, 0x3f, 0x66, 0x66, 0x66, 0x40, 0x66, 0xe6, 
    0x9a, 0x42, 0xc3, 0xf5, 0x30, 0x41},
  v8_int16 = {0x999a, 0x3f99, 0x6666, 0x4066, 0xe666, 0x429a, 0xf5c3, 0x4130},
  v4_int32 = {0x3f99999a, 0x40666666, 0x429ae666, 0x4130f5c3},
  v2_int64 = {0x406666663f99999a, 0x4130f5c3429ae666},
  uint128 = 0x4130f5c3429ae666406666663f99999a
}
```

## Section 7: Basic Math functions in assembly

59. Adding numbers in assembly
- addl: long, 32bit
- addw: word, 16bit
- addb: byte, 8bit
- Syntax: addx source,destination
```asm
.section .data
  value1: .int 4
  mybyte: .byte 7
.section .text
.globl _start
_start:
  movl $6,    %eax
  addl value1,%eax
  #
  movw $100,  %bx
  addw value1,%bx
  #
  movb $2,    %cl
  addb mybyte,%cl
  #exit
  movl $1, %eax
  movl $0, %ebx
  int  $0x80
```  
- Demo:
```bash
$ as as59.s -o as59.o
$ ld as59.o -o as59.exe
$ gdb -q as59.exe
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	mov    eax,0x6
   0x0000000000401005 <+5>:	add    eax,DWORD PTR ds:0x402000
   0x000000000040100c <+12>:	mov    bx,0x64
   0x0000000000401010 <+16>:	add    bx,WORD PTR ds:0x402000
   0x0000000000401018 <+24>:	mov    cl,0x2
   0x000000000040101a <+26>:	add    cl,BYTE PTR ds:0x402004
   0x0000000000401021 <+33>:	mov    eax,0x1
   0x0000000000401026 <+38>:	mov    ebx,0x0
   0x000000000040102b <+43>:	int    0x80
End of assembler dump.
gdb-peda$ b * _start
gdb-peda$ run
# Running <+0>
gdb-peda$ p/d $eax
$3 = 6
gdb-peda$ ni # now <+5>
gdb-peda$ p/d $eax
$4 = 10
...
# running <+16>
gdb-peda$ p/d $bx
$7 = 104
# running <+26>
gdb-peda$ p/d $cl
$10 = 9
```

60. Dealing with negative numbers in assembly
- How to negate a number
```asm
.section .text
.globl _start
_start:
  # add two byte integers
  movb $-3, %al
  movb $-10,%bl
  addb %al,%bl
  # add two 16bit (word) integers
  movw $-15,%ax
  movw $50, %bx
  addw %ax,%bx
  # add two 32bit integers
  movl $100,%eax
  movl $-30,%ebx
  addl %eax,%ebx
  # exit
  movl $1, %eax
  movl $0, %ebx
  int  $0x80
```
- Demo:
```bash

$ as as60.s -o as60.o
$ ld as60.o -o as60.exe
$ gdb -q as60.exe
Reading symbols from as60.exe...
(No debugging symbols found in as60.exe)
gdb-peda$ disassemble _start
Dump of assembler code for function _start:
   0x0000000000401000 <+0>:	mov    al,0xfd
   0x0000000000401002 <+2>:	mov    bl,0xf6
   0x0000000000401004 <+4>:	add    bl,al
   0x0000000000401006 <+6>:	mov    ax,0xfff1
   0x000000000040100a <+10>:	mov    bx,0x32
   0x000000000040100e <+14>:	add    bx,ax
   0x0000000000401011 <+17>:	mov    eax,0x64
   0x0000000000401016 <+22>:	mov    ebx,0xffffffe2
   0x000000000040101b <+27>:	add    ebx,eax
   0x000000000040101d <+29>:	mov    eax,0x1
   0x0000000000401022 <+34>:	mov    ebx,0x0
   0x0000000000401027 <+39>:	int    0x80
End of assembler dump.
gdb-peda$ b * _start
Breakpoint 1 at 0x401000
gdb-peda$ run
...
# running <+4>
gdb-peda$ p/d $bl
$5 = -13
# running <+14>
gdb-peda$ p/d $bx
$8 = 35
# running <+27>
gdb-peda$ p/d $ebx
$10 = 70
```

61. Concept of detecting crayyin addition operation

62. Writing assembly program for detecting carry flag in addition operation

63. Understanding overflow concept in addition arithmetic

64. Practical demonstration of overflow in addition to instruction in assembly

65. Detecting oveflow in signed integers addition in assembly

66. Concept of add carry instruction in assembly

67. Using ADC instruction in assembly

68. How binary subtraction works

69. Using sub instruction in assembly program

70. Increment and decrement instruction in assembly

71. Multiplication of two unsigned numbers in assembly

72. Different ways of multiplications of signed numbers in assembly

73. How division works in assembly

74. Bit shifting in assembly programming

75. Rotating bits in assembly programming

76. Logical operations in assembly

77. Using OR, AND, and XOR logical operations in assembly programming

## Section 8: Working with Strings

## Section 9: Using functions in assembly programming

## Section 10: Using system calls in assembly programming

## Section 11: Inline assembly programming
