## x86 Assembly Language Programming From Ground Up
- By Israel Gbati

2. Setting up the development environment
- The lecturer uses VS with MASM at Windows
- For Linux, use UASM: https://github.com/Terraspace/UASM
    - Ref: https://masm32.com/board/index.php?topic=9211.15
```
 sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key 10649EA8D069C51D
 curl -fsSL https://apt.navegos.net/pub.key | sudo apt-key add -
  sudo add-apt-repository "deb [arch=amd64] https://apt.navegos.net/ubuntu/uasm/ $(lsb_release -cs) main"
 sudo apt-get install uasm
 ```
 - We need hex editor or disassembler: sudo apt install ghex
- uasm doesn't work well in Ubuntu
- Let's convert masm code into nasm
    - https://left404.com/2011/01/04/converting-x86-assembly-from-masm-to-nasm-3/
- nasm is for Intel compiler(?). Let's use GAS or as to compile with gcc

3. Coding: Simple assembly program
- MASM
```
.386        # for 386 instructions
.model flat # flat memory model
.code       # coding starts here
start  PROC
		mov	eax,213
		add eax,432
		ret
start	endp  # close the procedure made above
end		start 
```
- nasm: nasm -felf64 ex3.nasm; gcc -g ex3.o -no-pie
```
section .text
global main
extern puts

main:
  mov eax, 213
  add eax, 432
  ret
```
- gas: as - ex3.o ex3.s; ld ex3.o ; gdb ./a.out; b _start; run ; nexti ; p $eax
	- Ref: https://cs.lmu.edu/~ray/notes/gasexamples/
```
.global _start
.text
_start:
  mov $213, %eax
  mov $432, %eax
  syscall
  mov $60, %rax # below is to exit(0)
  xor %rdi, %rdi
  syscall
```
- Disassembling the executable
```
objdump -d a.out 
a.out:     file format elf64-x86-64
Disassembly of section .init:
00000000004003b0 <_init>:
  4003b0:	48 83 ec 08          	sub    $0x8,%rsp
  4003b4:	48 8b 05 3d 0c 20 00 	mov    0x200c3d(%rip),%rax        # 600ff8 <__gmon_start__>
  4003bb:	48 85 c0             	test   %rax,%rax
  4003be:	74 02                	je     4003c2 <_init+0x12>
  4003c0:	ff d0                	callq  *%rax
  4003c2:	48 83 c4 08          	add    $0x8,%rsp
  4003c6:	c3                   	retq   
Disassembly of section .text:
```
- Going through gdb
```
$ gdb ./a.out 
(gdb) b main
Breakpoint 1 at 0x4004c0
(gdb) run
Starting program: /home/hpjeon/hw/class/udemy_X86Assembly/SimpleAddition/SimpleAddition/a.out 
Breakpoint 1, 0x00000000004004c0 in main ()
(gdb) p $eax
$1 = 4195520
(gdb) 
(gdb) nexti # next instruction
0x00000000004004c5 in main ()
(gdb) p $eax
$2 = 213
(gdb) nexti
0x00000000004004ca in main ()
(gdb) p $eax
$3 = 645
```

21. Simple x86 Assembly template
- MASM version:
```
; 
.386
.model flat, stdcall
.stack 4096
.code
main PROC
  mov eax, 10000h
  add eax, 40000h
  sub eax, 20000h
  ret
main ENDP
END main
```
- nasm conversion:
  - Flat memory model is default
```
section .text
global main
main:
  mov eax, 10000h
  add eax, 40000h
  sub eax, 20000h
  ret
```
- Compiling and debugging:
```
$ nasm -felf64 ex21.nasm
$ gcc -g ex21.o -no-pie
$ gdb ./a.out 
(gdb) b main
Breakpoint 1 at 0x4004a0
(gdb) run
Breakpoint 1, 0x00000000004004a0 in main ()
(gdb) nexti
0x00000000004004a5 in main ()
(gdb) p $eax
$1 = 65536  # corresponds to 10000h
(gdb) nexti
0x00000000004004aa in main ()
(gdb) p $eax
$2 = 327680  # corresponds to 50000h
(gdb) nexti
0x00000000004004af in main ()
(gdb) p $eax
$3 = 196608  # corresponds to 30000h
```
- gas conversion
```
.global _start
.text
_start:
	mov $0x10000, %eax
	add $0x40000, %eax
	sub $0x20000, %eax
	syscall
	mov $60, %eax # below is to exit(0)
	xor %edi, %edi # instead of rdi, edi works ok?
	syscall
```
	- command: as -o ex21.o ex21.s; ld ex21.o ; a.out # test through gdb
```
$ gdb a.out
(gdb) b _start
Breakpoint 1 at 0x400078
(gdb) run
Starting program: /home/hpjeon/hw/class/udemy_X86Assembly/a.out 
Breakpoint 1, 0x0000000000400078 in _start ()
(gdb) p $eax
$1 = 0
(gdb) nexti
0x000000000040007d in _start ()
(gdb) p $eax
$2 = 65536
(gdb) nexti
0x0000000000400082 in _start ()
(gdb) p $eax
$3 = 327680
(gdb) nexti
0x0000000000400087 in _start ()
(gdb) p $eax
$4 = 196608
```

22. Declaring variables
- MASM:
```
.386
.model flat
.data
num1	dword	11111111h
num2	dword	10101010h
ans		dword	0
.code
start	proc
		mov	eax,num1
		add eax,num2
		mov	ans,eax
		ret
start	endp		

end		start
```
- nasm conversion:
  - DB - 1 byte
  - DW - 2 bytes  
  - DD - 4 bytes. Doubleword
  - DQ - 8 bytes. Quadword
  - DT - 10 bytes
  - Ref: https://www.tutorialspoint.com/assembly_programming/assembly_variables.htm
```
section .data
num1 dq 11111111h
num2	dq 10101010h
ans	dq 0
section .text
global main
main:
  mov eax, num1
  add eax, num2
  mov [ans], eax
  ret
```
- [], square brackets work like a deference operetor, * in C
	- The above code might be wrong. GDB shows num1 and num2 value differently
- gas conversion:
```
.global _start
.data
num1: .quad 0x11111111
num2: .quad 0x10101010
ans:  .quad 0x0
.text
_start:
	movq num1, %rax
	add num2, %rax
	movq %rax, $ans
	syscall
	mov $60, %eax # below is to exit(0)
	xor %edi, %edi # instead of rdi, edi works ok?
	syscall
```
	- Running gdb
```
$ as -g -o ex22.o ex22.s
$ ld ex22.o
$ gdb a.out
(gdb) b _start
Breakpoint 1 at 0x4000b0: file ex22.s, line 8.
(gdb) run
Breakpoint 1, _start () at ex22.s:8
8		movq num1, %rax
(gdb) nexti
9		add num2, %rax
(gdb) p $rax
$1 = 286331153
(gdb) nexti
10		movq %rax, ans
(gdb) p $rax
$2 = 555819297
(gdb) nexti
11		syscall
(gdb) x &num1
0x6000d3:	0x11111111
(gdb) x &num2
0x6000db:	0x10101010
(gdb) x &ans
0x6000e3:	0x21212121
```	

24. Endianness
- Little vs Big Endian
- Differences in the order of storing digits

26. Mixing C/C++ and Assembly
- 
```
#include "pch.h"
#include <iostream>
#include <stdlib.h>
extern "C" void Reverser(int* y, const int *x, int n);
int main()
{
	const int n = 10;
	int x[n], y[n];
	for (int i = 0; i < n; i++) {
		x[i] = i;
	}
	Reverser(y, x, n);
  printf("\n----------------Array Reverser-----------\n");
	for (int i = 0; i < n; i++) {
		printf(" x: %5d	y: %5d\n", x[i], y[i]);
	}
	return 0;
}
```
- MASM
```
.386
.model flat,c
.code
Reverser     proc
			push ebp
			mov ebp,esp
			push esi
			push edi
			xor	eax,eax
			mov	edi,[ebp+8]
			mov	esi,[ebp+12]
			mov ecx,[ebp+16]
			test	ecx,ecx
			lea esi,[esi+ecx*4-4]
			pushfd
			std
@@:			lodsd
			mov	[edi],eax
			add edi,4
			dec ecx
			jnz	@B
  		popfd
			mov eax,1
			pop edi
			pop esi
			pop ebp
			ret
Reverser   endp
		   end		   
```
- gas conversion:
```
.global Reverser
.text
Reverser:
	push %rbp
	mov %rbp,%rsp
	push %rsi
	push %rdi
	xor	%rax,%rax
	mov	%di,$(%rbp+8)
	mov	%si,$(%rbp+12)
	mov %rcx,$(rbp+16)
	test	%rcx,%rcx
	lea %rsi,$(%rsi+%rcx*4-4)
	pushfd
	std
label1: lodsd
	mov	$(%rdi),%rax
	add %rdi,4
	dec %rcx
	jnz label1
	popfd
	mov %rax,1
	pop %rdi
	pop %rsi
	pop %rbp
	syscall
	mov $60, %eax # below is to exit(0)
	xor %edi, %edi # instead of rdi, edi works ok?
	syscall
```
	- Not working. TBD
- getmax.cpp
```
#include <cstdio>
#include <cinttypes>
int64_t getmax(int64_t a, int64_t b, int64_t c);
int main(){
  printf("%ld\n", getmax(40, -9, 67));
  printf("%ld\n", getmax( 0,  4, -7));
  printf("%ld\n", getmax(33, -99, 4));
  return 0;
}
```
- getmax_.s
```
.text
.globl getmax
getmax:
  mov   %rdi, %rax 
  cmp   %rsi, %rax
  cmovl %rsi, %rax
  cmp   %rdx, %rax
  cmovl %rdx, %rax
  ret
```
- Command:  
```
$ as -g -o getmax_.o getmax_.s
$ g++ -g -c getmax.cpp
$ g++ -o a.out getmax.o getmax_.o 
$ gdb ./a.out 
(gdb) b getmax
Breakpoint 1 at 0xa1b: file getmax_.s, line 5.
(gdb) run
Breakpoint 1, getmax () at getmax_.s:5
5	  mov   %rdi, %rax 
(gdb) p $rax
$1 = 93824992233690
(gdb) p $rdi
$2 = 40
(gdb) p $rsi
$3 = -9
(gdb) p $rdx
$4 = 67
```


