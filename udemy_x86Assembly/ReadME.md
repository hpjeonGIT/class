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
	- https://www.recurse.com/blog/7-understanding-c-by-learning-assembly
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

30. Memory address
- Effective address = BaseReg + IndexReg * ScaleFactor + Disp
```
Disp:                           mov eax, [MyVal]
BaseReg:                        mov eax, [ebx]
BaseReg + IndexReg + Disp:      mov eax, [ebx+esi+12]
BaseReg + IndexReg * SF + Disp: mov eax, [ebx+esi*4+20]
```

35. Summing Array elements
- MASM code:
```
.386
.model flat
.data
intArray DWORD 10000h,20000h,30000h,40000h;
.code
start proc
     mov edi,OFFSET intArray
	 mov ecx, LENGTHOF intArray
	 mov eax,0
LP:
	add eax,[edi]
	add edi,TYPE intArray
	loop LP
    ret
start	endp
end		start
```
- GAS conversion:
```
# https://www.tutorialspoint.com/assembly_programming/assembly_arrays.htm
.global _start
.data
intArray: 
    .long 0x10000, 0x20000, 0x30000, 0x40000
    arrsize = . - intArray
    arrlen = (. - intArray)/4 # long is 4 bytes
.text
_start:
    movl $arrlen, %edi
	movl $0, %eax
    movl $0, %ebx
    movl $intArray, %ecx
LP:
    add (%ecx, 4, %ebx), %eax
    inc %ebx
    dec %edi
    jnz LP

_end:
	mov $60, %eax # below is to exit(0)
	xor %edi, %edi # instead of rdi, edi works ok?
	syscall
```
- Ref: https://www.youtube.com/watch?v=oq7_jOu1Owc
- 0x10000 + 0x20000 + 0x30000 + 0x40000 = 65536 + 131072 + 196608 + 262144 = 655360
- gdb confirms 
```
(gdb) p $eax
$6 = 655360
```
- Q: is .long 4bytes? Not 8bytes?

43. Coding: Computing the sum of an array
``` C++
#include <iostream>
extern "C" int AdderGAS(int64_t a, int64_t b, int64_t c);

int main() {
	std::cout << AdderGAS(11,22,33) << std::endl;
}

int AdderCpp (int a, int b, int c) {
	return a+b+c;
}
```
``` AS
.global AdderGAS
.data
.text
AdderGAS:
	push %rbp
	mov %rsp, %rbp 
	mov %rdi, %rax
	add %rsi, %rax
	add %rdx, %rax
	pop %rbp
	ret
_end:
	mov $60, %eax # below is to exit(0)
	xor %edi, %edi # instead of rdi, edi works ok?
	syscall    
```
- ref: https://zims-en.kiwix.campusafrica.gos.orange.com/wikibooks_en_all_maxi/A/X86_Assembly/GAS_Syntax
- ref: https://cs.brown.edu/courses/cs033/docs/guides/x64_cheatsheet.pdf
- ref: http://6.s081.scripts.mit.edu/sp18/x86-64-architecture-guide.html
```
%rcx	used to pass 4th argument to functions
%rdx	used to pass 3rd argument to functions
%rsi	used to pass 2nd argument to functions
%rdi	used to pass 1st argument to functions
```
- Command:
```
$ as -g -o sum.o sum.s
$ g++ -c sum_.cpp
$ g++ -o a.out sum_.o sum.o
$ ./a.out
66
```

44. Coding: computing signed muliplication and division
```Cpp
#include <iostream>
extern "C" int IntegerMulDiv(int64_t a, int64_t b, int64_t * prod, 
	int64_t * quo, int64_t *rem);
int main()
{

	int a = -21, b = 9;
	int prod = 0, quo = 0, rem = 0;
	int rv;

    rv = IntegerMulDiv(a, b, &prod, &quo, &rem);

	printf("Input a : %4d b:	%4d\n", a, b);
	printf("Output rv: %4d  prod:   %4d    quo: %4d   rem:  %4d\n", rv, pro
d, quo, rem);

	return 0;
		 
}
```
``` ASM
.386
.model flat,c
.code

;Return : 0 Error (Division by zero)
;		: 1 Success
;
;Computation *prod = a*b
;			 *quo = a/b
;			 *rem = a %b

IntegerMulDiv proc
				
			 push ebp
			 mov ebp,esp
			 push ebx

			 xor eax,eax

			 mov ecx,[ebp+8]		;ecx ='a'
			 mov edx,[ebp+12]		;edx ='b'

			 or edx, edx
			 jz InvalidDivisor

			 imul edx,ecx   ;edx = 'a'*'b'

			 mov ebx,[ebp+16]  ; ebx ='prod'
			 mov [ebx],edx

			 mov eax,ecx			;eax ='a'
			 cdq					;edx:eax contains dividend

			idiv dword ptr[ebp+12]

			mov ebx,[ebp+20]
			mov [ebx],eax
			mov ebx,[ebp+24]
			mov [ebx],edx
			mov eax,1

	InvalidDivisor:
			pop ebx
			pop ebp
			ret
IntegerMulDiv  endp
			end

```

58. Conversion b/w F to C
- conv_.cpp
```
#include <iostream>
extern "C" double FtoC(double deg_f);
extern "C" double CtoF(double deg_c);
int main() {
	double ct = 30;
	double ft = 86;
	std::cout << "F to C: " << ft << "->" << FtoC(ft) << std::endl;
	std::cout << "C to F: " << ct << "->" << CtoF(ct) << std::endl;
}
```
- MASM
```
.model flat,c
.const
r8_SfFtoC real8 0.55555556
r8_SfCtoF real8 1.8
i4_32     dword 32
.code
FtoC  proc
	push ebp
	mov ebp,esp
	fld [r8_SfFtoC]
	fld real8 ptr[ebp+8]
	fild [i4_32]
	fsubp
	fmulp
	pop ebp
	ret
FtoC  endp
;
CtoF proc
	push ebp
	mov ebp,esp
	fld real8 ptr[ebp+8]
	fmul [r8_SfCtoF]
	fiadd [i4_32]
	pop ebp
	ret
CtoF endp
```
- GAS conversion
```
.global FtoC 
.global CtoF
.data
    CMULT: .double 0.55555556 # C = (F -32) * 0.555556
    FMULT: .double 1.8
    BIAS : .double   32
.text
FtoC: 
    push %rbp
	mov %rsp, %rbp
    movsd %xmm0, -0x8(%rbp)
    fldl -0x8(%rbp)
    fldl BIAS
    fsubrp %st, %st(1)
    fldl CMULT
    fmulp %st, %st(1)
    fstpl -0x10(%rbp)
    movsd -0x10(%rbp),%xmm0
    pop %rbp
	retq
CtoF:       # F = 1.8*C + 32
    push %rbp
	mov %rsp, %rbp
    movsd %xmm0, -0x8(%rbp)
    fldl -0x8(%rbp)
    fldl FMULT
    fmulp %st, %st(1)
    fldl BIAS
    faddp %st, %st(1)
    fstpl -0x10(%rbp)
    movsd -0x10(%rbp),%xmm0
    pop %rbp
	retq
_end:
	mov $60, %eax # below is to exit(0)
	xor %edi, %edi # instead of rdi, edi works ok?
	syscall
```
- Command:
```
$ as -o conv.o  conv.s
$ g++ -c conv_.cpp
$ g++ -no-pie -o a.out conv.o conv_.o
$ ./a.out 
F to C: 86->30
C to F: 30->22
```
- Discussion
	- For FPE, float/double arguments are handled through `%xmm`
	- To find GAS reference, write a simple C function and disassemble it

###  Extracting C++ function to objdump
```
double FtoC(double x) { 
	return (x - 32) * 0.5555556;
}
```
- Save as sample.cpp and disassemble as:
```
$ g++ -c -mfpmath=387  sampe.cpp 
$ objdump -d sampe.o
sampe.o:     file format elf64-x86-64
Disassembly of section .text:
0000000000000000 <_Z4FtoCd>:
   0:	55                   	push   %rbp
   1:	48 89 e5             	mov    %rsp,%rbp
   4:	f2 0f 11 45 f8       	movsd  %xmm0,-0x8(%rbp)
   9:	dd 45 f8             	fldl   -0x8(%rbp)
   c:	dd 05 00 00 00 00    	fldl   0x0(%rip)        # 12 <_Z4FtoCd+0x12>
  12:	de e9                	fsubrp %st,%st(1)
  14:	dd 05 00 00 00 00    	fldl   0x0(%rip)        # 1a <_Z4FtoCd+0x1a>
  1a:	de c9                	fmulp  %st,%st(1)
  1c:	dd 5d f0             	fstpl  -0x10(%rbp)
  1f:	f2 0f 10 45 f0       	movsd  -0x10(%rbp),%xmm0
  24:	5d                   	pop    %rbp
  25:	c3                   	retq   
```
- To read .rodata section, use -D option
- Ref: http://www.sig9.com/articles/att-syntax
	- https://github.com/Demkeys/x86_64AssemblyATTGASExamples/blob/master/MyProjects/EncryptAndDecryptFile/decryptprog.s
- Error message:
```
g++ -o a.out conv.o conv_.o
/usr/bin/ld: conv.o: relocation R_X86_64_32S against `.data' can not be used when making a PIE object; recompile with -fPIC
/usr/bin/ld: final link failed: Nonrepresentable section on output
collect2: error: ld returned 1 exit status
```
- Solution: use lea instead of mov. Ref: https://stackoverflow.com/questions/49434489/relocation-r-x86-64-32-against-data-can-not-be-used-when-making-a-shared-obje
- or use -no-pie when link using g++

## gdb command
```
(gdb) b FtoC
Breakpoint 1 at 0x400894
(gdb) run
Starting program: /home/hpjeon/hw/class/udemy_X86Assembly/a.out 
F to C
Breakpoint 1, 0x0000000000400894 in FtoC ()
(gdb) nexti
0x000000000040089b in FtoC ()
(gdb) display/i $pc
1: x/i $pc
=> 0x40089b <FtoC+11>:	fldl   0x0(%rip)        # 0x4008a1 <FtoC+17>
(gdb) p $rbp
$1 = (void *) 0x7fffffffd560
(gdb) x &BIAS
0x601060:	0x00000000
(gdb) x &FMULT
0x601058:	0xcccccccd
```

## ETC
- Ref: https://stackoverflow.com/questions/15786404/fld-instruction-x64-bit
	- for the float/double, arguments of functions are passed in XMM0, ...
