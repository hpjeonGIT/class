## x86 Assembly Language Programming From Ground Up
- By Israel Gbati

2. Setting up the development environment
- The lecturer uses VS with MASM
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
