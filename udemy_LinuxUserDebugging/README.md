## Learn Linux User Space Debugging
- Instructor: Linux Trainer

## Section 1: Linux User Space Debugging

1. GDB part 1
  - source code
```cpp
#include <iostream>
#include <cmath>
double comp(int i, int j) {
  return static_cast<double> (i*j);
}
int main() {
  double x;
  x = comp(3,4);
  std::cout << "x = " << x << std::endl;
}
```
  - Debug with arguments: `gdb --args a.exe arg1 arg2 arg3 ...  `
    - Or in the gdb: `run  arg1 arg2 arg3 ...`
      - or may use `set args`
  - Some keywords
    - `info functions`: enlists available functions
    - `break main`: injects a break point main
    - `b 9`: adds a break point at line 9
    - `info breakpoints`: enlists the breakpoints
    - `next`: next line
    - `step`: will step into a function
    - `continue`: continues up to the next break point
    - `finish`: finishes the current function
    - `list`: shows the code shot
  - Sample screen shot
```bash
(gdb) info functions
All defined functions:
File ex1.cxx:
double comp(int, int);
int main();
static void _GLOBAL__sub_I__Z4compii();
static void __static_initialization_and_destruction_0(int, int);
...
(gdb) break main
Breakpoint 1 at 0x929: file ex1.cxx, line 8.
(gdb) b 9
Breakpoint 2 at 0x941: file ex1.cxx, line 9.
(gdb) info breakpoints
Num     Type           Disp Enb Address            What
1       breakpoint     keep y   0x0000000000000929 in main() at ex1.cxx:8
2       breakpoint     keep y   0x0000000000000941 in main() at ex1.cxx:9
(gdb) list
4	  return static_cast<double> (i*j);
5	}
6	int main() {
7	  double x;
8	  x = comp(3,4);
9	  std::cout << "x = " << x << std::endl;
10	}(gdb) 
```
  - When an executable crashes in the gdb
    - `backtrace`
      - #1/#2/#3 ...: they are frame numbers
    - `frame 3`: selects frame 3
    - `p j`: prints the value of j
    - `p /x j`: prints in hexa
    - `p /t j`: prints in binary
    - `x &j`: shows the memory of j
    - `ptype j`: the type of j
    - `set j=3`: replaces the value of j as 3
```bash
(gdb) run
Starting program: /home/hpjeon/hw/class/udemy_gdb/a.out 
Program received signal SIGFPE, Arithmetic exception.
0x0000555555554918 in comp (i=3, j=0) at ex2.cxx:4
4	  return static_cast<double> (i/j);
(gdb) backtrace
#0  0x0000555555554918 in comp (i=3, j=0) at ex2.cxx:4
#1  0x0000555555554938 in main () at ex2.cxx:8
(gdb) frame 0
#0  0x0000555555554918 in comp (i=3, j=0) at ex2.cxx:4
4	  return static_cast<double> (i/j);
(gdb)  x &i
0x7fffffffd57c:	00000000000000000000000000000011
(gdb) ptype i
type = int
```
2. GDB part 2
  - When your program starts, the call stack has only one frame, that of function main
  - Each function call pushes a new frame into the stack
  - Recursive functions can generate many frames
```c
#include<stdio.h>	 	 
void func1();	 	 
void func2();	 	 
int main() 
{	 	 
	int i=10;	 	 
	func1();	 	 
	printf("In Main(): %d\n",i);	 	 
}	 	 
void func1() 
{	 	 
	int n=20;	 	 
	printf("In func1(): %d\n",n);	 	 
	func2();	 	 
}	 	 
void func2() 
{	 	 
	int n = 30;	 	 
	printf("In func2() : %d\n",n);	 	 
}	
```
  - using `info frame`
    - no argument is same to `info frame 0`
```bash
(gdb) bt
#0  func1 () at 6/backtrace.c:14
#1  0x0000555555554663 in main () at 6/backtrace.c:8
(gdb) list
9		printf("In Main(): %d\n",i);	 	 
10	}	 	 
11	
12	void func1() 
13	{	 	 
14		int n=20;	 	 
15		printf("In func1(): %d\n",n);	 	 
16		func2();	 	 
17	}	 	 
18	
(gdb) info frame
Stack level 0, frame at 0x7fffffffd590:
 rip = 0x555555554688 in func1 (6/backtrace.c:14); 
    saved rip = 0x555555554663
 called by frame at 0x7fffffffd5b0
 source language c.
 Arglist at 0x7fffffffd580, args: 
 Locals at 0x7fffffffd580, Previous frame's sp is 0x7fffffffd590
 Saved registers:
  rbp at 0x7fffffffd580, rip at 0x7fffffffd588
(gdb) info frame 1
Stack frame at 0x7fffffffd5b0:
 rip = 0x555555554663 in main (6/backtrace.c:8); 
    saved rip = 0x7ffff7a03c87
 caller of frame at 0x7fffffffd590
 source language c.
 Arglist at 0x7fffffffd5a0, args: 
 Locals at 0x7fffffffd5a0, Previous frame's sp is 0x7fffffffd5b0
 Saved registers:
  rbp at 0x7fffffffd5a0, rip at 0x7fffffffd5a8
```
  - Note that saved rip at frame 0 is same to rip at frame 1
  - `info variables` will print all system variables (too many)
  - `info locals` will print local variables
  - `info args` will print the arguments in the current frame
  - `info registers` will print all registers such as rax, rbx, rbp, ...
  - `condition 3 i==5` breaks at breakpoint 3 when i==5
```bash
(gdb) info breakpoints
Num     Type           Disp Enb Address            What
2       breakpoint     keep y   0x0000555555554663 in main 
                                                   at 6/backtrace.c:9
(gdb) condition 2 i==5
(gdb) info breakpoints
Num     Type           Disp Enb Address            What
2       breakpoint     keep y   0x0000555555554663 in main 
                                                   at 6/backtrace.c:9
	stop only if i==5
```
  - `enable/disable 3` : will enable disable breakpoint 3
  - ctrl+x+a or `tui enable` : tui window opens
  - ctrl+l to redraw tui window
  - `disassemble functionName`
```bash
(gdb) disassemble main
Dump of assembler code for function main:
   0x000055555555464a <+0>:	push   %rbp
   0x000055555555464b <+1>:	mov    %rsp,%rbp
   0x000055555555464e <+4>:	sub    $0x10,%rsp
   0x0000555555554652 <+8>:	movl   $0xa,-0x4(%rbp)
   0x0000555555554659 <+15>:	mov    $0x0,%eax
   0x000055555555465e <+20>:	callq  0x555555554680 <func1>
=> 0x0000555555554663 <+25>:	mov    -0x4(%rbp),%eax
   0x0000555555554666 <+28>:	mov    %eax,%esi
   0x0000555555554668 <+30>:	lea    0xf5(%rip),%rdi        # 0x555555554764
   0x000055555555466f <+37>:	mov    $0x0,%eax
   0x0000555555554674 <+42>:	callq  0x555555554520 <printf@plt>
   0x0000555555554679 <+47>:	mov    $0x0,%eax
   0x000055555555467e <+52>:	leaveq 
   0x000055555555467f <+53>:	retq   
End of assembler dump.
```

3. Coredump_Valgrind
  - `ulimit -c unlimited` # unlimit the size of core files
  - `gdb <exe> <corefiles>`
    - where
    - info locals
    - info registers
  - `kill -s SIGABRT <pid>` # terminate working process
  - Out of bonds memory access
    - Write overflow: write is attempted into a memory buffer after its legally accessible location
    - Write underflow: write is attempted into a memory buffer before its legally accessible location
    - Read overflow: read is attempted on a memory buffer after its last legally accessible location
    - Read underflow: read is attempted on a memory buffer after its first legally accessible location
  - Print array value
    - p &arr # data type
    - x &arr # array value
    - x/16 &arr
    - x/16xb &arr # 16 bytes value
    - x/32xb &arr # 32 bytes value

4. strace_ltrace 
  - strace: traces system calls and signals
    - As default, dumps to stderr, not stdout
    - In order to get screenshot, `strace pwd 2> ./dump.txt`
    - Very useful when:
      - no source code available
      - no log/stdout
      - gdb is not usable
      - to find interaction with filesystem
```bash      
$ gcc simple.c 
$ strace ./a.out 
execve("./a.out", ["./a.out"], 0x7ffe878fa620 /* 67 vars */) = 0
brk(NULL)                               = 0x55c92fe2d000
access("/etc/ld.so.nohwcap", F_OK)      = -1 ENOENT (No such file or directory)
access("/etc/ld.so.preload", R_OK)      = -1 ENOENT (No such file or directory)
...
access("/etc/ld.so.nohwcap", F_OK)      = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libc.so.6", O_RDONLY|O_CLOEXEC) = 3
read(3, "\177ELF\2\1\1\3\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\240\35\2\0\0\0\0\0"..., 832) = 832
fstat(3, {st_mode=S_IFREG|0755, st_size=2030928, ...}) = 0
mmap(NULL, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f2203f79000
mmap(NULL, 4131552, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7f220398f000
mprotect(0x7f2203b76000, 2097152, PROT_NONE) = 0
mmap(0x7f2203d76000, 24576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1e7000) = 0x7f2203d76000
mmap(0x7f2203d7c000, 15072, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7f2203d7c000
close(3)                                = 0
arch_prctl(ARCH_SET_FS, 0x7f2203f7a4c0) = 0
mprotect(0x7f2203d76000, 16384, PROT_READ) = 0
mprotect(0x5607ef2ed000, 4096, PROT_READ) = 0
mprotect(0x7f2203fa9000, 4096, PROT_READ) = 0
munmap(0x7f2203f7b000, 185734)          = 0
exit_group(-284240390)                  = ?
+++ exited with 250 +++
$ gcc simple.c --static
$ strace ./a.out 
execve("./a.out", ["./a.out"], 0x7ffc144329f0 /* 67 vars */) = 0
brk(NULL)                               = 0x13ff000
brk(0x14001c0)                          = 0x14001c0
arch_prctl(ARCH_SET_FS, 0x13ff880)      = 0
uname({sysname="Linux", nodename="hakune", ...}) = 0
readlink("/proc/self/exe", "/home/hpjeon/hw/class/udemy_gdbs"..., 4096) = 48
brk(0x14211c0)                          = 0x14211c0
brk(0x1422000)                          = 0x1422000
access("/etc/ld.so.nohwcap", F_OK)      = -1 ENOENT (No such file or directory)
exit_group(4197229)                     = ?
+++ exited with 109 +++
```
    - Building statically (--static) doesn't require dynamic library and strace shows very simple traces
    - strace -c ./a.out : prints statistics
    - strace -t ./a.out: prints time stamp
      - -tt to print microseconds
      - -ttt in terms of seconds unit
```bash
$ strace -tt ./a.out 
10:09:52.687345 execve("./a.out", ["./a.out"], 0x7ffdf2a4a678 /* 67 vars */) = 0
10:09:52.687959 brk(NULL)               = 0x8bb000
10:09:52.688038 brk(0x8bc1c0)           = 0x8bc1c0
10:09:52.688238 arch_prctl(ARCH_SET_FS, 0x8bb880) = 0
10:09:52.688338 uname({sysname="Linux", nodename="hakune", ...}) = 0
10:09:52.688393 readlink("/proc/self/exe", "/home/hpjeon/hw/class/udemy_gdbs"..., 4096) = 48
10:09:52.688483 brk(0x8dd1c0)           = 0x8dd1c0
10:09:52.688525 brk(0x8de000)           = 0x8de000
10:09:52.688579 access("/etc/ld.so.nohwcap", F_OK) = -1 ENOENT (No such file or directory)
10:09:52.688684 exit_group(4197229)     = ?
10:09:52.688857 +++ exited with 109 +++
```
    - Can trace scripts as well
    - -T prints walltime spent in <...>
      - Can couple with -t or -tt
```bash
$ strace -Ttt ./a.out 
10:14:06.270001 execve("./a.out", ["./a.out"], 0x7ffef7a84898 /* 67 vars */) = 0 <0.000223>
10:14:06.270432 brk(NULL)               = 0xc7d000 <0.000006>
10:14:06.270477 brk(0xc7e1c0)           = 0xc7e1c0 <0.000011>
10:14:06.270522 arch_prctl(ARCH_SET_FS, 0xc7d880) = 0 <0.000020>
10:14:06.270563 uname({sysname="Linux", nodename="hakune", ...}) = 0 <0.000009>
10:14:06.270596 readlink("/proc/self/exe", "/home/hpjeon/hw/class/udemy_gdbs"..., 4096) = 48 <0.000030>
10:14:06.270669 brk(0xc9f1c0)           = 0xc9f1c0 <0.000008>
10:14:06.270696 brk(0xca0000)           = 0xca0000 <0.000007>
10:14:06.270741 access("/etc/ld.so.nohwcap", F_OK) = -1 ENOENT (No such file or directory) <0.000011>
10:14:06.270798 exit_group(4197229)     = ?
10:14:06.270864 +++ exited with 109 +++
```
    - -ff to folks with separate output files per fork
    - strace -p <PID> to attach a running process
    - Tracing only specifi system calls
      - -e trace=ipc
      - -e trace=memory
      - -e trace=network
      - -e trace=process
      - -e trace=signal
      - -e trace=file
      - Only openat and close: `strace -e trace=openat,close pwd`
      - In order to exclude openat : `strace -e trace='!openat,close' pwd`
```bash
$ strace -e trace=openat,close pwd
openat(AT_FDCWD, "tls/haswell/x86_64/libc.so.6", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
...
openat(AT_FDCWD, "/usr/lib/locale/locale-archive", O_RDONLY|O_CLOEXEC) = 3
close(3)                                = 0
/home/hpjeon/hw/class/udemy_gdbstrace/sec4
close(1)                                = 0
close(2)                                = 0
+++ exited with 0 +++
```
    - -o to save screen log into a file: strace -o log pwd
    - -s 20 to have 20 characters per string
      - `read(3, "\177ELF\2\1\1\3\0\0\0\0\0\0\0\0\3\0>\0\1\0\0\0\240\35\2\0\0\0\0\0"..., 832) = 832` => `read(3, "\177ELF\2"..., 832)`
  - ltrace
    - command: ltrace a.exe arg1 arg2 ...
    - -e malloc: traces malloc only
    - -c: shows the statistics
```bash
$ ltrace -c  /bin/cat hello.c
#include <stdio.h>
int main() { printf("hello\n"); return 0;}
% time     seconds  usecs/call     calls      function
------ ----------- ----------- --------- --------------------
 11.16    0.000188         188         1 setlocale
  9.55    0.000161          40         4 __freading
  8.07    0.000136          68         2 fflush
  7.89    0.000133          66         2 fileno
  7.30    0.000123          61         2 read
  6.53    0.000110          55         2 fclose
  6.41    0.000108          54         2 __fxstat
  4.93    0.000083          41         2 __fpending
  4.27    0.000072          72         1 write
  4.15    0.000070          70         1 malloc
  4.09    0.000069          69         1 getpagesize
  3.74    0.000063          63         1 open
  3.68    0.000062          62         1 free
  3.38    0.000057          57         1 close
  2.97    0.000050          50         1 posix_fadvise
  2.85    0.000048          48         1 strrchr
  2.37    0.000040          40         1 getopt_long
  2.26    0.000038          38         1 bindtextdomain
  2.20    0.000037          37         1 textdomain
  2.20    0.000037          37         1 __cxa_atexit
------ ----------- ----------- --------- --------------------
100.00    0.001685                    29 total
$ ltrace -e malloc  /bin/cat hello.c
cat->malloc(135167)                    = 0x7f8439f57010
#include <stdio.h>
int main() { printf("hello\n"); return 0;}
+++ exited (status 0) +++
```

5. Linux commands
  - dmesg: print or control the kernel ring buffer
    - dmesg -c
    - May need sudo
  - ls
    - -l: regarding permission
    - -h: human readable format using kBytes
    - -a: showing hidden files
    - -t: sorts by modification time
    - -r: reverse the sorting
  - cd -: go to the previous location
  - touch: makes a file
  - man: an interface to the online reference manual
    1. Executable programs or shell commands
    2. System calls (functions by the kernel)
    3. Library calls
    4. Special files (such as /dev)
    5. File formats and convetions
```bash
$ man 1 printf # this is for the executable printf
    PRINTF(1)                User Commands               PRINTF(1)
NAME
       printf - format and print data
SYNOPSIS
       printf FORMAT [ARGUMENT]...
       printf OPTION
...
$ man 3 printf # this is for library calls
PRINTF(3)          Linux Programmer's Manual         PRINTF(3)
NAME
       printf,  fprintf,  dprintf, sprintf, snprintf, vprintf,
       vfprintf, vdprintf,  vsprintf,  vsnprintf  -  formatted
       output conversion
SYNOPSIS
       #include <stdio.h>
       int printf(const char *format, ...);
```
