## Title: Linux Binary Analysis for Ethical Hackers and Pentesters
- Instructor: Swapnil Singh

## Section 1: Introduction
1. Who should take this course

2. Lab Machine configuration
- Ubuntu 22.04.1 LTS

## Section 2: Anatomy of Binary (ELF) file

3. What is Binary (ELF) file?
- Binary vs ASCII
- Binary file: contains machine code
```bash
$ file hello.c 
hello.c: C source, ASCII text
$ file a.out 
a.out: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, BuildID[sha1]=e1499259c9761c2e2c0512aba7bca339e8a96f95, for GNU/Linux 3.2.0, not stripped
```

4. Binary (ELF) file compilation preprocessing stage
- Preprocessing -> compiling -> Link
- Preprocssesing only: gcc -E -P hello.c > preprocessed.c

5. Assembling the source code into assembly instructions of a source file
- Preprocessed file to compile: gcc -S preprocessed.c
  - Produces preprocssed.s, which is an assembly code

6. Generating Binary file (ELF) from object file
- gcc -s preprocessed.s
  - Produces preprocssed.o, which is an object file

7. Role of Linker in compilation process of Binary file (ELF)
- gcc preprocessed.o -o a.out
  - Produces the executable a.out
- Object file doesn't have the link address
```bash
$ readelf -x .rodata preprocessed.o
Hex dump of section '.rodata':
  0x00000000 68656c6c 6f20776f 726c6400          hello world.
$ readelf -x .rodata ./a.out 
Hex dump of section '.rodata':
  0x00002000 01000200 68656c6c 6f20776f 726c6400 ....hello world.
$ objdump -M intel -d preprocessed.o
preprocessed.o:     file format elf64-x86-64
Disassembly of section .text: 
0000000000000000 <main>:  ##<---- Note that address is zero
   0:	f3 0f 1e fa          	endbr64 
   4:	55                   	push   rbp
   5:	48 89 e5             	mov    rbp,rsp
   8:	48 8d 3d 00 00 00 00 	lea    rdi,[rip+0x0]        # f <main+0xf>
   f:	e8 00 00 00 00       	call   14 <main+0x14> ## goes to line 14, not calling a print function
  14:	b8 00 00 00 00       	mov    eax,0x0
  19:	5d                   	pop    rbp
  1a:	c3                   	ret    
$ objdump -M intel -d ./a.out
...
0000000000001149 <main>:  ## <------- Has a proper address
    1149:	f3 0f 1e fa          	endbr64 
    114d:	55                   	push   rbp
    114e:	48 89 e5             	mov    rbp,rsp
    1151:	48 8d 3d ac 0e 00 00 	lea    rdi,[rip+0xeac]        # 2004 <_IO_stdin_used+0x4>
    1158:	e8 f3 fe ff ff       	call   1050 <puts@plt> ## goes to line 1050, which is puts function
    115d:	b8 00 00 00 00       	mov    eax,0x0
    1162:	5d                   	pop    rbp
    1163:	c3                   	ret    
    1164:	66 2e 0f 1f 84 00 00 	nop    WORD PTR cs:[rax+rax*1+0x0]
    116b:	00 00 00 
    116e:	66 90                	xchg   ax,ax
...
```

## Section 3: Understanding ELF file format structure deeply

8. Structure of an ELF
- Understanding ELF
  - What does ELF contains?
    - Collection of hexadecimal bytes
    - Executable Header + Program Headers + Sections + Section Headers
- ref: /usr/include/elf.h
- Executable header: a structured series of bytes telling that it is an ELF file, what kind of ELF it is, and where in the file to find all the other contents (Program headers, sections, section headers)
```bash
$ readelf -h ./a.out 
ELF Header:
  Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00 
  Class:                             ELF64
  Data:                              2's complement, little endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              DYN (Shared object file)
  Machine:                           Advanced Micro Devices X86-64
  Version:                           0x1
  Entry point address:               0x1060
  Start of program headers:          64 (bytes into file)
  Start of section headers:          14720 (bytes into file)
  Flags:                             0x0
  Size of this header:               64 (bytes)
  Size of program headers:           56 (bytes)
  Number of program headers:         13
  Size of section headers:           64 (bytes)
  Number of section headers:         31
  Section header string table index: 30
```

9. ELF Header Magic Bytes
- From readelf command above:
  - ` Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00 `
  - First 4 bytes are called Magic bytes (7f 45 4c 46)
    - First byte: defines the kind of file 
    - Second byte: defines the type of file
    - ... Up to ABI version
  - They match with the results of hexdump
```bash
$ hexdump -C a.out |grep -A 3 ELF
00000000  7f 45 4c 46 02 01 01 00  00 00 00 00 00 00 00 00  |.ELF............|
00000010  03 00 3e 00 01 00 00 00  60 10 00 00 00 00 00 00  |..>.....`.......|
00000020  40 00 00 00 00 00 00 00  80 39 00 00 00 00 00 00  |@........9......|
00000030  00 00 00 00 40 00 38 00  0d 00 40 00 1f 00 1e 00  |....@.8...@.....|
```

10. ELF Header Indepth Analysis
```bash
  Start of section headers:          14720 (bytes into file)
  Flags:                             0x0
  Size of this header:               64 (bytes)
  Size of program headers:           56 (bytes)
  Number of program headers:         13
  Size of section headers:           64 (bytes)
  Number of section headers:         31
  Section header string table index: 30
```
- 13 program headers
- 31 section headers
  - Every section header is 64 bytes
- Start of header must be converted to hexa (shown as decimal)
- `this header` means the executable header

11. Section Headers and Sections of ELF
- Printing section headers
  - As shown above, there are 31 section headers in this a.out
```bash
$ readelf --wide -S a.out
There are 31 section headers, starting at offset 0x3980:
Section Headers:
  [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
  [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
  [ 1] .interp           PROGBITS        0000000000000318 000318 00001c 00   A  0   0  1
  [ 2] .note.gnu.property NOTE            0000000000000338 000338 000020 00   A  0   0  8
  [ 3] .note.gnu.build-id NOTE            0000000000000358 000358 000024 00   A  0   0  4
  [ 4] .note.ABI-tag     NOTE            000000000000037c 00037c 000020 00   A  0   0  4
  [ 5] .gnu.hash         GNU_HASH        00000000000003a0 0003a0 000024 00   A  6   0  8
  [ 6] .dynsym           DYNSYM          00000000000003c8 0003c8 0000a8 18   A  7   1  8
  [ 7] .dynstr           STRTAB          0000000000000470 000470 000082 00   A  0   0  1
  [ 8] .gnu.version      VERSYM          00000000000004f2 0004f2 00000e 02   A  6   0  2
  [ 9] .gnu.version_r    VERNEED         0000000000000500 000500 000020 00   A  7   1  8
  [10] .rela.dyn         RELA            0000000000000520 000520 0000c0 18   A  6   0  8
  [11] .rela.plt         RELA            00000000000005e0 0005e0 000018 18  AI  6  24  8
  [12] .init             PROGBITS        0000000000001000 001000 00001b 00  AX  0   0  4
  [13] .plt              PROGBITS        0000000000001020 001020 000020 10  AX  0   0 16
  [14] .plt.got          PROGBITS        0000000000001040 001040 000010 10  AX  0   0 16
  [15] .plt.sec          PROGBITS        0000000000001050 001050 000010 10  AX  0   0 16
  [16] .text             PROGBITS        0000000000001060 001060 000185 00  AX  0   0 16
  [17] .fini             PROGBITS        00000000000011e8 0011e8 00000d 00  AX  0   0  4
  [18] .rodata           PROGBITS        0000000000002000 002000 000010 00   A  0   0  4
  [19] .eh_frame_hdr     PROGBITS        0000000000002010 002010 000044 00   A  0   0  4
  [20] .eh_frame         PROGBITS        0000000000002058 002058 000108 00   A  0   0  8
  [21] .init_array       INIT_ARRAY      0000000000003db8 002db8 000008 08  WA  0   0  8
  [22] .fini_array       FINI_ARRAY      0000000000003dc0 002dc0 000008 08  WA  0   0  8
  [23] .dynamic          DYNAMIC         0000000000003dc8 002dc8 0001f0 10  WA  7   0  8
  [24] .got              PROGBITS        0000000000003fb8 002fb8 000048 08  WA  0   0  8
  [25] .data             PROGBITS        0000000000004000 003000 000010 00  WA  0   0  8
  [26] .bss              NOBITS          0000000000004010 003010 000008 00  WA  0   0  1
  [27] .comment          PROGBITS        0000000000000000 003010 00002b 01  MS  0   0  1
  [28] .symtab           SYMTAB          0000000000000000 003040 000618 18     29  46  8
  [29] .strtab           STRTAB          0000000000000000 003658 00020a 00      0   0  1
  [30] .shstrtab         STRTAB          0000000000000000 003862 00011a 00      0   0  1
Key to Flags:
  W (write), A (alloc), X (execute), M (merge), S (strings), I (info),
  L (link order), O (extra OS processing required), G (group), T (TLS),
  C (compressed), x (unknown), o (OS specific), E (exclude),
  l (large), p (processor specific)
```

12. Concept of Section and Section Headers in ELF
- The code and data in an ELF binary are logically divided into contiguous non-overlapping chunks called sections
- Section examples
  - .init: code section
  - .plt
  - .plt.got
  - .plt.sec
  - .text
  - .fini
  - .rodata: data section

13. How to view a Section from Section header info
```bash
$ readelf --wide -S a.out
There are 31 section headers, starting at offset 0x3980:
Section Headers:
  [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
  [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
  [ 1] .interp           PROGBITS        0000000000000318 000318 00001c 00   A  0   0  1
  [ 2] .note.gnu.property NOTE            0000000000000338 000338 000020 00   A  0   0  8
...
  [30] .shstrtab         STRTAB          0000000000000000 003862 00011a 00      0   0  1
Key to Flags:
  W (write), A (alloc), X (execute), M (merge), S (strings), I (info),
  L (link order), O (extra OS processing required), G (group), T (TLS),
  C (compressed), x (unknown), o (OS specific), E (exclude),
  l (large), p (processor specific)
```
- Flg or Flags definition is shown in the bottom
- Use `objdump --section <section_name>` to see each section
```bash
$ objdump --section .init -d ./a.out 
./a.out:     file format elf64-x86-64
Disassembly of section .init:
0000000000001000 <_init>:
    1000:	f3 0f 1e fa          	endbr64 
    1004:	48 83 ec 08          	sub    $0x8,%rsp
    1008:	48 8b 05 d9 2f 00 00 	mov    0x2fd9(%rip),%rax        # 3fe8 <__gmon_start__>
    100f:	48 85 c0             	test   %rax,%rax
    1012:	74 02                	je     1016 <_init+0x16>
    1014:	ff d0                	callq  *%rax
    1016:	48 83 c4 08          	add    $0x8,%rsp
    101a:	c3                   	retq   
```

14. Understanding init and fini sections in ELF
- `.init` works like a constructor, initializing
- `.fini` finishes all the program, cleaning memory
```bash
$ objdump --section .fini -d ./a.out 
./a.out:     file format elf64-x86-64
Disassembly of section .fini:
00000000000011e8 <_fini>:
    11e8:	f3 0f 1e fa          	endbr64 
    11ec:	48 83 ec 08          	sub    $0x8,%rsp
    11f0:	48 83 c4 08          	add    $0x8,%rsp
    11f4:	c3                   	retq   
```

15. Understanding text Section in ELF
- `.text` contains the source code
  - `<_start>` begins prior to `<main>`
```bash
$ objdump --section .text -d ./a.out 
./a.out:     file format elf64-x86-64
Disassembly of section .text:
0000000000001060 <_start>:
...
    1081:	48 8d 3d c1 00 00 00 	lea    0xc1(%rip),%rdi        # 1149 <main>  ## <--- calls main here
    1088:	ff 15 52 2f 00 00    	callq  *0x2f52(%rip)        # 3fe0 <__libc_start_main@GLIBC_2.2.5>
...
0000000000001149 <main>:
    1149:	f3 0f 1e fa          	endbr64 
    114d:	55                   	push   %rbp
...
```
16. Understanding Data Sections of ELF
- Using a following source code:
```c
#include<stdio.h>
#define MYG "54321\n"
char xyz[3] = "QXY";
int main() {
  printf("hello world\n");
  int abc[3] = { 5,4,3};
  char tmp[2];
  return 0; }
```
- `.rodata`: read-only data. In this sample code, `hello world`
```bash
$ objdump --section .rodata -d ./a.out2 
./a.out:     file format elf64-x86-64
Disassembly of section .rodata:
0000000000002000 <_IO_stdin_used>:
    2000:	01 00 02 00 68 65 6c 6c 6f 20 77 6f 72 6c 64 00     ....hello world.
```
- `.data`: global data. Not local array. 
```bash
$ objdump --section .data -d ./a.out2
./a.out2:     file format elf64-x86-64
Disassembly of section .data:
0000000000004000 <__data_start>:
	...
0000000000004008 <__dso_handle>:
    4008:	08 40 00 00 00 00 00 00                             .@......
0000000000004010 <xyz>:
    4010:	51 58 59                                            QXY
```
- Note that only `QXY` is found, not `{5,4,3}`
- `.bss`: uninitialized data
```bash
$ objdump --section .bss -d ./a.out2
./a.out2:     file format elf64-x86-64
Disassembly of section .bss:
0000000000004013 <completed.8061>:
    4013:	00 00 00 00 00                                      .....
```
- Uninitialized `tmp[2]` consumes 2\*8bytes=16 but disassembly shows 5\*8bytes, mismatching. When initialized, it will allocate the correct size

17. What is the use of plt and got sections in ELF?
- In `.text`, to execute `printf()`, we need the address of the function in C library
  - But cannot call the function directly
- `.plt` section gives an address to the print function in `.main`
  - This is the address of `.got` section
- In the `.got` section, it locates the address of puts() function, which is stored in the library (Global Offset Table)
- Summarizing, it needs two steps for printf() in `.main` to reach the puts function adress in the library

18. Tracing plt and got in gdb
```bash
$ gcc -g hello.c -o a.out3
$ gdb -q ./a.out3
Reading symbols from ./a.out3...
(gdb) set disassembly-flavor intel
(gdb) disassemble main
Dump of assembler code for function main:
   0x0000000000001169 <+0>:	endbr64 
   0x000000000000116d <+4>:	push   rbp
   0x000000000000116e <+5>:	mov    rbp,rsp
   0x0000000000001171 <+8>:	sub    rsp,0x20
   0x0000000000001175 <+12>:	mov    rax,QWORD PTR fs:0x28
   0x000000000000117e <+21>:	mov    QWORD PTR [rbp-0x8],rax
   0x0000000000001182 <+25>:	xor    eax,eax
   0x0000000000001184 <+27>:	lea    rdi,[rip+0xe79]        # 0x2004
   0x000000000000118b <+34>:	call   0x1060 <puts@plt> #<----- goes to plt section first
   0x0000000000001190 <+39>:	mov    DWORD PTR [rbp-0x18],0x5
   0x0000000000001197 <+46>:	mov    DWORD PTR [rbp-0x14],0x4
   0x000000000000119e <+53>:	mov    DWORD PTR [rbp-0x10],0x3
   0x00000000000011a5 <+60>:	mov    eax,0x0
   0x00000000000011aa <+65>:	mov    rdx,QWORD PTR [rbp-0x8]
   0x00000000000011ae <+69>:	xor    rdx,QWORD PTR fs:0x28
   0x00000000000011b7 <+78>:	je     0x11be <main+85>
   0x00000000000011b9 <+80>:	call   0x1070 <__stack_chk_fail@plt>
   0x00000000000011be <+85>:	leave  
   0x00000000000011bf <+86>:	ret    
End of assembler dump.
(gdb) x/3i 0x1060 # <--- investigating plt address
   0x1060 <puts@plt>:	endbr64 
   0x1064 <puts@plt+4>:	bnd jmp QWORD PTR [rip+0x2f5d]        # 0x3fc8 <puts@got.plt>
   0x106b <puts@plt+11>:	nop    DWORD PTR [rax+rax*1+0x0]
```

19. Understanding rel section in ELF
- Relocation in binary
  - Addresses in library function
  - `.rela.dyn`
  - `.rela.plt`
```
$ readelf --relocs ./a.out
Relocation section '.rela.dyn' at offset 0x568 contains 8 entries:
  Offset          Info           Type           Sym. Value    Sym. Name + Addend
000000003db0  000000000008 R_X86_64_RELATIVE                    1160
000000003db8  000000000008 R_X86_64_RELATIVE                    1120
000000004008  000000000008 R_X86_64_RELATIVE                    4008
000000003fd8  000100000006 R_X86_64_GLOB_DAT 0000000000000000 _ITM_deregisterTMClone + 0
000000003fe0  000400000006 R_X86_64_GLOB_DAT 0000000000000000 __libc_start_main@GLIBC_2.2.5 + 0
000000003fe8  000500000006 R_X86_64_GLOB_DAT 0000000000000000 __gmon_start__ + 0
000000003ff0  000600000006 R_X86_64_GLOB_DAT 0000000000000000 _ITM_registerTMCloneTa + 0
000000003ff8  000700000006 R_X86_64_GLOB_DAT 0000000000000000 __cxa_finalize@GLIBC_2.2.5 + 0
Relocation section '.rela.plt' at offset 0x628 contains 2 entries:
  Offset          Info           Type           Sym. Value    Sym. Name + Addend
000000003fc8  000200000007 R_X86_64_JUMP_SLO 0000000000000000 puts@GLIBC_2.2.5 + 0
```

20. What is the use of init array and fini array in ELF?
- `.init_array` section contains an array of pointers to functions to use as constructors
```bash
$ objdump --section .init_array -d ./a.out2
./a.out2:     file format elf64-x86-64
Disassembly of section .init_array:
0000000000003db0 <__frame_dummy_init_array_entry>:
    3db0:	60 11 00 00 00 00 00 00                             .......
```
  - This (60 11) points to <frame_dummy> as shown (1160):
```bash
0000000000001160 <frame_dummy>:
    1160:	f3 0f 1e fa          	endbr64 
    1164:	e9 77 ff ff ff       	jmpq   10e0 <register_tm_clones>
```
- `.fini_array` section contains an array of pointeres to functions to use as destructors
```bash
$ objdump --section .fini_array -d ./a.out2
./a.out2:     file format elf64-x86-64
Disassembly of section .fini_array:
0000000000003db8 <__do_global_dtors_aux_fini_array_entry>:
    3db8:	20 11 00 00 00 00 00 00                              .......
```
  - This (20 11) points to _do_golobal_dtors_aux as destructor (1120)
```bash
0000000000001120 <__do_global_dtors_aux>:
    1120:	f3 0f 1e fa          	endbr64 
    1124:	80 3d e8 2e 00 00 00 	cmpb   $0x0,0x2ee8(%rip)        # 4013 <completed.8061>
    112b:	75 2b                	jne    1158 <__do_global_dtors_aux+0x38>
    112d:	55                   	push   %rbp
```
21. Understanding string table sections in ELF
- `.symtab` section: containts a symbol table, which is a table of function or variable in ELF
```bash
$ readelf -s ./a.out
Symbol table '.dynsym' contains 8 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND 
     1: 0000000000000000     0 NOTYPE  WEAK   DEFAULT  UND _ITM_deregisterTMCloneTab
     2: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND puts@GLIBC_2.2.5 (2)
     3: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND __stack_chk_fail@GLIBC_2.4 (3)
     4: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND __libc_start_main@GLIBC_2.2.5 (2)
...
Symbol table '.symtab' contains 72 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND 
     1: 0000000000000318     0 SECTION LOCAL  DEFAULT    1 
...
$ readelf -p .symtab ./a.out
String dump of section '.symtab':
  [    38]  8^C
  [    50]  X^C
  [    68]  |^C
  [    c8]  &^E
...
```
- `.strtab`: contains strings having the symbolic names
```bash
$ readelf -p .strtab ./a.out
String dump of section '.strtab':
  [     1]  crtstuff.c
  [     c]  deregister_tm_clones
  [    21]  __do_global_dtors_aux
  [    37]  completed.8061
...
```
- `.shstrtab` section contains strings of section header
```bash
$ readelf -p .shstrtab ./a.out
String dump of section '.shstrtab':
  [     1]  .symtab
  [     9]  .strtab
  [    11]  .shstrtab
  [    1b]  .interp
  [    23]  .note.gnu.property
  [    36]  .note.gnu.build-id
```

22. Backtracing section bytes in hexdump in ELF
```bash
$ readelf --wide -S ./a.out
There are 36 section headers, starting at offset 0x42d0:
Section Headers:
  [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
  [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
  [ 1] .interp           PROGBITS        0000000000000318 000318 00001c 00   A  0   0  1
  [ 2] .note.gnu.property NOTE            0000000000000338 000338 000020 00   A  0   0  8
  [ 3] .note.gnu.build-id NOTE            0000000000000358 000358 000024 00   A  0   0  4
...
```
- As section headers starts at 0x42d0, we trace using hexdump:
```bash
$ hexdump -C ./a.out |grep -A7 42d0
000042d0  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................| ## <--- NULL part
*
00004310  1b 00 00 00 01 00 00 00  02 00 00 00 00 00 00 00  |................| ## <--- .interp and PROGBITS
00004320  18 03 00 00 00 00 00 00  18 03 00 00 00 00 00 00  |................| ## <--- Address of 0318  and offset 000318
00004330  1c 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00004340  01 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00004350  23 00 00 00 07 00 00 00  02 00 00 00 00 00 00 00  |#...............|
00004360  38 03 00 00 00 00 00 00  38 03 00 00 00 00 00 00  |8.......8.......|
```

23. What is a program header?
- Program header: chunk of segments containing section in a segmented way
  - Used by OS and dynamic linker
- Sections are for static linking purpose
```bash
$ readelf --wide -l ./a.out
Elf file type is DYN (Shared object file)
Entry point 0x1080
There are 13 program headers, starting at offset 64
Program Headers:
  Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
  PHDR           0x000040 0x0000000000000040 0x0000000000000040 0x0002d8 0x0002d8 R   0x8
  INTERP         0x000318 0x0000000000000318 0x0000000000000318 0x00001c 0x00001c R   0x1
      [Requesting program interpreter: /lib64/ld-linux-x86-64.so.2]
  LOAD           0x000000 0x0000000000000000 0x0000000000000000 0x000658 0x000658 R   0x1000
  LOAD           0x001000 0x0000000000001000 0x0000000000001000 0x000245 0x000245 R E 0x1000
...
  GNU_RELRO      0x002db0 0x0000000000003db0 0x0000000000003db0 0x000250 0x000250 R   0x1

 Section to Segment mapping:
  Segment Sections...
   00     
   01     .interp 
   02     .interp .note.gnu.property .note.gnu.build-id .note.ABI-tag .gnu.hash .dynsym .dynstr .gnu.version .gnu.version_r .rela.dyn .rela.plt 
   03     .init .plt .plt.got .plt.sec .text .fini 
   04     .rodata .eh_frame_hdr .eh_frame 
   05     .init_array .fini_array .dynamic .got
...
   09     .note.gnu.property 
   10     .eh_frame_hdr 
   11     
   12     .init_array .fini_array .dynamic .got 
```
- 13 segments contain section data
- Flg shows Read/Write/Executable
- 03rd segment contains .init .plt .plt.got .plt.sec .text .fini, which are executed
- DYNAMIC section is RW

## Section 4: Binary (ELF) analysis tools and techniques

24. Identifying hidden identity of files
- A challenge scenario: figure out and analyze a given file
```bash
$ file TOP_secret 
TOP_secret: ASCII text
```
- Let's decode: `base64 -d TOP_secret > decoded_file`
```bash
$ file decoded_file 
decoded_file: Zip archive data, at least v1.0 to extract
```
- This is a zip file
```bash
$ unzip decoded_file
Archive:  decoded_file
 extracting: 8754657                 
  inflating: top_secret              
$ file top_secret 
top_secret: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, BuildID[sha1]=ab009d7ad57bb8c013279dbc4b9e3a4c03e524d4, for GNU/Linux 3.2.0, not stripped
$ file 8754657
8754657: PNG image data, 206 x 24, 8-bit/color RGB, non-interlaced
```
- In this example, the image contains the passwd which is necessary to execute ./top_secret
- Lessons: 
  - Use `file` command to figure out the type of files

25. Investigating library files of the ELF
```bash
$ file Challenge2 
Challenge2: ASCII text
$ base64 -d Challenge2 > decoded_file
$ file decoded_file 
decoded_file: Zip archive data, at least v2.0 to extract
$ unzip decoded_file
Archive:  decoded_file
  inflating: test                    
  inflating: b1                      
$ file test
test: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, BuildID[sha1]=01636c6f2307f167ecc588431e2e5db1bbd42463, for GNU/Linux 3.2.0, not stripped
$ file b1
b1: PNG image data, 132 x 24, 8-bit/color RGB, non-interlaced
$ ldd test
./test: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found (required by ./test)
	linux-vdso.so.1 (0x00007ffd88ae1000)
	libconfidential.so => not found
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f841e40a000)
	/lib64/ld-linux-x86-64.so.2 (0x00007f841e62b000)
```
- libconfidential.so is missing to run test executable
- Let's check hexdump: `hexdump -C b1| less`
- Check any contents of ELF:
```bash
$ hexdump -C b1| grep ELF
000006a0  7f 45 4c 46 02 01 01 00  00 00 00 00 00 00 00 00  |.ELF............|
``` 
  - ELF is found. Let's extract those portions
```bash  
$ printf "%d\n" 0x6a0 # from hexdump of ELF portion
1696
$ dd skip=1696 if=b1 of=b2 bs=1
15578+0 records in
15578+0 records out
15578 bytes (16 kB, 15 KiB) copied, 0.0309297 s, 504 kB/s
$ file b2
b2: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked, BuildID[sha1]=0e66249c30b7b1cd2d24dab275bdc220f57866f1, not stripped
$ readelf -h b2
ELF Header:
  Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00 
  Class:                             ELF64
  Data:                              2's complement, little endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              DYN (Shared object file)
  Machine:                           Advanced Micro Devices X86-64
  Version:                           0x1
  Entry point address:               0x0
  Start of program headers:          64 (bytes into file)
  Start of section headers:          13656 (bytes into file)
  Flags:                             0x0
  Size of this header:               64 (bytes)
  Size of program headers:           56 (bytes)
  Number of program headers:         11
  Size of section headers:           64 (bytes)
  Number of section headers:         30
  Section header string table index: 29
```
- Now rename b2 as libconfidential.so
- `echo "__bitcode__" | base64 -d` to decode at CLI

26. Hidden identification and extraction of ELF
- Hidden binary inside of a binary
```bash
$ hexdump -C HelloWorld  | grep ELF
00000000  7f 45 4c 46 02 01 01 00  00 00 00 00 00 00 00 00  |.ELF............|
00003e60  00 00 00 42 58 3e 00 00  7f 45 4c 46 02 01 01 00  |...BX>...ELF....|
00007cd0  00 00 7f 45 4c 46 02 01  01 00 00 00 00 00 00 00  |...ELF..........|
```
- 3 ELF's are found
  - `.ELF` is magic 4bytes
  - 2nd ELF begins at 3e60+8 bytes. We need to convert to decimal
```bash
$ printf "%d\n" 0x3e68 # 3e60 + 8 bytes
15976
$ dd skip=15976 if=HelloWorld of=f1 bs=1
31940+0 records in
31940+0 records out
31940 bytes (32 kB, 31 KiB) copied, 0.0543647 s, 588 kB/s
$ readelf -h f1
ELF Header:
  Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00 
  Class:                             ELF64
  Data:                              2's complement, little endian
  Version:                           1 (current)
  ...
```
- To make sure of the validity of the produced binary, run `readelf -h`

27. Correct byte order extraction from the ELF
- How to extract exact bytes from a given file?
  - HelloHuman was 15960 bytes but now it shows 15962 bytes
- `readelf -h HelloHuman` and check the details
  - Size of section headers \* Number of section  headers = 64bytes \* 31 = 1984 bytes
  - Add this to start of section headers (13976 bytes) = 13976+ 1984 = 15960 bytes
  - This is the exact size of the file
- `dd skip=0 count=15960 if=HelloHuman of=original_HH bs=1`
  - Now the size matches and md5sum yields the same hash numbers

28. Identification of printable characters in ELF
```c
#include<stdio.h>
#include<string.h>
int main() {
  char pass[10]="P@ssWord";
  char input[10];
  int result;
  printf("Input the password:");
  scanf("%s",input);
  result = strcmp(pass,input);
  if (result ==0) printf("Correct\n");
  else printf("Wrong\n");
  return 0;
}
```
- We use `strings` command
  - Prints the sequences of printable characters in files
```bash
$ strings ./a.out
/lib64/ld-linux-x86-64.so.2
libc.so.6
__isoc99_scanf
...
_ITM_registerTMCloneTable
u+UH
P@ssWordH ## <---------- Found !!!
[]A\A]A^A_
```

29. Tracing Sys calls of an ELF
- Sample assembly file
```assembly
.text
.global _start
_start:
  mov $1, %rax
  mov $1, %rdi
  mov $msg, %rsi
  mov $len, %rdx
  syscall  # write syscall

  mov $60, %rax
  mov $0, %rdi
  syscall  # exit syscall
msg:
  .ascii "Hello World from write syscall\n"
len = . - msg
```
- Compiling:
```bash
$ as chap28.s -o h.o
$ ld h.o -o h.exe
$ ./h.exe 
Hello World from write syscall
```
- Using strace:
```bash
$ strace ./h.exe
execve("./h.exe", ["./h.exe"], 0x7fffd299c780 /* 62 vars */) = 0
write(1, "Hello World from write syscall\n", 31Hello World from write syscall
) = 31
exit(0)                                 = ?
+++ exited with 0 +++
$ strace -c ./h.exe  ## counts the number of syscalls (exit is not counted)
Hello World from write syscall
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
  0.00    0.000000           0         1           write
  0.00    0.000000           0         1           execve
------ ----------- ----------- --------- --------- ----------------
100.00    0.000000                     2           total
$ strace -e write ./h.exe  ## extracts write syscall only
write(1, "Hello World from write syscall\n", 31Hello World from write syscall
) = 31
+++ exited with 0 +++
```

30. Analysing instructions from disassembly of a binary
- Analyzing the results of objdump
  - Password inside of the executable
- objdump -d a.exe
  - From main:
```as
  401295:       e8 26 fe ff ff          callq  4010c0 <__isoc99_scanf@plt>
  40129a:       48 8d 45 90             lea    -0x70(%rbp),%rax  ## result of scan
  40129e:       48 89 c6                mov    %rax,%rsi
  4012a1:       48 8d 05 7d 0d 00 00    lea    0xd7d(%rip),%rax        # 402025 <_IO_stdin_used+0x25>
  4012a8:       48 89 c7                mov    %rax,%rdi
```
- Let's trace \%rax
```as
  4011d4:       48 b8 49 74 73 54 6f    movabs $0x6145306f54737449,%rax
  4011db:       30 45 61
  4011de:       48 ba 73 79 54 30 63    movabs $0x6361726330547973,%rdx
  4011e5:       72 61 63
```
- As this is Little Endian, we need to change the position of hexa number
  - `echo -e "\x49\x74\x73\x54\x6f\x30\x45\x61"`
  - Using this way, we can deduce the password

31. Dynamic analysis of Strings in ELF

32. Dynamic disassembly vs static disassembly
- Using gdb
  - info file: find the Entry point
    - Put a break point at the entry point
  - set pagination off
  - set logging on
  - diplay /i $pc
  - run
  - while 1
    - si
    - end
  - q
- Check gdb.txt -> dynamic disassembly analysis
- Using objdump is static disassembly analysis

## Section 5: Code injection techniques in ELF

33. Static code injectin in ELF
- sample.c:
```c
#include <stdio.h>
void abc() { printf("Hello from abc\n"); }
int main() { printf("Hello from main\n"); return 0; }
```
- How to run abc()?
- In hexa code, _start -> main
```as
0000000000001060 <_start>:
    1060:	f3 0f 1e fa          	endbr64 
    1064:	31 ed                	xor    %ebp,%ebp
    1066:	49 89 d1             	mov    %rdx,%r9
    1069:	5e                   	pop    %rsi
    106a:	48 89 e2             	mov    %rsp,%rdx
    106d:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    1071:	50                   	push   %rax
    1072:	54                   	push   %rsp
    1073:	4c 8d 05 76 01 00 00 	lea    0x176(%rip),%r8        # 11f0 <__libc_csu_fini>
    107a:	48 8d 0d ff 00 00 00 	lea    0xff(%rip),%rcx        # 1180 <__libc_csu_init>
    1081:	48 8d 3d d8 00 00 00 	lea    0xd8(%rip),%rdi        # 1160 <main>
    1088:	ff 15 52 2f 00 00    	callq  *0x2f52(%rip)        # 3fe0 <__libc_start_main@GLIBC_2.2.5>
    108e:	f4                   	hlt    
    108f:	90                   	nop
0000000000001160 <main>:
    1160:	f3 0f 1e fa          	endbr64 
    1164:	55                   	push   %rbp
    1165:	48 89 e5             	mov    %rsp,%rbp
    1168:	48 8d 3d a4 0e 00 00 	lea    0xea4(%rip),%rdi  
$ hexdump -C ./a.out |grep "48 8d 3d d8"
00001080  00 48 8d 3d d8 00 00 00  ff 15 52 2f 00 00 f4 90  |.H.=......R/....|
Go to 1080 in hexedit
0000000000001149 <abc>:
    1149:	f3 0f 1e fa          	endbr64 
    114d:	55                   	push   %rbp
    114e:	48 89 e5             	mov    %rsp,%rbp
    1151:	48 8d 3d ac 0e 00 00 	lea    0xeac(%rip),%rdi        # 2004 <_IO_stdin_used+0x4>
    1158:	e8 f3 fe ff f
```
- Not working in 64bit
    
34. Understanding code injection technique inside the ELF
- push point of the assembly is the entry point of the executable

