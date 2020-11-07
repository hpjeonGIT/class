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
  
