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
|----|
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

## Section 3: Hello World in Assembly

## Section 4: Moving Data

## Section 5: Controlling Execution flow in assembly

## Section 6: Using numbers in assembly language

## Section 7: Basic Math functions in assembly

## Section 8: Working with Strings

## Section 9: Using functions in assembly programming

## Section 10: Using system calls in assembly programming

## Section 11: Inline assembly programming
