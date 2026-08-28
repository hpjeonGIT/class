# Computer Architecture and Computer Organization Masterclass
- Instructor: Dr. Yasas Sri Wickramasinghe

## Section 1: Welcome to Your Course

### 1. Hello Everyone. Get to know your course structure

## Section 2: Introduction to Computer Organization and Architecture

###  2. Introduction to Computer Organization and Architecture
- Computer organization: physical aspects of computer systems such as circuite design, control signals, memory types - How does a computer work?
- Computer architecture: logical aspects of system as seen by the programmer
  - Instruction sets, instruction formats, data types, address modes - How do I design a computer?
- Equivalence of HW and SW: anything that can be deon with SW can also be done with HW, and anything that can be done with HW can alswo be done with SW
- A computer is a device consisting of three pieces
  - A processor: to interpret and execute a program
  - A memory: to store both data and programs
  - A mechanism: for transferring data to and from the outside world

### 3. Computer Level Hierarchy
- Moore's law: the density of transistors in an IC will double every year
- Rock's law: the cost of capital equipment to build semiconductors will double every four years
- Computer level hierarchy
  - Level 6: User - executable programs
  - Level 5: High-level language - C++, Java, Fortran, etc
  - Level 4: Assembly language 
  - Level 3: System software - OS, library code
  - Level 2: Machine - Instruction Set Architecture
  - Level 1: Control - Microcode or hardwired
  - Level 0: Digital logic - circuits,gates, etc


### 4. Introduction to Computer Organization and Architecture Lecture Materials


## Section 3: Fetch - Decode - Execute Cycle

### 5. Fetch Decode Execute Cycle Explained Part 1
- Fetch, decode and execute cycle
  - Program instructions are stored inside the main memory
  - The machine runs the programs sequentially
  - Each machine instruction is fetched, decoded and executed during one cycle known as the von Neumann execution cycle (also called the fetch-decode-execute cycle)
  - One iteration of the cycle is as follows:
    - The control unit fetches the next program instruction from the memory, using the program counter to determine where the instruction is located.
    - The instruction is decoded into a language the ALU can understand.
    - Any data operands required to execute the instruction are fetched from memory andplaced into registers within the CPU.
    - The ALU executes the instruction and places the results in registers or memory.

### 6. Fetch Decode Execute Cycle Explained Part 2
- Special purpose registers

| Register type | Notation | Purpose |
|---------------|----------|---------|
| Accumulator   | AC| Stores the results of calculations |
| Instruction Register | IR/CIR | Stores the address in RAM of the instruction to be processed|
| Memory Address Register| MAR | Stores the address in RAM of the data to be procssed |
| Memory Data Register | MDR | Stores the data that is being processed |
| Program Counter |PC| Stores the address in RAM of the next instruction|

<img src="./ch06_fetch.png" height="300">
<img src="./ch06_exe.png" height="300">

### 7. Fetch - Decode - Execute Cycle Lecture Materials

## Section 4: Assembly Language Programming with the Little Man Computer

### 8. What is the Little Man Computer
- https://peterhigginson.co.uk/lmc/

### 9. Programming the Little Man Computer
- LMC instructions: https://en.wikipedia.org/wiki/Little_Man_Computer
  - 1XX: ADD
  - 2XX: SUB
  - 3XX: STA
  - 5XX: LDA - Load
  - 6XX: BRA - Branch
  - 7XX: BRZ - Branch if 0
  - 8XX: BRP - Branch if +
  - 901: Input
  - 902: Output
  - 000: Halt
- Scenario
  - Memory cell 60 contains the value of 005
  - Memory cell 61 contains the value of 006
  - Memory cell 62 contains the value of 002
<img src="./ch09_mem.png" height="50">

  - Add three numbers and store the result in the memory cell id 65
- Assembly program
  - 560: LOAD memory in cell 60
  - 161: ADD memory in cell 61
  - 162: ADD memory in cell 62
  - 365: Store the value in the accumulator into cell 65
<img src="./ch09_isa.png" height="50">

### 10. Fetch Decode Execute Cycle Explained using the Little Man Computer
- Click RUN Icon
<img src="./ch10_result.png" height="500">
- Cell ID 65 has the value of 13 at PC=4
  - Instruction register is 3 which is STORE instruction
  - Address register is 65 (Cell ID)
  - Accumulator is 013

### 11. Writing Assembly Language Code
```asm
LDA 60
ADD 61
ADD 62
STA 65
HLT
```
- Click Submit and run
  - Cell data of 60,61,62 are given manually
<img src="./ch11_asm.png" height="500">

## Section 5: Instruction Set Architecture (ISA)

### 12. Introduction to ISA
- ISA
  - A well-defined HW/SW interface
  - The contract b/w SW and HW
    - Functional definition of storage location & operations
      - Storage locations: registers, memory
      - Operations: add, multiply, branch, load, store, etc
  - ISA can have multiple implementations
  - ISA allows SW to direct HW
  - ISA defines machine language

### 13. CISC and RISC
- Reduced Instruction set Computing - RISC
  - Fixed length instructions (mostly 32bit)
  - Small, highly optimized set of instructions
- Complex Instruction Set Computing - CISC
  - Single instruction represents multiple operations (low level -> IO, ALU, mem)
- CISC: x86
  - Larger instruction set
  - More complicated instructions built into HW
  - Variable length
  - Multiple clock cycles per instruction
- RISC: ARM
  - Small, highly optimized set of instructions
  - Memory accesses are specific instructions
  - One instruction per clock cycle
  - Instructions are of the same size and fixed format
<img src="./ch13_ciscrisc.png" height="200">
- CISC
```asm
MULT B, A
```
- RISC
```asm
LOAD A, eax
LOAD B, ebx
PROD eax, ebx
STORE ebx, A
```

### 14. Instructions
- Elements
  - Opcode: what to do
  - Operand : data sources/destinations
- Representation
  - Binary bits
    - 4 bits of opcode + 6bits of operand reference +  6bits of operand reference
    - Symbolic representation
      - ADD, SUB, LOAD etc
      - Ex) ADD X, Y
- Instruction length
  - Affected by memory size/organization, register numbers, bus structure etc
  - Flexibility vs implementation complexity
  - Memory-transfer consideration
  - Fixed vs non-fixed instructions

### 15. Number of Addressing
- 3 addresses
  - Operand 1, Operand 2, Result: a = b+c
- 2 addresses
  - One address doubles as operand and result: a = a+c
- 1 address
  - Implicit second address (accumulator)
- 0 address
  - All addresses are implicitly defined
  - Stack based computer

<img src="./ch15_ops.png" height="300">

### 16. Addressing Modes
- How is the address of an operand specified
- Different addressing mode
  - Immediate
    - Operand is a part of instruction: opcode + operand
    - Operand = address field
    - Ex) ADD 5
      - Add 5 to the content of accumulator
      - 5 is operand
    - No memory reference to fetch data
    - Fast
    - Limited range
  - Direct
    - Address field contains the address of operand
    - Effective address (EA) = address field (A)
    - Ex) ADD A
      - Add content of cell A to accumulator
      - Look in memory at address A for operand
    - Single memory reference to access data
    - No additional calculations to work out effective address
    - Limited address space
  - Indirect
    - Memory cell pointed to by address field contains the address of (pointer to) the operand
    - EA = (A)
      - Look in A, find address (A) and look there for operand
    - Ex) ADD (A)
      - Add contents of cell pointed to by contents of A to accumulator
    - Large address space
    - Unique memory size of 2^n where n = word length
    - May be nested, multi-level, cascaded
      - Ex) EA = (((A)))
    - Multiple memory accesses to find operand
    - Hence slower
  - Register
    - Operand in held in register named in address filed
    - EA = R
    - Limited number of registers
    - Very small address field needed
      - Shorter instructions
      - Faster instruction fetch
    - No memory access
    - Very fast execution
    - Very limited address space
    - Multiple registers help performance
      - Requires good assembly programming or compiler writing
      - Compare with direct addressing
  - Register indirect
    - EA = (R)
    - Operand is in memory cell pointed to by contents of register R
    - Compare with indirect addressing
      - Faster than indirect as R is a REGISTER
    - Large address space (2^n)
    - One fewer memory access than indirect addressing
  - Displacement
    - EA = A + (R)
    - Address field holds two values
      - A = base value
      - R = register that holds displacement
      - Or vice versa
  - Indexed addressing
    - A = base
    - R = displacement
    - EA = A + R
    - Good for accessing arrays
      - EA = A + R
      - R++
  - Stack
    - Operand is (implicitly) on top of stack
    - Ex) ADD: pop top wto items from stack and add

<img src="./ch16_summary.png" height="200">

### 17. Instruction Set Architecture Lecture Materials

## Section 6: CPU Benchmarking

### 18. Introduction to CPU Benchmarking
- What makes a good ISA?
  - Programmability
    - Easy to express programs efficiently?
  - Performance/implementability
    - Easy to design high-performance implementations?
    - More recently:
      - Easy to design low-power implementations?
      - Easy to design low-cost implementations?
  - Compatibility
    - Easy to maintain as languages, programs, and technology evolve?
    - x86 (IA32) generations: 8086, 286, 386, 486, Core2, Core i7, ...
- Programmability
  - Easy to express programs efficiently?
    - For whom?
  - Before 1980s: human
    - Most code was hand-assembled    
    - High-level coarse-grain instructions
  - After 1980s: compiler
    - Low-level fine-grain instructions
  - This shift changed what is considered a "good" ISA
- Implementability
  - Every ISA can be implemented
    - Not every ISA can be implemented efficiently
  - Classic high-performance implementations techniques
    - Pipelining, parallel execution, out-of-order execution
  - Certain ISA features make these difficult
    - Variable instruction lengths/formats: complicate decoding
    - Special purpose reigster: complicate compiler optimizations
    - Difficult to interrupt instructions: complicate many things
- Performance
  - How long does it take for a program to execute?
      1. How many instructions must execute to complete a program?
        - Instructions per program during execution
        - Dynamic instructions count
      2. How quickly does the processor cycle?
        - Clock frequency in Hz
        - Clock period in ns
        - Worst-case delay through circuit for a particular design
      3. How many cycles does each instruction take to execute?
        - Cycles per Instructions (CPI) or reciprocal, Instructions per Cycle (IPC)
  - Execution time = (instructions/program) * (seconds/cycle) * (cycles/instruction)

### 19. Calculating CPU Time

### 20. Understanding CPU Clock
### 21. Calculating CPU Time
### 22. Exercise - Solving CPU Time Calculations
### 23. Exercise - Solving CPI Calculations
### Coding Exercise 1: Coding Activity: Measuring CPU Benchmarking with Matrix Multiplication in Python
### 24. CPU Benchmarking Lecture Materials
### 25. Extra Resource

## Section 7: CPU Organization and Structure

### 26. Introduction to CPU Structure
### 27. Registers in CPU
### 28. Understanding CPU Interruptions
### 29. Techniques to Improve CPU Performance
### 30. CPU Organization and Structure Lecture Materials
### Quiz 1: Registers in the CPU
Not completed
Start
31. Extra Reading Material
2min

### 32. What is CPU Pipelining
### 33. Resource Hazards
### 34. Data Hazards
### 35. Control Hazards and Branch Prediction
### 36. Branch Prediction Strategies
### 37. Practical Example for Pipelining - Intel 80486
### 38. CPU Overclocking
### 39. CPU Pipelining Lecture Materials
1min

### 40. Introduction to I/O
### 41. I/O Mapping
### 42. Asynchronous Data Transfer
### 43. Modes of Data Transfer
### 44. Input-Output Organization Lecture Materials
1min

### 45. Introduction to Memory Hierarchy
### 46. Deep dive into Computer Memory Hierarchy
### 47. The Principal of Locality
### 48. Memory HIT rate and MISS rate
### 49. Cache Performance and Optimization
### 50. Exercise - Calculating Miss Rate
### 51. Memory Technology
### 52. DRAM Technology
### 53. How a DRAM Works
### 54. DRAM Read Cycle Deeply Explained Step by Step
### 55. SDRAM and DDR SDRAM Explained
### 56. Memory Organization Lecture Materials
### 57. Extra Reading Material
1min

### 58. Introduction to Hierarchical Bus Structures
### 59. Single and Multiple Bus Implementations and Examples
### 60. Bus Types, Timing, and Additional Details
### 61. Hierarchical Bus Organization Lecture Materials
1min

Not completed
Start
Practice Test 1: Mixed Instruction Benchmark in Computer Organization and Architecture
Not completed
Start
Practice Test 2: Mixed Instruction Benchmarking in Computer Organization and Architecture

### 62. Summary
### 63. Course Summary Short Note - Mind Map
### 64. Thank you
1min
