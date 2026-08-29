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
- Comparing machines
  - Metrics
    - Execution time
    - Throughput
    - CPU time
    - MIPS: millions of instructions per second
    - MFLOPS: millions of floating point operations per second
    
### 19. Calculating CPU Time
- Response time: the time b/w the start and completion of a task, including time spent on CPU, disk, memory, waiting for IO and other processes + OS overhead. Also referred as execution time
- Throughput: the total amount of work done in a given time
- CPU execution time: total time a CPU spends computing on a given task - excludes time for IO or running other programs - referred as CPU time

### 20. Understanding CPU Clock
- A computer clock runs at a constant rate and determines when events take placed in HW
- The clock cycle time is the amount of time for one clock period to eplase (e.g. 5ns)
- The clock rate is the inverse of the clock cycle time
  - 200MHz -> 5ns
- Is # of cycles ==  # of instructions?
  - No. Different instructions take different amounts of time on different machines  

### 21. Calculating CPU Time
- CPU time = CPU clock cycles x clock cycle time
- CPU time = CPU clock cycles / clock rate
- CPU clock cycles = (instructions/program) x (clock cycles/instruction) = Instruction count x CPI
  - CPU time = Instruction count x CPI x clock cycle time
  - CPU time = Instruction count x CPI / clock rate
- Which factors are affected by each of the following?

|   | Instr. Count | CPI | clock rate |
|----|-------------|-----|------------|
| Program | x      |     |            |
| Compiler |x      |  x  |            |
| ISA |x           |  x  |            |
| Organization |   |  x  |            |
| Technology   |   |     |  x         |

### 22. Exercise - Solving CPU Time Calculations
- Ex1
  - CPU clock rate is 1 MHz
  - Program takes 45 million cycles to execute
  - CPU time = 45e6 * (1/1e6) = 45 sec
- Ex2
  - CPU clock rate is 500 MHz
  - Program takes 45 million cycles to execute
  - CPU time = 45e6 * (1/500e6) = 0.09 sec

### 23. Exercise - Solving CPI Calculations
- Ex
  - A benchmark as 100 instructions
  - 25 instructions are loads/stores, each taking 2 cycles
  - 50 instructions are adds, taking 1 cycle each
  - 25 instructions are square root, each taking 50 cycles
  - CPI = (2 * 25/100) + (1 * 50/100) + (50 * 25/100) = 13.5
- Benchmark
  - Allows us to make performance comparisons based on execution times
  - Must
    - be representative of the type of applications run on the computer
    - not be overly depedent on one or two features of a computer
  - Can vary greatly in terms of their complexity and their usefulness

### Coding Exercise 1: Coding Activity: Measuring CPU Benchmarking with Matrix Multiplication in Python

### 24. CPU Benchmarking Lecture Materials

### 25. Extra Resource

## Section 7: CPU Organization and Structure

### 26. Introduction to CPU Structure
- Requirement of processor
  - Fetch instruction: reads an instruction from memory
  - Interpret instruction: determines what action to perform
  - Fetch data: if necessary read data from memory or an IO module
  - Process data: if necessary perform arithmetic/logical operation on data
  - Write data: if necessary write data to memroy or an IO module
- Major components of the processor
  - ALU (Arithmetic and Logic Unit): performs computation or processing of data
  - Control unit: moves data and instructions in and out of the processor. Also controls the operation of the ALU
  - Registers: internal memory
  - System bus: acting as a pathway b/w processor, memory, and IO module

<img src="./ch26_cpu.png" height="300">

### 27. Registers in CPU
- Registers in the processor perform two roles
  - User-visible registers
    - Used as internal memory by the assembly language programmer
  - Control and status registers
    - Used to control the operation of the processor
    - Used to check the status of the processor/ALU
- User-visible registers
  - Referenced by the programmer, categorized into 4 categories
  1. General purpose
      - Memory reference & backup
      - Register reference & backup
      - Data reference & backup
  2. Data
      - May be used only to hold data and cannot hold addresses
      - Must be able to hold values of most data types
      - Some machines allow two contiguous registers to be used, for holding double-length values
  3. Address
      - Used to hold addresses of stack pointer, program counter, index registers
      - Must be at least long enough to hold the largest address
  4. Condition codes/flags
      - Holding condition codes/flags which are bits set by processor as the result of operations
      - Condition code bits are collected into one or more control register
      - As an example, an arithmetic operation can produce: positive result, negative result, zero result, overflow result
- Control and status registers
  - Mostly not visible to the user
  - Program Counter (PC): contains instruction address to be fetched
  - Instruction Register (IR): contains the last instruction fetched
  - Memory Address Register (MAR): contains memory location address
  - Memory Buffer Register (MBR): contains a word of data to be written to memory or a word of data read from memory
  - Those registers are used for
    - Data movement b/w processor and memory
    - Within the processor, data must be presented to the ALU for processing
      - ALU may have direct access to the MBR and user-visible registers
      - Alternatively:
        - There may be additional buffering registers within ALU
        - These registers serve as input and output registers for the ALU
        - These registers exchange data with the MBR and user-visible registers

### 28. Understanding CPU Interruptions
- Interruption cycle
  - Contents of the PC must be saved
  - The contents of PC are:
    - Transferred to the MBR to be written into memory
    - Special memory location is loaded into MAR
      - E.g: Stack Pointer (SP)
    - PC is loaded with the address of the interrupt routine

### 29. Techniques to Improve CPU Performance
- How to increase processor performance?
  - Increase frequency - faster number of clock ticks per unit of time
  - Increase cache-levels - reduce number of read/writes from high latency memory
  - Multi-core architecture - parallel processing
  - Reduce physical size of the processor - electrical signals travel shorter distances
  - CPU pipelining

### 30. CPU Organization and Structure Lecture Materials

### Quiz 1: Registers in the CPU

### 31. Extra Reading Material

## Section 8: CPU Pipelining

### 32. What is CPU Pipelining
<img src="./ch32_pipelining.png" height="200">

- Pipelining breaks instruction execution down into several stages
  - Puts registers b/w stages to buffer data and control
  - Executes one instruction
  - As first starts second stage, executes second instruction, etc
  - Speeds up same as number of stages as long as pipe is full

<img src="./ch32_sample.png" height="300">

- Without pipelining, 9 instructions will take 9x6 = 54 time units
- With pipelining, all instructions can be done in 14 time units

### 33. Resource Hazards
- Hazards do not permit continued pipeline execution
  - Also called pipeline bubble
  - Types of hazards
    - Resource
    - Data
    - Control
- Resource hazards
  - Two or more instructions in pipeline need same resource (bus, memory, cache)
  - Executed in serial rather than parallel for part of pipeline
  - Also called structural hazard
  - If main memory has a single port, read or write cannot be performed in parallel with instruction fetch
  - Single ALU may have the same issue
  - Solutions
    - Multiple main memory ports
    - Multiple ALUs

### 34. Data Hazards
- Conflict in access of an operand location
- Two instructions to be executed in sequence
- Both access a particular memory or register operand
- If in strict sequence, no problem
- If in a pipeline, operand value could be updated so as to produce different result from strict sequential execution

<img src="./ch34_datahazard.png" height="300">

- Types of Data Hazard
  - Read After Write (RAW), or true dependency
    - An instruction modifies a register or memory location
    - Succeeding instruction reads data in that location
    - Hazard if read takes place before write complete
  - Write after read (RAW), or antidependency
    - An instruction reads a register or memory location
    - Succeeding instruction writes to location
    - Hazard if write completes before read takes place
  - Write after write (WAW), or output dependency
    - Two instructions both write to same location
    - Hazard if writes take place in reverse of order intended sequence
  - Previous example is RAW hazard
  
### 35. Control Hazards and Branch Prediction
- Control hazard
  - Known as branch hazard
  - Pipeline makes wrong decision on branch prediction
  - Brings instructions into pipeline that must subsequently be discarded
  - Dealing with branches
    - Multiple streams
    - Prefetch branch target
    - Loop buffer
    - Branch prediction
    - Delayed branching

### 36. Branch Prediction Strategies
- Predict never taken
  - Assumes that jump will not happen
  - Always fetch next instruction
  - 68020 & VAX 11/780
- Predict always taken
  - Assume that jump will happen
  - Always fetch target instruction
- Branch prediction strategies
  - Predict by opcode
    - Some instructions are more likely to result in a jump than others
    - Can get up to 75% success
  - Taken/not taken switch
    - Based on previous history
    - Good for loops
    - Refined by two-level or correlation-based branch history
  - Correlation-based
    - In loop-closing branches, history is good predictor
    - In more complex structures, branch direction correlates with that of related branches
      - Use recent branch history as well
  - Delayed branch
    - Do not take jump until you have to
    - Rearrange instructions

### 37. Practical Example for Pipelining - Intel 80486
- Fetch
  - From cache or external memory
  - Put in one of two 16-byte prefetch buffers
  - Fill buffer with new data as soon as old data consumed
  - Average 5 instructions fetched per load
- Independent of other stages to keep buffers full
  - Decode stage 1
  - Opcode & address-mode info
  - At most first 3 bytes of instruction
  - Can direct D2 stage to get rest of instruction
- Decode stage 2
  - Expand opcode into control signals
  - Computation of complex address modes
- Execute
  - ALU operations, cache access, register update
- Writeback
  - Update registers & flags
  - Results sent to cache & bus interface write buffers

### 38. CPU Overclocking

### 39. CPU Pipelining Lecture Materials

## Section 9: Input-Output Organization

### 40. Introduction to I/O
- Input or output devices attached to the computer are called "peripherals"
- IO interface
  - Provides a method for transferring information b/w internal storage (such as memory and CPU registers) and external IO devices
  - They are special HW components b/w CPU and peripherals to supervise and synchronize all input and output transfer
  - They are called interface units because they interface b/w the processor bus and the peripheral device

### 41. I/O Mapping

### 42. Asynchronous Data Transfer

### 43. Modes of Data Transfer

### 44. Input-Output Organization Lecture Materials

## Section 10: Memory Organization

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

## Section 11: Hierarchical Bus Organization

### 58. Introduction to Hierarchical Bus Structures
### 59. Single and Multiple Bus Implementations and Examples
### 60. Bus Types, Timing, and Additional Details
### 61. Hierarchical Bus Organization Lecture Materials

## Section 12: Course-level Practice Test

### Practice Test 1: Mixed Instruction Benchmark in Computer Organization and Architecture
### Practice Test 2: Mixed Instruction Benchmarking in Computer Organization and Architecture

## Section 13: Conclusion
### 62. Summary
### 63. Course Summary Short Note - Mind Map
### 64. Thank you
