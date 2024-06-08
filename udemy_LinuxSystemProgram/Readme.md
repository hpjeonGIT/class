## Linux System Programming Techniques & Concepts
- Instructor : Abhishek CSEPracticals, Shiwani Nigam, Ekta Ekta

## Section 1: Introduction

### 1. Introduction
- Prerequisite: C programming, knowledge of doubly linked lists

### 2. Join Telegram Group

### 3. Setting up Linux Development Environment

### 4. Begin - Doubly linked list as a library
- https://github.com/csepracticals/LibraryDesigning
- https://github.com/csepracticals/DevelopLibrary

### 5. Quick Compilation steps
- Using LibraryDesigning-master/ApplnIntegration/application.c

### 6. Summary

## Section 2: Understanding Header Files

### 7. What are Header Files and their Purpose

### 8. Relationship b/w Source and Header files
- Organization of code
  - Header file: anything needs to be exposed to other source files
  - Source files: anything need not be exposed to other files
- Function declaration vs definition

### 9. Text substitution
- Preprocessing
  - source code -> text substitution -> compilation
  - `#incldue <A.h>` will paste the content of A.h into the source file
    - Recursively done for the nested header files
  - `#define MACRO` will replace MACROs in all places in srouce files

### 10. Text substitution example
- After all text substitions are done, all of hash define inputs `#define XXXX` ar removed from source files then compilation is applied

### 11. Text substitution demonstration
- Using LibraryDesigning-master/Preprocessor/
- `gcc -E app.c -o app.i`
```c
# 1 "chap11/app.c"
# 1 "<built-in>"
# 1 "<command-line>"
# 31 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 32 "<command-line>" 2
# 1 "chap11/app.c"
# 1 "chap11/A.h" 1



int sum (int a, int b);
# 2 "chap11/app.c" 2
# 1 "chap11/B.h" 1




int multiply (int a, int b);
# 3 "chap11/app.c" 2

int foo(int b);
int foo(int b) {
}
int main(){
  int a = (15 * 15);
}
```

## Seciton 3: Processing Directives

### 12. Problem of Duplicate inclusion of header files
- May inject the same header files many times
  - Use preprocessor directives

### 13. What are pre-processing directives?
- C preprocessor directives (#include, #define) are just simple text substitution tools
  - #define, #include, #undef, #ifdef, #ifndef, #if, #else, #elif, #endif
  - May define a unique keyword in each header file, and may check if it is already defined (included) or not
```c
#ifndef __A__
#define __A__
#define max(a,b) (a> b? a:b)
int sum(int a, int b);
#endif
```

### 14. Solution to Duplicate inclusion of header files usig preprocessing directives

## Section 4: Correct way of using structures and functions

### 15. Structure definitions and use
- Rules for defining and using structures and functions
  - Definition first then use

### 16. Function declaration and use
- Declaration first then use
  - Definition might be done in other files
  - Compiler will compile anyway regardless of definition if it is declared

### 17. Recursive Dependency
- Pointer usage vs complete usage in structure definition
  - Complete usage: the corresponding structure must be defined ahead
  - Pointer data type will consume only pointer byte (4 or 8) and can be defined prior to the actual definition
    - in 32bit OS, a pointer has 4bytes
    - In 64bit OS, a pointer consumes 8bytes
```c
struct emp_t{ 
  struct occ_t occ;
};
struct occ_t {
  struct emp_t boss;
};
```
- Compilation fails

### 18. Solution to Recursive Dependency
```c
struct occ_t; /* Forward declaration */
struct emp_t{ 
  struct occ_t* occ;
};
struct occ_t {
  struct emp_t boss;
};
```
- Compilation works
- To resolve recursive dependency, use:
  - Forward declaration
  - Pointer usage

### 19. Summary
- Preprocessor directives are simple text substitution tool
- Using uniqueness check, avoid duplicated header files
- Avoid recursive dependency when available

## Section 5: Quick creation of static and dynamic libraries

### 20. Resuming with doubly linked list library

### 21. Static and dynamic libraries - quick creation
- Making static library
  - `gcc -c dll.c -o dll.o`
  - `gcc -c dll_util.c -o dll_util.o`
  - `ar rs libdll.a dll.o dll_util.o`
- Making dynamic library
  - `gcc -c -fPIC dll.c -o dll.o`
  - `gcc -c -fPIC dll_util.c -o dll_util.o`
  - `gcc dll.o dll_util.o -shared -o libdll.so`

### 22. Linking with static library
- `gcc -c application.c`
- `gcc application.o -o a.exe -L/folder -ldll`

### 23. Linking with dynamic library
- Put libdll.so on `LD_LIBRARY_PATH`
  - Or setup through `sudo ldconfig`
- `gcc application.o -o a.exe -ldll`

### 24. Summary

## Section 6: Four stages of compilation process

### 25. 4 Stages of Compilation process
- Preprocessing: text substitution, preprocessing directives
- Compilation: Generation of assembly code
- Assembler: Generates machine code from compiled code (*.o)
- Linking: Linking with dependent libraries then final executable is produced

### 26. 1 of 4 - Preprocessing stage

### 27. 2 of 4 - Compilation stage
- gcc -S test.c -o test.o
  - `-S` will generate assembly code

### 28. 3 of 4 - Assembler stage
- `objdump -D test.o > log`

### 29. 4 of 4 - Linking stage

## Section 7: Build Project using Makefiles

### 30. Introducing Makefile

### 31. Makefile analogy - Dependency Tree

### 32. Makefile Assignment - part 1

### 33. Makefile assignment - part 2

### 34. Final Makefile

## Section 8: Programmable libraries - Generics

### 35. Introduction

### 36. Revisiting DLL

### 37. Problem statement

### 38. Solution  - Responsibility Delegation
- For generic programming, use void * for arguments
- We send function pointer using void *

### 39. Using Programmable DLL library - code walk

### 40. Search Callback Summary

### 41. Comparison Callback
- Adding data into DLL structure in the sorted order

### 42. Comparison Callback Demo

### 43. Summary 

## Section 9: Iterative Macros

## Seciton 10: Glue Based Data structures

## Section 11: Opaque pointers

### 61. Introduction
- Opaque class: a class with all its instance variables are private

### 62. Typical Library Design

### 63. Problem statement

### 64. Solution Strategy
- Need to provide a memory allocation function as the access to private member data is not available

### 65. Conclusion

## Section 12: Bit programming

## Section 13: Machine Endianness

### 74. Machine Endianness
- Big Endian: larger digit byte in the lower address
- Little Endian: larger digit byte in the higher address

### 75. Program to find Machine Endianness

## Section 14: TLV Based Communication

## Section 15: Working with Linux Posix Timers

### 84. Agenda and Prerequisites

### 85. Timer Relevance

### 86. Timer Types
- One shot timers
  - Triggered only once
  - Ex) terminate process after 10sec
- Periodic timers
  - Triggered periodically at regular intervals
  - Ex) send packets every 5 sec
- Exponential back off timers
  - Triggered at exponetntially temporal points
  - Ex) send retry event at 1, 2, 4, 8, ... sec

### 87. Posix APIs for Timers
- timer_create(): create a timer data structure buit not fires it
- timer_settime(): start/stop
- timer_gettime(): return the time remaining for the timer to fire
- timer_delete(): delete the timer data structure

### 88. Timer Design

### 89. Timer Creatin Steps
```c
int timer_create(<TypeOfTimer>, <TimerControlParam>, <TimerPointer>);
```
- TypeOfTimer: Real
- TimerControlParam: `struct sigevent evp;`
  - evp.sigev_notify_function = ptr to callback function
  - evp.sigev_value.sival_ptr = address of arguments to callback arg
  - evp.sigev_notify = SIGEV_THREAD
- In POSIX:
```c
struct itimerspec ts;
struct itimerspec{
  struct timespec it_interval;
  struct timepsec it_value;
};
struct timespec {
  time_t tv_sec; /* seconds */
  long tv_nsec; /* nanoseconds */
}
```

### 90. Timer Implementation and Demo
```c
#include <signal.h>
#include <time.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
static void print_current_system_time() {
  time_t t;
  time(&t); /* Get the current system time */
  printf("%s ", ctime(&t));
}
typedef struct pair_ {
  int a;
  int b;
} pair_t;
pair_t pair = {10, 20};

void timer_callback(union sigval arg) {
  print_current_system_time();
  pair_t *pair = (pair_t *) arg.sival_ptr;
  printf("pair: [%u %u] \n", pair->a, pair->b); 
}
void timer_demo() {
  int ret;
  struct sigevent evp;
  timer_t timer;
  memset(&timer, 0, sizeof(timer_t));
  memset(&evp, 0, sizeof(struct sigevent));
  evp.sigev_value.sival_ptr = (void *) &pair;
  evp.sigev_notify = SIGEV_THREAD;
  evp.sigev_notify_function = timer_callback;
  ret = timer_create(CLOCK_REALTIME, &evp, &timer);
  if (ret <0) {
    printf("Timer creation failed, errno=%d\n", ret);
    exit(0);
  }
  struct itimerspec ts;
  ts.it_value.tv_sec = 5;
  ts.it_value.tv_nsec = 0;
  ts.it_interval.tv_sec = 0;
  ts.it_interval.tv_nsec = 0;
  ret = timer_settime(timer, 0, &ts, NULL);
  if (ret <0) { 
    printf("Timer start failed, errno=%d\n",ret);
  } else {
    print_current_system_time();
    printf("Timer alarmed successfully\n");
  }
}
int main(int argc, char **argv) {
  timer_demo();
  pause();
  return 0;
}
```
- Running:
```bash
$ gcc -g chap90/timer_demo.c -lrt 
$ ./a.out 
Sun Jul  3 21:23:47 2022
 Timer alarmed successfully
Sun Jul  3 21:23:52 2022
 pair: [10 20] 
```
- By adjusting `ts.it_interval.tv_sec = 5;`, re-firing can be customized

## Section 16: Memory Layout of Linux process

### 91. Agenda
- How stack memoryh/heap memory are managed by system?
- Stack memory
  - Procedure call
  - Procedure return
- Heap memory
  - malloc
  - free

### 92. Virtual Memory basics
- Virtual address space  
- Virtual Memory
  - Total amount of memory of the system for a process
  - Different from physical memory and computer architect dependent
  - 32bit system has `2^32` bytes per process
    - 100 processes will have up to `100*2^32` bytes memory
  - Every byte has an address
    - There are 2^32 virtual addresses in 32bit system
  - SW works with virtual memory only, not withy physical memory
    - Related with paging
  - Each process in execution is allotted virtual memory for its usage, which can grow up to 2^32 bytes in 32 bit system
- Memory layout of a process

### 93. Memory layout of Linux process
- Code: compiled process assembly code, fixed
- Initialized data: initialized global and static variables, fixed
- Uninitialized data (bss): uninitialized global and static variables, fixed
- Heap: grows from low address to high address
- Stack: cannot grow beyond a certain limit as it may conflict with heap memory growth. Stores all local variables, arguments passed, return address, caller's info
- Command line arguments: argc & argv[i]. Located at the highest memory address

### 94. Example: memory layout of Linux process
- Heap: calloc/malloc
- Stack: stores local variables, passed arguments, return address. Supports procedure calls and returns
- Data Segment: Global and static variables
- Stack memory grows from High Address to Low Address, and Heap memory grows from LA to HA

### 95. Exercise on size command
```bash
$ cat chap95/main.c 
#include <stdio.h>
int global=5;
int main(void) {
  static int s = 10;
  /*int x;*/
  /*printf("%d %d %d ", global, s, x);*/
  return 0;
}
$ gcc  chap95/main.c -o a.out
$ size a.out
   text	   data	    bss	    dec	    hex	filename
   1415	    552	      8	   1975	    7b7	a.out
```
- Test local variables and check how bss and data size change

## Section 17: Stack Memory Management

### 96. Stack Memory Basics
- Last-In/First-Out
- Stack Frame: a data which is added to stack memory when a new function call is invoked
  - main() -> func1() -> func2() ... each function call invokes a new stack frame
  - When a function returns, the associated stack frame is popped out

### 97. Stack memory Contents
- Stack frame contains:
  - Parameter passed to the callee
  - Return address of the caller func - 4 bytes
  - Base pointer - 4 bytes
  - Local variables of a function
- Stack Pointer
- Frame Pointer/Base pointer

### 98. Stack Overflow and Prevention
- When stack grows beyond the maximum stack fixed size
  - Recursive function may cause stack overflow
  - Very large array

### 99. Stack Memory Corruption
- Stack data might be corrupted by copying the data more than the memory limits
  - Array size violation then returns

### 100. Procedure call and return - getting started
- How function call is implemented in Linux using stack memory
  - When caller makes a call to callee, callee should start execute from beginning
  - When callee finishes or returns, caller resumes from the point where it left
  - Return value by callee, if any, should be available to caller
- Call stack: a collection of stack frames
- Frame pointer/Base pointer: points to the top most frame in the stack
- Stack pointer: points the end of the top-most frame in the stack
- Program Counter(PC)/Instruction pointer: points to the current instruction to be executed
- Procedure call: caller calling the callee, control transfer to callee
- Procedure return: callee terminates and control return back to a caller

### 101. Common CPU registers
- eip: Instruction pointer register which stores the address of very next instruction to be executed
- esp: Stack pointer register, which stores the address of top stack (lowest address)
- ebp: Base pointer register, which stores the starting address in the callee's stack frame where caller's base pointer value is copied

### 102. Procedure call mechansim
- Assume f0() -> f1() -> f2() ...
  - f1(arg1, arg2) {... f2(arg3, arg4);}
  - f2(arg1, arg2) {... f3(arg3, arg4);}
- stack frame of f1()
```
100 : arg2_f1
96  : arg1_f1
92  : Ret_add_f0
88  : ebp_f0 -> ebp=84
84  : local_var2_f1
80  : local var2_f1 -> esp=76
```
- stack frame of f2()
```
76 : arg2_f2
72 : arg1_f2
68 : store %eip
64 : 84 from ebp of f1() -> ebp=60
60 : local_var2_f2
56 : local_var1_f2 -> esp=52
```

### 103. Pupose of Base Pointer register (ebp)
- Use of ebp register 
  - For a frame in execution, ebp register value is used as a reference to access all local variables and local arguments of the frame
  - Assuming 32bit OS:
    - ebp + 0: address where caller's base pointer is saved
    - ebp + 4: address where caller's next instruction address
    - ebp + 8: arg1
- CPU accesses all of the data of current stack frame in execution through ebp register value

### 104. Formalizing Procedure call algorithm
- When caller calls the callee, following steps take place on most of Linux system
  - Caller: push the argument list in reverse order
    - push y
    - push x ...
  - Caller: Push the address of next instruction in caller as Return Address in the callee's stack frame
    - push %eip
  - Callee: Push the previous frame's base poniter and copy esp to ebp
    - push %ebp
    - mv %ebp,%esp
  - Callee: Set PC - next instruciton in callee to be executed
    - mov %eip, ...
  - Callee: push the local variables of callee
    - push temp1
    - push temp2
  - Callee: execute the callee
- Every push, esp is decremented
- Every pop, esp is incremented

### 105. Procedure Return - Goals

### 106. Procedure Return Explained - Step by Step

### 107. Formalizing Procedure Return Algorithm
- Callee: set the return value of the callee in eax register
- Callee: Increases the stack pointer by the amount of the size of all local variables fo the frame 
- Callee: Restore %ebp to pointer to caller's stack frame and POP the previous frame's base pointer from the stack
- Callee: set %eip = return address saved in callee's stack, and POP the saved return address from the stack
- Caller: POPs all the arguments it had passed onto the stack
- Caller: reads the value stored in eax register, and resumes execution from %eip + 1 (next instruction)

## Section 18: Heap Memory Management

### 108. Heap memory management - Goals and Introduction
- Dynamic memory allocation
- glibc API for heap:
  - malloc, calloc, free, realloc
  - system calls: brk, sbrk
- It is programmer's responsibility to free the dynamically allocated memory after usage

### 109. malloc() - quick revision
- Paging: Memory Management Unit (MMU) translates Virtual Address to Physical Address
- Note that malloc() gaurantees contiguous memory

### 110. Break pointer
- A pointer maintaned by OS per process, pointing to the top of heap memory segment
- Any memory above break pointer is not a valid memory to be used by the process
- Break pointer moves towards higher address as heap memory increases

### 111. brk and sbrk system calls
- brk: expands the heap memory segment using the memory address
- sbrk: expands as many as requested by an argument

### 112. malloc version 1.0
- Manual implementation of malloc/free
  - Wrapper over sbrk() system call
```c
void *malloc(int size) {
  void *p;
  p = sbrk(0);
  if (sbrk(size) == NULL) return NULL;
  return p;
}
void free(int size) {
  assert(size > 0);
  sbrk(size*-1);
}
```
- This free() cannot free memory from the middle of heap segment

### 113. Problem Statement
- How OS can free a memory block in the middle of heap segments?
  - The size of memory block must be book-kept by OS
    - MetaBlock

### 114. Heap memory management requirement

### 115. MetaBlock and DataBlock
- DataBlock: memory block containing user data
- OS allocates a very small amount to label DataBlock: MetaBlock
  - 12 bytes
- Actual malloc expands the heap memory as requested +  size(meta_block_t)
  - To have 4 bytes alignment, some extra bytes might be padded

### 116. Allocations and Deallocations

### 117. Block Splitting
- When a new memory request is given, it may allocate in the middle of memory segments if enough size is available. The left-over will remain as available but metaBlock is updated as well
  - Overhead of metaBlock

### 118. Block merging
- Merging multiple segments of free memory segments

### 119. Problem of fragmentation
- Internal fragmentation
- External fragmentation

## Section 19: Concepts of Paging

### 120. Introduction
- Paging: 
  - Backbone of modern OS
  - Paging create the illusion of every processing in execution as if system has 2^32 bytes of physical memory for its execution
  - Allows the process to store its data in non-contiguous addresses in physical memory
  - Allows multiple processes to re-use the same physical memory addresses to store its data, one processor at a time
  - Implemented through MMU (Memory Management Unit)

### 121. Byte Addressable Memory

### 122. What is 32 bit or 64 bit system?
- Size of data bus will be 32 or 64bit
- Size of address bus will be 32 or 64bit

### 123. Bus system
- Metal wires on a board connecting devices

### 124. CPU generates virtual address
- Variables in the programs are just symbolic names of addresses
  - Machine code operates using addresses of variables instead of variable names

### 125. Data bus and Address Bus
- CPU-MMU-Physical memory
- Address bus: b/w MMU and physical memory
  - In 32bit OS, address bus is 32bit
- Data bus: b/w CPU and physical memory
  - To read a long integer data, it will take 2 cpu clock for 32bit OS while 1 cpu clock cycle for 64bit OS

### 126. Mapping of Virtual Address to Physical Address
- When memory is allocated through malloc(), MMU translates Virtual Address to Physical Address (Paging)
  - No data is stored at virtual memory
  - Actual data is stored at physical memory

### 127. Physical Pages and Frames
- Physical memory is fragmented into frames
- Each frame is 4KB (4096B)
- 4GB RAM has 2^20 frames
- Snapshot of the data stored in a frame is called a Physical Page
- Size of page == size of frame

### 128. Page Swapping
- Main memory save the page in a frame to the secondar storage (disk?) and reload another page from secondary storage into the frame
  - Algorithm: FIFO, LRU, ...

### 129. Virtual Memory Pages
- Similar to page of main memory, virtual memory is also fragmented into pages of 4096B
- For 4GB RAM, Virtual Address Space is divided into 2^20 pages
  - We need 20bits to address a page
- Each page is 4096B and 12 bits are required to uniquely address

### 130. 1:1 Mapping b/w Physical and Virtual Page
- Virtual page is just a collection of virtual address 

### 131. Virtual Address Composition
- Virtual page number + offset

### 132. Page Table
- Data structure maintained by OS for every process running on the system
- Used to map the virtual address of process's Virtual Address Space to a physical address of RAM
- Virtual Memory Pages would be contiguous while Physical Memory Pages will not be contiguous
- Is composed as Virtual Page number + Physical Page number + Frame number

### 133. Paging in Action
- CPU generates virtual address and decomposes into 2 parts
- Using the first part, locate the frame number where physical memory is loaded
- Using 2nd part, offset is determined
- Starting address of frame is determined

### 134. Multiple Process Scenario
- Physical memroy frames are shred among processes running on the system

### 135. Resolve External Fragmentation
- External fragmentation: when frame size is larger than physical page size
- If size of frame is same to the physical page size, no external fragmentation

### 136. Page Allocation to a Process - part 1
- When PAGE_SIZE is 4096B
- When you invoke malloc(12)
  - 12 + MetaBlockSize bytes are required
- OS allocaets PAGE_SIZE and (12+MBS) bytes are assigned
- The remaining virtual memory is cached by glibc malloc implementation
  - The remaining memory can be used for another malloc()

### 137. Page Allocation to a Process - part 2
- Using cached memory from malloc is efficient as it doesn't call sbrk()

### 138. Shared Physical Pages
- Multiple processes have page tables, and they have their onw virtual page numbers while their matching physical page numbers might be shared
- mmap()
- Race-condition?
- Shared memory is one of the Inter Process Communication technique

### 139. Page tables Problems
- Drawbacks and solutions
- Page size
- Contiguous main memory allocation
- Page table hollowness for small processes

### 140. Page Table Problem 1 - Large Page Table Size Matters
- Scenario 1
  - 32 bit OS with 4GB main memory, 4kB page size, and 4B of page table entry size
  - Size of Page Table = 2^22 B = 4MB per process
  - 100 processes will consume 4x100 = 400MB memory
- Scenario 2
  - 64 bit OS with 8GB, 4kB page size, and 4B of page table entry size
  - Size of Page Table = 2^34 MB per process
  - Not feasible
  - Solution: multi-lavel paging

### 141. Page Table Problem 2 - Need for Contiguous Main Memory
- https://www.google.com/search?channel=fs&client=ubuntu&q=is+contiguous+memory+is+contiguous+on+virtual+address+or+physical+address
- Both of virtual and physical address need to be contiguous in a page
- With the increase in size of page tables, chances to find more available consecutive frames grows more rare
  - Solution: multilevel paging
  - Split the large page table into smaller size, load it in non-contiguous frames in main memory

### 142. Page Table Problem 3 - Page Table Hollowness
- 32 bit OS will have 4MB page table per process
- But small program like hello world doesn't consume any of heap or stack memory
  - Most of table rows would be empty
  - Solution: multilevel paging

## Section 20: Multi-level paging

### 143. Introduction
- Why multi-level paging
  - Larger size of page tables
  - Need for contiguous main memory
  - Hollow region of page talbes
- Goal of page table: given a virtual address, locate the physical frame, and then locate the exact physical address in main memory
- Analogy of multi-level indexing of TOC
  - Section
    - Unit
      - Chapter
        - ...
- Main idea is to split the large page tables into smaller sizes and fit each individual smaller page tables at dispersed location in main memory

### 144. Multi Level Paging in Action - part 1
- Scenario
  - Size of virtual address generated by CPU : 8bits
  - Page size : 4B
  - Main Memroy : 64B
  - Each page table entry size: 1B
  - Virtual address space size = 2^8 = 256B
  - Frame size = 4B
  - Virtual Address Composition = 6+2 bits
  - Page table size = PAGE size = 4B
  - No. of entries in Page table = 4
  - No. of bits required to index into a single page table = 2
  - Physical address size 6bits = 2(1st leve) + 2(2nd level) +2(3rd level)

### 145. Multi Level Paging in Action - part 2

| Single level paging scheme | Multi level paging scheme|
|-----|----|
| Page table size = PT Entry size * No. of virtual Pages of a process | Page table size = Frame size (4B) |
|No. of page tables per process = 1 | No. of page tables per process = 2^6+1 = 65|
| Memory references to map VA->PA = 1 | Memory references to map VA-> PA = 3 (Slow) |
| Most of page table rows are empty | Can create/delete page tables on demand| 

## Section 21: Paging on Demand

### 146. Problem statement
- Demand paging: keep only required physical pages of a process in main memory, rest swap them out to disk
- Benefits:
  - Increases multi-tasking
  - Less main memory is consumed per process
  - More users
- Additional infor in page table
  - Valid bit: shows if the physical page is present in a frame or has been swapped out of physical memory to disk

### 147. Demaind Paging steps
- Page fault: when page table dictates that a physical page is not present in a frame then a special signal is raised to CPU called trap, alo called **page fault**
  - CPU will check disk to find the memory and load back into main memory

### 148. Effective Access Time
- Page fault increases the memory access time by the CPU
- When P is the probability of page fault,
  - EAT(Effective access time) for memory access = (1-P) * memory_access_time + P*(Page_fault_overhead + swap_page_out + swap_page_in + restart_overhead)

## Section 22: Memory Management for Multi-threaded Processes

### 149. Introduction
- Threads share almost everything among other threads and parent process
  - Code segment
  - Data segment (initialized and uninitialized)
  - Open file descriptors
  - Heap memory
  - **BUT NOT STACK MEMORY**

### 150. Virtual Memory Management
- Newly created thread will have its own stack memory but the virtual memory of heap/uninitialized/initialized/code segments are shared

### 151. Page Table Management
- Thread inherits the page table of parent except stack memory

### 152. Thread Termination
- Virtual page of stack memory is freed
- Only physical page corresponding to stack memory is freed
