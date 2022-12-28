## Developing a C Compiler From Scratch - Module 1
- Instructor: Daniel McCarthy

## Section 1: Course Overview

1. Introduction

2. Overview of the course
- Module 1
  - We create a lexer takes a C source file then converts it into tokens 
  - We create a parser that creates many nodes forming an abstgract syntax tree
- Module 2
  - We create a resolver which is responsible for taking an expression and create a series of rules
  - We create a code generator that take in our abstract syntax rule and create NASM x86 32bit assembly language
- Module 3
  - We create a semantic validator that ensures valid works in the source file
  - We create preprocessor/macro systems so that #include, #ifdef, #define are supported

## Section 2: Installalation and setup

3. Installation and Setup
- Ubuntu with gcc, gdb, make

4. Preparing our project
- Helper functions from: https://github.com/nibblebits/DragonCompiler/tree/master/helpers
  - Get buffer.c/h and vector.c/h
- main.c
```c
 #include <stdio.h>
#include "compiler.h"
int main() 
{
  int res = compile_file("./test.c", "./test",0);
  if (res == COMPILER_FILE_COMPILED_OK)
  {
    printf("everything compiled file\n");
  } else if (res == COMPILER_FAILED_WITH_ERRORS)
  {
    printf("compile failed\n");
  } else 
  {
    printf("Unknown response for compile file\n");
  }
  return 0;
}
```  
- cprocess.c
```c
#include <stdio.h>
#include <stdlib.h>
#include "compiler.h"
struct compile_process* compile_process_create(const char* filename, const char*
 filename_out, int flags)
{
  FILE * file = fopen(filename, "r");
  if (!file)
  {
    return NULL;
  }
  FILE* out_file = NULL;
  if (filename_out) 
  {
    out_file = fopen(filename_out, "w");
    if (!out_file)
    {
      return NULL;
    }
  }
  struct compile_process* process = calloc(1, sizeof(struct compile_process));
  process-> flags = flags;
  process-> cfile.fp = file;
  process-> ofile = out_file;
  return process;
}
```
- compiler.c
```c
#include "compiler.h"
int compile_file(const char* filename, const char* out_filename, int flags)
{
  struct compile_process* process = compile_process_create(filename, out_filenam
e, flags);
  if (!process)
    return COMPILER_FAILED_WITH_ERRORS;
  // Perform lexical analysis
  // Perform parsing
  // Perform code generation
  return COMPILER_FILE_COMPILED_OK;
}
```
- compiler.h
```c
#ifndef PEACHCOMPILER_H
#define PEACHCOMPILER_H
#include <stdio.h>
enum {
  COMPILER_FILE_COMPILED_OK,
  COMPILER_FAILED_WITH_ERRORS
};
struct compile_process
{
  // The flags in regards this file should be compiled
  int flags;
  struct compile_process_input_file 
  {
    FILE* fp;
    const char* abs_path;
  } cfile;
  FILE * ofile;
};
int compile_file(const char* filename, const char* out_filename, int flags);
struct compile_process* compile_process_create(const char* filename, const char*
 filename_out, int flags);
#endif 
```
- Makefile
```
OBJECTS= ./build/compiler.o ./build/cprocess.o
INCLUDES= -I./
all: ${OBJECTS}
	gcc main.c ${INCLUDES} -g -o ./main ${OBJECTS}
./build/compiler.o: ./compiler.c
	gcc ./compiler.c ${INCLUDES} -o ./build/compiler.o -g -c
./build/cprocess.o: ./cprocess.c
	gcc ./cprocess.c ${INCLUDES} -o ./build/cprocess.o -g -c
clean:
	rm ./main
	rm -rf ${OBJECTS}
```
## Section 3: Lexical Analysis

5. What is Lexical analysis
- Turning strings into tokens
- A token has a type and a value
- Lexer does lexical analysis
- Token types
  - Identifier: variable name, function name, structure name
  - Keyword: unsigned, int, short, char, break, continue, ...
  - Operator: +, *, & { ) 
  - Symbol: "", ;, :
  - Number
  - String
  - Comment
  - Newline

6. Creating out token structures
```c
struct token
{
  int type;
  int flags;
  union 
  {
    char cval;
    const char* sval;
    unsigned int lnum;
    unsigned long long llnum;
    void * any;
  };
  bool whitespace;
  const char* between_brackets;
};
```

7. Preparing our lexer
```c
struct lex_process
{
  struct pos pos;
  struct vector* token_vec;
  struct compile_process* compiler;
};
```

8. Creating a number token

9. Creating a string token

## Section 4: Parsing the C programming language

23. What is parsing?
- Why parsers?
  - Provides structure for an input file
  - Nodes can branch off from each other providing stability in logic
  - Makes it eaasier to validate the code
  - Makes it easier to compile the input file

24. Creating our parser structures
- NODE_TYPE_EXPRESSION
- NODE_TYPE_EXPRESSION_PARENTHESES
- NODE_TYPE_NUMBER
- NODE_TYPE_IDENITFIER
- NODE_TYPE_STRING
- NODE_TYPE_VARIABLE
- NODE_TYPE_VARIABLE_LIST
- NODE_TYPE_FUNCTION
- NODE_TYPE_BODY
- NODE_TYPE_STATEMENT_RETURN
- NODE_TYPE_STATEMENT_IF
- NODE_TYPE_STATEMENT_ELSE
- NODE_TYPE_STATEMENT_WHILE

## Section 5: Module 1 Summary

102. Module 1 Summary
- https://dragonzap.com/course/creating-a-c-compiler-from-scratch

103. Module 2

# Below are from dragonzap.com

## Module 2: Code generator

104. The code generator
- Takes abstract syntax tree and converts it to assembly language or machine code
- Process
  - Loops through every root node in the abstract syntax tree
  - Generates code for a node. Loops through child nodes where applicable
  - Keeps the track of th eocd generation state such as if we are in a structure expression or not

105. Building the fundamentals


108. Understanding the label systems
- Assembly Labels
  - Allows you to declare a reference to a particular part of an assembly file. You can JUMP to this label and execute the code or read data from this label
- When we generate labels
  - for loop
  - if statement
  - while loop
  - do while loop
  - all other statements
  - all global variables
  - all strings
- Two stacks to be used
  - Entry point stack
  - Exit point stack
- Conclustion
  - State tracking is very important in compiler development
  - By tracking labels continue/break labels will know where to jump and we can easily exit a subroutine because the tracking functionality knows which label to jump to
- Q: is JMP expensive?
  - Ref: https://stackoverflow.com/questions/5127833/meaningful-cost-of-the-jump-instruction

112. Implementing numerical values for global variables

115. Stackframes
- A technical term tha describes a part of a stack whilst a function is running
- Holds subroutine return addresses
- Holds function arguments
- Holds local variables
- Sample C function:
```c
int sum_and_one(int x, int y) {
  int one = 1;
  return  (x*y) + one;
}
```
- Corresponding assembly code:
```asm
0000000000000000 <sum_and_one>:
   0:	55                   	push   rbp ; save old base pointer
   1:	48 89 e5             	mov    rbp,rsp  ; move stack pointer to the base pointer
   4:	89 7d ec             	mov    DWORD PTR [rbp-0x14],edi ; x
   7:	89 75 e8             	mov    DWORD PTR [rbp-0x18],esi ; y
   a:	c7 45 fc 01 00 00 00 	mov    DWORD PTR [rbp-0x4],0x1  ; one
  11:	8b 45 ec             	mov    eax,DWORD PTR [rbp-0x14]
  14:	0f af 45 e8          	imul   eax,DWORD PTR [rbp-0x18] ; x*y
  18:	89 c2                	mov    edx,eax
  1a:	8b 45 fc             	mov    eax,DWORD PTR [rbp-0x4]
  1d:	01 d0                	add    eax,edx                  ; x*y + one
  1f:	5d                   	pop    rbp
  20:	c3                   	ret   
```
- Using a stack frame mechanism in the compiler will ensure that when we are about to generate a "POP" assembly instruction we know exactly what element we are popping

117. The resolver explained
- 
