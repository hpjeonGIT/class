## Exploring the nm Tool: Understanding Symbols in Binaries
- By Can Özkan
- Ref: https://can-ozkan.medium.com/exploring-the-nm-tool-understanding-symbols-in-binaries-1ae03193212a

### 1. Introduction
- nm: a part of GNU Binutils package
  - Inspects the symbol table of object files, sstatic, and shared libraries, and exe
  - Reveals the names of functions and variables with their corresponding addresses and types
  - A window b/w source code and machine code
  - Reads **symbol table** embedded within a compiled file

### 2. Understanding Symbols in Binaries
- Compiler stores symbolic information - function names, variable names - inside the symbol table, along with their memory address and type (function, variable, external ref.)
- "Undefined reference": when the corresponding symbol is not found in the symbol table
- Classficiation of symbols
  1. Defined vs undefined
      - Defined symbols represent entities that are actually implemented in the file
      - Undefined symbols are references to entities defined elsewhere. The linker must latger resolve it by connecting it to the actual implementation in the library
  2. Global vs local
      - Global symbols are visible across different object files and can be accessed or refrenced by other modules, including public functions and global variables
      - Local symbols are internal to a single object file and cannot be referenced externally
- When code compiles, each source file is converted into a machine code, generating entries for every symbol the compiler encounters, both defined and undefined. During linking, the linker merges those symbol ables, resolves undefined symbols by matching them with defined ones, and produces a final symbol table for the executable. The final table might be static (compile-time) or dynamic (runtime)

### 3. Basic usage of nm
- hello.c:
```c
#include <stdio.h>
void hello() {
    printf("Hello, world!\n");
}
int main() {
    int a = 35;
    hello();
    return 0;
}
```
- gcc -c hello.c -O0 -o hello.o
- nm hello.o
```bash
0000000000000000 T hello
000000000000001a T main
                 U puts
```
- nm yields results using three columns:
  - Address (Value): the memory address or offset
  - Type: symbol type
    - `T`: defined functions located in the text section
    - `U`: undefined symbol. Linker will later resolve
  - Name: the name of symbols

### 4. Decoding Symbol Types
- Symbol type
  - `T`: defined functions located in the text section. Globally visible
  - `t`: Same as `T` but local to the object file (static function in C)
  - `U`: undefined symbol. Linker will later resolve (printf from libc)
  - `D`: A global or static variable with an initial value
  - `d`: Same as `D` but local
  - `B`: A global or static variable without an initial value, located in BSS
  - `b`: Same as `B` but local
  - `R`: Read-only data section (const variable)
  - `r`: Same as `R` but local
  - `W`: Weak symbol. Can be overridden by another definition with the same name
  - `w`: local weak symbol
  - `V`: Weak object symbol, in dynamic linking
  - `v`: local weak object symbol
  - `A`: The symbol's value in an absolute address
  - `?`: Unknown symbol type
- symbol.c
```c
// compilation: gcc -c symbols.c -o symbols.o
#include <stdio.h>
int global_var = 42;        // initialized global variable
static int counter;         // static variable (local to this file)
void display() {            // globally visible function
    printf("Value: %d\n", global_var);
}
static void helper() {      // local (static) function
    counter++;
}
- Demo:
```bash
$ gcc -c symbol.c -O0 -o symbol.o
$ nm symbol.o
0000000000000000 b counter
0000000000000000 T display
0000000000000000 D global_var
0000000000000027 t helper
                 U printf
```

### 5. Other use cases
- `nm -D exe`
  - Shows the lists of symbols needed at runtime for dynamic linking
- `nm -C main.o`
  - Demangles symbol name

### 6. Stripped vs unstripeed binaries
- `strip` will remove symbol tables from exe or object files
  - Will not accelerate the performance
  - Might be used for security purposes
