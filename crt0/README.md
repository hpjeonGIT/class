## Understanding the C Runtime: crt0, crt1, crti, and crtn
- https://www.inferara.com/en/blog/c-runtime/

### What is the C Runtime?
- CRT (C runtime) is a collection of startup routines, intialization code, standard library support, and system call wrappers
  - Most of them live outside your application's own source
- The compiler driver and linker implicitly include startup object files and libraries, including one or more CRT object files. These files contain assembly-level entry points and routines that:
    1. Initialize registers and the stack.
    2. Set up the program arguments (argc, argv, envp).
    3. Invoke global constructors (in C++ programs).
    4. Call your main() function.
    5. Handle the return from main() and pass the exit status to the operating system.

### The role of crt0.o (or crt1.o in Modern toolchains)
- Historically, crt0.o is a small object file containing the actual entry point routine, often named `_start`. Its responsibilities include:
    1. Program initialization
    2. Transferring control to main()
    3. Cleaning up
- But crt0.o was often a large, monolithic file, now many modern toolchains split it up into more modular components
  - Typical content of crt1.o/crt1.o
    - Low-level assembly code responsible for setting up the runtime
    - `_start` that acts as the entry point
    - A call to main()

### Additional runtime files
- Moden toolchains provide:
  - crti.o: intialization
  - crtn.o: termination
  - crt1.o: entry point
- Steps of runtime
    1. _start
    2. Initialization code from crti.o, then jump into main()
    3. When main() returns, the epilogue from crtn.o is executed
    4. A final exist syscall terminates the process with the return value from main()
