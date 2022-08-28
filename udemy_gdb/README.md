## Hands on Debugging in C and C++
- Instructor: Mike Shah

## Section 1: Introduction

1. Introduction to Debugging

2. A Working Example in GDB
```bash
$ gcc -g -Wall ../Resources/array_of_list.c -o array_of_list
$ gdb ./array_of_list 
(gdb) start
(gdb) print map
$2 = (hashmap_t *) 0x0
(gdb) whatis map
type = hashmap_t *
(gdb) br hashmap_insert
Breakpoint 2 at 0x5555555547e5: file ../Resources/array_of_list.c, line 66.
(gdb) info breakpoints
Num     Type           Disp Enb Address            What
2       breakpoint     keep y   0x00005555555547e5 in hashmap_insert at ../Resources/array_of_list.c:66
...
   ┌──../Resources/array_of_list.c────────────────────────────────────────────────────────────────────────────────────┐
   │61      void hashmap_insert(hashmap_t* in, char* key_in,int value){                                               │
   │62              // Perhaps logic on handling if the key already exists                                            │
   │63              // and just return...                                                                             │
   │64              // and more error handling if the hashmap is NULL.                                                │
   │65                                                                                                                │
B+ │66              pair_t* newpair = (pair_t*)malloc(sizeof(pair_t));                                                │
   │67              newpair->key = (char*)malloc(strlen(key_in)*sizeof(char)+1);                                      │
   │68              newpair->value = value;                                                                           │
   │69              // Copy the string                                                                                │
  >│70              strcpy(newpair->key,key_in);                                                                      │
   │71              // Create our new node                                                                            │
   │72              node_t* newnode = (node_t*)malloc(sizeof(node_t));                                                │
   │73              newnode->next = NULL;                                                                             │
   └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
native process 9355 In: hashmap_insert                                                        L70   PC: 0x55555555481f 
(gdb) s

Breakpoint 2, hashmap_insert (in=0x555555757260, key_in=0x555555554b63 "Mike", value=10)
    at ../Resources/array_of_list.c:66
(gdb) n
(gdb) n
(gdb) n
(gdb) 
(gdb) p value
$2 = 10
(gdb) info args
in = 0x555555757260
key_in = 0x555555554b63 "Mike"
value = 10
(gdb) define log
Type commands for definition of "log".
End with a line saying just "end".
>print "hello"
>info args 
>bt
>end
(gdb) list hashmap_insert
(gdb) n
(gdb) n
(gdb) finish
Run till exit from #0  hashmap_insert (in=0x555555757260, key_in=0x555555554b63 "Mike", value=10)
    at ../Resources/array_of_list.c:73
main () at ../Resources/array_of_list.c:127
(gdb) show language
The current source language is "auto; currently c".
```
 - 0x0 means null

3. Course Objectives
  - Debugging skills
  - Practice using tools
    - compiler
    - strace
    - ltrace
    - valgrind

4. A story about the first bug

5. Write your code neatly

6. GDB(linux), LLDB(Mac) or Visual Studio (windows)

7. GDB for D, Objective-C, OpenCL, Rust, etc?

8. Resources

## Section 2: Respecting the Compiler

9. Compile time vs Run-time debugging
  - Compile time bug
    - Before our program runs
    - (static) assert
    - Compiler
  - Runtime bug
    - While our program runs
    - Memory leaks
    - Segmentation fault
    - Performance related
    - Exception

10. Compiler Errors

11. assert statement and static_assert
  - static_assert: works at compile time
```cpp
#include <iostream>
#include <cassert>
int main() {
  assert (1==2 && "yes, math is working"); // compiles OK but aborts at execution
  static_assert(sizeof(int) == 5 && "yes int is 4bytes"); // compilation fails
  return 0;
}
```

12. Treat Compiler Warning as Errors
  - Use `-Wall -Werror` at gcc

13. Trick: Leverage multiple compilers
  - Try gcc/clang to see different warning/error messages

## Section 3: Your First Debugging Techniques

14. [Practice] printf or std::cout debugging messages
  - Helper or log function to print/dum the state of object would be good to have

15. [Theory] Delta debugging Technique
  - How to shrink search space
    - By logging or looking in nearby places

16. [Concept] Understanding common errors - the segmentation fault
  - Dereferencing null value
  - Occurs when you access memory that your process does not own

17. [Concept] Understanding common errors - the memory leak
  - Failure to reclaim memory while our program runs

## Section 4: Using a Debugger

18. GNU Debugger
```bash
$ gcc -g -Wall array_of_list.c -o array_of_list
$ gdb ./array_of_list
(gdb) start
Temporary breakpoint 1 at 0x988: file Resources/array_of_list.c, line 124.
Starting program: /home/hpjeon/hw/class/udemy_HandsOnDebugging/array_of_list 

Temporary breakpoint 1, main () at Resources/array_of_list.c:124
124	    map = hashmap_create(8);
(gdb) list
119	
120	int main(){
121	
122	    // Create our hashmap
123	    hashmap_t* map;
124	    map = hashmap_create(8);
125		// Insert some values
126	    hashmap_insert(map,"Mike",10);
127	    hashmap_insert(map,"Jacob",11);
128	    hashmap_insert(map,"Matt",12);
(gdb) next
126	    hashmap_insert(map,"Mike",10);
(gdb) n
127	    hashmap_insert(map,"Jacob",11);
(gdb) n
128	    hashmap_insert(map,"Matt",12);
(gdb) 
129	    hashmap_insert(map,"Nathan",13);
(gdb) step
(gdb) n
(gdb) finish
Run till exit from #0  hashmap_insert (in=0x555555757260, key_in=0x555555554b73 "Nathan", value=13)
    at Resources/array_of_list.c:68
main () at Resources/array_of_list.c:130
130	    hashmap_insert(map,"Carter",14);
(gdb) continue
Continuing.
Bucket# 0
	Key=Campbell	Values=16
Bucket# 1
	Key=Dr. House	Values=23
Bucket# 2
Bucket# 3
	Key=Jeff Probst	Values=40
Bucket# 4
	Key=Lebron James	Values=23
	Key=Matt	Values=12
	Key=Mike	Values=10
Bucket# 5
	Key=Jacob	Values=11
Bucket# 6
	Key=Michael Jordan	Values=23
	Key=Andrew	Values=15
	Key=Carter	Values=14
	Key=Nathan	Values=13
Bucket# 7
[Inferior 1 (process 23170) exited normally]
(gdb) 
```
  - ctrl+l to clear gdb screen

19. Whey we compile with Debug Symbols
  - What `-g` does

20. Print Value and Listing Source Code
```bash
(gdb) set listsize 5
(gdb) list
129	    hashmap_insert(map,"Nathan",13);
130	    hashmap_insert(map,"Carter",14);
131	    hashmap_insert(map,"Andrew",15);
132	    hashmap_insert(map,"Campbell",16);
133	    hashmap_insert(map,"Michael Jordan",23);
```

21. Figuring out a variable type with `whatis`
```bash
(gdb) start
Temporary breakpoint 1 at 0x988: file Resources/array_of_list.c, line 124.
Starting program: /home/hpjeon/hw/class/udemy_HandsOnDebugging/array_of_list 

Temporary breakpoint 1, main () at Resources/array_of_list.c:124
124	    map = hashmap_create(8);
(gdb) br hashmap_insert
(gdb) c
(gdb) list
61	void hashmap_insert(hashmap_t* in, char* key_in,int value){
62		// Perhaps logic on handling if the key already exists
63		// and just return...
64		// and more error handling if the hashmap is NULL.
65	
66		pair_t* newpair = (pair_t*)malloc(sizeof(pair_t));
67		newpair->key = (char*)malloc(strlen(key_in)*sizeof(char)+1);
68		newpair->value = value;
69		// Copy the string 
70		strcpy(newpair->key,key_in);	
(gdb) n
67		newpair->key = (char*)malloc(strlen(key_in)*sizeof(char)+1);
(gdb) whatis newpair
type = pair_t *
(gdb) whatis hashmap_insert
type = void (hashmap_t *, char *, int)
(gdb) whatis in
type = hashmap_t *
```

22. GDB the Text User Interface (TUI)
```
$ gdb --tui ./array_of_list
(gdb) start
Temporary breakpoint 1 at 0x988: file Resources/array_of_list.c, line 124.
Starting program: /home/hpjeon/hw/class/udemy_HandsOnDebugging/array_of_list

Temporary breakpoint 1, main () at Resources/array_of_list.c:124
(gdb) 
(gdb) winheight src -4

```
  - ctrl+x+a to enable/disable tui window
  - `winheight src -4` to change the size of tui window

23. Breakpoints
```
(gdb) br printf
Breakpoint 2 at 0x5e0
(gdb) info breakpoints
Num     Type           Disp Enb Address            What
1       breakpoint     keep y   0x0000000000000988 in main at Resources/array_of_list.c:124
2       breakpoint     keep y   0x00000000000005e0 <printf@plt>
(gdb) del 2
(gdb) info breakpoints
Num     Type           Disp Enb Address            What
1       breakpoint     keep y   0x0000000000000988 in main at Resources/array_of_list.c:124
```
  - ~~`br printf` will set breakpoints for every printf()~~
  - use `del` to delete any breakpoints

24. Conditional Breakpoints
```cpp
#include <iostream>
int main() {
  int c = 0;
  for (int i=0; i< 1000; i++) {
    if (i%2 ==0 ) c += i;
  }
  std::cout << "counter = " << c << std::endl;
  return 0;
}
```
  - **How to break when i > 899?**
```bash
(gdb) list
1	#include <iostream>
2	int main() {
3	  int c = 0;
4	  for (int i=0; i< 1000; i++) {
5	    if (i%2 ==0 ) c += i;
6	  }
7	  std::cout << "counter = " << c << std::endl;
8	  return 0;
9	}(gdb) br 5 if i==899
Breakpoint 1 at 0x929: file section24/cond_brk.cpp, line 5.
(gdb) info breakpoints
Num     Type           Disp Enb Address            What
1       breakpoint     keep y   0x0000000000000929 in main() at section24/cond_brk.cpp:5
	stop only if i==899
(gdb) start
(gdb) c
Continuing.
Breakpoint 1, main () at section24/cond_brk.cpp:5
5	    if (i%2 ==0 ) c += i;
(gdb) p i
$1 = 899
```

25. Watching Variables
  - `watch variable`
    - When the variable is changed, it prints old/new value
```bash  
(gdb) watch c
Hardware watchpoint 2: c
...
(gdb) n
4	  for (int i=0; i< 1000; i++) {
(gdb) n
5	    if (i%2 ==0 ) c += i;
(gdb) n
Hardware watchpoint 2: c
Old value = 2
New value = 6
main () at section24/cond_brk.cpp:4
4	  for (int i=0; i< 1000; i++) {
```

26. What is a call stack?
  - Keeps the track of execution of functions in your program

27. Navigating the call stack with backtrace
  - `bt` works anytime (not necessarily crashed)
```bash
(gdb) s
add (a=0, b=1) at Resources/add.c:6
6	    return a+b;
(gdb) bt
#0  add (a=0, b=1) at Resources/add.c:6
#1  0x000055555555468b in addAndSquare (a=0, b=1) at Resources/add.c:14
#2  0x00005555555546bd in main () at Resources/add.c:21
```
  - Use `up` to go up of the current backtrace

28. Getting help in GDB

## Section 5: More Interactive Debugging

29. GDB Debug Cycle - rerunning and reviewing software
  - `file ./a.exe` will load the executable
  - Previously configured breakpoints will be still available
```bash
(gdb) info breakpoints
Num     Type           Disp Enb Address            What
2       breakpoint     keep y   0x000055555555467c in addAndSquare at Resources/add.c:14
	breakpoint already hit 2 times
(gdb) file ./a.out
A program is being debugged already.
Are you sure you want to change the file? (y or n) y
Load new symbol table from "./a.out"? (y or n) y
Reading symbols from ./a.out...done.
(gdb) info breakpoints
Num     Type           Disp Enb Address            What
2       breakpoint     keep y   0x000000000000066e in addAndSquare at Resources/add.c:13
	breakpoint already hit 2 times
```

30. Calling functions
- In the interactive session, we can use the functions of the source directly
```bash
   │9       int square(int x){                                 │
   │10          return x*x;                                    │
   │11      }                                                  │
   │12                                                         │
   │13      int addAndSquare(int a, int b){                    │
b+ │14          return square(add(a,b));                       │
   │15      }                                                  │
   │16                                                         │
   │17      int main(){                                        │
   │18                                                         │
  >│19          for(int i=0; i < 7; i++){                      │
   └───────────────────────────────────────────────────────────┘
native process 20433 In: main          L19   PC: 0x55555555469c 
Start it from the beginning? (y or n) y
Temporary breakpoint 3 at 0x69c: file Resources/add.c, line 19.
Starting program: /home/hpjeon/hw/class/udemy_HandsOnDebugging/a
Temporary breakpoint 3, main () at Resources/add.c:19
(gdb) call square(3)
$3 = 9
```

31. Attaching the debugger to a running process
```bash
$ ps -ef |grep a.out
hpjeon   21153 18874  0 19:30 pts/1    00:00:00 ./a.out
hpjeon   21382 26831  0 19:31 pts/0    00:00:00 grep --color=auto a.out
$ sudo gdb -p 21153
...
done.
0x00007ff0fad25654 in __GI___nanosleep (
    requested_time=requested_time@entry=0x7ffd6f2a8910, 
    remaining=remaining@entry=0x7ffd6f2a8910)
    at ../sysdeps/unix/sysv/linux/nanosleep.c:28
28	../sysdeps/unix/sysv/linux/nanosleep.c: No such file or directory.
(gdb) list
23	in ../sysdeps/unix/sysv/linux/nanosleep.c
(gdb) bt
#0  0x00007ff0fad25654 in __GI___nanosleep (
    requested_time=requested_time@entry=0x7ffd6f2a8910, 
    remaining=remaining@entry=0x7ffd6f2a8910)
    at ../sysdeps/unix/sysv/linux/nanosleep.c:28
#1  0x00007ff0fad2555a in __sleep (seconds=0)
    at ../sysdeps/posix/sleep.c:55
#2  0x00005606485f96af in main () at Resources/infinite.c:9
(gdb) up
#1  0x00007ff0fad2555a in __sleep (seconds=0)
    at ../sysdeps/posix/sleep.c:55
55	../sysdeps/posix/sleep.c: No such file or directory.
(gdb) up
#2  0x00005606485f96af in main () at Resources/infinite.c:9
9	        sleep(2);
(gdb) list
4	int main(){
5	
6	    int counter =0;
7	    while(1){
8	        printf("hi\n");
9	        sleep(2);
10	        printf("again\n");
11	        counter++;
12	    }
13	
(gdb) br 8
Breakpoint 1 at 0x5606485f9699: file Resources/infinite.c, line 8.
(gdb) p counter
$1 = 61
(gdb) p $pc  # program counter
$2 = (void (*)()) 0x5606485f96af <main+37>
(gdb) p $rsp  # stack pointer
$3 = (void *) 0x7ffd6f2a8950
```

32. Core dumped - and how to look at those files
- Core files: snapshot of a program's memory when it crashed
- Using gcore: dump the running process into core files
  - sudo gcore <pid>
```bash
$ ps -ef |grep a.out
hpjeon   23369 18874  0 19:40 pts/1    00:00:00 ./a.out
hpjeon   23372 26831  0 19:40 pts/0    00:00:00 grep --color=auto a.out
$ sudo gcore 23369
0x00007f496242d654 in __GI___nanosleep (requested_time=requested_time@entry=0x7fff30f5aa50, remaining=remaining@entry=0x7fff30f5aa50) at ../sysdeps/unix/sysv/linux/nanosleep.c:28
28	../sysdeps/unix/sysv/linux/nanosleep.c: No such file or directory.
Saved corefile core.23369
$ ls core.23369 
core.23369
# 23369 process is still running. 
$ gdb -c core.23369 ./a.out
GNU gdb (Ubuntu 8.1.1-0ubuntu1) 8.1.1
Copyright (C) 2018 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
and "show warranty" for details.
This GDB was configured as "x86_64-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<http://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
<http://www.gnu.org/software/gdb/documentation/>.
For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from ./a.out...done.
[New LWP 23369]
Core was generated by `./a.out'.
#0  0x00007f496242d654 in __GI___nanosleep (
    requested_time=requested_time@entry=0x7fff30f5aa50, 
    remaining=remaining@entry=0x7fff30f5aa50)
    at ../sysdeps/unix/sysv/linux/nanosleep.c:28
28	../sysdeps/unix/sysv/linux/nanosleep.c: No such file or directory.
(gdb) list
23	in ../sysdeps/unix/sysv/linux/nanosleep.c
(gdb) whatis main
type = int ()
(gdb) up
#1  0x00007f496242d55a in __sleep (seconds=0)
    at ../sysdeps/posix/sleep.c:55
55	../sysdeps/posix/sleep.c: No such file or directory.
(gdb) list
50	in ../sysdeps/posix/sleep.c
(gdb) up
#2  0x000055624b0126af in main () at Resources/infinite.c:9
9	        sleep(2);
(gdb) list
4	int main(){
5	
6	    int counter =0;
7	    while(1){
8	        printf("hi\n");
9	        sleep(2);
10	        printf("again\n");
11	        counter++;
12	    }
13	
(gdb) p counter
$1 = 6
```

33. Using Python within GDB
  - We may run python environment 
  - Not line by line but lie a batch script per each python session
  - Use ctrl+d to exit python session
```bash
(gdb) python
>import gdb
>gdb.execute('start')
>gdb.execute('n')
>gdb.execute('list')
# ctrl+d to exit python. Then the python commands get started to run sequentially
>Temporary breakpoint 1 at 0x692: file Resources/infinite.c, line 6.
Temporary breakpoint 1, main () at Resources/infinite.c:6
6	    int counter =0;
8	        printf("hi\n");
3	
4	int main(){
5	
6	    int counter =0;
7	    while(1){
8	        printf("hi\n");
9	        sleep(2);
10	        printf("again\n");
11	        counter++;
12	    }
(gdb) 
```

34. Redirecting output from GDB to other terminal 
  - tty: shows the current terminal
```bash
$ tty
/dev/pts/1
$ gdb ./a.out 
GNU gdb (Ubuntu 8.1.1-0ubuntu1) 8.1.1
Copyright (C) 2018 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
and "show warranty" for details.
This GDB was configured as "x86_64-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<http://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
<http://www.gnu.org/software/gdb/documentation/>.
For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from ./a.out...done.
(gdb) tty /dev/pts/0
(gdb) r # now all script output is printed in the terminal of /dev/pts/0
```

## Section 6: GDB Commands and Scripts

35. Define your own commands
  - Start with `define <command_name>`
  - `end` to exit the definition
```bash
(gdb) define log
Type commands for definition of "log".
End with a line saying just "end".
>print "hello"
>print counter
>end
(gdb) log
$1 = "hello"
$2 = 0
```

36. Breakpoints and commands - save time!
  - `br` then `commands`
```bash
(gdb) br 4
Breakpoint 2 at 0x55555555464e: file section23/brk_study.c, line 4.
(gdb) commands 
Type commands for breakpoint(s) 2, one per line.
End with a line saying just "end".
>print "hello line 4"
>print $pc
>end
(gdb) br 6
Breakpoint 3 at 0x555555554661: file section23/brk_study.c, line 6.
(gdb) commands
Type commands for breakpoint(s) 3, one per line.
End with a line saying just "end".
>print "hello line 6"
>print z
>end
(gdb) info breakpoints
Num     Type           Disp Enb Address            What
2       breakpoint     keep y   0x000055555555464e in main 
                                                   at section23/brk_study.c:4
        print "hello line 4"
        print $pc
3       breakpoint     keep y   0x0000555555554661 in main 
                                                   at section23/brk_study.c:6
        print "hello line 6"
        print z
```
- If `commands` is given without `br`, then it will use the current line as the triggering point

37. gdb scripts
  - Edit `~/.gdbinit` then it wil be executed everytime when gdb runs
```txt
define log
bt
print $pc
end
```
  - Now running gdb will define `log` from the start
```bash
(gdb) start
Temporary breakpoint 1 at 0x642: file section23/brk_study.c, line 3.
Starting program: /home/hpjeon/hw/class/udemy_HandsOnDebugging/a.out 
Temporary breakpoint 1, main () at section23/brk_study.c:3
3	  printf("hello world\n");
(gdb) list
1	#include <stdio.h>
2	int main() {
3	  printf("hello world\n");
4	  int z=123;
5	  printf("hello world2\n");
6	  int y=123;
7	  printf("hello world3\n");
8	  int x=123;
9	  return 0;
10	}
(gdb) log
#0  main () at section23/brk_study.c:3
$1 = (void (*)()) 0x555555554642 <main+8>
```

## Section 7: Experimental Debugging Techniques

38. Reverse Debugging
  - target record-full
  - next
  - next
  - reverse-next

39. Set a variable value
  - set var i=123

## Section 8: Other useful Debugging Tools

40. DDD - Data Display Debugger
  - sudo apt install ddd
  - ddd ./a.out

41. strace and ltrace
  - strace ./a.out
  - strace -c ./a.out # statistics
  - ltrace ./a.out

42. Using valgrind
  - valgrind --leak-check=full ./a.out

43. Using Valgrind and GDB together to fix a segfault and memory leak
  - `p sizeof(var1)` at gdb

## Section 9: Conclusion

44. The conclusion and your next steps

## Section 10: Going Further with GDB Extra Features

45. GDB Command Debug Levels
  - g0 produces no debug information
  - g1 produces some information, but no line numbers
  - g2 is the default level that we have been using (same as -g). We get line numbers, symbols, file information at this level.
  - g3 can include further information such as macros
  - ggdb produces debug information specific to gdb. This is like -g3, and will try to generate as much information as possible. 

46. Inspecting the Virtual Table for Inheritance
```cpp
#include <iostream>
class Base {
  public:
    Base() { std::cout <<"base\n";}
    virtual void VirtualMember() {}
};
class Derived: public Base {
  public:
    Derived() { std::cout <<"derived\n";}
    void VirtualMember() {}
};
int main() {
  Base* obj = new Base;
  Base* obj2 = new Derived;
  delete obj;
  delete obj2;
  return 0;
}
```
  - obj and obj2 are type of Base but allocated using Base or Derived
```bash
(gdb) info vtbl obj
vtable for 'Base' @ 0x555555755d48 (subobject @ 0x555555768e70):
[0]: 0x555555554c1e <Base::VirtualMember()>
(gdb) info vtbl obj2
vtable for 'Base' @ 0x555555755d30 (subobject @ 0x5555557692a0):
[0]: 0x555555554c66 <Derived::VirtualMember()>
```
