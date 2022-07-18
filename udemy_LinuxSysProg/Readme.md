## Linux System Programming - A programmers/Practical Approach
- Instructor: Mohan P

## Section 1: Introduction to Linux

1. Course Contents

2. Introduction to Linux system Programming and Kernel
- Linux Kernel functions
  - File operation
  - Memory management
  - Virtual memory management
  - Process management
  - Thread management
  - Interprocess communications

3. Outcome of this course

## Section 2: System Programming Concepts

4. User mode and kernel mode
- Or user space vs kernel space
- user mode cannot access h/w
- Kernel mode can access h/w through privileged mode

5. Library functions
- Application SW <-> C stnadard library <-> System calls <-> H/W

6. Part A: System calls
- An interface that allows user application code to enter Kernel mode
- Can be invoked through
  - Direct system call: open(), close(), read(), write()
  - Use code calls the library function which calls system calls: printf()
- Instead of `printf("hello\n");`, `write(1, buf, msg);` will do the same work

7. Part B: System Calls
- write() in user code -> wrapper for write() from glibc
- Kernel model gets system call number for write()
- Trap handler is triggered and calls sys_write()
- sys_write() is executed
- Then trap handler closes and returns to user code

## Section 3: File Operations

8. Preliminary concepts of File
- File permission
  - Read has 4 units
  - Write has 2 units
  - Execute has 1 unit
- File type
  - `-`: regular file
  - d: directory
  - c: character device file
  - b: block device file
  - s: local socket file
  - p: named pipe
  - l: symbolic link
```bash
$ ls -al
total 12
drwxrwxr-x  2 hpjeon hpjeon 4096 Jul 15 20:17 .
drwxr-xr-x 42 hpjeon hpjeon 4096 Jul 15 20:17 ..
-rw-rw-r--  1 hpjeon hpjeon 2160 Jul 16 16:08 Readme.md
```

9. File open() - opening a file
- system calls involved in file operations
  - Open()
  - Read()
  - Write()
  - Close()
  - Lseek()
- File descriptors
  - All system calls for performing IO refer to open files using a file descriptor, a non-negative interger
  - All file related operations are performed via file descriptor (fd)
  - 3 standard file descriptor in Linux

| File Descriptor | Description | Posix Name | Stdio stream|
|---|---|---|---|
| 0 | Standard Input | STDIN_FILENO | stdin|
| 1 | Standard Output | STDOUT_FILENO | stdout|
| 2 | Standard Error | STDERR_FILENO | stderr|

- Open()
  - fd=0: standard input (keyboard)
  - fd=1: standard output (screen)
  - fd=2: standard error (screen/file)
  - fd=3: user file ...
- Parameters of Open()
  - Path
  - Flags
    - O_WRONLY, O_RDONLY, O_RDWR
    - O_CREAT: create fiel if it doesn't exist
  - Mode: specifies the file creation with access permissions
- Return value
  - `-1` for error
    - errno is set as EACCESS, EEXIST, EISDIR
  - When successful, returns non-zero value, which is file descriptor
- Sample code
  - `fd = open("newFile.log", O_RDWR | O_CREAT, 0774);`
  - `fd = open("newFile.log", O_RDWR | O_CREAT, S_IRWXU|S_IRWXG|S_IROTH);`

10. File read() - Reading a file
- `ssize_t read(int fd, void * buffer, size_t count);`
  - `buffer` must be allocaed beforehand
- read() systemcalls are applied on files like regular files, PIPES, sockets, FIFO
```c
#include <stdlib.h>
#include <stdio.h> 
#include <unistd.h>
#include <fcntl.h> 
int main() 
{ 
  int fd, sz; 
  char buf[20] = {0};
  fd = open("input.txt", O_RDONLY); 
  if (fd < 0){
      perror("Error:"); exit(1); 
  } 
  sz = read(fd, buf, 10); 
  printf("call 1 - called read. fd = %d,  %d bytes  were read.\n", fd, sz); 
  buf[sz] = '\0'; 
  printf("Read bytes are as follows: \n<%s>\n", buf); 
  printf("\n Note the next set of bytes read from file, it is continuos\n");
  sz = read(fd, buf, 11); 
  printf("call 2 - called read. fd = %d,  %d bytes  were read.\n", fd, sz); 
  buf[sz] = '\0'; 
  printf("Read bytes are as follows:\n<%s>\n", buf); 
  sz = read(fd, buf, 10); 
  printf("call 3 - called read. fd = %d,  %d bytes  were read.\n", fd, sz); 
  if(sz == 0){
      printf("EOF Reached\n");
  }
  close(fd);
} 
```
- In `sz = read(fd, buf, 10);`, sz could be smaller than 10 when the entire contents are almost read
  
11. File write() - writing to a file
- `ssize_t write(int fd, void* buffer, size_t count);`
- O_TRUNC: overwrites
- O_APPEND: appends
- `int fd = open("output.txt", O_WRONLY  | O_TRUNC  );`

12. File lseek() and close() system call
- lseek() is a system call that changes the location of read/write pointer of a file descriptor. The location can be set either in absolute or relative terms
- `off_t lseek(int fd, off_t offset, int whence);`
- Whence
  - SEEK_SET: offset is set to offset bytes
  - SEEK_CUR: offset is set to its current location plus offset bytes
  - SEEK_END: offset is set to the size of file plus offset bytes
- Return value: returns the offset of the pointer from the beginning of the file. -1 when there is an error moving the pointer

13. Tips

## Section 4: Advanced IO

14. Race condition

15. Atomicity

16. Pre-Emptive and Non Pre-emptive concept
- Ref: https://www.geeksforgeeks.org/preemptive-and-non-preemptive-scheduling/
- Preemptive scheduling is used when a process switches from running state to ready state or from the waiting state to ready state. The resources (mainly CPU cycles) are allocated to the process for a limited amount of time and then taken away, and the process is again placed back in the ready queue if that process still has CPU burst time remaining. That process stays in the ready queue till it gets its next chance to execute. 
  - ref: https://www.geeksforgeeks.org/time-slicing-in-cpu-scheduling/
  - time slice (or quantum): time frame for which process is allotted to run
- Non-preemptive Scheduling is used when a process terminates, or a process switches from running to the waiting state. In this scheduling, once the resources (CPU cycles) are allocated to a process, the process holds the CPU till it gets terminated or reaches a waiting state. In the case of non-preemptive scheduling does not interrupt a process running CPU in the middle of the execution. Instead, it waits till the process completes its CPU burst time, and then it can allocate the CPU to another process. 

17. Part A: File descriptor table and open file table
- File descriptor table, file table, and Inode table
- Each process will have its own **file descriptor table**
  - It holds all open file of that process
  - Another column for pointer to **open file table**
- Open file Table (system wide)
  - Visible to all processes
  - File offset, file status flags, pointer to i-node
- **i-Node table**
  - file type (PIPE, FIFO, regular)
  - UID
  - GID
  - File size
  - Time
  - Address of first 12 disk blocks : 12*4k = 48kB assuming 4kByte block size
  - Single indirection: 4kB * 1kB = 4MB memory can be mapped
  - Double indirection: 4kB * 1kB * 1kB = 4GB memory can be mapped
  - Triple indirection: 4kB * 1kB * 1kB * 1kB = 4TB memory can be mapped
- Disk block size may be 4K bytes, 8K bytes, ...
- General block size in Linux is 4K bytes
- For i-Node info, use `stat` command
```bash
$ stat ./author.txt 
  File: ./author.txt
  Size: 90        	Blocks: 8          IO Block: 4096   regular file
Device: 806h/2054d	Inode: 21506631    Links: 1
Access: (0664/-rw-rw-r--)  Uid: ( 1000/  hpjeon)   Gid: ( 1000/  hpjeon)
Access: 2022-07-16 17:19:07.105245912 -0400
Modify: 2022-07-16 17:19:04.793291770 -0400
Change: 2022-07-16 17:19:04.793291770 -0400
 Birth: -
```

18. Part B: File descriptor table and oepn file table
- File descriptor table (per processor) <-> Open file table (system wide) <-> i-Node table (system wide)

19. Duplicating File descriptor - dup() system call
- `int dup(int oldfd);`
- `int dup2(int oldfd, int newfd);`
- Q: why we need this?
  - Can redirect to screen or from screen to file

20. Use Case Scenario

## Section 5: Introduction to Process

21. Introduction to Process
- A process is an instance of an executing program

22. Process ID and parent process ID
- In Linux startup sequence
  - Process 0 is swapper process
  - Process 1 is `init` process, creating and monitoring set of other processes
    - `init` process becomes the parent of any Orphan process
- Process ID
  - `pid_t getpid(void);`
- Linux kernel limits process IDs as less than or equal to 32,767 (default on 32bit)
  - /proc/sys/kernel/pid_max
- Parent Process ID
  - `pid_t getppid(void);`
```bash
ps -ef |grep gedit
hpjeon   23715  8285  0 16:38 pts/0    00:00:03 gedit 
```
  - PID is 23715 and parent PID is 8285

23. Process States
- A process is CREATED state using fork() system call
- Running state: running in main memory
- Ready to Run in Main/swap memory
- Sleep state in Main/swap memory
- Blocked state in Main/swap memory
- Terminated state

24. Process Memory Layout - Part A
- Text segment: code resides
- Data segment: data variables during compile time
  - Initialized data segment
  - Uninitialised data segment (BSS)
- Stack segment: local variables
- Heap segment: dynamic memory data

25. Process Memory Layout - Part B

26. Tips

## Seciton 6: Virtual Memory of Process

27. The Big Picture

28. Virtual Memory Management and Page Table
- Each process has a private user space memory, which other process cannot access directly
- Each process has separate memory segments
  - Text segment
  - Data segment (initialized and uninitialised)
  - Stack segment
  - Heap segment
- The process virtual memory has a user space + kernel space. This memory region is configurable but usually 3GB for user space + 1GB for kernel space at 32bit system
  - 0xFFFF FFFF
  - kernel memory in the top
  - stack
  - heap
  - uninitalised data segment
  - initialized data segment
  - Text/Program segment
  - 0x0000 0000
- Page frame
  - A virtual memory of each process is split into small, fixed size units called pages (4096 bytes)
  - Similarly, physical memory or RAM is divided into a series of page frames of the same size
- Page fault
  - Whe a process tries to access a page that is not currently present in physical memory (and the page table). Kernel will suspend the process while the page is loaded from swap memroy into main memory

29. Command Line Arguments of Process
```c
#include <stdio.h>
#include <string.h>
void main(int argc, char *argv[]){
    int count = 0;
    printf("\nDemonstrate the command line arguments");
    printf("\n the value of argc is (%d)", argc);
    while(count < argc){
        printf("\n (%d) th string is (%s)",count,argv[count]);
        count++;
    }
}
```

30. Environment of Process
- name-value pairs of list
```c
#include <stdio.h>
extern char **environ;
int main(int argc, char *argv[])
{
    char **ep;
    for (ep = environ; *ep != NULL; ep++)
        printf("\n (%s)",*ep);
    return 0;
}
```
- Yields the equivalent results of `env` command
```bash
$ gcc chap30/env1.c 
$ ./a.out 
 (CLUTTER_IM_MODULE=xim)
 (LD_LIBRARY_PATH=:/usr/local/cuda/lib64:/usr/lib/nvidia-430)
 ...
 (LESSCLOSE=/usr/bin/lesspipe %s %s)
 (XDG_MENU_PREFIX=gnome-)
 (LANG=en_US.UTF-8)
 (DISPLAY=:1)
 (GNOME_SHELL_SESSION_MODE=ubuntu)
 (GTK2_MODULES=overlay-scrollbar)
```
- Using getenv():
```c
#include <stdio.h>
#include <stdlib.h>
int main () {
   printf("PATH : (%s)\n", getenv("PATH"));
   printf("HOME : (%s)\n", getenv("HOME"));
   return(0);
}
```
- putenv() to configure an environmental variable

31. Summary

## Section 7: Memory Allocation

32. Memory allocation - part A
- malloc(): library function call
- calloc(): library function call
- realloc(): library function call
  - May expand the memory depending on old block location, and care must be taken as it may give wrong results
- alloca()
  - Allocates memory dynamically but on **stack**
- brk(): system call
- sbrk(): system call
- free(): library function call
- Program break: current limit of heap
  - Q: is this **break pointer** from an other Udemy class?
  - The location of program break goes up as more heap memory is allocated
- brk()/sbrk()
  - Sets Program break to new memory location (brk) or increment as given (sbrk)

33. Memory allocation - part B
- free(): explicitly releases the memory occupied
  - Does NOT lower the program break

34. Memory allocation example programs

35. Summary

## Section 8: Process Programming

36. Process creation - fork() and Example program
- Process creation
  - Creating new process: fork()
  - Dividing tasks up often makes application design simpler
  - Fork() system call creates a new process (child), which is almost exact duplicate of the calling process (parent)
```c
#include <sys/types.h>
#include <unistd.h>
pid_t fork(void);
```
- fork() returns twice on success, one in parent process, and the other in child process
  - For parent: ID of child process on success. -1 or Error
  - For child: returns 0 on success
- Child's memory contents are intially exact duplicates of parent
```c
#include <stdio.h> 
#include <sys/types.h> 
#include <unistd.h> 
#include <stdlib.h>
int main() 
{ 
    pid_t id;
    printf("Parent Process : Executed by parent process before fork() - PID = (%d)\n",getpid()); 
    id = fork();  // from this point of code, the child and parent process both execute 
    if (id < 0 ){
        printf("\nfork failed\n");
        exit(-1);
    }
    if(id > 0){
        printf("\nParent Process: I have created child process withID = (%d)\n",id);

    }else
    {
        printf("\nI am child process,  id value is (%d)\n",id) ;
        printf("\nchild process id is (%d)\n",getpid());
        printf("\nThe creator of child process is (%d)\n",getppid());
    }
    return 0; 
} 
```
- From `id=fork()`, parent process gets ID of child, while the child gets id==0. After fork(), 2 different a.out's are executed

## Section 9: Process Monitor

37. wait(), waitpid() and Process termination
- wait(): system call. Syncs parent and child process and gets the exit status of child process
  - Each process has an entry in process table
  - When a process ends, all of the memory is deallocated. The parent can read the child's exit status by executing the wait system call
    - Even though memory of child process deallocated, the **entry** is not deallocated yet. This may cause a zombie process
  - The parent process receives signal SIGCHLD when child process is terminated
```c
#include<stdio.h> 
#include<stdlib.h> 
#include<sys/wait.h> 
#include<unistd.h> 
int main() 
{ 
    pid_t cpid; 
    int status = 0;
    cpid = fork();
    if(cpid == -1)
        exit(-1);           /* terminate child */
    if(cpid == 0){
        printf("\nchild executing first its pid = (%d)\n",getpid());
        sleep(2);
        printf("Child pid = %d\n", getpid()); 
        exit(1);
    } 
    else{    
        printf("\n Parent executing before wait()\n");
        //cpid = wait(NULL); 
        cpid = wait(&status); 
        printf("\n wait() in parent done\nParent pid = %d\n", getpid()); 
        printf("\n cpid returned is (%d)\n",cpid);
        printf("\n status is (%d)\n",status);
    }
    return 0; 
} 
```
  - `status` is determined by the normal exit or signal killed. Number needs to be converted into binary for analysis
    - 256 => 0001 0000 0000 : normal exit
    - If the child process is killed manually (kill command), status prints 15 => 1111
  - When 2 child processes are made, which child pid is met by wait() ?
    - First come first served
      - No control on specific child process
      - waitpid() can control the specific child process
    - The corresponding child pid is returned from wait()    
  - `ret_pid = waitpid(cpid, &status, WNOHANG);` : non-blocking
- exit(): system call. Terminates the process

38. Orphan, Zombie, and sleeping Process
- If a parent process exits or returns while a child processor, orphaned process is made
```c
#include <stdio.h> 
#include <sys/types.h> 
#include <unistd.h> 
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
int main() 
{ 
    pid_t id;
    printf("Parent Process : Executed by parent process before fork() - PID = (%d)\n",getpid()); 
    id = fork();  // from this point of code, the child and parent process both execute 
    if (id < 0 ){
        printf("\nfork failed\n");
        exit(-1);
    }
    if(id > 0){ // Parent process, killing before child executes
        printf("\nParent process exited\n");
        return (0);
    }else
    { // Child process
        printf("\nChild process executing\n");
        sleep(10);
        printf("\nI am child process,  id value is (%d)\n",id) ;
        printf("\nchild process id is (%d)\n",getpid());
    }
    return 0; 
} 
```
  - In this example, a.out of the child process will be terminated after 10sec, without manual killing
- When a child process exits while the parent is not yet, the entry of the child process becomes a zombie process
  - Use wait() or waitpid() to avoid zombie processes
    - Zombie processes are still made when the child entry still exists but eventually it is removed by the parent process

## Section 10: Advanced Process Programming

39. Exeucting new program - exec()
- execve() - system call which loads a new program into a process's memory
  - Replaces old program's stack, data, heap with new program
  - Returns -1 on error while nothing returned on success
- Library functions using execve()
  - execl()
  - execlp()
  - execle()
  - execv()
  - execvp()
  - execvpe()

40. Examples of exec functions
- execl(): runs an external executable within a C code
  - Like subprocess of python
  - But replaces the parent code
  - The left-over of the parent code will not be executed

41. Example of execv()
- execv()
```c
char *args[]={"arg1","arg2","arg3",NULL};
re = execv("./p2",args); 
```
  - When successful, nothing is returned. The left-over of the source code is ignored

42. Example of execve()

43. Exec() and Fork()
- When a child process runs execl()
  - The virtual memory of the child process is replaced with the given executable
  - The left-over of the code in child process is not loaded
  - Parent process will get the status of child process through wait()

44. Process Table and file descriptor b/w Parent and Child
- When a fork() is run, the child receives the duplicates of all of parent's file descriptors
- Both of parent/child process will have the same file descriptors 
  - Potential race condition?
- Process table has pointers for Process Control Block table
- Process Control Block (PCB)
  - Data structure of each process
  - Program counter
  - Process number
  - Process state
  - CPU Registers
  - Process Priority
  - Memory management information
  - List of open files of the process

## Section 11: Signals

45. Signals in Linux
- SW interrupts that provide a mechanism for handling asynchronous events
- Originates from outside of the system - ctrl+c, ...
- Different signals with unique numbers
- Signals have a very precise lifecycle. First a signal is raised, then kernel stores the signal until it is able to deliver. Finally the signals are delivered to the corresponding process
- Kernel can perform one of three actions, depending on what the process asked it to do
  - Ignore: no action is taken
    - SIGKILL and SIGSTOP cannot be ignored
  - Catch and handle the signal: the kernel will suspend execution of the process's current code path and jump to a previously registered function. The process will then execute this function when the process returns from this function. It will jump back to wherever it was when it caught the signal
  - Perform the default action: every signal will have its default behavior in case the process has not handled it
- How signal handler work
  - Main program runs
  - Signal is delivered to a process from kernel
  - Jumps to signal handler
  - Singal handlers execute
  - Return to the main program
- Signal Identifiers
  - `SIG` for prefix
  - All definitions are at signal.h
  - Every signal has DEFAULT ACTION
  - SIGABRT: abort() function sends this signal to the process, which terminates the process and generates a core dump file
  - SIGALRM: alarm() and setitimer() send this signal to the process, when an alam expires
  - SIGBUS: Bus error. Raised when a process has a memory access error
  - SIGCHLD: whenever a process terminates or stops, this signal is sent to the parent process
  - SIGCONT: When stopped process needs to be resumed
  - SIGFPE: When arithmetic exception happened
  - SIGILL: When a process attempts to execute an illegal machine instruction
  - SIGINT: interrupt signal by ctrl+c
  - SIGKILL: sent from kill() system call
  - SIGPIPE: When a process writes to a pipe in which the reader has terminated
  - SIGSEGV: Segmentation violation. Sent to a process when it attempts an invalid memory access
  - SIGUSR1/2: For user-defined purposes. Kernel never raises them

46. Programming with Signals - Part A
- signal() function defines or configures signum behavior. Raising or triggering is handled by kernel, not application code
```
#include <signal.h>
typedef void (*sighandler_t)(int);
sighandler_t signal(int signum, sighandler_t handler);
```
- When a signal is delivered to a process, using signal(), the signal can either perform default action or ignore the particular signal
  - handler is a function pointer
- SIG_DFL: sets the behavior of the signal given by signum to its default
- SIG_IGN: ignores the signal given by signum
- kill(): sends a signal from one process to another
  - Not necessarily to terminate process all the time
  - Just the default signum is TERM
  - `int kill(pid_t pid, int signum);`

47. Programming with Signals - Part B
```c
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
static void signal_handler (int signo)
{
    if (signo == SIGINT)
        printf ("Caught SIGINT!\n");
    else if (signo == SIGTERM)
        printf ("Caught SIGTERM!\n");
    exit (EXIT_SUCCESS);
}
int main (void)
{
    printf("\n process ID is(%d)\n",getpid());
    if (signal (SIGINT, signal_handler) == SIG_ERR) {
        fprintf (stderr, "Cannot handle SIGHUP!\n");
        exit (-1);
    }
    if (signal (SIGTERM, signal_handler) == SIG_ERR) {
        fprintf (stderr, "Cannot handle SIGTERM!\n");
        exit (-1);
    }
    while(1);
}
```
  - Note while(1) in the bottom
  - Entering ctrl+c will invoke SIGINT and exits
    - When SIGINT (ctrl+c) is raised, signal() will call signal_handler function
  - If we disable `exit(EXIT_SUCCESS)`, ctrl+c doesn't terminate a.out
- When  `if (signal (SIGINT, SIG_IGN) == SIG_ERR) ` is implemented, ctrl+c (SIGINT) will be ignored, doing nothing
- When  `if (signal (SIGINT, SIG_DFL) == SIG_ERR) ` is implemented, ctrl+c (SIGINT) will just kill the process, as default behavior. signal_handler is not called
- To activate SIGTERM above, kill <pid> from another CLI
```c
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
void display_message(int s) {
     printf("Generated SIGALARM\n" );
     //signal(SIGALRM, SIG_IGN);
     signal(SIGALRM, SIG_DFL);
     alarm(2);    //for every second
}
int main(void) {
    signal(SIGALRM, display_message);
    alarm(2);
    while (1);
}
```
- First alarm() will trigger display_message(), which configures SIGARM as default behavior. Therefore, 2nd alarm() will default alarm and terminates

48. Programming using SIGUSR signals
- parent code
```c
#include<stdio.h> 
#include<stdlib.h> 
#include<sys/wait.h> 
#include<unistd.h> 
static void signal_handler (int signo)
{
    if (signo == SIGUSR1)
        printf ("Parent: Caught SIGUSR1 in parent!\n");
}
int main() 
{ 
    pid_t cpid; 
    int status = 0;
    int num  = 5;
    cpid = fork();
    if(cpid == -1)
        exit(-1);           /* terminate */
    if(cpid == 0){
        printf("\nChild: Before exec\n");
        execl("./program2","arg1","arg2",NULL);
        printf("\n Child: line is not printed\n");
    } 
    else{   
        if (signal (SIGUSR1, signal_handler) == SIG_ERR) {
            fprintf (stderr, "Cannot handle SIGUSR1!\n");
            exit (-1);
        }
        printf("\nParent: Parent executing before wait(), child process created by parent is = (%d)\n",cpid);
        sleep(2);
        kill(cpid,SIGUSR2);
        cpid = wait(&status); /* waiting for child process to exit*/
        printf("\nParent: wait() in parent done\nParent pid = %d\n", getpid()); 
        printf("\nParent: cpid returned is (%d)\n",cpid);
        printf("\nParent: status is (%d)\n",status);
    }
    return 0; 
} 
```
  - `kill(cpid,SIGUSR2);` Send SIGUSR2 the child process
    - Not necessarily kill always but the default signal is TERM
- External code
```c
#include <stdio.h> 
#include <unistd.h> 
#include <signal.h>
#include <stdlib.h>
static void signal_handler (int signo)
{
    if (signo == SIGUSR2)
        printf ("Child: Caught SIGUSR2 in child!\n");
}
int main(int argc, char *argv[]) 
{ 
    int i = 0; 
    printf("Child: I am new process called by execl() \n"); 
    printf("Child: new program pid  = (%d)\n",getpid()); 
    if (signal (SIGUSR2, signal_handler) == SIG_ERR) {
        fprintf (stderr, "Cannot handle SIGUSR2!\n");
        exit (-1);
    }
    for(i = 0; i < argc; i++){
        printf("\nChild: argv[%d] = (%s)\n",i,argv[i]);
    }
    sleep(5);
    printf("\nChild: sending sigusr1 to parent\n");
    kill(getppid(),SIGUSR1);
    sleep(10);
    printf("\nChild: exiting\n");
    return 0; 
} 
```

## Section 12: Threads

49. The Big Picture
- Threads can share the virtual memory (stack, heap, data, code segment) with a process
  - Each thread may have its own stack segment

50. Thread creation and termination
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
static void * threadFunc(void *arg)
{
    char *str = (char *) arg;
    printf("ThreadFunc: arguments passed to thread are: %s , pid = (%d)\n", str,getpid());
    (void)sleep(2);
    printf("\nthreadFunc: exiting now\n");
    return(0);
}
int main(int argc, char *argv[])
{
    pthread_t t1;
    void *res;
    int s;
    s = pthread_create(&t1, NULL, threadFunc, "Hello world");
    if (s != 0)
        perror("Thread create error");
    printf("main thread: Message from main() , pid = (%d)\n",getpid());
    sleep(5);
    printf("\nmain thread: exiting now\n");
    exit(0);
}
```

51. Pthread join
- `int pthread_join(pthread_t thread, void** retval);`

52. pthread_cancel() and detaching a thread
- By default, a thread is joinable. When it terminates, another thread can obtain its return value using pthread_join()
- If a joinable thread is not joined by calling pthread_join(), the terminated thread becomes a zombie thread, consuming memory resource
- pthread_detach() cleans up terminated thread, without pthread_join()
- `int pthread_detach(pthread_t, thread);`
- 0 returned on sucess while positive error number or error

53. Example programs

54. Threads vs Process

## Section 13: Thread Synchronization

55. Synchronization using Mutex
- How to avoid race conditions
  - Critical section where atomic execution should be done
- MUTEX(Mutual Exclusion)
  - A kind of lock
  - Two states: locked and unlocked
  - Steps
    - Lock the mutex
    - Operation on shared resource
    - Unlock the mutex
- Avoiding dead locks
  - Build a hierarchy of mutex

56. Condition Variables
- Signals from one thread to other thread regarding the changes in the state of a shared variable
- Can help to define the sequence of thread execution
- Can be allocated statically or dynamically
  - `pthread_cond_t cond = PTHREAD_COND_INITIALIZER;`
- The principle of condition variable is `signal and wait`

## Section 14: IPC - Introduction

57. A brief overview of IPC
- Send/receive data b/w processes
- Sync b/w processes
- Communication based IPC
  - Data transfer based: PIPE, FIFO, message queue, socket
  - Memory sharing based: Shared memory
- Synchronization based IPC
  - Semaphore
  - Mutex (in threads)
  - Condition variables (in threads)

## Section 15: PIPES and FIFO - Inter process communication

58. PIPE - IPC
- A byte stream used for IPC
- 2 ends
  - read end
  - write end
- Uni-directional
- Limited capacity (64Kbytes usually)
  - A pipe is simply a buffer maintained in kernel memory
- When a pipe is full, further writes to PIPE is blocked until the receive end process removes the data from PIPE
- Write to PIPE puts data while reading data from PIPE removes the data from PIPE
```c
#include <unistd.h>
int pipe(int pipefd[2]);
```
  - Needs 2 file descriptors
- After fork(), child process inherits the copies of file descriptor of parent
  - parent process: fd[0], fd[1]
  - child process: fd[0], fd[1]
  - At PIPE, let parent process write at fd[1] and child process read at fd[0]
- SIGPIPE: is sent to write end process when read end is closed
```c
#include <fcntl.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h> 
#include <string.h>
int main() 
{ 
    int fd[2];  
    char buff[50]; 
    char data[50] = {}; 
    //Open a pipe
    if (pipe(fd) == -1)  
    { 
        perror("pipe"); // error in pipe 
        exit(1); // exit from the program 
    } 
    // fd[0] contains descriptor to read end of pipe, fd[1] contains descriptor to write end of pipe 
    sprintf(buff,"PIPE data flow demo:");
    // writing to pipe 
    write(fd[1], buff, strlen(buff));  
    printf("\n"); 
    // reading pipe , and storing data in data buffer
    read(fd[0], data, 5); 
    printf("%s\n", data);  
    read(fd[0], data, 5); 
    printf("%s\n", data);  
    read(fd[0], data, 10); 
    printf("%s\n", data);  
} 
```
- PIPE/data/flow demo: are printed sequentially

59. FIFO - IPC
- First In First Out
  - Similar to a PIPE
  - Aka `Name PIPES`
- FIFO has a name within the file system and is opened in the same way as a regular file, where as PIPES does not have a name
- FIFO is used b/w unrelated processes (ex: client vs server) while PIPE is for b/w related processes (parent-child)
- When a FIFO is opened, IO system calls of read(), write(), close() are used
- Smiliar to PIPE, FIFO has write/read ends
```c
#include <sys/stat.h>
int mkfifo(const char * pathname, mode_t mode);
```
  - Returns 0 on success while -1 on error
- Similar to sharing a scratch file but no actual file at disk

## Section 16: POSIX - Message Queue

60. Message Queue operations
- Passes messages b/w processes
- This is message oriented IPC
  - PIPEs and FIFO are byte oriented IPC
- Readers and writers communicate each other in units of messages
- POSIX message queues permit each message to be assigned a priority
- Priority allows high-priority messages to be queued ahead of low priority messages
- Message queue entry is present in the file system in /dev/mqueue
- System calls of message queue
  - mq_open(): creates a new message queue or opens an existing queue, returning a message queue descriptor for use in later calls
  - mq_close(): closes an opened message queue
  - mq_unlink(): removes a message queue name and marks the queue for deletion
  - mq_send(), mq_receive() : writes/reads a message to/from a queue
  - mq_setattr(), mq_getattr()
  - Need to compile with `-lrt`
- Message queue starts with `/`
- Example
- sender.c
```c
#include <fcntl.h>
#include <limits.h>
#include <mqueue.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
int main(int argc, char **argv)
{
    mqd_t queue;
    struct mq_attr attrs;
    size_t msg_len;
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s <queuename> <message>\n", argv[0]);
        return 1;
    }
    queue = mq_open(argv[1], O_WRONLY | O_CREAT, 0660, NULL); // Open message queue with default attributes
    if (queue == (mqd_t)-1)
    {
        perror("mq_open");
        return 1;
    }
    if (mq_getattr(queue, &attrs) == -1)
    {
        perror("mq_getattr");
        mq_close(queue);
        return 1;
    }
    // print the attribute values
    printf("\n message queue mq_maxmsg = (%d), mq_msgsize is (%d)\n",(int)attrs.mq_maxmsg, (int)attrs.mq_msgsize);
    msg_len = strlen(argv[2]);
    if ( (long)msg_len > attrs.mq_msgsize)
    {
        fprintf(stderr, "Your message is too long for the queue.\n");
        mq_close(queue);
        return 1;
    }
    if (mq_send(queue, argv[2], strlen(argv[2]), 0) == -1) // 0 is the priority that can be set based on message priority 0 is least priority
    {
        perror("mq_send");
        mq_close(queue);
        return 1;
    }
    return 0;
}
#if 0
Assignment
1. create priority Message queue with different priority for each message, and check output
#endif
```
- rec.c
```c
#include <fcntl.h>
#include <mqueue.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
int main(int argc, char **argv)
{
    mqd_t queue;
    struct mq_attr attrs;
    char *msg_ptr;
    ssize_t recvd;
    size_t i;
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <queuename>\n", argv[0]);
        return 1;
    }
    queue = mq_open(argv[1], O_RDONLY | O_CREAT, S_IRUSR | S_IWUSR, NULL);
    if (queue == (mqd_t)-1)
    {
        perror("mq_open");
        return 1;
    }
    if (mq_getattr(queue, &attrs) == -1)
    {
        perror("mq_getattr");
        mq_close(queue);
        return 1;
    }
    msg_ptr = calloc(1, attrs.mq_msgsize);
    if (msg_ptr == NULL)
    {
        perror("calloc for msg_ptr");
        mq_close(queue);
        return 1;
    }
    recvd = mq_receive(queue, msg_ptr, attrs.mq_msgsize, NULL);
    if (recvd == -1)
    {
        perror("mq_receive");
        return 1;
    }
    printf("\n Received messsage in msg queue is (%s)\n",msg_ptr);
}
```
- gcc send.c -o send -lrt
- gcc rec.c -o rec -lrt
```bash
$ ./sender /test "hello world"
 message queue mq_maxmsg = (10), mq_msgsize is (8192)
$ ./rec /test
 Received messsage in msg queue is (hello world)
```

## Section 17: POSIX - Semaphore

61. Semaphore operations - named semaphore
- Allows processes and threads to synchronize access to shared resource
- Executes the critical section in an atomic manner
- Has name as specified in sem_open()
  - /dev/shm
- Un-named semaphores reside at an location in memory
- List of functions
  - sem_open(): opens or creates a semaphore, initializing semaphore and returning a handle for use in later calls
  - sem_post(): increments a semaphore's value
  - sem_wait(): decrements a semaphore's value
  - sem_getvalue(): retrieves a semaphore's current value
  - sem_close(): removes the calling process's association with an opened semaphore
  - sem_unlink(): removes a semaphore name and marks the semaphre for deletion when all processes have closed it
- The name of semaphore begins with `/`
- In order to delete /dev/shm/sem.sem1, `sem_unlink("/sem1");` is necessary in the end of the code

62. Un-named semaphore 
- Variables of type sem_t that are stored in memory allocatd by the application
  - Named semaphore is present in file system similar to regular files
- Use same functions of sem_wait(), sem_post, sem_getvalue() and there are two more extra functions 
  - sem_init()
  - sem_destory()

## Section 18: POSIX - Shared Memory

63. Shared Memory Concepts
- Ref: https://stackoverflow.com/questions/9701757/when-to-use-pipes-vs-when-to-use-shared-memory
- Same physical memory is mapped into virtual memory of multiple processes
- At /dev/shm
- The fastest IPC mechanism
- Steps to use POSIX shared memory object
  - Use shm_open() to open an object with a specified name
    - Creates a new shared memory object or opens an existing one
    - Returns a file descriptor referring to the shared memory
    - Newly created shared memory object is 0 byte
  - Define the length of shared memory
  - File descriptor returned in the above step is referenced in mmap()
  - This maps the shared memory object into the process's virtual address space
- Compile needs `-lrt`
    
64. Shared Memory Operations
- shwrite.c
```c
#include <fcntl.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
int main(int argc, char *argv[])
{
  int fd;
  size_t len;
  char *addr;
  fd = shm_open("/shm_1", O_RDWR | O_CREAT, 0660);
  if (fd == -1){
    printf("\nError creating shm\n");
    perror("shm_open");
    exit(-1);
  }
  /* Open existing object */
  printf("\n shm open success\n");
  len = strlen(argv[1]);
  if (ftruncate(fd, len) == -1){
      perror("ftruncate");
      exit(-1);
  }
  printf("Resized to %ld bytes\n", (long) len);
  addr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED){
    perror("mmap");
    exit(-1);
  }
  if (close(fd) == -1){
      perror("close");
  }
  memcpy(addr, argv[1], len);
  exit(0);
}
```
- shread.c
```c
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
int main(int argc, char *argv[])
{
    int fd;
    char *addr;
    struct stat len;
    fd = shm_open("/shm_1", O_RDONLY, 0);
    if (fd == -1)
    {
        printf("\n shm open error\n");
        return -1;
    }
    fstat(fd, &len);
    addr = mmap(NULL, len.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED)
    {
        printf("\n mmap error\n");
        return -1;
    }
    printf("\nRead data from shared memory - (%s)\n",addr);
    exit(0);
}
```
- Compilation and command:
```bash
$ gcc chapt64/shread.c -o sr -lrt
$ gcc chapt64/shwrite.c -o sw -lrt
$ ./sw "hello world"
 shm open success
Resized to 11 bytes
$ ./sr
Read data from shared memory - (hello world)
```

## Section 19: Closing Note

65. Closing Note and Further Reading
- The design of Unix operating system by Maurice J Bach
- The Linux programming interface  by Michael Kerrisk
- Advanced Programming in the Unix environment by W. Richard Stevens

## Section 20: Bonus - Students Q&A
