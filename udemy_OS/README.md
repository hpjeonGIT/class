# Table of classes

1. Udemy: The Complete Operating Systems Course: From Zero to Hero!
2. Udemy: Building a Toy Graphical Operating System from Scratch
3. Udemy: Write Your Own Operating System From Scratch - Step by Step


# Udemy: The Complete Operating Systems Course: From Zero to Hero!
- By Lucas Bazilio

## Section 1: Basic Fundamentals

### 1. Introduction to Operating Systems
- OS is a SW that manages HW resources
- The kernel is a computer program at the core of a computer's OS and generally has complete control over everything in the system
  - Kernel internals: defines data structures to manage HW and algorithms to decide how to use it
  - Kernel API: offers a set of functions to ask for the system services
- What to expect from OS?
  - Usable environment
    - Abstracts the user from the different kind of systems
  - Offers a safe/robust execution environment
  - Offers an efficient execution management
- Following sections
  - Process management
    - Concept of process
    - Context of a process
    - The life of a process
    - Context change mechanism
    - Event management (signals)
    - Flow concept
    - Communication b/w processes
  - Memory management
    - Address translation support
    - Dynamic memory
    - Virtual memory management

### 2. Operating System Steps
- Boot
  - Executed when system is switched on, the kernel code is loaded in memory
  - Interrupts and basic HW configurations are initialized
  - It starts the system access mechanism: login, shell, etc
- Usage
  - Develop new applications
  - Execute applications
- Shutdown
  - Executed when system is switched off
  - Saves persistent information, stop devices, etc

### 3. Running Executables

## Section 2: Processes

### 4. Introduction to Processes
- A process is the OS representation of a program during its execution
- When a user starts a program execution, a new (unique) process is created
  - The kernel assigns resources to it: physical memory, some slot of CPU time and allows file access
  - The kernel reserves and initializes a new process data structure with dynamic information
    - Each OS uses a name for the data structure, in general, we will refer to it as PCB
    - Each new process has a unique identifier, PID

### 5. Process States
- run: the process is assigned a cpu and is running
- ready: the process is ready to run but is waiting for a CPU to be allocated
- blocked: the process does not have/consumes CPU, it is blocked waiting for an input/output to finish or the arrival of an event
- zombie: the process has finished executing but has not yet disappeared from the kernel data structures

### 6. Process Creation - Fork() System call
- System calls allow users to ask for:
  - Creating new processes
  - Changing the executable file associated with a process
  - Edning its execution
  - Waiting until a specific process ends
  - Sending events from one process to another
- Linux system calls for process management

| Description                         | syscall|
|-------------------------------------|--------|
|Process creation                     | fork|
|Changing executable file             | exec (execlp) |
| End process                         | exit |
| Wait for a chid process (can block) | wait/waitpid |
| PID of the calling process          | getpid |
| Parent's PID of the calling process | getppid |

- pcreate.c
```c
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
int main() 
{
  int rv = fork();
  if (rv > 0)
   printf(" I am parent. my pid is %d\n", getpid());
  else if (rv <0)
   printf("fork error\n");
  else  // rv == 0
   printf(" I am child. my pid is %d\n", getpid());
  return 0;
}
```
- Run
```bash
$ gcc pcreate.c 
$ ./a.out 
 I am parent. my pid is 786073
 I am child. my pid is 786074
```

### 7. Executable Change - Execlp() System call
- exec() family
  - replaces the current process with a new process image
  - replaces the currently shell operation without creating a new process
- Changinge the executable file
  - The  exec() family of functions replaces the current process image with a new process image
    - Same process, different image
  - Steps the kernel must do:
    - Release the memory currently in use
    - Load (from disc) the new executable file in the new reserved memory area
    - PC and SP registers are initialized to the main function call
  - Since the process is the same:
    - Resource usage account information is not modified
    - The signal actions table is reset
- exe.c:
```c
#include <stdio.h>
#include <unistd.h>
#include <string.h>
void main()
{
  char buffer[80];
  sprintf(buffer, "I am the process: %d\n", getpid());
  write(1,buffer,strlen(buffer));
  execlp("./a.out", "a.out",(char *)0);
  // below lines are not executed as execlp() replaces the current process, and they are wiped out from memory
  sprintf(buffer, "I am the process: %d\n", getpid());
  write(1,buffer,strlen(buffer));
}
```
- Demo:
```bash
$ gcc exe.c -o b.out
$ ./b.out 
I am the process: 18190
 I am parent. my pid is 18190
 I am child. my pid is 18191
```
- In `execlp()`, `(char *) 0` is a null pointer sentinel value that marks the end of the argument list

### 8. Process Termination - Exit() and Waitpid()
- `void exit(int status);`: terminates the process voluntarily
  - Child process sends SIGCHLD to the parent process
- `pid_t waitpid(pid_t pid, int *status, int options);`: suspends execution of the calling process until one of its children terminates
  - `waitpid(-1,NULL,0)`: wait for any child process
    - Not all of them. Whichever child finishes first
    - You may a loop to wait for all the children processes, to avoid zombie processes
      - `while (waitpid(-1,NULL,0) > 0);`
      - `void sigchld_handlre(int sig) { while (waitpid(-1,NULL,WNOHANG) > 0 );}`
  - `waitpid(pid_child,NULL,0)`: wait for a child process with PID=pid_child
    - If status is not null, the kernel stores status information in the int to which it points
    - Options can be:
      - 0 means the calling process will block
      - WNOHANG means the calling process will return immediately if no child process has exited
- How to have sequential scheme:
![seq_scheme](./seq_scheme.png)
```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#define num_procs 5
void main() 
{
  int ret;
  for (int i=0;i<num_procs; ++i) 
  {
    ret = fork();
    if (ret == 0) {
        // CHILD code
        printf("Hi I am the child with pid: %d\n", getpid());
        exit(0); // finishes child process
    }
    if (ret <0) {printf("Error\n");}
    // PARENT code
    waitpid(-1,NULL,0);
  }
}
```
- Demo:
```bash
$ gcc code.c -o seq_scheme.exe
$ ./seq_scheme.exe 
Hi I am the child with pid: 12206
Hi I am the child with pid: 12207
Hi I am the child with pid: 12208
Hi I am the child with pid: 12209
Hi I am the child with pid: 12210
```
- How to have concurrent scheme
![conc_scheme](./conc_scheme.png)
```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#define num_procs 5
void main() 
{
  int ret;
  for (int i=0;i<num_procs; ++i) 
  {
    ret = fork();
    if (ret == 0) {
        // CHILD code
        printf("Hi I am the child with pid: %d\n", getpid());
        exit(0); // finishes child process
    }
    if (ret <0) {printf("Error\n");}
    // PARENT code
  }
  // wait all of the children to finish
  while(waitpid(-1,NULL,0) > 0);  
}
```
- Demo:
```bash
$ ./conc_scheme.exe 
Hi I am the child with pid: 12891
Hi I am the child with pid: 12892
Hi I am the child with pid: 12893
Hi I am the child with pid: 12894
Hi I am the child with pid: 12895
```

### 9. Exercise - Process Treee
- How many processes are created from the following code?
```c
#include<stdlib.h>
#include<stdio.h>
#include<unistd.h>
int main(int argc, char * argv[])
{
  for (int i=0; i< 3; ++i)  
  {
    int p = fork();
  }
  printf("Process PID: %d\n", getpid());
  return 0;
}
```
- 8 processes are created:
![process_tree](./process_tree.png)
- Demo:
```bash
$ ./a.out 
Process PID: 13703
Process PID: 13707
Process PID: 13705
Process PID: 13704
Process PID: 13706
Process PID: 13709
Process PID: 13708
Process PID: 13710
```

## Section 3: Laboratory 1 - Processes

### 10. Introduction to Laboratory 1

### 11. Problem 1
- Create a C program that sequentially creates as many processes as there are arguments minus 1. If executed with no arguments, an error message is printed then exits.
  - Each child process screens the name of argument and exits
  - The parent process waits for its child to finish, writes the child's pid to the scree, and exits
```c
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <sys/wait.h>
void Usage() {
    printf("Usage: /processes1 arg1 [arg2...argn]\n");
    printf("This program creates as many processes as it receives arguments minus 1 sequentially\n");
}
void error_and_exit(char *s, int error_status) {
    perror(s);
    exit(error_status);
}
void main(int argc,char *argv[]) {
    int pid;
    if (argc >= 2) {
        for (int i = 1; i < argc; ++i) {
            pid = fork();
            if (pid == 0) {
                // CHILD CODE
                printf("The argument %d is %s\n",i,argv[i]);
                exit(0);
            }
            if (pid < 0) {
                error_and_exit("Error in the fork\n",1);
            }
            // PARENT CODE
            waitpid(-1,NULL,0);
            printf("The child that has finished has been %d\n",pid);
        }
    }
    else {
        Usage();
    }
}
```  
- Demo:
```bash
$ ./a.out 
Usage: /processes1 arg1 [arg2...argn]
This program creates as many processes as it receives arguments minus 1 sequentially
$ ./a.out hello world
The argument 1 is hello
The child that has finished has been 21967
The argument 2 is world
The child that has finished has been 21968
```

### 12. Problem 2
- Create a C program that concurrently creates as many processes as there are argumens - 1. 
- Each child i mutates to the linux ls command. The argument of ls will be argument of the program on the same position i (-l -s -i -C ...)
- The parent must wait for all its children
```c
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <sys/wait.h>
void Usage() {
    printf("Usage: /processes2 arg1 [arg2...argn]\n");
    printf("This program creates as many processes as it receives arguments minus 1 concurrently\n");
}
void error_and_exit(char *s, int exit_status) {
    perror(s);
    exit(exit_status);
}
void main(int argc,char *argv[]) {
    if (argc >=2) {
        for (int i = 1; i <argc; ++i) {
            pid = fork();
            if (pid == 0) {
                // CHILD CODE
                execlp("ls","ls",argv[i],(char*) 0);
                // Note that exit(0) is not necessary as after ls is executed, it exits anyway
            }
            if (pid < 0) {
                error_and_exit("Error in the fork\n",1);
            }
        }
    // We created all of the children
    // Now we wait for all of them to terminate
    int ret;
    while((ret=waitpid(-1,NULL,0)) > 0) {
        printf("PARENT: child %d terminated\n",ret);
    }    
    }    
    else {
        Usage();
    }    
}
```
- Demo:
```bash
$ ./a.out -t -C
Hi I am the child with pid: 23051
Hi I am the child with pid: 23052
a.out	 code.c		  part11.c     seq_scheme.exe
b.out	 conc_scheme.exe  part12.c
code2.c  exe.c		  pcreate.c
code3.c  part11_0.c	  proc_tree.c
a.out	    proc_tree.c      seq_scheme.exe  pcreate.c
part12.c    code2.c	     code.c
part11_0.c  code3.c	     b.out
part11.c    conc_scheme.exe  exe.c
parent process exits now
```

### 13. Problem 3
- Create a C program that has a global variable 'A' and a local variable 'a'. Then create a child process with the fork system call that modifies these values to 'Z' and 'z' respectively
  - The child will print the variables
  - The parent waits the child to terminate then it prints those variables
```c
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <sys/wait.h>
char x = 'A';
int main(int argc, char * argv[])
{
  char y = 'a';
  printf("from parent : %c %c \n", x, y);      
  int p = fork();  
  if (p == 0) {
    // CHILD CODE
    x = 'Z';
    y = 'z';
    printf("from child : %c %c \n", x, y);
    exit(0);
  }
  if (p < 0) {
    printf("Error in the fork\n");
  }
  // PARENT CODE
  waitpid(-1,NULL,0);
  printf("from parent : %c %c \n", x, y);    
  return 0;
}
```  
- Result:
```bash
$ ./a.out 
from parent : A a 
from child : Z z 
from parent : A a 
```
- Modification from child process doesn't change the memory of the parent process

### 14. Problem 4
- What would be the outputs of a following file:
```c
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
char global_variable = 'A';
void
Write_global_variable (char *whoiam)
{
  char buffer[80];
  sprintf (buffer, "%s:The global variable value is %c\n", whoiam,
           global_variable);
  write (1, buffer, strlen (buffer));
}
int
main (int argc, char *argv[])
{
  int pid;
  int i;
  char buffer[80];
  sprintf (buffer, "Before fork: I am the process %d\n", getpid ());
  write (1, buffer, strlen (buffer));
  pid = fork ();
  switch (pid)
    {                           
    case 0:                    
      sprintf (buffer, "CHILD: I am the process %d\n", getpid ());
      write (1, buffer, strlen (buffer));
      /* Test if we have access from a function */
      for (i = 0; i < 100000; i++)
        {
          /* Every 10000 iterations we change the value of the variable */
          if ((i % 10000) == 0)
            global_variable++;
          Write_global_variable ("CHILD");
        }  
      exit (0);
    case -1:                   
      strcpy (buffer, "Error has been produced\n");
      write (1, buffer, strlen (buffer));
      break;
    default:                   /* (pid != 0) && (pid != -1) */
      sprintf (buffer, "PARENT: I am the process %d\n", getpid ());
      write (1, buffer, strlen (buffer));
      /* Test if we have access from a function */
      /* Change the value of the variable */
      global_variable = 'a';
      for (i = 0; i < 100000; i++)
        {
          /* Every 10000 iterations we change the value of the variable */
          if ((i % 10000) == 0)
            global_variable++;
          Write_global_variable ("PARENT");
        }
    }
  sprintf (buffer, "Only parent executes: I am the process %d\n",
           getpid ());
  write (1, buffer, strlen (buffer));
  return 0;
}
```
- ? What is the purpose of this exercise?

### 15. Problem 5
- Create a program called levels.c that receives arguments of N and M. Then create N children at level 1. Each of these children will create M children at level 2.
- Each child print its PID and its parent PID
```c
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <sys/wait.h>
void print_pid_ppid(int level) {
    printf("Level = %d, pid = %d, ppid=%d\n", level, getpid(), getppid());
}
void main(int argc,char *argv[]) {
    if (argc != 3) {printf("Requires N and M as arguments\n"); exit(0);}
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    printf ("N and M are %d %d\n", N, M);
    int ret, rv, rv2, stat;
    for (int i = 0; i <N; ++i) {
        rv = fork();
        if (rv == 0) {
            print_pid_ppid(0);
            for(int j=0; j<M; ++j) {
                rv2 = fork();                
                if (rv2 == 0){
                    print_pid_ppid(1);
                    exit(0); // finish child process
                }
                if (rv2 <0) {printf("Error in the fork2\n");}
            }
            for (int j=0;j<M;++j) {
                waitpid(-1,&stat,0);
            }
            exit(0); // finishes child process
        }
        if (rv < 0) {
            printf("Error in the fork\n");
        }
    }
    // We created all of the children
    // Now we wait for all of them to terminate
    for (int i=0;i<N;++i) {
                waitpid(-1,&stat,0);
    }
}        
```
- If a child process is not terminated appropriately (no exit(0)), then a dangling process will be made
- Demo:
```bash
$ ./a.out 2 3
N and M are 2 3
Level = 0, pid = 16775, ppid=16774
Level = 0, pid = 16776, ppid=16774
Level = 1, pid = 16777, ppid=16775
Level = 1, pid = 16778, ppid=16776
Level = 1, pid = 16779, ppid=16775
Level = 1, pid = 16780, ppid=16776
Level = 1, pid = 16781, ppid=16775
Level = 1, pid = 16782, ppid=16776
```

## Section 4: Signals

### 16. Introduction to Signals
- Signals: notification that an event has occurred
- Signals received by a process can be sent by the kernel or by other processes of the same user
- Types of signals and management
  - Each type of event has an associated signal
    - Types of events and associated signals are defined by the kernel
    - The type of signal is a number, but there exist constants that can be used inside programs or in the command line
    - There are two signals that are not associated to any event, so the programmer can assign any meaning to them -> SIGUSR1 SIGUSR2
  - Each process has associated management to each signal
    - A process can catch to all type of signal except SIGKILL and SIGSTOP
- Types of signals and management

| Name | Default management | Event |
|------|--------------------|-------|
| SIGCHLD | IGNORE | A child process has finished the execution or has been stopped |
| SIGCONT |      | To continue a stopped process|
| SIGSTOP | STOP | To stop a process |
| SIGINT  | END  | Interrupted from the keyboard (ctr+c)|
| SIGALRM | END | Timer programmed by alarm has expired |
|SIGKILL | END | Finish the process|
|SIGSEGV | CORE | Invalid memroy access |
|SIGUSR1 | END | Defined by the process |
|SIGUSR2 | END | Defined by the process |

- We use mainly signals for:
  - Process synchronization
  - Time control (alarms)

### 17. Signals Basic Interface

| Service               | System call |
|-----------------------|-------------|
| Send a signal         | kill        |
| Catch a signal        | sigaction   |
| Block/unblock signals | sigprocmask |
|Wait for a signal (this system call is blocking) | sigsuspend |
|Timer setting (to send a SIGALRM to the process) | alarm|

- Interface: sending and catching signals
  - sending: `int kill(int pid, int signum)`
    - signum: SIGUSR1, SIGURS2, etc
  - Catching signals: defining the management of the signal for the current process: `int sigaction(int signum, struct sigaction *mngt, struct siganction *old_mngt)`
- target.c:  
```c
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
int main(int argc, char * argv[])
{
  char buffer[80];
  sprintf(buffer, "I am PID: %d\n", getpid());
  write(1,buffer,strlen(buffer));
  sleep(100); // sleep for 100 sec
  printf("Hi");
  return 0;
}
```
- killer.c:
```c
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
int main(int argc, char * argv[])
{
  int pid = atoi(argv[1]);
  kill(pid,SIGSTOP);
  return 0;
}
```
- Demo:
```bash
$ ./a.out 
I am PID: 22454
# Now run ./killer 22454 from another terminal
[1]+  Stopped                 ./a.out
```

### 18. Sigaction Structure
- Definition of sigaction
  - Struct sigaction: several fields but we use only 3 fields in this course
  - sa_handler: can take 3 different values
    - SIG_IGN: ignore the signal when it is received
    - SIG_DFL: use the default management for this signal
    - User function with the following signature: void function_name(int s);
  - sa_mask: signals to add to the mask of blocked signals of the process just during the execution of the signal management
    - If this mask is empty, the only signal added is the signal catched by the sigaction system call
    - When the signal management ends, the mask of blocked signals used before the signal management is restarted
  - sa_flags: to configure the behaviour of the signal management (if this field is 0, the default configuration is used)
    - SA_RESETHAND: signal catching is only valid for one signal delivering. After managing one signal, the default signal management is restored
    - SA_RESTART: if a process blocked in a system call receives a signal, the blocking system call is restarted after the signal management

### 19. Signal Management
- Signal management is per process: the management information is in the PCB. Each process has:
  - A table of signals (1 entry per signal)
    - Action to perform when the signal is received
  - A bitmap of pending events (1 bit per signal)
    - It is not a counter, is like a boolean
  - A timer to manage alarms
    - If a process programs two timers, the only valid is the last one
  - A mask to indicate which signals the process want to receive and manage
- Sending and delivering
  - What really happens? The kernel offers the service to pass the information
- Signal mask management
  - sigemtyset: intializes a mask without any signal
  - sigfillset: initializes a mask with all signals
  - sigaddset: add a signal to the mask parameter
  - sigdelset: removes a signal from the mask paramter
  - sigismember: returns true if signal is set in mask
- Example: catching the signal  
```c
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
void f_sigint(int s) {
  char buffer[128];
  sprintf(buffer,"SIGINT has been received\n");
  write(1,buffer,strlen(buffer));
  exit(0);
}
void main() {
  struct sigaction manage;
  sigset_t mask;
  sigemptyset(&mask);
  manage.sa_mask = mask;
  manage.sa_flags = 0;
  manage.sa_handler = f_sigint;
  sigaction(SIGINT,&manage,NULL);
  char buffer[128];
  while(1) {
    sprintf(buffer,"Doing something\n");
    write(1,buffer,strlen(buffer));
  }
}
```
- Demo:
```bash
$ ./a.out
Doing something
...
Doing something # ctrl+c entered
^CSIGINT has been receive
```
- Example: sending the signal  
```c
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
void f_sigint(int s) {
  char buffer[128];
  sprintf(buffer,"SIGINT has been received\n");
  write(1,buffer,strlen(buffer));
  exit(0);
}
void main() {
  char buffer[128];
  struct sigaction manage;
  sigset_t mask;
  sigemptyset(&mask);
  manage.sa_mask = mask;
  manage.sa_flags = 0;
  manage.sa_handler = f_sigint;
  sigaction(SIGINT,&manage,NULL);
  sprintf(buffer,"I am sleeping\n");
  write(1,buffer,strlen(buffer));
  sleep(5);
  kill(getpid(),SIGINT);
  sprintf(buffer,"Message after SIGINT\n"); // Not printed as the process is terminated above
  write(1,buffer,strlen(buffer));
}
```
- Demo:
```bash
$ ./part19_2 
I am sleeping
SIGINT has been received
```
- What if the handler doesn't exit?
  - Instead of terminating process, let it continue:
```c
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
void f_sigint(int s) {
  char buffer[128];
  sprintf(buffer,"SIGINT has been received\n");
  write(1,buffer,strlen(buffer));
  //exit(0); // # even after SIGINT entered, the program continues to run
}
void main() {
  char buffer[128];
  struct sigaction manage;
  sigset_t mask;
  sigemptyset(&mask);
  manage.sa_mask = mask;
  manage.sa_flags = 0;
  manage.sa_handler = f_sigint;
  sigaction(SIGINT,&manage,NULL);
  sprintf(buffer,"I am sleeping\n");
  write(1,buffer,strlen(buffer));
  sleep(5);
  kill(getpid(),SIGINT);
  sprintf(buffer,"Message after SIGINT\n"); // Now print works
  write(1,buffer,strlen(buffer));
}
```
- Demo:
```bash
$ ./part19_3 
I am sleeping
SIGINT has been received
Message after SIGINT
```  
- Let's redo target vs killer above:
- target2.c:  
```c
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
void f_sigint(int s) {
  char buffer[128];
  sprintf(buffer,"SIGINT has been received\n");
  write(1,buffer,strlen(buffer));
  //exit(0);
}
void main() {
  char buffer[128];
  struct sigaction manage;
  sigset_t mask;
  sigemptyset(&mask);
  manage.sa_mask = mask;
  manage.sa_flags = 0;
  manage.sa_handler = f_sigint;
  sigaction(SIGINT,&manage,NULL);
  printf("my PID= %d\n", getpid());
  sprintf(buffer,"I am sleeping\n");
  write(1,buffer,strlen(buffer));
  sleep(10);
  sprintf(buffer,"Message after SIGINT\n");
  write(1,buffer,strlen(buffer));
}
```
- killer2.c:
```c
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
int main(int argc, char * argv[])
{
  int pid = atoi(argv[1]);
  kill(pid,SIGINT); // Note SIGINT, not SIGSTOP
  return 0;
}
```
- Demo:
```bash
$ ./target2 
my PID= 26389
I am sleeping # ./killer2 26389 from another terminal
SIGINT has been received
Message after SIGINT
$ ./target2
my PID= 9077
I am sleeping  # ctrl+c in the current terminal
^CSIGINT has been received
Message after SIGINT # Still runs through 
```

## Section 5: Laboratory 2 - Signals

### 20. Introduction to Laboratory 2
- Management of events as a mechanism for communication and synchronization b/w processes
- Be able to re-program/wait/send events using the Unix interface

| | |
|----------|-------------------------------------|
| sigaction | Reprogram the action associated to a particular event |
| kill (system call) | Send an event to a particular process (not necessarily terminate process!!!) |
| sigsuspend | Blocks the process that executes it until it receives a signal (the signals which treatment is to be ignored don't block the process) |
|sigprocmask | Modifies the mask of blocked signals of the calling process |
|alarm | Program the sending of a SIGALRM signal in N seconds |
| sleep | C library function that blocks the process during the time passed as a parameter|
| `/bin/kill` | Send an event to a process |
| `ps` | Shows information about the process in the system |
| waitpid | Waits for the finalization of a process |

```bash
$ kill -L
 1) SIGHUP	 2) SIGINT	 3) SIGQUIT	 4) SIGILL 5) SIGTRAP
 6) SIGABRT	 7) SIGBUS	 8) SIGFPE	 9) SIGKILL	10) SIGUSR1
11) SIGSEGV	12) SIGUSR2	13) SIGPIPE	14) SIGALRM	15) SIGTERM
16) SIGSTKFLT	17) SIGCHLD	18) SIGCONT	19) SIGSTOP	20) SIGTSTP
21) SIGTTIN	22) SIGTTOU	23) SIGURG	24) SIGXCPU	25) SIGXFSZ
26) SIGVTALRM	27) SIGPROF	28) SIGWINCH	29) SIGIO30) SIGPWR
31) SIGSYS	34) SIGRTMIN	35) SIGRTMIN+1	36) SIGRTMIN+2	37) SIGRTMIN+3
38) SIGRTMIN+4	39) SIGRTMIN+5	40) SIGRTMIN+6	41) SIGRTMIN+7	42) SIGRTMIN+8
43) SIGRTMIN+9	44) SIGRTMIN+10	45) SIGRTMIN+11	46) SIGRTMIN+12	47) SIGRTMIN+13
48) SIGRTMIN+14	49) SIGRTMIN+15	50) SIGRTMAX-14	51) SIGRTMAX-13	52) SIGRTMAX-12
53) SIGRTMAX-11	54) SIGRTMAX-10	55) SIGRTMAX-9	56) SIGRTMAX-8	57) SIGRTMAX-7
58) SIGRTMAX-6	59) SIGRTMAX-5	60) SIGRTMAX-4	61) SIGRTMAX-3	62) SIGRTMAX-2
63) SIGRTMAX-1	64) SIGRTMAX	
```

### 21. Problem 1
- alarm1.c
```c
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
/* THIS PROCESS SETS A TIMER IN 10 SECONDS AND BLOCKS TO WAIT FOR IT TO ARRIVE */
/* THE DEFAULT ACTION OF SIGALRM IS TO END THE PROCESS */
int
main (int argc, char *argv[])
{
  sigset_t mask;
  sigemptyset (&mask); // empty mask
  sigaddset (&mask, SIGALRM); // Adds SIGALRM
  sigprocmask (SIG_BLOCK, &mask, NULL); // blocks SIGALRM given above - stand-by mask before ALRM reaches
  alarm (10);  // alarm will reach in 10 sec but mask will block it as done above
  sigfillset (&mask); // fill all signals to mask, implying blocking all signals
  sigdelset (&mask, SIGALRM); // mask can receive SIGALRM now (removed from the blocked list)
  sigdelset (&mask, SIGINT);  // mask can receive SIGINT now (removed from the blocked list)
  sigsuspend (&mask); // wait for unblocked signals - SIGALRM or SIGINT
  exit (1);
}
```
- Execute alarm1 then what happens after 10sec?
- Run it then send SIGKILL signal from another terminal using `kill -9 <pid>`
- Which process is in charge of showing message when alarm1 process dies?
- Demo:
```bash
$ ./alarm1.exe # wait 10sec then:
Alarm clock
$ ./alarm1.exe  # kill -kill 11878 from another terminal
Killed
```

### 22. Problem 2

### 23. Problem 3

### 24. Problem 4
### 25. Problem 5
### 26. Problem 6
### 27. Problem 7
### 28. Problem 8
### 29. Problem 9
### 30. Problem 10
### 31. Problem 11
### 32. Problem 12

------------------------------------------------------------------

# Udemy: Building a Toy Graphical Operating System from Scratch
- By Muhammad Ahmad

## Section 1: Introduction

### 1. Introduction

### 2. Software Installation
- nasm, gcc, qemu, qemu-kvm
- sudo apt install qemu qemu-system
### 3. QEMU Windows

## Section 2: Bootloader

### 4. First Modified Hello Wrold Program
- os/boot/boot.asm
```asm
[org 0x00]
[bits 16]
section code
.main:
  mov eax, 0xb800
  mov es, eax
  mov byte [es:0x00], 'H' ; 0xb800 + 0x00 = 0xb800
  mov byte [es:0x01], 0x40 ; controls the color of the selected prompt
jmp $
times 510 - ($ - $$) db 0x00 ; Pads the file with 0s, making it the right size
db 0x55
db 0xaa
```
- os/boot/Makefile
```bash
all: bootloader
bootloader: 
  nasm boot/boot.asm -o boot/boot.img
clear:
  rm -f boot/boot.img
run:
  qemu-system-x86_64 -m 64 -fda ./boot/boot.img
```
- `make` wil produce os/boot/boot.img
- `make run`
![boot_screen](./sec4.png)
- ctrl+alt+g to release Grub

### 5. Clear the Screen
- os/boot/boot.asm
```asm
[org 0x00]
[bits 16]
section code
.init: 
  mov eax, 0xb800
  mov es, eax
  mov eax, 0; set eax to 0-> i=0
.clear: 
  mov byte [es:eax], 0; move blank character to current text address
  inc eax
  mov byte [es:eax], 0x30; move the bgcolor and char color to the next address
  inc eax
  cmp eax, 2*25*80
  jl .clear
.main:
  mov byte [es:0x00], 'H' ; 0xb800 + 0x00 = 0xb800, prints 'H'
  mov byte [es:0x01], 0x40 ; Red when prompted
jmp $
times 510 - ($ - $$) db 0x00 ; Pads the file with 0s, making it the right size
db 0x55
db 0xaa
```
![boot_screen_sec5](./sec5.png)
- ctrl+alt+g to release Grub

### 6. Making a Print Utility
```asm
[org 0x00]
[bits 16]
section code
.init: 
  mov eax, 0x07c0
  mov ds, eax
  mov eax, 0xb800
  mov es, eax
  mov eax, 0; set eax to 0-> i=0
  mov ebx, 0; index of the char in the string printed
  mov ecx, 0; actual address of the char on the screen
  mov dl, 0; store the actual val
.clear: 
  mov byte [es:eax], 0; move blank character to current text address
  inc eax
  mov byte [es:eax], 0x30; move the bgcolor and char color to the next address
  inc eax
  cmp eax, 2*25*80
  jl .clear
mov eax, text1
push .end
.print: 
  mov dl, byte [eax+ebx]
  cmp dl,0 ; check the end of text
  je .print_end
  mov byte [es:ecx], dl
  inc ebx
  inc ecx
  inc ecx
  jmp .print
.print_end:
  ret
.end:
  jmp $
text: db 'Hello world!', 0
text1 db 'This is another text!', 0
times 510 - ($ - $$) db 0x00 ; Pads the file with 0s, making it the right size
db 0x55
db 0xaa
```
![boot_screen_sec6](./sec6.png)
- ctrl+alt+g to release Grub

### 7. Improving Print Utility
```asm
[org 0x00]
[bits 16]
section .code
.init:
    mov eax, 0x07c0
    mov ds, eax
    mov eax, 0xb800
    mov es, eax
    mov eax, 0 ; set eax to 0 -> i = 0
    mov ebx, 0 ; Index of the character in the string that we are printing
    mov ecx, 0 ; Actual address of the chracter on the screen memory
    mov dl, 0 ; Actual value of the character that we are printing
.clear:
    mov byte [es:eax], 0
    inc eax
    mov byte [es:eax], 0xB0
    inc eax
    cmp eax, 2 * 25 * 80
    jl .clear
mov eax, .welcome
mov ecx, 0 * 2 * 80
push .end
call .print
.end:
    jmp $
.print:
    mov ebx, 0
.print_main:
    mov dl, byte [eax + ebx]
    cmp dl, 0
    je .print_end
    mov byte [es:ecx], dl
    inc ebx
    inc ecx
    inc ecx
    jmp .print_main
.print_end:
    ret
.welcome: db 'Welcome to SaphireOS.', 0
times 510 - ($ - $$) db 0x00 ; Pads the file with 0s, making the file the right size
db 0x55
db 0xaa
```

### 8. Switching to Protected Mode
- Using 32bit mode 
- Global Descriptor Table
  - A table of 8 byte entries, defining characteristics of the various memory areas
  - Ref: https://en.wikipedia.org/wiki/Global_Descriptor_Table#:~:text=The%20Global%20Descriptor%20Table%20(GDT,privileges%20like%20executability%20and%20writability.
```asm
[org 0x7c00]
[bits 16]
section code
.init:
    mov eax, 0xb800
    mov es, eax
    mov eax, 0 ; set eax to 0 -> i = 0
    mov ebx, 0 ; Index of the character in the string that we are printing
    mov ecx, 0 ; Actual address of the chracter on the screen memory
    mov dl, 0 ; Actual value of the character that we are printing
.clear:
    mov byte [es:eax], 0
    inc eax
    mov byte [es:eax], 0xB0
    inc eax
    cmp eax, 2 * 25 * 80
    jl .clear
mov eax, welcome
mov ecx, 0 * 2 * 80
call .print
jmp .switch
.print:
    mov ebx, 0
.print_main:
    mov dl, byte [eax + ebx]
    cmp dl, 0
    je .print_end    
    mov byte [es:ecx], dl
    inc ebx
    inc ecx
    inc ecx    
    jmp .print_main
.print_end:
    ret
.switch:
    cli ; Turh off the interrupts
    lgdt [gdt_descriptor] ; Load the GDT Table
    mov eax, cr0
    or eax, 0x1
    mov cr0, eax ; Make the switch
    jmp protected_start
welcome: db 'Welcome to SaphireOS.', 0
[bits 32]
protected_start:
    mov ax, data_seg
    mov ds, ax
    mov ss, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    ; Update the stack pointer
    mov ebp, 0x90000
    mov esp, ebp
    jmp $
gdt_begin:
gdt_null_descriptor:
    dd 0x00
    dd 0x00
gdt_code_seg:
    dw 0xffff
    dw 0x00
    db 0x00
    db 10011010b
    db 11001111b
    db 0x00
gdt_data_seg:
    dw 0xffff
    dw 0x00
    db 0x00
    db 10010010b
    db 11001111b
    db 0x00
gdt_end:
gdt_descriptor:
    dw gdt_end - gdt_begin - 1
    dd gdt_begin
code_seg equ gdt_code_seg - gdt_begin
data_seg equ gdt_data_seg - gdt_begin
times 510 - ($ - $$) db 0x00 ; Pads the file with 0s, making the file the right size
db 0x55
db 0xaa
```

### 9. Integrating C code
- boot.asm:
```asm
[org 0x7c00]
[bits 16]
section code
.switch:
    mov bx, 0x1000 ; This is the location where the code is loaded from hard disk
    mov ah, 0x02
    mov al, 30 ; The number of sectors to read from hard disk
    mov ch, 0x00
    mov dh, 0x00
    mov cl, 0x02
    int 0x13
    cli ; Turh off the interrupts
    lgdt [gdt_descriptor] ; Load the GDT Table
    mov eax, cr0
    or eax, 0x1
    mov cr0, eax ; Make the switch
    jmp code_seg:protected_start
welcome: db 'Welcome to SaphireOS.', 0
[bits 32]
protected_start:
    mov ax, data_seg
    mov ds, ax
    mov ss, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    ; Update the stack pointer
    mov ebp, 0x90000
    mov esp, ebp
    call 0x1000
    jmp $
gdt_begin:
gdt_null_descriptor:
    dd 0x00
    dd 0x00
gdt_code_seg:
    dw 0xffff
    dw 0x00
    db 0x00
    db 10011010b
    db 11001111b
    db 0x00
gdt_data_seg:
    dw 0xffff
    dw 0x00
    db 0x00
    db 10010010b
    db 11001111b
    db 0x00
gdt_end:
gdt_descriptor:
    dw gdt_end - gdt_begin - 1
    dd gdt_begin
code_seg equ gdt_code_seg - gdt_begin
data_seg equ gdt_data_seg - gdt_begin
times 510 - ($ - $$) db 0x00 ; Pads the file with 0s, making the file the right size
db 0x55
db 0xaa
```
- kernel_entry.asm
```asm
[bits 32]
START:
[extern start]
    call start
    jmp $
```
- main.c:
```c
char* video_memory;
int start() {
    video_memory = (char*) 0xb8000;
    for (int i = 0; i < 2 * 25 * 80; i += 2) {
        *(video_memory + i) = 0; // clearing screen
        *(video_memory + i + 1) = 0xB0;
    }
}
```
- Makefile
```bash
all: bootloader
bootloader:
  mkdir boot/bin
  nasm boot/boot.asm -f bin -o boot/bin/boot.bin
  nasm boot/kernel_entry.asm -f elf -o boot/bin/kernel_entry.bin	
  gcc -m32 -ffreestanding -c boot/main.c -o boot/bin/kernel.o
  ld -m elf_i386 -o boot/bin/kernel.img -Ttext 0x1000 boot/bin/kernel_entry.bin boot/bin/kernel.o
  objcopy -O binary -j .text boot/bin/kernel.img boot/bin/kernel.bin
  cat boot/bin/boot.bin boot/bin/kernel.bin > os.img
clear:
  rm -f os.img
run:
  qemu-system-x86_64 -drive format=raw,file=os.img
```
![boot_screen_sec9](./sec9.png)
- ctrl+alt+g to release Grub

## Section 3: Graphics

### 10. Switch to Graphics Mode and Clear the screen
- boot.asm
```asm
[org 0x7c00]
[bits 16]
section code
.switch:
    mov ax, 0x4f01 ; querying the VBE
    mov cx, 0x117 ; Mode we want
    mov bx, 0x0800 ; Offset for the vbe info structure
    mov es, bx
    mov di, 0x00
    int 0x10
    ; Make the switch to graphics mode
    mov ax, 0x4f02
    mov bx, 0x117
    int 0x10
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov bx, 0x1000 ; This is the location where the code is loaded from hard disk
    mov ah, 0x02
    mov al, 1 ; The number of sectors to read from hard disk
    mov ch, 0x00
    mov dh, 0x00
    mov cl, 0x02
    int 0x13
    cli ; Turh off the interrupts
    lgdt [gdt_descriptor] ; Load the GDT Table
    mov eax, cr0
    or eax, 0x1
    mov cr0, eax ; Make the switch
    jmp code_seg:protected_start
welcome: db 'Welcome to SaphireOS.', 0
[bits 32]
protected_start:
    mov ax, data_seg
    mov ds, ax
    mov ss, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    ; Update the stack pointer
    mov ebp, 0x90000
    mov esp, ebp
    call 0x1000
    jmp $
gdt_begin:
gdt_null_descriptor:
    dd 0x00
    dd 0x00
gdt_code_seg:
    dw 0xffff
    dw 0x00
    db 0x00
    db 10011010b
    db 11001111b
    db 0x00
gdt_data_seg:
    dw 0xffff
    dw 0x00
    db 0x00
    db 10010010b
    db 11001111b
    db 0x00
gdt_end:
gdt_descriptor:
    dw gdt_end - gdt_begin - 1
    dd gdt_begin
code_seg equ gdt_code_seg - gdt_begin
data_seg equ gdt_data_seg - gdt_begin
times 510 - ($ - $$) db 0x00 ; Pads the file with 0s, making the file the right size
db 0x55
db 0xaa
```
- kernet_entry.asm/Makefile are same as above
- main.c
```c
typedef struct VBEInfoBlockStruct {
    unsigned short mode_attribute;
    unsigned char win_a_attribute;
    unsigned char win_b_attribute;
    unsigned short win_granuality;
    unsigned short win_size;
    unsigned short win_a_segment;
    unsigned short win_b_segment;
    unsigned int win_func_ptr;
    unsigned short bytes_per_scan_line;
    unsigned short x_resolution;
    unsigned short y_resolution;
    unsigned char char_x_size;
    unsigned char char_y_size;
    unsigned char number_of_planes;
    unsigned char bits_per_pixel;
    unsigned char number_of_banks;
    unsigned char memory_model;
    unsigned char bank_size;
    unsigned char number_of_image_pages;
    unsigned char b_reserved;
    unsigned char red_mask_size;
    unsigned char red_field_position;
    unsigned char green_mask_size;
    unsigned char green_field_position;
    unsigned char blue_mask_size;
    unsigned char blue_field_position;
    unsigned char reserved_mask_size;
    unsigned char reserved_field_position;
    unsigned char direct_color_info;
    unsigned int screen_ptr;
} VBEInfoBlock;
#define VBEInfoAddress 0x8000
VBEInfoBlock* gVBE;
// R - 4 bits
// G - 5 bits
// B - 4 bits
int rgb(int r, int g, int b) {
    r = (int)(r / 3);
    g = (int)(g / 2);
    b = (int)(b / 3);
    return r << 11 | g << 5 | b;
}
int start() {
    gVBE = (VBEInfoBlock*) VBEInfoAddress;
    for (int i = 0; i < gVBE->x_resolution * gVBE->y_resolution; i++) {
        *((unsigned short*)gVBE->screen_ptr + i) = rgb(255, 0, 0);
    }
}
```
![boot_screen_sec10](./sec10.png)
- A graphic window, not text

### 11. Make Graphics Module
- graphics.h
```c
#ifndef GRAPHICS_H
#define GRAPHICS_H
typedef struct VBEInfoBlockStruct {
    unsigned short mode_attribute;
    unsigned char win_a_attribute;
    unsigned char win_b_attribute;
    unsigned short win_granuality;
    unsigned short win_size;
    unsigned short win_a_segment;
    unsigned short win_b_segment;
    unsigned int win_func_ptr;
    unsigned short bytes_per_scan_line;
    unsigned short x_resolution;
    unsigned short y_resolution;
    unsigned char char_x_size;
    unsigned char char_y_size;
    unsigned char number_of_planes;
    unsigned char bits_per_pixel;
    unsigned char number_of_banks;
    unsigned char memory_model;
    unsigned char bank_size;
    unsigned char number_of_image_pages;
    unsigned char b_reserved;
    unsigned char red_mask_size;
    unsigned char red_field_position;
    unsigned char green_mask_size;
    unsigned char green_field_position;
    unsigned char blue_mask_size;
    unsigned char blue_field_position;
    unsigned char reserved_mask_size;
    unsigned char reserved_field_position;
    unsigned char direct_color_info;
    unsigned int screen_ptr;
} VBEInfoBlock;
#define VBEInfoAddress 0x8000
int rgb(int r, int g, int b);
void Draw(int x, int y, int r, int g, int b);
void ClearScreen(int r, int g, int b);
void DrawRect(int x, int y, int width, int height, int r, int g, int b);
#endif
```
- graphics.c
```c
#include "graphics.h"
// R - 4 bits
// G - 5 bits
// B - 4 bits
int rgb(int r, int g, int b) {
    r = (int)(r / 3);
    g = (int)(g / 2);
    b = (int)(b / 3);
    return r << 11 | g << 5 | b;
}
void Draw(int x, int y, int r, int g, int b) {
    VBEInfoBlock* VBE = (VBEInfoBlock*) VBEInfoAddress;
    int index = y * VBE->x_resolution + x;
    *((unsigned short*)VBE->screen_ptr + index) = rgb(r, g, b);
}
void ClearScreen(int r, int g, int b) {
    VBEInfoBlock* VBE = (VBEInfoBlock*) VBEInfoAddress;
    for (int y = 0; y < VBE->y_resolution; y++) {
        for (int x = 0; x < VBE->x_resolution; x++) {
            Draw(x, y, r, g, b);
        }
    }
}
void DrawRect(int x, int y, int width, int height, int r, int g, int b) {
    for (int j = y; j < (y + height); j++) {
        for (int i = x; i < (x + width); i++) {
            Draw(i, j, r, g, b);
        }
    }
}
```
- final.c
```c
#include "graphics.c"
#include "main.c"
```
- main.c
```c
#include "graphics.h"
int start() {
  ClearScreen(255, 90, 255);
}
```
- Makefile
```bash
all: bootloader
bootloader:
  nasm boot/boot.asm -f bin -o boot/bin/boot.bin
  nasm boot/kernel_entry.asm -f elf -o boot/bin/kernel_entry.bin	
  gcc -m32 -ffreestanding -c boot/final.c -o boot/bin/kernel.o
  ld -m elf_i386 -o boot/bin/kernel.img -Ttext 0x1000 boot/bin/kernel_entry.bin boot/bin/kernel.o
  objcopy -O binary -j .text boot/bin/kernel.img boot/bin/kernel.bin
  cat boot/bin/boot.bin boot/bin/kernel.bin > os.img
clear:
  rm -f boot/boot.img
run:
  qemu-system-x86_64.exe -L "C:/Program Files/qemu" -drive format=raw,file=os.img
```

### 12. Printing Text 1

### 13. Printing Text 2

### 14. Printing Text 3

### 15. Printing a Sting of Text

### 16. Implementing Double Buffering

## Section 4: Mouse and Keyboard

### 17. Mouse Interrupts

### 18. Mouse Cursor

### 19. Keyboard

### 20. Fix String Issue

## Section 5: Task System

### 21. Implementing a Simple Task System

### 22. Fixing Shift Key Bug

### 23. Improving the Task System

## Section 6: Graphical Elements

### 24. Drawing-a-Window

### 25. Closing a Window

### 26. Making a Button

### 27. Fixing Mouse Possession

### 28. Fixing Mouse Possession 2

### 29. Fixing Mouse Possession 3

## Section 7: Taskbar and Bouncing Ball

### 30. Making a Taskbar

### 31. Bouncing Ball

----------------------------------------------------------

# Udemy: Write Your Own Operating System From Scratch - Step by Step
- Instructor: x-BIT Development

## References
- [1] https://www.cs.bham.ac.uk/~exr/lectures/opsys/10_11/lectures/os-dev.pdf
- [2] https://github.com/cfenollosa/os-tutorial

## Section 1: Introduction

### 1. Introduction
- Requirements
  - X86 assembly
  - C language

### 2. MUST SEE
- A test computer
  - CPU
  - No HDD
  - PS/2 keyboard
- A USB flash drive
  - To boot
  - Use Rufus

### 3. OS concepts
- Address space
  - 64bit address
  - Byte addressing
    - 0, 1, 2, ... 2^64-1
- Process
  - Kernel (code, data)
  - Stack
  - Heap
  - Data
  - Instructions
- When multiple rocesses run, each process has Kernel-stack-heap-data-instructions structure each
- Operation modes
  - Kernel mode
    - Interface w/ HW
  - User mode
    - Where application runs

### 4. OS structures
- Monolithic system
  - System call + scheduling + Memory management + Device drivers at Kernel space
  - Direct communication among modules
  - An error in a module will panic entire OS
- Microkernel
  - OS is divided into small modules
  - Memory management+Device Drivers are at user space
  - Only Scheduling and Interprocess communication at Kernel space
  - Kernel is very small

## Section 2: Installation and Setup

### 5. Working with Windows10
- Windows subsystem for Linux
  - Run Ubuntu 
  - Install nasm
- Bochs x86 PC emulator
  - Make a `boot.img`
  - Enable 1GB page memory
  - ATA channel 9 20/16/63 for cylindeerrs/heads/Sectors per track

### 6. Working with MacOS

### 7. Working with Linux (Ubuntu)
- gcc, nasm
- `sudo apt install bochs`
- `sudo apt install bochs-x`
- bximage: 1-> hd -> flat -> 512 -> 10 -> boot.img
- bochs
  - Enable 1G page 
```bash
$ bximage  
========================================================================
                                bximage
                  Disk Image Creation Tool for Bochs
          $Id: bximage.c 11315 2012-08-05 18:13:38Z vruppert $
========================================================================

Do you want to create a floppy disk image or a hard disk image?
Please type hd or fd. [hd] 

What kind of image should I create?
Please type flat, sparse or growing. [flat] 

Enter the hard disk size in megabytes, between 1 and 8257535
[10] 

I will create a 'flat' hard disk image with
  cyl=20
  heads=16
  sectors per track=63
  total sectors=20160
  total size=9.84 megabytes

What should I name the image?
[c.img] boot.img

Writing: [] Done.

I wrote 10321920 bytes to boot.img.

The following line should appear in your bochsrc:
  ata0-master: type=disk, path="boot.img", mode=flat, cylinders=20, heads=16, spt=63
$ bochs
========================================================================
                        Bochs x86 Emulator 2.6
            Built from SVN snapshot on September 2nd, 2012
========================================================================
00000000000i[     ] LTDL_LIBRARY_PATH not set. using compile time default '/usr/lib/bochs/plugins'
00000000000i[     ] BXSHARE not set. using compile time default '/usr/share/bochs'
00000000000i[     ] lt_dlhandle is 0x55ad64847c70
00000000000i[PLGIN] loaded plugin libbx_unmapped.so
00000000000i[     ] lt_dlhandle is 0x55ad648492e0
00000000000i[PLGIN] loaded plugin libbx_biosdev.so
00000000000i[     ] lt_dlhandle is 0x55ad6484ac50
00000000000i[PLGIN] loaded plugin libbx_speaker.so
00000000000i[     ] lt_dlhandle is 0x55ad6484b470
00000000000i[PLGIN] loaded plugin libbx_extfpuirq.so
00000000000i[     ] lt_dlhandle is 0x55ad6484bc50
00000000000i[PLGIN] loaded plugin libbx_parallel.so
00000000000i[     ] lt_dlhandle is 0x55ad6484d890
00000000000i[PLGIN] loaded plugin libbx_serial.so
00000000000i[     ] lt_dlhandle is 0x55ad64851640
00000000000i[PLGIN] loaded plugin libbx_gameport.so
00000000000i[     ] lt_dlhandle is 0x55ad64851f00
00000000000i[PLGIN] loaded plugin libbx_iodebug.so
00000000000e[     ] Switching off quick start, because no configuration file was found.
------------------------------
Bochs Configuration: Main Menu
------------------------------

This is the Bochs Configuration Interface, where you can describe the
machine that you want to simulate.  Bochs has already searched for a
configuration file (typically called bochsrc.txt) and loaded it if it
could be found.  When you are satisfied with the configuration, go
ahead and start the simulation.

You can also start bochs with the -q option to skip these menus.

1. Restore factory default configuration
2. Read options from...
3. Edit options
4. Save options to...
5. Restore the Bochs state from...
6. Begin simulation
7. Quit now

Please choose one: [2] 3
------------------
Bochs Options Menu
------------------
0. Return to previous menu
1. Optional plugin control
2. Logfile options
3. Log options for all devices
4. Log options for individual devices
5. CPU options
6. CPUID options
7. Memory options
8. Clock & CMOS options
9. PCI options
10. Bochs Display & Interface options
11. Keyboard & Mouse options
12. Disk & Boot options
13. Serial / Parallel / USB options
14. Network card options
15. Sound card options
16. Other options
17. User-defined options

Please choose one: [0] 6

-------------
CPUID Options
-------------
0. Return to previous menu
1. CPUID vendor string: GenuineIntel
2. CPUID brand string:               Intel(R) Pentium(R) 4 CPU        
3. Stepping ID: 3
4. Model ID: 3
5. Family ID: 6
6. Support for MMX instruction set: yes
7. APIC configuration: xapic
8. Support for SSE instruction set: sse2
9. Support for AMD SSE4A instructions: no
10. Support for SYSENTER/SYSEXIT instructions: yes
11. Support for MOVBE instruction: no
12. Support for ADX instructions: no
13. Support for AES instruction set: no
14. Support for XSAVE extensions: no
15. Support for XSAVEOPT instruction: no
16. Support for AVX instruction set: 0
17. Support for AVX F16 convert instructions: no
18. Support for AVX FMA instructions: no
19. Support for BMI instructions: 0
20. Support for AMD XOP instructions: no
21. Support for AMD four operand FMA instructions: no
22. Support for AMD TBM instructions: no
23. x86-64 and long mode: yes
24. 1G pages support in long mode: no
25. PCID support in long mode: no
26. FS/GS BASE access instructions support: no
27. Supervisor Mode Execution Protection support: no
28. MONITOR/MWAIT instructions support: yes
29. Support for Intel VMX extensions emulation: 1

Please choose one: [0] 24

1G pages support in long mode? [no] yes

-------------
CPUID Options
-------------
0. Return to previous menu
1. CPUID vendor string: GenuineIntel
2. CPUID brand string:               Intel(R) Pentium(R) 4 CPU        
3. Stepping ID: 3
4. Model ID: 3
5. Family ID: 6
6. Support for MMX instruction set: yes
7. APIC configuration: xapic
8. Support for SSE instruction set: sse2
9. Support for AMD SSE4A instructions: no
10. Support for SYSENTER/SYSEXIT instructions: yes
11. Support for MOVBE instruction: no
12. Support for ADX instructions: no
13. Support for AES instruction set: no
14. Support for XSAVE extensions: no
15. Support for XSAVEOPT instruction: no
16. Support for AVX instruction set: 0
17. Support for AVX F16 convert instructions: no
18. Support for AVX FMA instructions: no
19. Support for BMI instructions: 0
20. Support for AMD XOP instructions: no
21. Support for AMD four operand FMA instructions: no
22. Support for AMD TBM instructions: no
23. x86-64 and long mode: yes
24. 1G pages support in long mode: yes
25. PCID support in long mode: no
26. FS/GS BASE access instructions support: no
27. Supervisor Mode Execution Protection support: no
28. MONITOR/MWAIT instructions support: yes
29. Support for Intel VMX extensions emulation: 1

Please choose one: [0] 
------------------
Bochs Options Menu
------------------
0. Return to previous menu
1. Optional plugin control
2. Logfile options
3. Log options for all devices
4. Log options for individual devices
5. CPU options
6. CPUID options
7. Memory options
8. Clock & CMOS options
9. PCI options
10. Bochs Display & Interface options
11. Keyboard & Mouse options
12. Disk & Boot options
13. Serial / Parallel / USB options
14. Network card options
15. Sound card options
16. Other options
17. User-defined options

Please choose one: [0] 7

--------------
Memory Options
--------------
0. Return to previous menu
1. Standard Options
2. Optional ROM Images
3. Optional RAM Images

Please choose one: [0] 1

----------------
Standard Options
----------------
1. RAM size options
2. BIOS ROM options
3. VGABIOS ROM options

Please choose one: [0] 1

----------------
RAM size options
----------------

Enter memory size (MB): [32] 1024

Enter host memory size (MB): [32] 1024

----------------
Standard Options
----------------
1. RAM size options
2. BIOS ROM options
3. VGABIOS ROM options

Please choose one: [0] 

--------------
Memory Options
--------------
0. Return to previous menu
1. Standard Options
2. Optional ROM Images
3. Optional RAM Images

Please choose one: [0] 
------------------
Bochs Options Menu
------------------
0. Return to previous menu
1. Optional plugin control
2. Logfile options
3. Log options for all devices
4. Log options for individual devices
5. CPU options
6. CPUID options
7. Memory options
8. Clock & CMOS options
9. PCI options
10. Bochs Display & Interface options
11. Keyboard & Mouse options
12. Disk & Boot options
13. Serial / Parallel / USB options
14. Network card options
15. Sound card options
16. Other options
17. User-defined options

Please choose one: [0] 12

------------------
Bochs Disk Options
------------------
0. Return to previous menu
1. First Floppy Drive
2. Second Floppy Drive
3. ATA channel 0
4. First HD/CD on channel 0
5. Second HD/CD on channel 0
6. ATA channel 1
7. First HD/CD on channel 1
8. Second HD/CD on channel 1
9. ATA channel 2
10. First HD/CD on channel 2 (disabled)
11. Second HD/CD on channel 2 (disabled)
12. ATA channel 3
13. First HD/CD on channel 3 (disabled)
14. Second HD/CD on channel 3 (disabled)
15. Boot Options

Please choose one: [0] 3

-------------
ATA channel 0
-------------

Channel is enabled: [yes] 

Enter new ioaddr1: [0x1f0] 

Enter new ioaddr2: [0x3f0] 

Enter new IRQ: [14] 

------------------
Bochs Disk Options
------------------
0. Return to previous menu
1. First Floppy Drive
2. Second Floppy Drive
3. ATA channel 0
4. First HD/CD on channel 0
5. Second HD/CD on channel 0
6. ATA channel 1
7. First HD/CD on channel 1
8. Second HD/CD on channel 1
9. ATA channel 2
10. First HD/CD on channel 2 (disabled)
11. Second HD/CD on channel 2 (disabled)
12. ATA channel 3
13. First HD/CD on channel 3 (disabled)
14. Second HD/CD on channel 3 (disabled)
15. Boot Options

Please choose one: [0] 4

------------------------
First HD/CD on channel 0
------------------------

Device is enabled: [no] yes

Enter type of ATA device, disk or cdrom: [disk] 

Enter new filename: [] boot.img

Enter mode of ATA device, (flat, concat, etc.): [flat] 

Enter number of cylinders: [0] 20

Enter number of heads: [0] 16

Enter number of sectors per track: [0] 63

Enter new model name: [Generic 1234]

Enter bios detection type: [auto]

Enter translation type: [auto]

------------------
Bochs Disk Options
------------------
0. Return to previous menu
1. First Floppy Drive
2. Second Floppy Drive
3. ATA channel 0
4. First HD/CD on channel 0
5. Second HD/CD on channel 0
6. ATA channel 1
7. First HD/CD on channel 1
8. Second HD/CD on channel 1
9. ATA channel 2
10. First HD/CD on channel 2 (disabled)
11. Second HD/CD on channel 2 (disabled)
12. ATA channel 3
13. First HD/CD on channel 3 (disabled)
14. Second HD/CD on channel 3 (disabled)
15. Boot Options

Please choose one: [0] 15

------------
Boot Options
------------
0. Return to previous menu
1. Boot drive #1: floppy
2. Boot drive #2: none
3. Boot drive #3: none
4. Skip Floppy Boot Signature Check: no
5. 32-bit OS Loader Hack

Please choose one: [0] 1

Boot from floppy drive, hard drive or cdrom ? [floppy] disk

------------
Boot Options
------------
0. Return to previous menu
1. Boot drive #1: disk
2. Boot drive #2: none
3. Boot drive #3: none
4. Skip Floppy Boot Signature Check: no
5. 32-bit OS Loader Hack

Please choose one: [0] 

------------------
Bochs Disk Options
------------------
0. Return to previous menu
1. First Floppy Drive
2. Second Floppy Drive
3. ATA channel 0
4. First HD/CD on channel 0
5. Second HD/CD on channel 0
6. ATA channel 1
7. First HD/CD on channel 1
8. Second HD/CD on channel 1
9. ATA channel 2
10. First HD/CD on channel 2 (disabled)
11. Second HD/CD on channel 2 (disabled)
12. ATA channel 3
13. First HD/CD on channel 3 (disabled)
14. Second HD/CD on channel 3 (disabled)
15. Boot Options

Please choose one: [0] 
------------------
Bochs Options Menu
------------------
0. Return to previous menu
1. Optional plugin control
2. Logfile options
3. Log options for all devices
4. Log options for individual devices
5. CPU options
6. CPUID options
7. Memory options
8. Clock & CMOS options
9. PCI options
10. Bochs Display & Interface options
11. Keyboard & Mouse options
12. Disk & Boot options
13. Serial / Parallel / USB options
14. Network card options
15. Sound card options
16. Other options
17. User-defined options

Please choose one: [0] 
------------------------------
Bochs Configuration: Main Menu
------------------------------

This is the Bochs Configuration Interface, where you can describe the
machine that you want to simulate.  Bochs has already searched for a
configuration file (typically called bochsrc.txt) and loaded it if it
could be found.  When you are satisfied with the configuration, go
ahead and start the simulation.

You can also start bochs with the -q option to skip these menus.

1. Restore factory default configuration
2. Read options from...
3. Edit options
4. Save options to...
5. Restore the Bochs state from...
6. Begin simulation
7. Quit now

Please choose one: [6] 4
Save configuration to what file?  To cancel, type 'none'.
[none] bochsrc
00000000000i[     ] write current configuration to bochsrc
Wrote configuration to 'bochsrc'.
------------------------------
Bochs Configuration: Main Menu
------------------------------

This is the Bochs Configuration Interface, where you can describe the
machine that you want to simulate.  Bochs has already searched for a
configuration file (typically called bochsrc.txt) and loaded it if it
could be found.  When you are satisfied with the configuration, go
ahead and start the simulation.

You can also start bochs with the -q option to skip these menus.

1. Restore factory default configuration
2. Read options from...
3. Edit options
4. Save options to...
5. Restore the Bochs state from...
6. Begin simulation
7. Quit now

Please choose one: [6] 7
00000000000i[CTRL ] quit_sim called with exit code 1
```
- Produced bochsrc:
```bash
# configuration file generated by Bochs
plugin_ctrl: unmapped=1, biosdev=1, speaker=1, extfpuirq=1, parallel=1, serial=1
, gameport=1, iodebug=1
config_interface: textconfig
display_library: x
memory: host=1024, guest=1024
romimage: file="/usr/share/bochs/BIOS-bochs-latest"
vgaromimage: file="/usr/share/bochs/VGABIOS-lgpl-latest"
boot: disk
floppy_bootsig_check: disabled=0
# no floppya
# no floppyb
ata0: enabled=1, ioaddr1=0x1f0, ioaddr2=0x3f0, irq=14
ata0-master: type=disk, mode=flat, translation=auto, path="boot.img", cylinders=
20, heads=16, spt=63, biosdetect=auto, model="Generic 1234"
ata1: enabled=1, ioaddr1=0x170, ioaddr2=0x370, irq=15
ata2: enabled=0
ata3: enabled=0
pci: enabled=1, chipset=i440fx
vga: extension=vbe, update_freq=5
cpu: count=1, ips=4000000, model=bx_generic, reset_on_triple_fault=1, cpuid_limi
t_winnt=0, ignore_bad_msrs=1, mwait_is_nop=0
cpuid: family=6, model=0x03, stepping=3, mmx=1, apic=xapic, sse=sse2, sse4a=0, s
ep=1, aes=0, xsave=0, xsaveopt=0, movbe=0, adx=0, smep=0, avx=0, avx_f16c=0, avx
_fma=0, bmi=0, xop=0, tbm=0, fma4=0, vmx=1, x86_64=1, 1g_pages=1, pcid=0, fsgsba
se=0, mwait=1
cpuid: vendor_string="GenuineIntel"
cpuid: brand_string="              Intel(R) Pentium(R) 4 CPU        "

print_timestamps: enabled=0
debugger_log: -
magic_break: enabled=0
port_e9_hack: enabled=0
private_colormap: enabled=0
clock: sync=none, time0=local, rtc_sync=0
# no cmosimage
# no loader
log: -
logprefix: %t%e%d
panic: action=ask
error: action=report
info: action=report
debug: action=ignore
keyboard: type=mf, serial_delay=250, paste_delay=100000, keymap=
user_shortcut: keys=none
mouse: enabled=0, type=ps2, toggle=ctrl+mbutton
parport1: enabled=1, file=""
parport2: enabled=0
com1: enabled=1, mode=null, dev=""
com2: enabled=0
com3: enabled=0
com4: enabled=0
```

### 8. How to use Resources

## Section 3: Boot Up

### 9. The first program
- BIOS
  - Basic Input Output System
  - Services run in real mode
  - Print characters
  - Disk services
  - Memory map
  - Video mode
- Boot
  - Find boot device and read the first sector from it into memroy location 0x7c00 (MBR code)
  - Jump to 0x7c00
- BIOS mode
  - Real Mode: boot process
  - Protected Mode: prepares for long mode
  - Long Mode: 64bit mode. Compatibility mode
    - Cannot print
- UEFI
- Real Mode
  - Load Kernel
  - Retrieve information about HW
  - Sgement Registeres (16bit)
    - cs : Code Segment
    - ds : Data Segment
    - es : Extra Segment
    - ss : Stack Segment
  - Address Format
    - Segment Register: offset(logical address)
    - Segment Register x 16 + Offset = Physical address
  - General purpose registers
    - 8bit: al ah bl bh
    - 16bit: ax bx cx dx
    - 32bit: eax ebx ecx edx
    - 64bit: rax rbx rcx rdx are not available in Real Mode
```assembly
[BITS 16]
[ORG 0x7c00]
start:
    xor ax,ax   
    mov ds,ax
    mov es,ax  
    mov ss,ax
    mov sp,0x7c00 ; 
PrintMessage:
    mov ah,0x13  ; service 13 - print string. For tty mode, mov ah,0x0e
    mov al,1
    mov bx,0xa ; 0xa means the character is printed in bright screen
    xor dx,dx
    mov bp,Message
    mov cx,MessageLen 
    int 0x10  ; 
End:
    hlt    
    jmp End
Message:    db "Hello"
MessageLen: equ $-Message
times (0x1be-($-$$)) db 0
    db 80h      ; boot indicator
    db 0,2,0    ; starting CHS
    db 0f0h     ; type
    db 0ffh,0ffh,0ffh ; ending CHS
    dd 1        ; starting sector 
    dd (20*16*63-1)  ; size
    times (16*3) db 0
    db 0x55     ; signature
    db 0xaa	    ; signature
```
- [2]: `int 0x10` is a general interrupt for video services
- Build script (build.sh)
```bash
nasm -f bin -o boot.bin boot.asm
dd if=boot.bin of=boot.img bs=512 count=1 conv=notrunc
```

### 10. Testing on Windows 10

### 11. Testing on Linux (Ubuntu)
```bash
$ bash ./build.sh 
1+0 records in
1+0 records out
512 bytes copied, 0.000165192 s, 3.1 MB/s
$ bochs
...
<bochs:1> c # enter c to continue
```
![first_bochs](./bochs_11.png)
- "Hello" is printed
  - If `Message: dlopen failed for module 'x': file not found` is found at bochs, install bochs-x

- How to make a bootable USB to boot from a physical/actual computer:
```
$ sudo fdisk -l
$ sudo dd if=boot.img of=/dev/sdb bs=512 count=1
```

### 12. Testing on MacOS

### 13. Test Disk Extension Service
- [2]: When the computer boots, the BIOS doesn't know how to load the OS, so it delegates that task to the boot sector. Thus, the boot sector must be placed in a known, standard location. That location is the first sector of the disk (cylinder 0, head 0, sector 0) and it takes 512 bytes. To make sure that the "disk is bootable", the BIOS checks that bytes 511 and 512 of the alleged boot sector are bytes **0xAA55**.
- Adding the check of disk extension
```assembly
[BITS 16]
[ORG 0x7c00] ; global offset
; [2]  since offsetting 0x7c00 everywhere is very inconvenient, assemblers let us define a "global offset" for every memory location, with the ORG command
start:
    xor ax,ax   
    mov ds,ax
    mov es,ax  
    mov ss,ax
    mov sp,0x7c00
TestDiskExtension:
    mov [DriveId],dl
    mov ah,0x41 ; ref=https://www.ctyme.com/intr/rb-0706.htm
    mov bx,0x55aa
    int 0x13
    jc NotSupport
    cmp bx,0xaa55
    jne NotSupport
PrintMessage:
    mov ah,0x13
    mov al,1
    mov bx,0xa
    xor dx,dx
    mov bp,Message
    mov cx,MessageLen 
    int 0x10
NotSupport:
End:
    hlt    
    jmp End    
DriveId:    db 0
Message:    db "Disk extension is supported"
MessageLen: equ $-Message
times (0x1be-($-$$)) db 0
    db 80h
    db 0,2,0
    db 0f0h
    db 0ffh,0ffh,0ffh
    dd 1
    dd (20*16*63-1)	
    times (16*3) db 0
    db 0x55
    db 0xaa
```
- BIOS places memory at 0x7c00
- ./build.sh and bochs again

## Section 4: Loading the Loader and Switching to Long Mode

### 14. Loader
- Loader retrieves information about HW
- Prepares for 64-bit mode and switch to it
- Loader loads kernel in main memroy
- Jump to kernel
- Loader doesn't have 512 limit
- MBR code at 0x7c00
- Loader at 0x7e00
- Ref: https://www.ctyme.com/intr/rb-0708.htm
- boot.asm:
```assembly
[BITS 16]
[ORG 0x7c00]
start:
    xor ax,ax   
    mov ds,ax
    mov es,ax  
    mov ss,ax
    mov sp,0x7c00
TestDiskExtension:
    mov [DriveId],dl
    mov ah,0x41
    mov bx,0x55aa
    int 0x13
    jc NotSupport
    cmp bx,0xaa55
    jne NotSupport
LoadLoader:
    mov si,ReadPacket     ; offset field
    mov word[si],0x10     ; 0      size
    mov word[si+2],5      ; 2      number of sectors
    mov word[si+4],0x7e00 ; 4      offset
    mov word[si+6],0      ; 6      segment
    mov dword[si+8],1     ; 8      address lo
    mov dword[si+0xc],0   ; 12     address hi
    mov dl,[DriveId]
    mov ah,0x42
    int 0x13
    jc  ReadError
    mov dl,[DriveId]
    jmp 0x7e00 
ReadError:
NotSupport:
    mov ah,0x13
    mov al,1
    mov bx,0xa
    xor dx,dx
    mov bp,Message
    mov cx,MessageLen 
    int 0x10
End:
    hlt    
    jmp End    
DriveId:    db 0
Message:    db "We have an error in boot process"
MessageLen: equ $-Message
ReadPacket: times 16 db 0
times (0x1be-($-$$)) db 0
    db 80h
    db 0,2,0
    db 0f0h
    db 0ffh,0ffh,0ffh
    dd 1
    dd (20*16*63-1)	
    times (16*3) db 0
    db 0x55
    db 0xaa
```
- loader.asm:
  - DriveId from boot.asm must be delivered to loader.asm
```assembly
[BITS 16]
[ORG 0x7e00]
start:
    mov ah,0x13
    mov al,1
    mov bx,0xa
    xor dx,dx
    mov bp,Message
    mov cx,MessageLen 
    int 0x10    
End:
    hlt
    jmp End
Message:    db "loader starts"
MessageLen: equ $-Message
```
- build.sh
```bash
nasm -f bin -o boot.bin boot.asm
nasm -f bin -o loader.bin loader.asm
dd if=boot.bin of=boot.img bs=512 count=1 conv=notrunc
dd if=loader.bin of=boot.img bs=512 count=5 seek=1 conv=notrunc
```
- Build image
```bash
$ bash build.sh
1+0 records in
1+0 records out
512 bytes copied, 9.4243e-05 s, 5.4 MB/s
0+1 records in
0+1 records out
33 bytes copied, 7.9545e-05 s, 415 kB/s
```

### 15. Long Mode Support
- Checks if the CPU supports Long Mode
- loader.asm:
```assembly
[BITS 16]
[ORG 0x7e00]
start:
    mov [DriveId],dl
    mov eax,0x80000000
    cpuid
    cmp eax,0x80000001
    jb NotSupport
    mov eax,0x80000001
    cpuid
    test edx,(1<<29)
    jz NotSupport
    test edx,(1<<26)
    jz NotSupport
    mov ah,0x13
    mov al,1
    mov bx,0xa
    xor dx,dx
    mov bp,Message
    mov cx,MessageLen 
    int 0x10
NotSupport:
End:
    hlt
    jmp End
DriveId:    db 0
Message:    db "long mode is supported"
MessageLen: equ $-Message
```

### 16. Load Kernel File
- Memory map
```
Free (reserved for HW)
--------- 0x100000
Reserved
--------- 0x80000
Kernel
--------- 0x10000
Free (reserved for HW)
Loader
--------- 0x7e00
Boot
--------- 0x7c00
Free
BIOS data Vectors
--------- 0
```
- loader.asm:
```assembly
[BITS 16]
[ORG 0x7e00]
start:
    mov [DriveId],dl
    mov eax,0x80000000
    cpuid
    cmp eax,0x80000001
    jb NotSupport
    mov eax,0x80000001
    cpuid
    test edx,(1<<29)
    jz NotSupport
    test edx,(1<<26)
    jz NotSupport
LoadKernel:
    mov si,ReadPacket
    mov word[si],0x10
    mov word[si+2],100
    mov word[si+4],0       ; offset
    mov word[si+6],0x1000  ; Segment 0x1000:0 = 0x1000*16+0 = 0x100000 Kernel memory map shown above
    mov dword[si+8],6
    mov dword[si+0xc],0
    mov dl,[DriveId]
    mov ah,0x42
    int 0x13
    jc  ReadError
    mov ah,0x13
    mov al,1
    mov bx,0xa
    xor dx,dx
    mov bp,Message
    mov cx,MessageLen 
    int 0x10
ReadError:
NotSupport:
End:
    hlt
    jmp End
DriveId:    db 0
Message:    db "kernel is loaded"
MessageLen: equ $-Message
ReadPacket: times 16 db 0
```

### 17. Get Memory Map
- Ref: https://www.ctyme.com/intr/rb-1741.htm
- loader.asm:
```assembly
[BITS 16]
[ORG 0x7e00]
start:
    mov [DriveId],dl
    mov eax,0x80000000
    cpuid
    cmp eax,0x80000001
    jb NotSupport
    mov eax,0x80000001
    cpuid
    test edx,(1<<29)
    jz NotSupport
    test edx,(1<<26)
    jz NotSupport
LoadKernel:
    mov si,ReadPacket
    mov word[si],0x10
    mov word[si+2],100
    mov word[si+4],0
    mov word[si+6],0x1000
    mov dword[si+8],6
    mov dword[si+0xc],0
    mov dl,[DriveId]
    mov ah,0x42
    int 0x13
    jc  ReadError
GetMemInfoStart:
    mov eax,0xe820
    mov edx,0x534d4150
    mov ecx,20
    mov edi,0x9000
    xor ebx,ebx
    int 0x15
    jc NotSupport
GetMemInfo:
    add edi,20
    mov eax,0xe820
    mov edx,0x534d4150
    mov ecx,20
    int 0x15
    jc GetMemDone
    test ebx,ebx
    jnz GetMemInfo
GetMemDone:
    mov ah,0x13
    mov al,1
    mov bx,0xa
    xor dx,dx
    mov bp,Message
    mov cx,MessageLen 
    int 0x10
ReadError:
NotSupport:
End:
    hlt
    jmp End
DriveId:    db 0
Message:    db "Get memory info done"
MessageLen: equ $-Message
ReadPacket: times 16 db 0
```
- We don't print the detailed memory map yet as print function is not implemented. Check Section 7 to print

### 18. Test A20 Line
- Ref: https://en.wikipedia.org/wiki/A20_line
   - For legacy BIOS boot loader
   - UEFI boot loaders use 32 bit protected mode or 64bit long mode

### 19. Set Video Mode
- Use ASCII code and set foreground/background color
- loader.asm:
```asm
[BITS 16]
[ORG 0x7e00]
start:
    mov [DriveId],dl
    mov eax,0x80000000
    cpuid
    cmp eax,0x80000001
    jb NotSupport
    mov eax,0x80000001
    cpuid
    test edx,(1<<29)
    jz NotSupport
    test edx,(1<<26)
    jz NotSupport
LoadKernel:
    mov si,ReadPacket
    mov word[si],0x10
    mov word[si+2],100
    mov word[si+4],0
    mov word[si+6],0x1000
    mov dword[si+8],6
    mov dword[si+0xc],0
    mov dl,[DriveId]
    mov ah,0x42
    int 0x13
    jc  ReadError
GetMemInfoStart:
    mov eax,0xe820
    mov edx,0x534d4150
    mov ecx,20
    mov edi,0x9000
    xor ebx,ebx
    int 0x15
    jc NotSupport
GetMemInfo:
    add edi,20
    mov eax,0xe820
    mov edx,0x534d4150
    mov ecx,20
    int 0x15
    jc GetMemDone
    test ebx,ebx
    jnz GetMemInfo
GetMemDone:
TestA20:
    mov ax,0xffff
    mov es,ax
    mov word[ds:0x7c00],0xa200
    cmp word[es:0x7c10],0xa200
    jne SetA20LineDone
    mov word[0x7c00],0xb200
    cmp word[es:0x7c10],0xb200
    je End
SetA20LineDone:
    xor ax,ax
    mov es,ax
SetVideoMode:
    mov ax,3
    int 0x10        
    mov si,Message
    mov ax,0xb800
    mov es,ax
    xor di,di
    mov cx,MessageLen
PrintMessage:
    mov al,[si]
    mov [es:di],al
    mov byte[es:di+1],0xa
    add di,2
    add si,1
    loop PrintMessage
ReadError:
NotSupport:
End:
    hlt
    jmp End
DriveId:    db 0
Message:    db "Text mode is set"
MessageLen: equ $-Message
ReadPacket: times 16 db 0
```
![video_bochs](./bochs_19.png)
- Printed without BIOS service

### 20. Protected Mode
- Entering protected mode
  - Load GDT
  - Load IDT
  - Enable Protected Mode
  - Jump to PM Entry
- GDT (Global Descriptor Table)
  - Segment registers: cs ds es ss
- IDT (Interrupt Descriptor Table)
  - Up to 256 entries
  - Handling keyboard, input, ...
- 4 levels of privilege
  - Level 0: kernel
  - Level 1/2:
  - Level 3: user program
- How to figure out the current privilege level?
  - Current Privilege Level: stored in cs and ss
  - Descriptor Privilege LeveL: stored in descriptors
  - Requested Privilege Level: stored in segment selectors
- Descriptor entry
  - Code segment
  - Data segment
- loader.asm:
```assembly
[BITS 16]
[ORG 0x7e00]
start:
    mov [DriveId],dl
    mov eax,0x80000000
    cpuid
    cmp eax,0x80000001
    jb NotSupport
    mov eax,0x80000001
    cpuid
    test edx,(1<<29)
    jz NotSupport
    test edx,(1<<26)
    jz NotSupport
LoadKernel:
    mov si,ReadPacket
    mov word[si],0x10
    mov word[si+2],100
    mov word[si+4],0
    mov word[si+6],0x1000
    mov dword[si+8],6
    mov dword[si+0xc],0
    mov dl,[DriveId]
    mov ah,0x42
    int 0x13
    jc  ReadError
GetMemInfoStart:
    mov eax,0xe820
    mov edx,0x534d4150
    mov ecx,20
    mov edi,0x9000
    xor ebx,ebx
    int 0x15
    jc NotSupport
GetMemInfo:
    add edi,20
    mov eax,0xe820
    mov edx,0x534d4150
    mov ecx,20
    int 0x15
    jc GetMemDone
    test ebx,ebx
    jnz GetMemInfo
GetMemDone:
TestA20:
    mov ax,0xffff
    mov es,ax
    mov word[ds:0x7c00],0xa200
    cmp word[es:0x7c10],0xa200
    jne SetA20LineDone
    mov word[0x7c00],0xb200
    cmp word[es:0x7c10],0xb200
    je End    
SetA20LineDone:
    xor ax,ax
    mov es,ax
SetVideoMode:
    mov ax,3
    int 0x10    
    cli
    lgdt [Gdt32Ptr]
    lidt [Idt32Ptr]
    mov eax,cr0
    or eax,1
    mov cr0,eax
    jmp 8:PMEntry  ; The index of the selector is 8
                   ; 00001000(binary)
                   ; index T RPL
ReadError:
NotSupport:
End:
    hlt
    jmp End
[BITS 32]
PMEntry:
    mov ax,0x10
    mov ds,ax
    mov es,ax
    mov ss,ax
    mov esp,0x7c00
    mov byte[0xb8000],'P'
    mov byte[0xb8001],0xa
PEnd:
    hlt
    jmp PEnd    
DriveId:    db 0
ReadPacket: times 16 db 0
Gdt32:
    dq 0
Code32:         ; Code segment
    dw 0xffff
    dw 0
    db 0
    db 0x9a
    db 0xcf
    db 0
Data32:         ; Data segment
    dw 0xffff
    dw 0
    db 0
    db 0x92
    db 0xcf
    db 0    
Gdt32Len: equ $-Gdt32
Gdt32Ptr: dw Gdt32Len-1
          dd Gdt32
Idt32Ptr: dw 0
          dd 0
```
- Both of code segment and data segment are necessary

### 21. Long Mode
- 64 bit mode and compatibility mode
  - OS runs in 64-bit mode
  - OS kernel and applications run in 64-bit mode
- Entering Long mode
  - Load GDT
  - Load IDT
  - Prepare for paging and enable it
  - Enable long mode
  - Jump to LM entry
- Global Descriptor Table
  - Segment registeres: cs ds es ss
- Descriptor entry
  - Code segment
  ![code_segment](./long_mode_code_seg.png)
  - Data segment
  ![data_segment](./long_mode_data_seg.png)
- Canonical address: the bits 63 through to the most-significant implemented bit are either all ones or all zeros
- loader.asm:
```assembly
[BITS 16]
[ORG 0x7e00]
start:
    mov [DriveId],dl
    mov eax,0x80000000
    cpuid
    cmp eax,0x80000001
    jb NotSupport
    mov eax,0x80000001
    cpuid
    test edx,(1<<29)
    jz NotSupport
    test edx,(1<<26)
    jz NotSupport
LoadKernel:
    mov si,ReadPacket
    mov word[si],0x10
    mov word[si+2],100
    mov word[si+4],0
    mov word[si+6],0x1000
    mov dword[si+8],6
    mov dword[si+0xc],0
    mov dl,[DriveId]
    mov ah,0x42
    int 0x13
    jc  ReadError
GetMemInfoStart:
    mov eax,0xe820
    mov edx,0x534d4150
    mov ecx,20
    mov edi,0x9000
    xor ebx,ebx
    int 0x15
    jc NotSupport
GetMemInfo:
    add edi,20
    mov eax,0xe820
    mov edx,0x534d4150
    mov ecx,20
    int 0x15
    jc GetMemDone
    test ebx,ebx
    jnz GetMemInfo
GetMemDone:
TestA20:
    mov ax,0xffff
    mov es,ax
    mov word[ds:0x7c00],0xa200
    cmp word[es:0x7c10],0xa200
    jne SetA20LineDone
    mov word[0x7c00],0xb200
    cmp word[es:0x7c10],0xb200
    je End    
SetA20LineDone:
    xor ax,ax
    mov es,ax
SetVideoMode:
    mov ax,3
    int 0x10
    cli
    lgdt [Gdt32Ptr]
    lidt [Idt32Ptr]
    mov eax,cr0
    or eax,1
    mov cr0,eax
    jmp 8:PMEntry
ReadError:
NotSupport:
End:
    hlt
    jmp End
[BITS 32]
PMEntry:
    mov ax,0x10
    mov ds,ax
    mov es,ax
    mov ss,ax
    mov esp,0x7c00
    cld
    mov edi,0x70000
    xor eax,eax
    mov ecx,0x10000/4
    rep stosd    
    mov dword[0x70000],0x71007
    mov dword[0x71000],10000111b
    lgdt [Gdt64Ptr]
    mov eax,cr4
    or eax,(1<<5)
    mov cr4,eax
    mov eax,0x70000
    mov cr3,eax
    mov ecx,0xc0000080
    rdmsr
    or eax,(1<<8)
    wrmsr
    mov eax,cr0
    or eax,(1<<31)
    mov cr0,eax
    jmp 8:LMEntry
PEnd:
    hlt
    jmp PEnd
[BITS 64]
LMEntry:
    mov rsp,0x7c00
    mov byte[0xb8000],'L'
    mov byte[0xb8001],0xa
LEnd:
    hlt
    jmp LEnd    
DriveId:    db 0
ReadPacket: times 16 db 0
Gdt32:
    dq 0
Code32:
    dw 0xffff
    dw 0
    db 0
    db 0x9a
    db 0xcf
    db 0
Data32:
    dw 0xffff
    dw 0
    db 0
    db 0x92
    db 0xcf
    db 0    
Gdt32Len: equ $-Gdt32
Gdt32Ptr: dw Gdt32Len-1
          dd Gdt32
Idt32Ptr: dw 0
          dd 0
Gdt64:
    dq 0
    dq 0x0020980000000000
Gdt64Len: equ $-Gdt64
Gdt64Ptr: dw Gdt64Len-1
          dd Gdt64
```

## Section 5: Exceptions and Interrupts Handling on the x86

### 22. Jumping to Kernel
- loader.asm
```asm
[BITS 16]
[ORG 0x7e00]
start:
    mov [DriveId],dl
    mov eax,0x80000000
    cpuid
    cmp eax,0x80000001
    jb NotSupport
    mov eax,0x80000001
    cpuid
    test edx,(1<<29)
    jz NotSupport
    test edx,(1<<26)
    jz NotSupport
LoadKernel:
    mov si,ReadPacket
    mov word[si],0x10
    mov word[si+2],100
    mov word[si+4],0
    mov word[si+6],0x1000
    mov dword[si+8],6
    mov dword[si+0xc],0
    mov dl,[DriveId]
    mov ah,0x42
    int 0x13
    jc  ReadError
GetMemInfoStart:
    mov eax,0xe820
    mov edx,0x534d4150
    mov ecx,20
    mov edi,0x9000
    xor ebx,ebx
    int 0x15
    jc NotSupport
GetMemInfo:
    add edi,20
    mov eax,0xe820
    mov edx,0x534d4150
    mov ecx,20
    int 0x15
    jc GetMemDone
    test ebx,ebx
    jnz GetMemInfo
GetMemDone:
TestA20:
    mov ax,0xffff
    mov es,ax
    mov word[ds:0x7c00],0xa200
    cmp word[es:0x7c10],0xa200
    jne SetA20LineDone
    mov word[0x7c00],0xb200
    cmp word[es:0x7c10],0xb200
    je End    
SetA20LineDone:
    xor ax,ax
    mov es,ax
SetVideoMode:
    mov ax,3
    int 0x10    
    cli
    lgdt [Gdt32Ptr]
    lidt [Idt32Ptr]
    mov eax,cr0
    or eax,1
    mov cr0,eax
    jmp 8:PMEntry
ReadError:
NotSupport:
End:
    hlt
    jmp End
[BITS 32]
PMEntry:
    mov ax,0x10
    mov ds,ax
    mov es,ax
    mov ss,ax
    mov esp,0x7c00
    cld
    mov edi,0x70000
    xor eax,eax
    mov ecx,0x10000/4
    rep stosd
    mov dword[0x70000],0x71007
    mov dword[0x71000],10000111b
    lgdt [Gdt64Ptr]
    mov eax,cr4
    or eax,(1<<5)
    mov cr4,eax
    mov eax,0x70000
    mov cr3,eax
    mov ecx,0xc0000080
    rdmsr
    or eax,(1<<8)
    wrmsr
    mov eax,cr0
    or eax,(1<<31)
    mov cr0,eax
    jmp 8:LMEntry
PEnd:
    hlt
    jmp PEnd
[BITS 64]
LMEntry:
    mov rsp,0x7c00
    cld
    mov rdi,0x200000 ; destination
    mov rsi,0x10000  ; source
    mov rcx,51200/8  ; this is counter
    rep movsq        ; copy 
    jmp 0x200000     ; we jump to kernel
LEnd:
    hlt
    jmp LEnd    
DriveId:    db 0
ReadPacket: times 16 db 0
Gdt32:
    dq 0
Code32:
    dw 0xffff
    dw 0
    db 0
    db 0x9a
    db 0xcf
    db 0
Data32:
    dw 0xffff
    dw 0
    db 0
    db 0x92
    db 0xcf
    db 0    
Gdt32Len: equ $-Gdt32
Gdt32Ptr: dw Gdt32Len-1
          dd Gdt32
Idt32Ptr: dw 0
          dd 0
Gdt64:
    dq 0
    dq 0x0020980000000000
Gdt64Len: equ $-Gdt64
Gdt64Ptr: dw Gdt64Len-1
          dd Gdt64
```
- kernel.asm:
```asm
[BITS 64]
[ORG 0x200000]
start:
    mov byte[0xb8000],'K'
    mov byte[0xb8001],0xa
End:
    hlt
    jmp End
```
- build.sh:
```bash
nasm -f bin -o boot.bin boot.asm
nasm -f bin -o loader.bin loader.asm
nasm -f bin -o kernel.bin kernel.asm
dd if=boot.bin of=boot.img bs=512 count=1 conv=notrunc
dd if=loader.bin of=boot.img bs=512 count=5 seek=1 conv=notrunc
dd if=kernel.bin of=boot.img bs=512 count=100 seek=6 conv=notrunc
```
- command:
    - bash build.sh
    - Enter c in the prompt
    - Find K in the bochs terminal

### 23. Reload GDT
- kernel.asm
```asm
[BITS 64]
[ORG 0x200000]
start:
    ; cs selector: 0x7c00
    ; offset     : RSP
    ;
    lgdt [Gdt64Ptr]
    push 8
    push KernelEntry
    db 0x48
    retf
KernelEntry:
    mov byte[0xb8000],'K'
    mov byte[0xb8001],0xa
End:
    hlt
    jmp End
Gdt64:
    dq 0
    dq 0x0020980000000000
Gdt64Len: equ $-Gdt64
Gdt64Ptr: dw Gdt64Len-1
          dq Gdt64
```
- lgdt/lidt: Load Global/Interrupt Descriptor Table Register
    - Ref: https://www.felixcloutier.com/x86/lgdt:lidt
- retf: far return. Return to the same privilege level

### 24. Exceptions and Interrupts Handling
- Interrupts: HW interrupts
- Exceptions: internal errors

![interrupt_handling](./interrupt_handling.png)
![interrupt_handling2](./interrupt_handling2.png)

- IST is not used in the lecture
- DPL in attributes will be edited
- There are 256 different interrupt handling

![interrupt_handling3](./interrupt_handling3.png)

- kernel.asm:
```asm
[BITS 64]
[ORG 0x200000]
start:
    ;  63....32  31...16  15...0  bit
    ;  xxxxxxxx  xxxxxxx  xxxxxx  handler0 address
    mov rdi,Idt       ; IDT, interrupt handler
    mov rax,Handler0
    mov [rdi],ax     ; first 2 bytes (15...0)
    shr rax,16        
    mov [rdi+6],ax   ; 31...16
    shr rax,16
    mov [rdi+8],eax  ; 63...32
    lgdt [Gdt64Ptr]
    lidt [IdtPtr]
    push 8
    push KernelEntry
    db 0x48
    retf
KernelEntry:
    mov byte[0xb8000],'K'
    mov byte[0xb8001],0xa
    xor rbx,rbx
    div rbx   ; yields divide by zero to make an interrupt
End:
    hlt
    jmp End
Handler0:
    mov byte[0xb8000],'D'  ; We print 'D' in the screen
    mov byte[0xb8001],0xc  ; character in red
    jmp End
    iretq
Gdt64:
    dq 0
    dq 0x0020980000000000
Gdt64Len: equ $-Gdt64
Gdt64Ptr: dw Gdt64Len-1  ; 2byte
          dq Gdt64       ; 8byte
Idt:
    %rep 256  ; repeat 256 times
        dw 0
        dw 0x8  ; attribute field 6byte
        db 0
                 ; P DPL TYPE 
        db 0x8e  ; 1 00  01110 
        dw 0
        dd 0
        dd 0
    %endrep
IdtLen: equ $-Idt
IdtPtr: dw IdtLen-1 ; 2byte
        dq Idt      ; 8byte
```
- Kernel code runs -> divide by zero -> Handler0 is activated

### 25. Saving Registers
- kernel.asm:
```asm
[BITS 64]
[ORG 0x200000]
start:
    mov rdi,Idt
    mov rax,Handler0
    mov [rdi],ax
    shr rax,16
    mov [rdi+6],ax
    shr rax,16
    mov [rdi+8],eax
    lgdt [Gdt64Ptr]
    lidt [IdtPtr]
    push 8
    push KernelEntry
    db 0x48
    retf
KernelEntry:
    mov byte[0xb8000],'K'
    mov byte[0xb8001],0xa
    xor rbx,rbx
    div rbx
End:
    hlt
    jmp End
Handler0:
    push rax
    push rbx  
    push rcx
    push rdx  	  
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15    
    mov byte[0xb8000],'D'
    mov byte[0xb8001],0xc
    jmp End
    pop	r15
    pop	r14
    pop	r13
    pop	r12
    pop	r11
    pop	r10
    pop	r9
    pop	r8
    pop	rbp
    pop	rdi
    pop	rsi  
    pop	rdx
    pop	rcx
    pop	rbx
    pop	rax
    iretq
Gdt64:
    dq 0
    dq 0x0020980000000000
Gdt64Len: equ $-Gdt64
Gdt64Ptr: dw Gdt64Len-1
          dq Gdt64
Idt:
    %rep 256
        dw 0
        dw 0x8
        db 0
        db 0x8e
        dw 0
        dd 0
        dd 0
    %endrep
IdtLen: equ $-Idt
IdtPtr: dw IdtLen-1
        dq Idt
```

### 26. Setting Up the Interrupt Controller
- HW interrupt
    - The programmable interrupt controller
        - Manages the interrupt requests (IRQ)
    - The programmable interval timer
        - Periodic timer
- kernel.asm:
```asm
[BITS 64]
[ORG 0x200000]
start:
    mov rdi,Idt
    mov rax,Handler0
    mov [rdi],ax
    shr rax,16
    mov [rdi+6],ax
    shr rax,16
    mov [rdi+8],eax
    mov rax,Timer
    add rdi,32*16
    mov [rdi],ax
    shr rax,16
    mov [rdi+6],ax
    shr rax,16
    mov [rdi+8],eax
    lgdt [Gdt64Ptr]
    lidt [IdtPtr]
    push 8
    push KernelEntry
    db 0x48
    retf
KernelEntry:
    ; 7 6 5 4 3 2 1 0 bit
    ; 00   11  010  0
    mov byte[0xb8000],'K'
    mov byte[0xb8001],0xa
InitPIT:
    mov al,(1<<2)|(3<<4) ; 1193182/100 = 11931
    out 0x43,al
    mov ax,11931
    out 0x40,al
    mov al,ah
    out 0x40,al
InitPIC:
    mov al,0x11
    out 0x20,al
    out 0xa0,al
    ; Initialization Commmand word 2
    ; 7 6 5 4 3 2 1 0 bit
    ; 0 0 1 0 0 0 0 0 = 32
    mov al,32
    out 0x21,al
    mov al,40
    out 0xa1,al
    ; Initialization Commmand word 3
    ; 7 6 5 4 3 2 1 0 bit
    ; 0 0 0 0 0 1 0 0 = 4
    ; 0 0 0 0 0 0 1 0 = 2
    mov al,4
    out 0x21,al
    mov al,2
    out 0xa1,al
    ; Initialization Commmand word 4
    ; 7 6 5 4 3 2 1 0 bit
    ; 0 0 0 0 0 0 0 1
    mov al,1
    out 0x21,al
    out 0xa1,al
    mov al,11111110b
    out 0x21,al
    mov al,11111111b
    out 0xa1,al
    sti
End:
    hlt
    jmp End
Handler0:
    push rax
    push rbx  
    push rcx
    push rdx  	  
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15
    mov byte[0xb8000],'D'
    mov byte[0xb8001],0xc
    jmp End
    pop	r15
    pop	r14
    pop	r13
    pop	r12
    pop	r11
    pop	r10
    pop	r9
    pop	r8
    pop	rbp
    pop	rdi
    pop	rsi  
    pop	rdx
    pop	rcx
    pop	rbx
    pop	rax
    iretq
Timer:
    push rax
    push rbx  
    push rcx
    push rdx  	  
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15
    mov byte[0xb8020],'T'
    mov byte[0xb8021],0xe
    jmp End   
    pop	r15
    pop	r14
    pop	r13
    pop	r12
    pop	r11
    pop	r10
    pop	r9
    pop	r8
    pop	rbp
    pop	rdi
    pop	rsi  
    pop	rdx
    pop	rcx
    pop	rbx
    pop	rax
    iretq
Gdt64:
    dq 0
    dq 0x0020980000000000
Gdt64Len: equ $-Gdt64
Gdt64Ptr: dw Gdt64Len-1
          dq Gdt64
Idt:
    %rep 256
        dw 0
        dw 0x8
        db 0
        db 0x8e
        dw 0
        dd 0
        dd 0
    %endrep
IdtLen: equ $-Idt
IdtPtr: dw IdtLen-1
        dq Idt
```

### 27. Getting to Ring3
- How to jump from Ring0 to Ring3
    - Lower 2bit of cs register
- Ref: https://stackoverflow.com/questions/18717016/what-are-ring-0-and-ring-3-in-the-context-of-operating-systems
    - CPU is in one of 4 rings
    - Linux kernel only uses 0 and 3
        - 0 for kernel. Can do anything
        - 3 for users
```asm
[BITS 64]
[ORG 0x200000]
start:
    mov rdi,Idt
    mov rax,Handler0
    mov [rdi],ax
    shr rax,16
    mov [rdi+6],ax
    shr rax,16
    mov [rdi+8],eax
    mov rax,Timer
    add rdi,32*16
    mov [rdi],ax
    shr rax,16
    mov [rdi+6],ax
    shr rax,16
    mov [rdi+8],eax
    lgdt [Gdt64Ptr]
    lidt [IdtPtr]
    push 8
    push KernelEntry
    db 0x48
    retf
KernelEntry:
    mov byte[0xb8000],'K'
    mov byte[0xb8001],0xa
    ; High Address: ss Selector
    ;               RSP
    ;               Rflags
    ;               cs selector
    ; Low Adddress: RIP <= RSP
InitPIT:
    mov al,(1<<2)|(3<<4)
    out 0x43,al
    mov ax,11931
    out 0x40,al
    mov al,ah
    out 0x40,al
InitPIC:
    mov al,0x11
    out 0x20,al
    out 0xa0,al
    mov al,32
    out 0x21,al
    mov al,40
    out 0xa1,al
    mov al,4
    out 0x21,al
    mov al,2
    out 0xa1,al
    mov al,1
    out 0x21,al
    out 0xa1,al
    mov al,11111110b
    out 0x21,al
    mov al,11111111b
    out 0xa1,al
    push 0x18|3
    push 0x7c00
    push 0x2
    push 0x10|3
    push UserEntry
    iretq
End:
    hlt
    jmp End
UserEntry:
    mov ax,cs
    and al,11b
    cmp al,3
    jne UEnd
    mov byte[0xb8010],'U'
    mov byte[0xb8011],0xE
UEnd:
    jmp UEnd
Handler0:
    push rax
    push rbx  
    push rcx
    push rdx  	  
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15    
    mov byte[0xb8000],'D'
    mov byte[0xb8001],0xc
    jmp End
    pop	r15
    pop	r14
    pop	r13
    pop	r12
    pop	r11
    pop	r10
    pop	r9
    pop	r8
    pop	rbp
    pop	rdi
    pop	rsi  
    pop	rdx
    pop	rcx
    pop	rbx
    pop	rax
    iretq
Timer:
    push rax
    push rbx  
    push rcx
    push rdx  	  
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15
    mov byte[0xb8020],'T'
    mov byte[0xb8021],0xe
    jmp End
    pop	r15
    pop	r14
    pop	r13
    pop	r12
    pop	r11
    pop	r10
    pop	r9
    pop	r8
    pop	rbp
    pop	rdi
    pop	rsi  
    pop	rdx
    pop	rcx
    pop	rbx
    pop	rax
    iretq
Gdt64:
    dq 0                   ; D L P DPL 1 1 C
    dq 0x0020980000000000  ; 0 1 1  00 1 1 0
    dq 0x0020f80000000000  ; 0 1 1  11 1 1 0
                           ; P DPL 1 0 0 W 0
    dq 0x0000f20000000000  ; 1  11 1 0 0 1 0
Gdt64Len: equ $-Gdt64
Gdt64Ptr: dw Gdt64Len-1
          dq Gdt64
Idt:
    %rep 256
        dw 0
        dw 0x8
        db 0
        db 0x8e
        dw 0
        dd 0
        dd 0
    %endrep
IdtLen: equ $-Idt
IdtPtr: dw IdtLen-1
        dq Idt
```

### 28. Interrupts Handling in Ring3 Part I
- kernel.asm:
```asm
[BITS 64]
[ORG 0x200000]
start:
    mov rdi,Idt
    mov rax,Handler0
    mov [rdi],ax
    shr rax,16
    mov [rdi+6],ax
    shr rax,16
    mov [rdi+8],eax
    mov rax,Timer
    add rdi,32*16
    mov [rdi],ax
    shr rax,16
    mov [rdi+6],ax
    shr rax,16
    mov [rdi+8],eax
    lgdt [Gdt64Ptr]
    lidt [IdtPtr]
SetTss:
    mov rax,Tss
    mov [TssDesc+2],ax
    shr rax,16
    mov [TssDesc+4],al
    shr rax,8
    mov [TssDesc+7],al
    shr rax,8
    mov [TssDesc+8],eax
    mov ax,0x20
    ltr ax
    push 8
    push KernelEntry
    db 0x48
    retf
KernelEntry:
    mov byte[0xb8000],'K'
    mov byte[0xb8001],0xa
InitPIT:
    mov al,(1<<2)|(3<<4)
    out 0x43,al
    mov ax,11931
    out 0x40,al
    mov al,ah
    out 0x40,al
InitPIC:
    mov al,0x11
    out 0x20,al
    out 0xa0,al
    mov al,32
    out 0x21,al
    mov al,40
    out 0xa1,al
    mov al,4
    out 0x21,al
    mov al,2
    out 0xa1,al
    mov al,1
    out 0x21,al
    out 0xa1,al
    mov al,11111110b
    out 0x21,al
    mov al,11111111b
    out 0xa1,al
    ;sti
    push 0x18|3
    push 0x7c00
    push 0x202
    push 0x10|3
    push UserEntry
    iretq
End:
    hlt
    jmp End
UserEntry:
    mov ax,cs
    and al,11b
    cmp al,3
    jne UEnd
    mov byte[0xb8010],'U'
    mov byte[0xb8011],0xE
UEnd:
    jmp UEnd
Handler0:
    push rax
    push rbx  
    push rcx
    push rdx  	  
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15    
    mov byte[0xb8000],'D'
    mov byte[0xb8001],0xc
    jmp End
    pop	r15
    pop	r14
    pop	r13
    pop	r12
    pop	r11
    pop	r10
    pop	r9
    pop	r8
    pop	rbp
    pop	rdi
    pop	rsi  
    pop	rdx
    pop	rcx
    pop	rbx
    pop	rax
    iretq
Timer:
    push rax
    push rbx  
    push rcx
    push rdx  	  
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15
    mov byte[0xb8020],'T'
    mov byte[0xb8021],0xe
    jmp End
    pop	r15
    pop	r14
    pop	r13
    pop	r12
    pop	r11
    pop	r10
    pop	r9
    pop	r8
    pop	rbp
    pop	rdi
    pop	rsi  
    pop	rdx
    pop	rcx
    pop	rbx
    pop	rax
    iretq
Gdt64:
    dq 0
    dq 0x0020980000000000
    dq 0x0020f80000000000
    dq 0x0000f20000000000
TssDesc:
    dw TssLen-1
    dw 0
    db 0
    db 0x89
    db 0
    db 0
    dq 0
Gdt64Len: equ $-Gdt64
Gdt64Ptr: dw Gdt64Len-1
          dq Gdt64
Idt:
    %rep 256
        dw 0
        dw 0x8
        db 0
        db 0x8e
        dw 0
        dd 0
        dd 0
    %endrep
IdtLen: equ $-Idt
IdtPtr: dw IdtLen-1
        dq Idt
Tss:
    dd 0
    dq 0x150000
    times 88 db 0
    dd TssLen
TssLen: equ $-Tss
```
![HW_interrupt](./hw_interrupt.png)

### 29. Interrupts Handling in Ring3 Part II

### 30. Spurious Interrupt Handling
- Not real interrupt

## Section 6: Working with C

### 31. Kernel main
- build.sh:
```bash
nasm -f bin -o boot.bin boot.asm
nasm -f bin -o loader.bin loader.asm
nasm -f elf64 -o kernel.o kernel.asm
gcc -std=c99 -mcmodel=large -ffreestanding -fno-stack-protector -mno-red-zone -c main.c 
ld -nostdlib -T link.lds -o kernel kernel.o main.o
objcopy -O binary kernel kernel.bin 
dd if=boot.bin of=boot.img bs=512 count=1 conv=notrunc
dd if=loader.bin of=boot.img bs=512 count=5 seek=1 conv=notrunc
dd if=kernel.bin of=boot.img bs=512 count=100 seek=6 conv=notrunc
```
- link.lds:
    - Ref: https://bravegnu.org/gnu-eprog/lds.html
```
OUTPUT_FORMAT("elf64-x86-64")
ENTRY(start)
SECTIONS
{
    . = 0x200000;
    .text : {
        *(.text)
    }
    .rodata : {
        *(.rodata)
    }
    . = ALIGN(16);
    .data : {
        *(.data)
    }
    .bss : {
        *(.bss)
    }
}
```
- kernel.asm:
```
section .data
Gdt64:
    dq 0
    dq 0x0020980000000000
    dq 0x0020f80000000000
    dq 0x0000f20000000000
TssDesc:
    dw TssLen-1
    dw 0
    db 0
    db 0x89
    db 0
    db 0
    dq 0
Gdt64Len: equ $-Gdt64
Gdt64Ptr: dw Gdt64Len-1
          dq Gdt64
Tss:
    dd 0
    dq 0x190000
    times 88 db 0
    dd TssLen
TssLen: equ $-Tss
section .text
extern KMain
global start
start:
    lgdt [Gdt64Ptr]
SetTss:
    mov rax,Tss
    mov [TssDesc+2],ax
    shr rax,16
    mov [TssDesc+4],al
    shr rax,8
    mov [TssDesc+7],al
    shr rax,8
    mov [TssDesc+8],eax
    mov ax,0x20
    ltr ax
InitPIT:
    mov al,(1<<2)|(3<<4)
    out 0x43,al
    mov ax,11931
    out 0x40,al
    mov al,ah
    out 0x40,al
InitPIC:
    mov al,0x11
    out 0x20,al
    out 0xa0,al
    mov al,32
    out 0x21,al
    mov al,40
    out 0xa1,al
    mov al,4
    out 0x21,al
    mov al,2
    out 0xa1,al
    mov al,1
    out 0x21,al
    out 0xa1,al
    mov al,11111110b
    out 0x21,al
    mov al,11111111b
    out 0xa1,al
    push 8
    push KernelEntry
    db 0x48
    retf
KernelEntry:
    mov rsp,0x200000
    call KMain
End:
    hlt
    jmp End
```
- main.c
```c
void KMain(void)
{
    char* p = (char*)0xb8000; 
    p[0] = 'C';
    p[1] = 0xa;
}
```
- Text mode video memory is a buffer located at 0xB8000 for color monitor and 0xB00000 for monochrome
    - Ref: http://kernelx.weebly.com/text-console.html
- Color code: https://gist.github.com/Pwootage/1396569
    - 0xa is green
 
### 32. Putting it all together
- System V AMD64 calling convention
    - The first 6 parameters -> RDI RSI RDX RCX R8 R9 others are on the stack
    - Return value -> RAX
    - RAX RCX RDX RSI RDI R8 R9 R10 R11 Caller saved registers
    - RBX RBP R12 R13 R14 R15 Callee saved registers
- trap.h
```c
#ifndef _TRAP_H_
#define _TRAP_H_
#include "stdint.h"
struct IdtEntry {
    uint16_t low; /* bottom 2bytes as offset */
    uint16_t selector;
    uint8_t res0;
    uint8_t attr;
    uint16_t mid;
    uint32_t high;
    uint32_t res1;
};
struct IdtPtr {
    uint16_t limit;
    uint64_t addr;
} __attribute__((packed));
struct TrapFrame {
    int64_t r15;
    int64_t r14;
    int64_t r13;
    int64_t r12;
    int64_t r11;
    int64_t r10;
    int64_t r9;
    int64_t r8;
    int64_t rbp;
    int64_t rdi;
    int64_t rsi;
    int64_t rdx;
    int64_t rcx;
    int64_t rbx;
    int64_t rax;
    int64_t trapno;
    int64_t errorcode;
    int64_t rip;
    int64_t cs;
    int64_t rflags;
    int64_t rsp;
    int64_t ss;
};
void vector0(void);
void vector1(void);
void vector2(void);
void vector3(void);
void vector4(void);
void vector5(void);
void vector6(void);
void vector7(void);
void vector8(void);
void vector10(void);
void vector11(void);
void vector12(void);
void vector13(void);
void vector14(void);
void vector16(void);
void vector17(void);
void vector18(void);
void vector19(void);
void vector32(void);
void vector39(void);
void init_idt(void);
void eoi(void);
void load_idt(struct IdtPtr *ptr);
unsigned char read_isr(void);
#endif
```
- Using `__attribute__((packed))`
    - No insertion of padding
    - Ref: https://stackoverflow.com/questions/8568432/is-gccs-attribute-packed-pragma-pack-unsafe
- trap.asm
```asm
section .text
extern handler
global vector0
global vector1
global vector2
global vector3
global vector4
global vector5
global vector6
global vector7
global vector8
global vector10
global vector11
global vector12
global vector13
global vector14
global vector16
global vector17
global vector18
global vector19
global vector32
global vector39
global eoi
global read_isr
global load_idt
Trap:
    push rax
    push rbx  
    push rcx
    push rdx  	  
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15
    inc byte[0xb8010]
    mov byte[0xb8011],0xe
    mov rdi,rsp  ; first parameter
    call handler ; from C code
TrapReturn:
    pop	r15
    pop	r14
    pop	r13
    pop	r12
    pop	r11
    pop	r10
    pop	r9
    pop	r8
    pop	rbp
    pop	rdi
    pop	rsi  
    pop	rdx
    pop	rcx
    pop	rbx
    pop	rax       
    add rsp,16   ; traps here
    iretq
vector0:
    push 0
    push 0
    jmp Trap
vector1:
    push 0
    push 1
    jmp Trap
vector2:
    push 0
    push 2
    jmp Trap
vector3:
    push 0
    push 3	
    jmp Trap 
vector4:
    push 0
    push 4	
    jmp Trap   
vector5:
    push 0
    push 5
    jmp Trap    
vector6:
    push 0
    push 6	
    jmp Trap      
vector7:
    push 0
    push 7	
    jmp Trap  
vector8:
    ; no 0 push as it will be done from C code
    push 8
    jmp Trap  
; vector9 is reserved
vector10:
    push 10	
    jmp Trap                    
vector11:
    push 11	
    jmp Trap    
vector12:
    push 12	
    jmp Trap                    
vector13:
    push 13	
    jmp Trap    
vector14:
    push 14	
    jmp Trap
; vector15 is reserved 
vector16:
    push 0
    push 16	
    jmp Trap                    
vector17:
    push 17	
    jmp Trap                         
vector18:
    push 0
    push 18	
    jmp Trap                   
vector19:
    push 0
    push 19	
    jmp Trap
vector32:
    push 0
    push 32
    jmp Trap
vector39:
    push 0
    push 39
    jmp Trap
eoi:
    mov al,0x20  ; processing the interrupt 
    ; ref: https://stackoverflow.com/questions/14848645/what-does-this-x86-assembly-language-code-with-io-ports-0x61-and-0x20-do
    out 0x20,al
    ret
read_isr:
    mov al,11
    out 0x20,al
    in al,0x20
    ret
load_idt:       ; no idt command in C so we write in asm
    lidt [rdi]
    ret
```
- trap.c
```c
#include "trap.h"
static struct IdtPtr idt_pointer;
static struct IdtEntry vectors[256];
static void init_idt_entry(struct IdtEntry *entry, uint64_t addr, uint8_t attribute)
{
    entry->low = (uint16_t)addr;
    entry->selector = 8;
    entry->attr = attribute;
    entry->mid = (uint16_t)(addr>>16);
    entry->high = (uint32_t)(addr>>32);
}
void init_idt(void)
{
    init_idt_entry(&vectors[0],(uint64_t)vector0,0x8e);
    init_idt_entry(&vectors[1],(uint64_t)vector1,0x8e);
    init_idt_entry(&vectors[2],(uint64_t)vector2,0x8e);
    init_idt_entry(&vectors[3],(uint64_t)vector3,0x8e);
    init_idt_entry(&vectors[4],(uint64_t)vector4,0x8e);
    init_idt_entry(&vectors[5],(uint64_t)vector5,0x8e);
    init_idt_entry(&vectors[6],(uint64_t)vector6,0x8e);
    init_idt_entry(&vectors[7],(uint64_t)vector7,0x8e);
    init_idt_entry(&vectors[8],(uint64_t)vector8,0x8e);
    init_idt_entry(&vectors[10],(uint64_t)vector10,0x8e);
    init_idt_entry(&vectors[11],(uint64_t)vector11,0x8e);
    init_idt_entry(&vectors[12],(uint64_t)vector12,0x8e);
    init_idt_entry(&vectors[13],(uint64_t)vector13,0x8e);
    init_idt_entry(&vectors[14],(uint64_t)vector14,0x8e);
    init_idt_entry(&vectors[16],(uint64_t)vector16,0x8e);
    init_idt_entry(&vectors[17],(uint64_t)vector17,0x8e);
    init_idt_entry(&vectors[18],(uint64_t)vector18,0x8e);
    init_idt_entry(&vectors[19],(uint64_t)vector19,0x8e);
    init_idt_entry(&vectors[32],(uint64_t)vector32,0x8e);
    init_idt_entry(&vectors[39],(uint64_t)vector39,0x8e);
    idt_pointer.limit = sizeof(vectors)-1;
    idt_pointer.addr = (uint64_t)vectors;
    load_idt(&idt_pointer);
}
void handler(struct TrapFrame *tf)
{
    unsigned char isr_value;
    switch (tf->trapno) {
        case 32:
            eoi();
            break;            
        case 39:
            isr_value = read_isr();
            if ((isr_value&(1<<7)) != 0) {
                eoi();
            }
            break;
        default:
            while (1) { }
    }
}
```
- kernel.asm
```asm
section .data
Gdt64:
    dq 0
    dq 0x0020980000000000
    dq 0x0020f80000000000
    dq 0x0000f20000000000
TssDesc:
    dw TssLen-1
    dw 0
    db 0
    db 0x89
    db 0
    db 0
    dq 0
Gdt64Len: equ $-Gdt64
Gdt64Ptr: dw Gdt64Len-1
          dq Gdt64
Tss:
    dd 0
    dq 0x190000
    times 88 db 0
    dd TssLen
TssLen: equ $-Tss
section .text
extern KMain
global start
start:
    lgdt [Gdt64Ptr]
SetTss:
    mov rax,Tss
    mov [TssDesc+2],ax
    shr rax,16
    mov [TssDesc+4],al
    shr rax,8
    mov [TssDesc+7],al
    shr rax,8
    mov [TssDesc+8],eax
    mov ax,0x20
    ltr ax
InitPIT:
    mov al,(1<<2)|(3<<4)
    out 0x43,al
    mov ax,11931
    out 0x40,al
    mov al,ah
    out 0x40,al
InitPIC:
    mov al,0x11
    out 0x20,al
    out 0xa0,al
    mov al,32
    out 0x21,al
    mov al,40
    out 0xa1,al
    mov al,4
    out 0x21,al
    mov al,2
    out 0xa1,al
    mov al,1
    out 0x21,al
    out 0xa1,al
    mov al,11111110b
    out 0x21,al
    mov al,11111111b
    out 0xa1,al
    push 8
    push KernelEntry
    db 0x48
    retf
KernelEntry: 
    xor ax,ax
    mov ss,ax
    
    mov rsp,0x200000
    call KMain
    sti
End:
    hlt
    jmp End
```
- build.sh
```
nasm -f bin -o boot.bin boot.asm
nasm -f bin -o loader.bin loader.asm
nasm -f elf64 -o kernel.o kernel.asm
nasm -f elf64 -o trapa.o trap.asm
gcc -std=c99 -mcmodel=large -ffreestanding -fno-stack-protector -mno-red-zone -c main.c 
gcc -std=c99 -mcmodel=large -ffreestanding -fno-stack-protector -mno-red-zone -c trap.c 
ld -nostdlib -T link.lds -o kernel kernel.o main.o trapa.o trap.o
objcopy -O binary kernel kernel.bin 
dd if=boot.bin of=boot.img bs=512 count=1 conv=notrunc
dd if=loader.bin of=boot.img bs=512 count=5 seek=1 conv=notrunc
dd if=kernel.bin of=boot.img bs=512 count=100 seek=6 conv=notrunc
```

### 33. Simple library functions

### 34. Print function

### 35. Assertion
- debug.h
```c
#ifndef _DEBUG_H_
#define _DEBUG_H_
#include "stdint.h"
#define ASSERT(e) do {                      \
        if (!(e))                           \
            error_check(__FILE__,__LINE__); \
} while (0) 
void error_check(char *file, uint64_t line);
#endif
```
- debug.c
```c
void error_check(char *file, uint64_t line)
{
    printk("\n------------------------------------------\n");
    printk("             ERROR CHECK");
    printk("\n------------------------------------------\n");
    printk("Assertion Failed [%s:%u]", file, line);
    while (1) { } /* infinite loop */
}
```

## Section 7: Memory Management

### 36. Retrieve Memory Map

![memory_map](./memory_map.png)

- In loader.asm:
```asm
GetMemInfoStart:
    mov eax,0xe820
    mov edx,0x534d4150
    mov ecx,20
    mov dword[0x9000],0
    mov edi,0x9008
    xor ebx,ebx
    int 0x15
    jc NotSupport
GetMemInfo:
    add edi,20
    inc dword[0x9000] 
    test ebx,ebx
    jz GetMemDone
-----------------------------Ti    mov eax,0xe820
    mov edx,0x534d4150
    mov ecx,20
    int 0x15
    jnc GetMemInfo
```
- memory.h:
```c
#ifndef _MEMORY_H_
#define _MEMORY_H_
#include "stdint.h"
struct E820 {
    uint64_t address;
    uint64_t length;
    uint32_t type;
} __attribute__((packed));
struct FreeMemRegion {
    uint64_t address;
    uint64_t length;
};
void init_memory(void);
#endif
```
- memory.c:
```c
#include "memory.h"
#include "print.h"
#include "debug.h"
static struct FreeMemRegion free_mem_region[50];
void init_memory(void)
{
    int32_t count = *(int32_t*)0x9000;
    uint64_t total_mem = 0;
    struct E820 *mem_map = (struct E820*)0x9008;	
    int free_region_count = 0;
    ASSERT(count <= 50);
  for(int32_t i = 0; i < count; i++) {        
        if(mem_map[i].type == 1) {
            free_mem_region[free_region_count].address = mem_map[i].address;
            free_mem_region[free_region_count].length = mem_map[i].length;
            total_mem += mem_map[i].length;
            free_region_count++;
        }        
        printk("%x  %uKB  %u\n", mem_map[i].address, mem_map[i].length/1024, (uint64_t)mem_map[i].type);
  }
    printk("Total memory is %uMB\n", total_mem/1024/1024);
}
```

### 37. Paging
![new_memory_map](./new_memory_map.png)
- Mapping virtual page into physical memory page

### 38. Memory Allocator
- Collects free memory based on physical memory, not virtual

### 39. Memory Pages

### 40. Free Memory Page
```c
#include "memory.h"
#include "print.h"
#include "debug.h"
#include "lib.h"
#include "stddef.h"
#include "stdbool.h"
static void free_region(uint64_t v, uint64_t e);
static struct FreeMemRegion free_mem_region[50];
static struct Page free_memory;
static uint64_t memory_end;
uint64_t page_map;
extern char end;
void init_memory(void)
{
    int32_t count = *(int32_t*)0x9000;
    uint64_t total_mem = 0;
    struct E820 *mem_map = (struct E820*)0x9008;	
    int free_region_count = 0;
    ASSERT(count <= 50);
  for(int32_t i = 0; i < count; i++) {        
        if(mem_map[i].type == 1) {			
            free_mem_region[free_region_count].address = mem_map[i].address;
            free_mem_region[free_region_count].length = mem_map[i].length;
            total_mem += mem_map[i].length;
            free_region_count++;
        }
        printk("%x  %uKB  %u\n",mem_map[i].address,mem_map[i].length/1024,(uint64_t)mem_map[i].type);
  }
    for (int i = 0; i < free_region_count; i++) {                  
        uint64_t vstart = P2V(free_mem_region[i].address);
        uint64_t vend = vstart + free_mem_region[i].length;
        if (vstart > (uint64_t)&end) {
            free_region(vstart, vend);
        } 
        else if (vend > (uint64_t)&end) {
            free_region((uint64_t)&end, vend);
        }       
    }   
    memory_end = (uint64_t)free_memory.next+PAGE_SIZE;   
    printk("%x\n",memory_end);
}
static void free_region(uint64_t v, uint64_t e)
{
    for (uint64_t start = PA_UP(v); start+PAGE_SIZE <= e; start += PAGE_SIZE) {        
        if (start+PAGE_SIZE <= 0xffff800040000000) {            
           kfree(start);
        }
    }
}
void kfree(uint64_t v)
{
    ASSERT(v % PAGE_SIZE == 0);
    ASSERT(v >= (uint64_t) & end);
    ASSERT(v+PAGE_SIZE <= 0xffff800040000000);

    struct Page *page_address = (struct Page*)v;
    page_address->next = free_memory.next;
    free_memory.next = page_address;
}
void* kalloc(void)
{
    struct Page *page_address = free_memory.next;
    if (page_address != NULL) {
        ASSERT((uint64_t)page_address % PAGE_SIZE == 0);
        ASSERT((uint64_t)page_address >= (uint64_t)&end);
        ASSERT((uint64_t)page_address+PAGE_SIZE <= 0xffff800040000000);

        free_memory.next = page_address->next;            
    }   
    return page_address;
}
static PDPTR find_pml4t_entry(uint64_t map, uint64_t v, int alloc, uint32_t attribute)
{
    PDPTR *map_entry = (PDPTR*)map;
    PDPTR pdptr = NULL;
    unsigned int index = (v >> 39) & 0x1FF;
    if ((uint64_t)map_entry[index] & PTE_P) {
        pdptr = (PDPTR)P2V(PDE_ADDR(map_entry[index]));       
    } 
    else if (alloc == 1) {
        pdptr = (PDPTR)kalloc();          
        if (pdptr != NULL) {     
            memset(pdptr, 0, PAGE_SIZE);     
            map_entry[index] = (PDPTR)(V2P(pdptr) | attribute);           
        }
    } 
    return pdptr;    
}
static PD find_pdpt_entry(uint64_t map, uint64_t v, int alloc, uint32_t attribute)
{
    PDPTR pdptr = NULL;
    PD pd = NULL;
    unsigned int index = (v >> 30) & 0x1FF;
    pdptr = find_pml4t_entry(map, v, alloc, attribute);
    if (pdptr == NULL)
        return NULL;       
    if ((uint64_t)pdptr[index] & PTE_P) {      
        pd = (PD)P2V(PDE_ADDR(pdptr[index]));      
    }
    else if (alloc == 1) {
        pd = (PD)kalloc();  
        if (pd != NULL) {    
            memset(pd, 0, PAGE_SIZE);       
            pdptr[index] = (PD)(V2P(pd) | attribute);
        }
    } 
    return pd;
}
bool map_pages(uint64_t map, uint64_t v, uint64_t e, uint64_t pa, uint32_t attribute)
{
    uint64_t vstart = PA_DOWN(v);
    uint64_t vend = PA_UP(e);
    PD pd = NULL;
    unsigned int index;
    ASSERT(v < e);
    ASSERT(pa % PAGE_SIZE == 0);
    ASSERT(pa+vend-vstart <= 1024*1024*1024);
    do {
        pd = find_pdpt_entry(map, vstart, 1, attribute);    
        if (pd == NULL) {
            return false;
        }
        index = (vstart >> 21) & 0x1FF;
        ASSERT(((uint64_t)pd[index] & PTE_P) == 0);
        pd[index] = (PDE)(pa | attribute | PTE_ENTRY);
        vstart += PAGE_SIZE;
        pa += PAGE_SIZE;
    } while (vstart + PAGE_SIZE <= vend); 
    return true;
}
void switch_vm(uint64_t map)
{
    load_cr3(V2P(map));   
}
static void setup_kvm(void)
{
    page_map = (uint64_t)kalloc();
    ASSERT(page_map != 0);
    memset((void*)page_map, 0, PAGE_SIZE);        
    bool status = map_pages(page_map, KERNEL_BASE, memory_end, V2P(KERNEL_BASE), PTE_P|PTE_W);
    ASSERT(status == true);
}
void init_kvm(void)
{
    setup_kvm();
    switch_vm(page_map);
    printk("memory manager is working now");
}
void free_pages(uint64_t map, uint64_t vstart, uint64_t vend)
{
    unsigned int index; 
    ASSERT(vstart % PAGE_SIZE == 0);
    ASSERT(vend % PAGE_SIZE == 0);
    do {
        PD pd = find_pdpt_entry(map, vstart, 0, 0);
        if (pd != NULL) {
            index = (vstart >> 21) & 0x1FF;
            ASSERT(pd[index] & PTE_P);           
            kfree(P2V(PTE_ADDR(pd[index])));
            pd[index] = 0;
        }
        vstart += PAGE_SIZE;
    } while (vstart+PAGE_SIZE <= vend);
}
static void free_pdt(uint64_t map)
{
    PDPTR *map_entry = (PDPTR*)map;
    for (int i = 0; i < 512; i++) {
        if ((uint64_t)map_entry[i] & PTE_P) {            
            PD *pdptr = (PD*)P2V(PDE_ADDR(map_entry[i]));           
            for (int j = 0; j < 512; j++) {
                if ((uint64_t)pdptr[j] & PTE_P) {
                    kfree(P2V(PDE_ADDR(pdptr[j])));
                    pdptr[j] = 0;
                }
            }
        }
    }
}
static void free_pdpt(uint64_t map)
{
    PDPTR *map_entry = (PDPTR*)map;
    for (int i = 0; i < 512; i++) {
        if ((uint64_t)map_entry[i] & PTE_P) {          
            kfree(P2V(PDE_ADDR(map_entry[i])));
            map_entry[i] = 0;
        }
    }
}
static void free_pml4t(uint64_t map)
{
    kfree(map);
}
void free_vm(uint64_t map)
{   
    //free_pages(map,vstart,vend);
    free_pdt(map);
    free_pdpt(map);
    free_pml4t(map);
}
```

### 41. User Space

## Section 8: Processes

### 42. The first process
- 2 processes
    - each process has the same kernel space
    - User spaces are different
- process.h
```c
#ifndef _PROCESS_H_
#define _PROCESS_H_
#include "trap.h"
struct Process {
    int pid;
  int state;
  uint64_t page_map;	
  uint64_t stack;
  struct TrapFrame *tf;
};
struct TSS {
    uint32_t res0;
    uint64_t rsp0;
    uint64_t rsp1;
    uint64_t rsp2;
  uint64_t res1;
  uint64_t ist1;
  uint64_t ist2;
  uint64_t ist3;
  uint64_t ist4;
  uint64_t ist5;
  uint64_t ist6;
  uint64_t ist7;
  uint64_t res2;
  uint16_t res3;
  uint16_t iopb;
} __attribute__((packed));
#define STACK_SIZE (2*1024*1024)
#define NUM_PROC 10
#define PROC_UNUSED 0
#define PROC_INIT 1
void init_process(void);
void launch(void);
void pstart(struct TrapFrame *tf);
#endif
```
- process.c
```c
#include "process.h"
#include "trap.h"
#include "memory.h"
#include "print.h"
#include "lib.h"
#include "debug.h"
extern struct TSS Tss; 
static struct Process process_table[NUM_PROC];
static int pid_num = 1;
void main(void);
static void set_tss(struct Process *proc)
{
    Tss.rsp0 = proc->stack + STACK_SIZE;    
}
static struct Process* find_unused_process(void)
{
    struct Process *process = NULL;
    for (int i = 0; i < NUM_PROC; i++) {
        if (process_table[i].state == PROC_UNUSED) {
            process = &process_table[i];
            break;
        }
    }
    return process;
}
static void set_process_entry(struct Process *proc)
{
    uint64_t stack_top;
    proc->state = PROC_INIT;
    proc->pid = pid_num++;
    proc->stack = (uint64_t)kalloc();
    ASSERT(proc->stack != 0);
    memset((void*)proc->stack, 0, PAGE_SIZE);   
    stack_top = proc->stack + STACK_SIZE;
    proc->tf = (struct TrapFrame*)(stack_top - sizeof(struct TrapFrame)); 
    proc->tf->cs = 0x10|3;
    proc->tf->rip = 0x400000;
    proc->tf->ss = 0x18|3;
    proc->tf->rsp = 0x400000 + PAGE_SIZE;
    proc->tf->rflags = 0x202;    
    proc->page_map = setup_kvm();
    /*
                Kernel stack
    low address: rip (0x400000) <=RSP
                 cs
                 rflags
                 rsp (0x600000)
    high address:ss
    */
    ASSERT(proc->page_map != 0);
    ASSERT(setup_uvm(proc->page_map, (uint64_t)main, PAGE_SIZE));
}
void init_process(void)
{  
    struct Process *proc = find_unused_process();
    ASSERT(proc == &process_table[0]);

    set_process_entry(proc);
}
void launch(void)
{
    set_tss(&process_table[0]);
    switch_vm(process_table[0].page_map);
    pstart(process_table[0].tf);
}
void main(void)
{
    char *p = (char*)0xffff800000200020;
    *p = 1;
}
```

### 43. System call
- Using interrupt, run print function
- From user mode: print function -> int 0x80
- In the kernel mode: handler -> system call -> write screen
- syscall.h
```c
#ifndef _SYSCALL_H_
#define _SYSCALL_H_
#include "trap.h"
typedef int (*SYSTEMCALL)(int64_t *argptr);
void init_system_call(void);
void system_call(struct TrapFrame *tf);
#endif
```
- syscall.c
```c
#include "syscall.h"
#include "print.h"
#include "debug.h"
#include "stddef.h"
static SYSTEMCALL system_calls[10];
static int sys_write(int64_t *argptr)
{    
    write_screen((char*)argptr[0], (int)argptr[1], 0xe);  
    return (int)argptr[1];
}
void init_system_call(void)
{
    system_calls[0] = sys_write;
}
void system_call(struct TrapFrame *tf)
{
    int64_t i = tf->rax;
    int64_t param_count = tf->rdi;
    int64_t *argptr = (int64_t*)tf->rsi;
    if (param_count < 0 || i != 0) { 
        tf->rax = -1;
        return;
    }    
    ASSERT(system_calls[i] != NULL);
    tf->rax = system_calls[i](argptr);
}
```

### 44. Scheduling
- How to switch from one process to another
- Every 10ms, timer interrupt runs and user mode is trapped
- process.c
```c
#include "process.h"
#include "trap.h"
#include "memory.h"
#include "print.h"
#include "lib.h"
#include "debug.h"
extern struct TSS Tss; 
static struct Process process_table[NUM_PROC];
static int pid_num = 1;
static struct ProcessControl pc;
static void set_tss(struct Process *proc)
{
    Tss.rsp0 = proc->stack + STACK_SIZE;    
}
static struct Process* find_unused_process(void)
{
    struct Process *process = NULL;
    for (int i = 0; i < NUM_PROC; i++) {
        if (process_table[i].state == PROC_UNUSED) {
            process = &process_table[i];
            break;
        }
    }
    return process;
}
static void set_process_entry(struct Process *proc, uint64_t addr)
{
    uint64_t stack_top;
    proc->state = PROC_INIT;
    proc->pid = pid_num++;
    proc->stack = (uint64_t)kalloc();
    ASSERT(proc->stack != 0);
    memset((void*)proc->stack, 0, PAGE_SIZE);   
    stack_top = proc->stack + STACK_SIZE;
    proc->context = stack_top - sizeof(struct TrapFrame) - 7*8;   
    *(uint64_t*)(proc->context + 6*8) = (uint64_t)TrapReturn;
    proc->tf = (struct TrapFrame*)(stack_top - sizeof(struct TrapFrame)); 
    proc->tf->cs = 0x10|3;
    proc->tf->rip = 0x400000;
    proc->tf->ss = 0x18|3;
    proc->tf->rsp = 0x400000 + PAGE_SIZE;
    proc->tf->rflags = 0x202;   
    proc->page_map = setup_kvm();
    ASSERT(proc->page_map != 0);
    ASSERT(setup_uvm(proc->page_map, P2V(addr), 5120));
    proc->state = PROC_READY;    
}
static struct ProcessControl* get_pc(void)
{
    return &pc;
}
void init_process(void)
{
    struct ProcessControl *process_control;
    struct Process *process;
    struct HeadList *list;
    uint64_t addr[2] = {0x20000, 0x30000};
    process_control = get_pc();
    list = &process_control->ready_list;
    for (int i = 0; i < 2; i++) {
        process = find_unused_process();
        set_process_entry(process, addr[i]);
        append_list_tail(list, (struct List*)process);
    }
}
void launch(void)
{
    struct ProcessControl *process_control;
    struct Process *process;
    process_control = get_pc();
    process = (struct Process*)remove_list_head(&process_control->ready_list);
    process->state = PROC_RUNNING;
    process_control->current_process = process;   
    set_tss(process);
    switch_vm(process->page_map);
    pstart(process->tf);
}
static void switch_process(struct Process *prev, struct Process *current)
{
    set_tss(current);
    switch_vm(current->page_map);
    swap(&prev->context, current->context); /* from trap.asm */
}
static void schedule(void)
{
    struct Process *prev_proc;
    struct Process *current_proc;
    struct ProcessControl *process_control;
    struct HeadList *list;
    process_control = get_pc();
    prev_proc = process_control->current_process;
    list = &process_control->ready_list;
    ASSERT(!is_list_empty(list));    
    current_proc = (struct Process*)remove_list_head(list);
    current_proc->state = PROC_RUNNING;   
    process_control->current_process = current_proc;
    switch_process(prev_proc, current_proc);   
}
void yield(void)
{
    struct ProcessControl *process_control;
    struct Process *process;
    struct HeadList *list;
        process_control = get_pc();
    list = &process_control->ready_list;
    if (is_list_empty(list)) {
        return;
    }
    process = process_control->current_process;
    process->state = PROC_READY;
    append_list_tail(list, (struct List*)process);
    schedule();
}
```

### 45. Sleep and Wake up
- In process.c:
```c
void sleep(int wait)
{
    struct ProcessControl *process_control;
    struct Process *process;    
    process_control = get_pc();
    process = process_control->current_process;
    process->state = PROC_SLEEP;
    process->wait = wait;
    append_list_tail(&process_control->wait_list, (struct List*)process);
    schedule();
}
void wake_up(int wait)
{
    struct ProcessControl *process_control;
    struct Process *process;
    struct HeadList *ready_list;
    struct HeadList *wait_list;
    process_control = get_pc();
    ready_list = &process_control->ready_list;
    wait_list = &process_control->wait_list;
    process = (struct Process*)remove_list(wait_list, wait);
    while (process != NULL) {       
        process->state = PROC_READY;
        append_list_tail(ready_list, (struct List*)process);
        process = (struct Process*)remove_list(wait_list, wait);
    }
}
```

### 46. Exit and Wait
- In process.c:
```c
void exit(void)
{
    struct ProcessControl *process_control;
    struct Process* process;
    struct HeadList *list;
    process_control = get_pc();
    process = process_control->current_process;
    process->state = PROC_KILLED;
    list = &process_control->kill_list;
    append_list_tail(list, (struct List*)process);
    wake_up(1);
    schedule();
}
void wait(void)
{
    struct ProcessControl *process_control;
    struct Process *process;
    struct HeadList *list;
    process_control = get_pc();
    list = &process_control->kill_list;
    while (1) {
        if (!is_list_empty(list)) {
            process = (struct Process*)remove_list_head(list); 
            ASSERT(process->state == PROC_KILLED);
            kfree(process->stack);
            free_vm(process->page_map);            
            memset(process, 0, sizeof(struct Process));   
        }
        else {
            sleep(1);
        }
    }
}
```

### 47. Terminate a process
- In trap.c:
```c
void handler(struct TrapFrame *tf)
{
    unsigned char isr_value;
    switch (tf->trapno) {
        case 32:  
            timer_handler();   
            eoi();
            break;            
        case 39:
            isr_value = read_isr();
            if ((isr_value&(1<<7)) != 0) {
                eoi();
            }
            break;
        case 0x80:                   
            system_call(tf);
            break;
        default:
            if ((tf->cs & 3) == 3) {
                printk("Exception is %d\n", tf->trapno);
                exit();
            }
            else {                                      
                while (1) { }
            }
    }
    if (tf->trapno == 32) {       
        yield();
    }
}
```

## Section 9: Keyboard and console

### 48. Write a PS/2 keyboard driver 1
- Kyeboard scan code: https://www.win.tue.nl/~aeb/linux/kbd/scancodes-1.html
- PS2 keyboard through IRQ 1
- keyboard.c
```c
#include "keyboard.h"
#include "print.h"
#include "process.h"
static unsigned char shift_code[256] = {
    [0x2A] = SHIFT, [0x36] = SHIFT, [0xAA] = SHIFT, [0xB6] = SHIFT
};
static unsigned char lock_code[256] = {
    [0x3A] = CAPS_LOCK
};
static char key_map[256] = {
    0, 0, '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    '-', '=', '\b', 0, 'q', 'w', 'e', 'r', 't', 'y', 'u',
    'i', 'o', 'p', '[', ']', '\n', 0, 'a', 's', 'd', 'f',
    'g', 'h', 'j', 'k', 'l', ';', '\'', '`', 0, '\\', 'z',
    'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', 0, '*', 0, ' '
};
static char shift_key_map[256] = {
    0, 1, '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',
    '_', '+', '\b', '\t', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U',
    'I', 'O', 'P', '{', '}', '\n', 0, 'A', 'S', 'D', 'F', 'G',
    'H', 'J', 'K', 'L', ':', '"', '~', 0, '|', 'Z', 'X', 'C',
    'V', 'B', 'N', 'M', '<', '>', '?', 0, '*', 0, ' '
};
static unsigned int flag;
static char keyboard_read(void)
{
    unsigned char scan_code;
    char ch;
    scan_code = in_byte(0x60);
    if (scan_code == 0xE0) {
        flag |= E0_SIGN;   
        return 0;
    }
    if (flag & E0_SIGN) {
        flag &= ~E0_SIGN;
        return 0;
    }
    if (scan_code & 0x80) {
        flag &= ~(shift_code[scan_code]);
        return 0;
    }
    flag |= shift_code[scan_code];
    flag ^= lock_code[scan_code];
    if (flag & SHIFT) {
        ch = shift_key_map[scan_code];
    }
    else {
        ch = key_map[scan_code];
    }
    if (flag & CAPS_LOCK) { 
        if('a' <= ch && ch <= 'z')
            ch -= 32;
        else if('A' <= ch && ch <= 'Z')
            ch += 32;
    }
    return ch;
}
void keyboard_handler(void)
{
    char ch[2] = { 0 };
    ch[0] = keyboard_read();
    if (ch[0] > 0) {
        printk("%s", ch);
    }
}
```

### 49. Write a PS/2 keyboard driver 2
- Key buffer
- keyboard.h
```c
#ifndef _KEYBOARD_H_
#define _KEYBOARD_H_
#include "stdint.h"
struct KeyboardBuffer {
    char buffer[500];
    int front;
    int end;
    int size;
};
#define E0_SIGN (1 << 0)
#define SHIFT (1 << 1)
#define CAPS_LOCK (1 << 2)
char read_key_buffer(void);
void keyboard_handler(void);
unsigned char in_byte(uint16_t port);
#endif
```

### 50. Interact with kernel using console
- Yields prompt and reads command
    - Doesn't actually execute as no command is implemented
- main.c:
```c
#include "lib.h"
#include "console.h"
#include "stdint.h"    
static void cmd_get_total_memory(void)
{
    uint64_t total;   
    total = get_total_memoryu();
    printf("Total Memory is %dMB\n", total);
}
static int read_cmd(char *buffer)
{
    char ch[2] = { 0 };
    int buffer_size = 0;
    while (1) {
        ch[0] = keyboard_readu();
        
        if (ch[0] == '\n' || buffer_size >= 80) {
            printf("%s", ch);
            break;
        }
        else if (ch[0] == '\b') {    
            if (buffer_size > 0) {
                buffer_size--;
                printf("%s", ch);    
            }           
        }          
        else {     
            buffer[buffer_size++] = ch[0]; 
            printf("%s", ch);        
        }
    }
    return buffer_size;
}
static int parse_cmd(char *buffer, int buffer_size)
{
    int cmd = -1;
    if (buffer_size == 8 && (!memcmp("totalmem", buffer, 8))) {
        cmd = 0;
    }
    return cmd;
}
static void execute_cmd(int cmd)
{ 
    CmdFunc cmd_list[1] = {cmd_get_total_memory};   
    if (cmd == 0) {       
        cmd_list[0]();
    }
}
int main(void)
{
    char buffer[80] = { 0 };
    int buffer_size = 0;
    int cmd = 0;
    while (1) {
        printf("shell# ");
        buffer_size = read_cmd(buffer);
        if (buffer_size == 0) {
            continue;
        }        
        cmd = parse_cmd(buffer, buffer_size);        
        if (cmd < 0) {
            printf("Command Not Found!\n");
        }
        else {
            execute_cmd(cmd);             
        }            
    }
    return 0;
}
```

## Section 10: Bonus: File system

### 51. Introduction
- FAST16
    - File system structure
    - Open or read files
    - Load programs
- How to load the file system in the boot process
- In this class, we load all disk into memory as mounting disk is quite complex

### 52. Working with Windows 10

### 53. Working with Linux (Ubuntu)
- File system image using FreeDOS
    - Download iso image (FD12CD.iso)
    - bximage -> 1 (Convert hard disk image to other format) -> hd -> flat -> 512 -> 100 -> os.img
    - bochs -> 3 (Edit) -> 12 -> 4 (First HD/CD on channel 0) -> disk -> os.img (New file name) -> flat -> 203 -> 16 -> 63 -> 512 (sector size) -> Generic 1234 (new model name) -> auto -> auto -> 5 (Second HD/CD on channel 0) -> cdrom -> FD12CD.iso (downloaded FreeDOS image) -> inserted -> Generic 1234 -> auto -> 15 (Boot options) -> 1 -> cdrom (Boot from) -> 0 -> 0 (return to Main Menu) -> 4 (Save options to) -> bochsrc -> 6 -> c
    - Now FreeDOS runs
- gnome-disk-image-mounter --writable os.img
- hexdump -C os.img

### 54. Working with MacOS

### 55. The New loader
- Real mode can use 1MB while FS is 100MB
- We use Big Real Mode -> Protected Mode load FS (4GB limit) -> Real Mode FS(4G Limit)

### 56. FAST16 Structure

### 57. Loading Files from the FAST16 Image

### 58. The idle process

### 59. The File Module

### 60. Fork

### 61. Exec

### 62. The New Console

### 63. ls command

## Section 11: Conclusion

### 64. Conclusion

