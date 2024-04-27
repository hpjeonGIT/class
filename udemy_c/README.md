# The Beginner's guide to Advanced C programming for Linux
- Instructor: dr Jonas Birch, C programming ninja

## Section 1: Beginner

### 1. Writing output to the screen
```c
#include<stdio.h>
int main() 
{
  printf("hello world\n");  
  return 0;
}
```

### 2. Reading input from the keyboard
```c
#include<stdio.h>
int main() 
{
  char name[32];
  printf("what is your name\n");
  scanf("%s", &name);
  printf("hello %s\n", name);  
  return 0;
}
```

### 3. Integer numbers

### 4. Decimal (float) numbers

### 5. Troubleshooting your code

### 6. While loops

### 7. If statements

### 8. Functions

### 9. Random numbers

### 10. Sleep

### 11. Countdown

### 12. ASSIGNMENT: dice game

### 13. SOLUTION: dice game

### 14. Structures

### 15. Switch

### 16. Pointers

### 17. For loops

## Section 2: Advanced

### 18. Introduction of Advanced section

### 19. Dynamic memory allocation
```c
#include<stdlib.h>
#include<stdio.h>
int main() 
{
  char *name;
  name = malloc(32);
  printf("what is your name\n");
  scanf("%s", name);
  printf("hello %s\n", name);  
  free(name);
  return 0;
}
```

### 20. Read from keywboard with a timeout
- Ref: https://www.gnu.org/software/libc/manual/html_node/Waiting-for-I_002fO.html
- FD_ZERO()
- FD_SET()

### 21. XOR encryption

### 22. Ncurses 1: Screens
```c
#include <ncurses.h>
int main() 
{
  char letter;
  //
  initscr();
  printw("Press any key:");
  refresh();
  letter = getch();
  clear();
  printw("You pushed: '%c'", letter);
  refresh();
  getch();
  endwin();
  //
  return 0;
```
- Demo:
```bash
$ gcc nc1.c -I/usr/include -L/usr/lib/x86_64-linux-gnu -lcurses
$ ./a.out 
Press any key:
You pushed: 'a'
```

### 23. Ncurses 2: X and Y
```c
#include <ncurses.h>
int main() 
{
  int x, y;
  //
  initscr();
  clear();
  getyx(stdscr, y, x);
  printw("x = %d\n"
         "y = %d", x,y);
  refresh();
  y = 2;
  x = 10;
  move(y,x);
  printw("Over here!");
  refresh();
  getch();
  endwin();
  //
  return 0;
}
```
- Demo:
```bash
$ gcc nc2.c -lncurses
$ ./a.out
x = 0
y = 0
          Over here!
```

### 24. Ncurse 3: "Arrow"
```c
#include <ncurses.h>
int main() 
{
  int key, x, y;
  //
  initscr();
  keypad(stdscr, TRUE);
  noecho();
  x = y = 5;
  while (key != 'q') 
  {
    clear();
    move (0,0);
    printw("Please left or right arrow - exit by pressing: q");
    move(y,x);
    printw("O");
    refresh();
    key = getch();
    if (key == KEY_LEFT)
    {
      x--;
      if (x <0) x = 0;
    }
    else if (key == KEY_RIGHT)
    { 
      x++;
      if (x> 30) x = 30;
    }
  }
  endwin();
  //
  return 0;
}
```

### 25. Function pointers

### 26. Linked lists

### 27. The "&"

### 28. Sockets 1: Building a simple TCP client

### 29. Sockets 2: Creating a sample TCP server

### 30. Forking your code

### 31. Build your own webserver 1: Accepting connections

### 32. Build your own webserver 2: Parsing HTTP requests

### 33. Build your own webserver 3: Handling routes with an HTTP response

### 34. Build your own webserver 4: Reading and sending files

### ASSIGNMENT: Fix the webserver

### 35. Solution: Build your own webserver 5: Finishing the webserver

## Section 3: Misc

### 36. Introduction
