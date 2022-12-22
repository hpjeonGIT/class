## Basic class
- Ref: https://dev.to/tbhaxor/introduction-to-ncurses-part-1-1bk5

```cxx
#include <ncurses.h>
using namespace std;
int main(int argc, char ** argv)
{
    // init screen and sets up screen
    initscr();
    // print to screen
    printw("Hello World");
    // refreshes the screen
    refresh();
    // pause the screen output
    getch();
    // deallocates memory and ends ncurses
    endwin();
    return 0;
}
```
- `g++ hello-world.cc -lncurses`
```cxx
#include <ncurses.h>
using namespace std;
int main (int argc, char ** argv)
{
    initscr();
    // moving cursor, x = 20, y = 10
    move(10, 20);
    printw("I am here...");
    move(21, 10);
    printw("Now i am here");
    refresh();
    getch();
    endwin();
    return 0;
}
```
```cxx
#include <ncurses.h>
using namespace std;
int main(int argc, char **argv)
{
    initscr();
    // creating a window;
    // with height = 15 and width = 10
    // also with start x axis 10 and start y axis = 20
    WINDOW *win = newwin(15, 17, 2, 10);
    refresh();
    // making box border with default border styles
    box(win, 0, 0);
    // move and print in window
    mvwprintw(win, 0, 1, "Greeter");
    mvwprintw(win, 1, 1, "Hello");
    // refreshing the window
    wrefresh(win);
    getch();
    endwin();
    return 0;
}
```
```bash
          ┌Greeter────────┐
          │Hello          │
          │               │
          │               │
          │               │
          │               │
          │               │
          │               │
          │               │
          │               │
          │               │
          │               │
          │               │
          │               │
          └───────────────┘
```

## Sample menu
- http://www.linuxfocus.org/English/March2002/article233.shtml
```bash
Menu1(F1)           Menu2(F2)
┌─────────────────┐
│Item1            │open the menus. ESC quits.
│Item2            │
│Item3            │
│Item4            │
│Item5            │
│Item6            │
│Item7            │
│Item8            │
└─────────────────┘
```

## Youtube lecture
- Ncurses Tutorial 0 - Hello World (initscr, endwin, refresh, getch, printw)

- cbreak(): accept ctrl+c to exit
- raw(): reads ctrl+c as character. No exit
- Drawing a box with character:
```
char x = 'x';
char z = 'z';
box(win, (int)c, (int)z)
```
- Current cursor info
    - getyx()
    - getbegyx()
    - getmaxyx()
- Detecting up arrow button:
```cxx
#include <ncurses.h>
#include <string>
using namespace std;
int main() {
  initscr();
  noecho();
  cbreak();
  int yMax, xMax;
  getmaxyx(stdscr, yMax, xMax);
  WINDOW * inputwin = newwin(3, xMax-12, yMax-5,5);
  box(inputwin,0,0);
  refresh();
  wrefresh(inputwin);
  keypad(inputwin, true);
  int c = wgetch(inputwin);
  if (c == KEY_UP) {
    mvwprintw(inputwin, 1,1, "You pressed UP");
    wrefresh(inputwin); 
  }
  getch();
  endwin();
  return 0;
}
```
- Result:
```bash
     ┌──────────────────────────────────────────────────────────────────┐
     │You pressed UP                                                    │
     └──────────────────────────────────────────────────────────────────┘

```
