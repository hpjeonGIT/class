# NCURSES Tutorial
- Instructor: Qi Liang


## Section 1: Introduction

### 1. Introduction
- Ncurses
  - API for TUI
  - Basic extensions like panel, menu, and form
- Maintained at: https://invisible-island.net/ncurses
- Wrappers
  - ncurses++
  - Python
  - ncurses-rs

### 2. Preparation
- This class uses a docker with podman
- At Ubuntu: sudo apt install libncurses-dev
- https://invisible-island.net/ncurses/howto/NCURSES-Programming-HOWTO.html
- hello.c:
```c
#include <curses.h>
int
main(void)
{
    initscr();                  /* Start curses mode              */
    printw("Hello World !!!");  /* Print Hello World              */
    refresh();                  /* Print it on to the real screen */
    getch();                    /* Wait for user input */
    endwin();                   /* End curses mode                */

    return 0;
}
```
- gcc hello.c -lncurses; a.out
- Example program here
  - A file namanger
  - Menu in the top
  - File List window left-middle
  - Contents window right-middle
  - Debug output window in the bottom
  - https://github.com/std3lqi/ncurses-tutorial
- Linux documentation:
  - sudo apt install ncurses-doc
  - man initscr

## Section 2: NCURSES Library

### 3. Basics
- terminfo at /usr/shre/terminfo
```bash
$ echo $TERM
xterm-256color
```
- Screen: a terminal screen with width/height in character cells stored in COLS and LINES
- Window: 
  - The position of the left-top corner(y,x)
  - The height (h) and width (w) not outside the screen
- Pad: 
- Default window
  - After ncurses is initialized, as special window named stdscr is created
  - stdscr has the same size as the terminal screen
  - stdscr is the default window used by the APIs without argument WINDOW
  - printw("Hello"): print Hello to the default window stdscr
- Initialization and de-initialization
  - First API to call to initialize: WINDOW *initscr(void);
  - Exit curses mode and free the data structure: init endwin(void);
  - Return true if endwin() is called: bool isendwin(void);
- From https://github.com/std3lqi/ncurses-tutorial, download step0 Branches
  - git clone https://github.com/std3lqi/ncurses-tutorial.git
  - cd ncurses-tutorial/
  - git branch -a
  - git checkout step0
  - mkdir build
  - make
  - ./main # does nothing as no WINDOW is made
- Line buffering
  - `int cbreak(void);` is preferred over `int raw(void);` as ctrl+C is accepted
- Echoing user input
  - `int noecho(void);` to disable user input, `int echo(void);` to show user input in the screen
- Function keys
  - `int keypad(WINDOW *win, bool bf);`
- Cursor
  - `int curs_set(int visibility);`
  - Cursor state to inivisible, normal, or very visible by 0,1,2
- Output a chtype character
  - `int addch(const chtype ch);`
  - `int waddch(WINDOW *win, const chtype ch);`
  - `int mvaddch(int y, in x, const chtype ch);`
  - `int mvaddch(WINDOW *win, int y, int x, const chtype ch);`
  - addch()/waddch() are followed by refresh/wrefresh()
- Data type chtype
  - A combination of 8-bit character with attributes
  - addch('A');
  - addch('A' | A_BOLD);
  - start_color(); init_pair(1,COLOR_YELLOW,COLOR_GREEN); addch('A'|A_BOLD|COLOR_PAIR(1));
- ch03.c:
```c
#include <curses.h>
#include <panel.h>
int main(int argc, char *argv[]) {
    initscr();
    cbreak();
    noecho(); // no user input shown
    keypad(stdscr, TRUE);
    curs_set(0);
    addch('N');
    addch('S'|A_STANDOUT);
    addch('U'|A_UNDERLINE);
    addch('R'|A_REVERSE);
    addch('K'|A_BLINK);
    addch('D'|A_DIM);
    addch('B'|A_BOLD);
    addch('P'|A_PROTECT);
    addch('H'|A_INVIS);
    addch('A'|A_ALTCHARSET);
    addch('I'|A_ITALIC);
    refresh();
    getch(); //wait to get input
    endwin();
    return 0;
}
```
  - gcc ch03.c -lncurses
  ![demo](./ch03.png)
- Family of getch
  - int getch(void);
  - int wgetch(WINDOW *win);
  - int mvgetch(int y, int x);
  - int mvwgetch(WINDOW *win, int y, int x);

### 4. Window
- Create a new window: WINDOW *newin(int nlines, int ncols, int begin_y, int begin_x);
- Delete the window: int delwin(WINDOW *win);
- Move the window: int mvwin(WINDOW *win, int y, int x);
- Window border: int border(chtype ls, chtype rs, chtype ts, chtype bs, chtype tl, chtype tr, chtype bl, chtype br)
- Shortcut border function: int box(WINDOW *win, chtype verch, chtype horch);
```c
#include <curses.h>
#include <panel.h>
static WINDOW *win = NULL;
void create_menu_bar_window(int h, int w, int y, int x) {
    win = newwin(h, w, y, x);
    wrefresh(win);
}
void delete_menu_bar_window() { delwin(win);     win = NULL; }
void create_file_list_window(int h, int w, int y, int x) {
    win = newwin(h, w, y, x);
    wborder(win, 0, 0, 0, 0, 0, 0, 0, 0);
    wrefresh(win);
}
void delete_file_list_window() {    delwin(win);    win = NULL;}
void create_contents_window(int h, int w, int y, int x) {
    win = newwin(h, w, y, x);
    box(win, 0, 0);
    wrefresh(win);
}
void delete_contents_window() {    delwin(win);    win = NULL;}
void create_debug_window(int h, int w, int y, int x) {
    win = newwin(h, w, y, x);
    wborder(win, '|', '|', '-', '-', '+', '+', '+', '+');
    wrefresh(win);
}
void delete_debug_window() {    delwin(win);    win = NULL;}
int main(int argc, char *argv[]) {
    initscr();      // Enter curses mode
    cbreak();       // Disable line buffering
    noecho();       // Disable echoing
    keypad(stdscr, TRUE);   // Enable function keys like F1
    curs_set(0);            // Hide cursor
    refresh();
    //
    int h, w;
    getmaxyx(stdscr, h, w);
    int h_files = h - 1 - 6;
    int w_files = w * 0.3;
    create_menu_bar_window(1, w, 0, 0); // top 
    create_file_list_window(h_files, w_files, 1, 0); // left-center
    create_contents_window(h_files, w - w_files, 1, w_files); // right-center
    create_debug_window(6, w, 1 + h_files, 0); // bottom
    //
    getch();
    //
    delete_menu_bar_window();
    delete_file_list_window();
    delete_contents_window();
    delete_debug_window();

    endwin();       // Exit from curses mode
    return 0;
}
```
![basic_windows](./ch04.png)

### 5. Output
- Lab task
  - Output window titles
  - Provide a debug function
```c
#include <curses.h>
#include <panel.h>
static WINDOW *win = NULL;
void create_menu_bar_window(int h, int w, int y, int x) {
    win = newwin(h, w, y, x);
    wrefresh(win);
}
void delete_menu_bar_window() { delwin(win);     win = NULL; }
void create_file_list_window(int h, int w, int y, int x) {
    win = newwin(h, w, y, x);
    wborder(win, 0, 0, 0, 0, 0, 0, 0, 0);
    mvwaddstr(win, 0, 1, "File List");
    wrefresh(win);
}
void delete_file_list_window() {    delwin(win);    win = NULL;}
void create_contents_window(int h, int w, int y, int x) {
    win = newwin(h, w, y, x);
    box(win, 0, 0);
    mvwaddstr(win, 0, 1, "Contents");
    wrefresh(win);
}
void delete_contents_window() {    delwin(win);    win = NULL;}
void create_debug_window(int h, int w, int y, int x) {
    win = newwin(h, w, y, x);
    wborder(win, '|', '|', '-', '-', '+', '+', '+', '+');
    mvwaddstr(win, 0, 1, "Debug");
    wrefresh(win);
}
void delete_debug_window() {    delwin(win);    win = NULL;}
static int count = 0;
void debug_line(const char *fmt, ...) {
    // 1: This is a debug line
    // 2: This is another debug line
    int y = 1 + count;
    int x = 1;
    mvwprintw(win, y, x, "%d: ", count);
    count++;
    va_list varglist;
    va_start(varglist, fmt);
    vw_printw(win, fmt, varglist);
    va_end(varglist);
    wrefresh(win);
}

int main(int argc, char *argv[]) {
    initscr();      // Enter curses mode
    cbreak();       // Disable line buffering
    noecho();       // Disable echoing
    keypad(stdscr, TRUE);   // Enable function keys like F1
    curs_set(0);            // Hide cursor
    refresh();
    //
    int h, w;
    getmaxyx(stdscr, h, w);
    int h_files = h - 1 - 6;
    int w_files = w * 0.3;
    create_menu_bar_window(1, w, 0, 0); // top 
    create_file_list_window(h_files, w_files, 1, 0); // left-center
    create_contents_window(h_files, w - w_files, 1, w_files); // right-center
    create_debug_window(6, w, 1 + h_files, 0); // bottom
    debug_line("This is a debug message"); // inject strings into the debug window
    debug_line("Hello %s", "ncurses");
    //
    getch();
    //
    delete_menu_bar_window();
    delete_file_list_window();
    delete_contents_window();
    delete_debug_window();
    //
    endwin();       // Exit from curses mode
    return 0;
}
```
![window_demo](./ch05_win.png)

### 6. List directory
- Lab task
  - Take the directory to list from command line
  - Use scandir() to list the directory entries
  - Show the current directory path name on window top border
  - Show the count of entries on window bottom border
  - Show these entries in File List window
  - Press 'q' to exit
```c
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <curses.h>
#include <panel.h>
static WINDOW *men_win = NULL;
static WINDOW *fil_win = NULL;
static WINDOW *con_win = NULL;
static WINDOW *dbg_win = NULL;
void create_menu_bar_window(int h, int w, int y, int x) {
    men_win = newwin(h, w, y, x);
    wrefresh(men_win);
}
void delete_menu_bar_window() { delwin(men_win);     men_win = NULL; }
void create_file_list_window(int h, int w, int y, int x) {
    fil_win = newwin(h, w, y, x);
    wborder(fil_win, 0, 0, 0, 0, 0, 0, 0, 0);
    mvwaddstr(fil_win, 0, 1, "File List");
    wrefresh(fil_win);
}
static int entry_count = 0;
static struct dirent **entry_list = NULL; // from <dirent.h>
void delete_file_list_window() {    delwin(fil_win);    fil_win = NULL; free(entry_list); entry_list = NULL; entry_count = 0;}
void list_dir_in_file_list_window(const char *dir) {
    if (entry_list != NULL) free(entry_list);
    entry_count = scandir(dir, &entry_list, NULL, NULL);
    int h = getmaxy(fil_win);
    wborder(fil_win,0,0,0,0,0,0,0,0);
    mvwaddstr(fil_win, 0, 1, dir); // print title
    int y = 1;
    if (entry_count == 0) {  
         mvwaddstr(fil_win,1,1,"Nothing");
    } 
    else { 
         for (int row = 0; row < h - 2; row++) { 
            int i=row;
            if (i>=entry_count) {
                break;
            }
            struct dirent *entry = entry_list[i];
            mvwaddstr(fil_win,y,1,entry->d_name);
            y++;
      }
    }
    y = h-1;
    mvwprintw(fil_win, y,1,"%d", entry_count); // print footer
    wrefresh(fil_win);
}
void create_contents_window(int h, int w, int y, int x) {
    con_win = newwin(h, w, y, x);
    box(con_win, 0, 0);
    mvwaddstr(con_win, 0, 1, "Contents");
    wrefresh(con_win);
}
void delete_contents_window() { delwin(con_win); con_win = NULL;}
void create_debug_window(int h, int w, int y, int x) {
    dbg_win = newwin(h, w, y, x);
    wborder(dbg_win, '|', '|', '-', '-', '+', '+', '+', '+');
    mvwaddstr(dbg_win, 0, 1, "Debug");
    wrefresh(dbg_win);
}
void delete_debug_window() {    delwin(dbg_win);    dbg_win = NULL;}
int main(int argc, char *argv[]) {
   if (argc < 2) { printf("Usage: ./main <dir>\n"); exit(1); } // check n. of arguments
    initscr();      // Enter curses mode
    cbreak();       // Disable line buffering
    noecho();       // Disable echoing
    keypad(stdscr, TRUE);   // Enable function keys like F1
    curs_set(0);            // Hide cursor
    refresh();
    char *dir = argv[1];
    int len = strlen(dir);
    if (dir[len-1] == '/') {dir[len-1] = '\0';} 
    //
    int h, w;
    getmaxyx(stdscr, h, w);
    int h_files = h - 1 - 6;
    int w_files = w * 0.3;
    create_menu_bar_window(1, w, 0, 0); // top 
    create_file_list_window(h_files, w_files, 1, 0); // left-center
    create_contents_window(h_files, w - w_files, 1, w_files); // right-center
    create_debug_window(6, w, 1 + h_files, 0); // bottom
    list_dir_in_file_list_window(dir);
    //
    int ch;
    while ((ch = getch()) != 'q') {    }
    //
    delete_menu_bar_window();
    delete_file_list_window();
    delete_contents_window();
    delete_debug_window();
    endwin();       // Exit from curses mode
    return 0;
}
```
![ch06_demo](./ch06_demo.png)

### 7. List directory - enhancement
- How to handle long strings
```c
void list_dir_in_file_list_window(const char *dir) {
    if (entry_list != NULL) free(entry_list);
    entry_count = scandir(dir, &entry_list, NULL, NULL);
    int h = getmaxy(fil_win);
    int w = getmaxx(fil_win);
    wborder(fil_win,0,0,0,0,0,0,0,0);
    if (strlen(dir) > w-2) { // print title
       mvwaddnstr(fil_win, 0,1,dir,w-2-3);
       waddstr(fil_win, "...");
    } else { mvwaddstr(fil_win,0,1,dir); }
    int y = 1;
    if (entry_count == 0) {  
         mvwaddstr(fil_win,1,1,"Nothing");
    } 
    else { 
         for (int row = 0; row < h - 2; row++) { 
            int i=row;
            if (i>=entry_count) {
                break;
            }
            struct dirent *entry = entry_list[i];
            mvwaddstr(fil_win,y,1,entry->d_name);
            y++;
      }
    }
    y = h-1;
    mvwprintw(fil_win, y,1,"%d", entry_count); // print footer
    wrefresh(fil_win);
}
```
- Now "/lib/x86_64-linux-gnu/b" is shown as "/lib/x86_64-linux-g..."

### 8. Attributes
- Support active window
  - Register all windows in a window manager
  - Press TAB key to move focus around the windows
  - Highlight window title when the window gains focus
- APIs
  - int attroff(int attrs);
  - int wattroff(WINDOW *win, int attrs);
  - int attron(int attrs);
  - int wattron(WINDOW *win, int attrs);
  - int attrset(int attrs);
  - int wattrset(WINDOW *win, int attrs);
  - int attr_get(attr_t, *attrs, short *pair, void *opts);
- Erase/clear window
  - int erase(void);
  - int werase(WINDOW *win);
  - int clear(void);
  - int wclear(WINDOW *win);
```c
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <limits.h>
#include <curses.h>
#include <panel.h>
#define WINDOWS 4
static WINDOW *windows[WINDOWS];
static int count = 0;
static int current_window_index = -1;
void init_win_manager() {
    for (int i = 0; i < WINDOWS; i++) {
        windows[i] = NULL;
    }
}
void register_window(WINDOW *win) {
    windows[count] = win;
    count++;
    if (current_window_index == -1) {
        current_window_index = 0;
    }
}
void next_window() {
    if (current_window_index < count - 1) {
        current_window_index++;
    } else {
        current_window_index = 0;
    }
}
bool is_current_window(WINDOW *win) {
    return windows[current_window_index] == win;
}
static WINDOW *men_win = NULL;
static WINDOW *fil_win = NULL;
static WINDOW *con_win = NULL;
static WINDOW *dbg_win = NULL;
void refresh_menu_bar_window() {
    if (is_current_window(men_win)) {wattron(men_win, A_REVERSE);}
    mvwaddstr(men_win,0,1,"Menu");
    if (is_current_window(men_win)) {wattroff(men_win, A_REVERSE);}
    wrefresh(men_win);
}
void create_menu_bar_window(int h, int w, int y, int x) {
    men_win = newwin(h, w, y, x);
    register_window(men_win);
    refresh_menu_bar_window();
}
void delete_menu_bar_window() { delwin(men_win);     men_win = NULL; }
static int entry_count = 0;
static struct dirent **entry_list = NULL; // from <dirent.h>
static char current_path[PATH_MAX];
void create_file_list_window(int h, int w, int y, int x) {
    fil_win = newwin(h, w, y, x);
    register_window(fil_win);
    current_path[0] = '\0';
    wborder(fil_win, 0, 0, 0, 0, 0, 0, 0, 0);
    mvwaddstr(fil_win, 0, 1, "File List");
    wrefresh(fil_win);
}
void delete_file_list_window() {    delwin(fil_win);    fil_win = NULL; free(entry_list); entry_list = NULL; entry_count = 0;}
void refresh_file_list_window() {
    int h = getmaxy(fil_win);
    int w = getmaxx(fil_win);
    wborder(fil_win,0,0,0,0,0,0,0,0);
    if (is_current_window(fil_win)) { wattron(fil_win, A_REVERSE); }
    if (strlen(current_path) > w-2) { // print title
       mvwaddnstr(fil_win, 0,1,current_path,w-2-3);
       waddstr(fil_win, "...");
    } else { mvwaddstr(fil_win,0,1,current_path); }
    if (is_current_window(fil_win)) { wattroff(fil_win, A_REVERSE); }
    int y = 1;// Print entries
    if (entry_count == 0) {  
         mvwaddstr(fil_win,1,1,"Nothing");
    } 
    else { 
         for (int row = 0; row < h - 2; row++) { 
            int i=row;
            if (i>=entry_count) {
                break;
            }
            struct dirent *entry = entry_list[i];
            mvwaddstr(fil_win,y,1,entry->d_name);
            y++;
      }
    }
    y = h-1;
    mvwprintw(fil_win, y,1,"%d", entry_count); // print footer
    wrefresh(fil_win);
}
void list_dir_in_file_list_window(const char *dir) {
    if (entry_list != NULL) free(entry_list);
    entry_count = scandir(dir, &entry_list, NULL, NULL);
    strcpy(current_path, dir);
    refresh_file_list_window();
}
void refresh_contents_window() {
    box(con_win, 0, 0);
    if (is_current_window(con_win)) {wattron(con_win, A_REVERSE);}    
    mvwaddstr(con_win, 0, 1, "Contents");
    if (is_current_window(con_win)) {wattroff(con_win, A_REVERSE);}    
    wrefresh(con_win);
}
void create_contents_window(int h, int w, int y, int x) {
    con_win = newwin(h, w, y, x);
    register_window(con_win);
    refresh_contents_window();
}
void delete_contents_window() { delwin(con_win); con_win = NULL;}
void refresh_debug_window() {
    wborder(dbg_win, '|', '|', '-', '-', '+', '+', '+', '+');
    if (is_current_window(dbg_win)) {wattron(dbg_win, A_REVERSE);}    
    mvwaddstr(dbg_win, 0, 1, "Debug");
    if (is_current_window(dbg_win)) {wattroff(dbg_win, A_REVERSE);}        
    wrefresh(dbg_win);
}
void create_debug_window(int h, int w, int y, int x) {
    dbg_win = newwin(h, w, y, x);
    register_window(dbg_win);    
    refresh_debug_window();
}
void delete_debug_window() {    delwin(dbg_win);    dbg_win = NULL;}
int main(int argc, char *argv[]) {
   if (argc < 2) { printf("Usage: ./main <dir>\n"); exit(1); } // check n. of arguments
    initscr();      // Enter curses mode
    cbreak();       // Disable line buffering
    noecho();       // Disable echoing
    keypad(stdscr, TRUE);   // Enable function keys like F1
    curs_set(0);            // Hide cursor
    refresh();
    char *dir = argv[1];
    int len = strlen(dir);
    if (dir[len-1] == '/') {dir[len-1] = '\0';} 
    //
    int h, w;
    getmaxyx(stdscr, h, w);
    int h_files = h - 1 - 6;
    int w_files = w * 0.3;
    create_menu_bar_window(1, w, 0, 0); // top 
    create_file_list_window(h_files, w_files, 1, 0); // left-center
    create_contents_window(h_files, w - w_files, 1, w_files); // right-center
    create_debug_window(6, w, 1 + h_files, 0); // bottom
    list_dir_in_file_list_window(dir);
    //
    int ch;
    while ((ch = getch()) != 'q') { 
        switch (ch)  {
            case '\t':
               next_window();                
               refresh_menu_bar_window();
               refresh_file_list_window();
               refresh_debug_window();
               refresh_contents_window();
               break;
        }
    }
    //
    delete_menu_bar_window();
    delete_file_list_window();
    delete_contents_window();
    delete_debug_window();
    endwin();       // Exit from curses mode
    return 0;
}
```
![ch08_demo](./ch08_demo.png)

### 9. Color
- Lab task
  - Show the directory entry in yello
  - Press Enter key to enter the selected directory
  - Set background of Debug Window to blue
- Colors API
  - has_colors() to check if color is supported in the current terminal
  - start_colors() before using color
  - Check global variable COLORS
- Window background
  - void bkgdset(chtype ch);
  - void wbkgdset(WINDOW *win, chtype ch);
  - int bkgd(chtype ch);
  - in wbkgd(WINDOW *win, chtype ch);
```c
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <limits.h>
#include <curses.h>
#include <panel.h>
#define WINDOWS 4
#define COLOR_OF_DIR    1
#define COLOR_OF_DEBUG_WIN  2
typedef void (*WIN_KEY_HANDLE)(int);
static WINDOW *windows[WINDOWS];
static WIN_KEY_HANDLE key_handle_funcs[WINDOWS];
static int count = 0;
static int current_window_index = -1;
void init_win_manager() {
    for (int i = 0; i < WINDOWS; i++) {
        windows[i] = NULL;
    }
}
void register_window_with_key_handle(WINDOW *win, 
    WIN_KEY_HANDLE handle) {
    windows[count] = win;
    key_handle_funcs[count] = handle;
    count++;
    if (current_window_index == -1) {
        current_window_index = 0;
    }
}
void register_window(WINDOW *win) {
    windows[count] = win;
    count++;
    if (current_window_index == -1) {
        current_window_index = 0;
    }
}
void next_window() {
    if (current_window_index < count - 1) {
        current_window_index++;
    } else {
        current_window_index = 0;
    }
}
bool is_current_window(WINDOW *win) {
    return windows[current_window_index] == win;
}
void handle_key_in_current_window(int ch) {
    if (0 <= current_window_index && current_window_index < count) {
        if (key_handle_funcs[current_window_index]) {
            key_handle_funcs[current_window_index](ch);
        }
    }
}
static WINDOW *men_win = NULL;
static WINDOW *fil_win = NULL;
static WINDOW *con_win = NULL;
static WINDOW *dbg_win = NULL;
void refresh_menu_bar_window() {
    if (is_current_window(men_win)) {wattron(men_win, A_REVERSE);}
    mvwaddstr(men_win,0,1,"Menu");
    if (is_current_window(men_win)) {wattroff(men_win, A_REVERSE);}
    wrefresh(men_win);
}
void create_menu_bar_window(int h, int w, int y, int x) {
    men_win = newwin(h, w, y, x);
    register_window(men_win);
    refresh_menu_bar_window();
}
void delete_menu_bar_window() { delwin(men_win);     men_win = NULL; }
static int entry_count = 0;
static struct dirent **entry_list = NULL; // from <dirent.h>
static char current_path[PATH_MAX];
static int index_of_selected = -1;
static void handle_key(int ch); // see below for the function definition
void create_file_list_window(int h, int w, int y, int x) {
    fil_win = newwin(h, w, y, x);    
    register_window_with_key_handle(fil_win, handle_key);
    current_path[0] = '\0';
    wborder(fil_win, 0, 0, 0, 0, 0, 0, 0, 0);
    mvwaddstr(fil_win, 0, 1, "File List");
    wrefresh(fil_win);
}
void delete_file_list_window() {    
	delwin(fil_win);    
	fil_win = NULL; 
	free(entry_list); 
	entry_list = NULL; 
	entry_count = 0;
}
void refresh_file_list_window() {
    int h = getmaxy(fil_win);
    int w = getmaxx(fil_win);
    wborder(fil_win,0,0,0,0,0,0,0,0);
    if (is_current_window(fil_win)) { wattron(fil_win, A_REVERSE); }
    if (strlen(current_path) > w-2) { // print title
       mvwaddnstr(fil_win, 0,1,current_path,w-2-3);
       waddstr(fil_win, "...");
    } else { mvwaddstr(fil_win,0,1,current_path); }
    if (is_current_window(fil_win)) { wattroff(fil_win, A_REVERSE); }
    int y = 1;// Print entries
    if (entry_count == 0) {  
         mvwaddstr(fil_win,1,1,"Nothing");
    } 
    else { 
         for (int row = 0; row < h - 2; row++) { 
            int i=row;
            if (i>=entry_count) {
                break;
            }
            struct dirent *entry = entry_list[i];
            if (i == index_of_selected) {
                wattron(fil_win, A_REVERSE);
            }
            if (entry->d_type == DT_DIR) {
                wattron(fil_win, COLOR_PAIR(COLOR_OF_DIR));
            }
            mvwaddstr(fil_win, y, 1, entry->d_name);
            if (entry->d_type == DT_DIR) {
                wattroff(fil_win, COLOR_PAIR(COLOR_OF_DIR));
            }
            if (i == index_of_selected) {
                wattroff(fil_win, A_REVERSE);
            }
            y++;
      }
    }
    y = h-1;
    mvwprintw(fil_win, y,1,"%d", entry_count); // print footer
    wrefresh(fil_win);
}
void list_dir_in_file_list_window(const char *dir) {
    if (entry_list != NULL) free(entry_list);
    entry_count = scandir(dir, &entry_list, NULL, NULL);
    strcpy(current_path, dir);
    refresh_file_list_window();
}
void enter_dir(struct dirent *entry) {
    if (strcmp(entry->d_name, ".") == 0) {
        return;
    }
    char new_dir[PATH_MAX];
    if (strcmp(entry->d_name, "..") == 0) {
        strcpy(new_dir, current_path);
        char *end = strrchr(new_dir, '/');
        *end = '\0';        
    } else {
        sprintf(new_dir, "%s/%s", current_path, entry->d_name);
    }
    list_dir_in_file_list_window(new_dir);
}
static void handle_key(int ch) {
    switch (ch) {
        case KEY_DOWN:
            if (index_of_selected < entry_count - 1) {
                index_of_selected++;
                refresh_file_list_window();
            }
            break;
        case KEY_UP:
            if (index_of_selected > 0) {
                index_of_selected--;
                refresh_file_list_window();
            }
            break;
        case '\n':
            if (0 <= index_of_selected && index_of_selected < entry_count) {
                struct dirent *entry = entry_list[index_of_selected];
                if (entry->d_type == DT_DIR) {
                    enter_dir(entry);
                }
            }
            break;
    }
}
void refresh_contents_window() {
    box(con_win, 0, 0);
    if (is_current_window(con_win)) {wattron(con_win, A_REVERSE);}    
    mvwaddstr(con_win, 0, 1, "Contents");
    if (is_current_window(con_win)) {wattroff(con_win, A_REVERSE);}    
    wrefresh(con_win);
}
void create_contents_window(int h, int w, int y, int x) {
    con_win = newwin(h, w, y, x);
    register_window(con_win);
    refresh_contents_window();
}
void delete_contents_window() { delwin(con_win); con_win = NULL;}
void refresh_debug_window() {    
    wbkgd(dbg_win, COLOR_PAIR(COLOR_OF_DEBUG_WIN));
    wborder(dbg_win, '|', '|', '-', '-', '+', '+', '+', '+');
    if (is_current_window(dbg_win)) {wattron(dbg_win, A_REVERSE);}    
    mvwaddstr(dbg_win, 0, 1, "Debug");
    if (is_current_window(dbg_win)) {wattroff(dbg_win, A_REVERSE);}        
    wrefresh(dbg_win);
}
void create_debug_window(int h, int w, int y, int x) {
    dbg_win = newwin(h, w, y, x);
    register_window(dbg_win);    
    refresh_debug_window();
}
void delete_debug_window() {    delwin(dbg_win);    dbg_win = NULL;}
int main(int argc, char *argv[]) {
   if (argc < 2) { printf("Usage: ./main <dir>\n"); exit(1); } // check n. of arguments
    initscr();      // Enter curses mode
    cbreak();       // Disable line buffering
    noecho();       // Disable echoing
    keypad(stdscr, TRUE);   // Enable function keys like F1
    curs_set(0);            // Hide cursor
    refresh();
    if (has_colors()) {
        start_color();
        init_pair(COLOR_OF_DIR, COLOR_YELLOW, COLOR_BLACK);
        init_pair(COLOR_OF_DEBUG_WIN, COLOR_WHITE, COLOR_BLUE);
    }

    char *dir = argv[1];
    int len = strlen(dir);
    if (dir[len-1] == '/') {dir[len-1] = '\0';} 
    //
    int h, w;
    getmaxyx(stdscr, h, w);
    int h_files = h - 1 - 6;
    int w_files = w * 0.3;
    create_menu_bar_window(1, w, 0, 0); // top 
    create_file_list_window(h_files, w_files, 1, 0); // left-center
    create_contents_window(h_files, w - w_files, 1, w_files); // right-center
    create_debug_window(6, w, 1 + h_files, 0); // bottom
    list_dir_in_file_list_window(dir);
    //
    int ch;
    while ((ch = getch()) != 'q') { 
        switch (ch)  {
            case '\t':
               next_window();                
               refresh_menu_bar_window();
               refresh_file_list_window();
               refresh_debug_window();
               refresh_contents_window();
               break;
        }
    }
    //
    delete_menu_bar_window();
    delete_file_list_window();
    delete_contents_window();
    delete_debug_window();
    endwin();       // Exit from curses mode
    return 0;
}
```
![ch09_etc](./ch09_etc.png)

### 10. Input
- Lab task
  - Show a help window
  - Print ncurses version and terminal names
  - Allow user to input new directory path to list
- getch and ungetch
  - Read a character:
    - int getch(void);
    - int wgetch(WINDOW *win);
    - int mvgetch(int y, int x);
    - int mvwgetch(WINDOW *win, int y, int x)
  - Put character ch into input queue so that the next getch() will read it
    - int ungetch(int ch);
- Read characters
  - int getstr(char *str);
  - int wgetstr(WINDOW *win, char *str);
  - int mvegetstr(int y, int x, char *str);
  - int mvwgetstr(WINDOW *win, int y, int x, char *str);
- Read string from cursor position
  - int instr(char *str);
  - int winstr(WINDOW *win, char *str);
- Read chtype String from cursor position
  - int inchsgtr(chtype *chstr);
  - int winchstr(WINDOW *win, chtype *chstr);
- Read from the formatted string
```c
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <limits.h>
#include <curses.h>
#include <panel.h>
#define WINDOWS 4
#define COLOR_OF_DIR    1
#define COLOR_OF_DEBUG_WIN  2
static WINDOW *men_win = NULL;
static WINDOW *fil_win = NULL;
static WINDOW *con_win = NULL;
static WINDOW *dbg_win = NULL;
static WINDOW *hlp_win = NULL;
void show_help_window(char *dir, int len) {
    int h = 20;
    int w = 60;
    WINDOW *hlp_win = newwin(h, w, (LINES - h) / 2, (COLS - w) / 2);
    box(hlp_win, 0, 0);
    mvwaddstr(hlp_win, 0, 1, "Help");
    int y = 3;
    int x = 3;
    mvwprintw(hlp_win, y++, x, "ncurses : %s", curses_version());
    mvwprintw(hlp_win, y++, x, "terminal: %s", longname());
    mvwprintw(hlp_win, y++, x, "terminal: %s", termname());
    // Dir: _____________________
    mvwprintw(hlp_win, y, x, "Dir: ");
    x += strlen("Dir: ");
    wbkgdset(hlp_win, A_UNDERLINE);
    waddstr(hlp_win, "                   ");
    wbkgdset(hlp_win, A_NORMAL);
    wmove(hlp_win, y, x);
    curs_set(1);
    echo(); // print back
    nocbreak();
    wgetnstr(hlp_win, dir, len);
    curs_set(0);
    noecho();
    cbreak();
    wrefresh(hlp_win);
    delwin(hlp_win);
}
typedef void (*WIN_KEY_HANDLE)(int);
static WINDOW *windows[WINDOWS];
static WIN_KEY_HANDLE key_handle_funcs[WINDOWS];
static int count = 0;
static int current_window_index = -1;
void init_win_manager() {
    for (int i = 0; i < WINDOWS; i++) {
        windows[i] = NULL;
    }
}
void register_window_with_key_handle(WINDOW *win, 
    WIN_KEY_HANDLE handle) {
    windows[count] = win;
    key_handle_funcs[count] = handle;
    count++;
    if (current_window_index == -1) {
        current_window_index = 0;
    }
}
void register_window(WINDOW *win) {
    windows[count] = win;
    count++;
    if (current_window_index == -1) {
        current_window_index = 0;
    }
}
void next_window() {
    if (current_window_index < count - 1) {
        current_window_index++;
    } else {
        current_window_index = 0;
    }
}
bool is_current_window(WINDOW *win) {
    return windows[current_window_index] == win;
}
void handle_key_in_current_window(int ch) {
    if (0 <= current_window_index && current_window_index < count) {
        if (key_handle_funcs[current_window_index]) {
            key_handle_funcs[current_window_index](ch);
        }
    }
}
void refresh_menu_bar_window() {
    if (is_current_window(men_win)) {wattron(men_win, A_REVERSE);}
    mvwaddstr(men_win,0,1,"Menu");
    if (is_current_window(men_win)) {wattroff(men_win, A_REVERSE);}
    wrefresh(men_win);
}
void create_menu_bar_window(int h, int w, int y, int x) {
    men_win = newwin(h, w, y, x);
    register_window(men_win);
    refresh_menu_bar_window();
}
void delete_menu_bar_window() { delwin(men_win);     men_win = NULL; }
static int entry_count = 0;
static struct dirent **entry_list = NULL; // from <dirent.h>
static char current_path[PATH_MAX];
static int index_of_selected = -1;
static void handle_key(int ch); // see below for the function definition
void create_file_list_window(int h, int w, int y, int x) {
    fil_win = newwin(h, w, y, x);    
    register_window_with_key_handle(fil_win, handle_key);
    current_path[0] = '\0';
    wborder(fil_win, 0, 0, 0, 0, 0, 0, 0, 0);
    mvwaddstr(fil_win, 0, 1, "File List");
    wrefresh(fil_win);
}
void delete_file_list_window() {    
	delwin(fil_win);    
	fil_win = NULL; 
	free(entry_list); 
	entry_list = NULL; 
	entry_count = 0;
}
void refresh_file_list_window() {
    int h = getmaxy(fil_win);
    int w = getmaxx(fil_win);
    wclear(fil_win);
    wborder(fil_win,0,0,0,0,0,0,0,0);
    if (is_current_window(fil_win)) { wattron(fil_win, A_REVERSE); }
    if (strlen(current_path) > w-2) { // print title
       mvwaddnstr(fil_win, 0,1,current_path,w-2-3);
       waddstr(fil_win, "...");
    } else { mvwaddstr(fil_win,0,1,current_path); }
    if (is_current_window(fil_win)) { wattroff(fil_win, A_REVERSE); }
    int y = 1;// Print entries
    if (entry_count == 0) {  
         mvwaddstr(fil_win,1,1,"Nothing");
    } 
    else { 
         for (int row = 0; row < h - 2; row++) { 
            int i=row;
            if (i>=entry_count) {
                break;
            }
            struct dirent *entry = entry_list[i];
            if (i == index_of_selected) {
                wattron(fil_win, A_REVERSE);
            }
            if (entry->d_type == DT_DIR) {
                wattron(fil_win, COLOR_PAIR(COLOR_OF_DIR));
            }
            mvwaddstr(fil_win, y, 1, entry->d_name);
            if (entry->d_type == DT_DIR) {
                wattroff(fil_win, COLOR_PAIR(COLOR_OF_DIR));
            }
            if (i == index_of_selected) {
                wattroff(fil_win, A_REVERSE);
            }
            y++;
      }
    }
    y = h-1;
    mvwprintw(fil_win, y,1,"%d", entry_count); // print footer
    wrefresh(fil_win);
}
void list_dir_in_file_list_window(const char *dir) {
    if (entry_list != NULL) free(entry_list);
    entry_list = NULL;
    entry_count = scandir(dir, &entry_list, NULL, NULL);
    index_of_selected = 0;
    strcpy(current_path, dir);
    refresh_file_list_window();
}
void enter_dir(struct dirent *entry) {
    if (strcmp(entry->d_name, ".") == 0) {
        return;
    }
    char new_dir[PATH_MAX];
    if (strcmp(entry->d_name, "..") == 0) {
        strcpy(new_dir, current_path);
        char *end = strrchr(new_dir, '/');
        *end = '\0';        
    } else {
        sprintf(new_dir, "%s/%s", current_path, entry->d_name);
    }
    list_dir_in_file_list_window(new_dir);
}

static void handle_key(int ch) {
    switch (ch) {
        case KEY_DOWN:
            if (index_of_selected < entry_count - 1) {
                index_of_selected++;
                refresh_file_list_window();
            }
            break;
        case KEY_UP:
            if (index_of_selected > 0) {
                index_of_selected--;
                refresh_file_list_window();
            }
            break;
        case '\n':
            if (0 <= index_of_selected && index_of_selected < entry_count) {
                struct dirent *entry = entry_list[index_of_selected];
                if (entry->d_type == DT_DIR) {
                    enter_dir(entry);
                }
            }
            break;
    }
}
void refresh_contents_window() {
    box(con_win, 0, 0);
    if (is_current_window(con_win)) {wattron(con_win, A_REVERSE);}    
    mvwaddstr(con_win, 0, 1, "Contents");
    if (is_current_window(con_win)) {wattroff(con_win, A_REVERSE);}    
    wrefresh(con_win);
}
void create_contents_window(int h, int w, int y, int x) {
    con_win = newwin(h, w, y, x);
    register_window(con_win);
    refresh_contents_window();
}
void delete_contents_window() { delwin(con_win); con_win = NULL;}
void refresh_debug_window() {    
    wbkgd(dbg_win, COLOR_PAIR(COLOR_OF_DEBUG_WIN));
    wborder(dbg_win, '|', '|', '-', '-', '+', '+', '+', '+');
    if (is_current_window(dbg_win)) {wattron(dbg_win, A_REVERSE);}    
    mvwaddstr(dbg_win, 0, 1, "Debug");
    if (is_current_window(dbg_win)) {wattroff(dbg_win, A_REVERSE);}        
    wrefresh(dbg_win);
}
void create_debug_window(int h, int w, int y, int x) {
    dbg_win = newwin(h, w, y, x);
    register_window(dbg_win);    
    refresh_debug_window();
}
void delete_debug_window() {    delwin(dbg_win);    dbg_win = NULL;}
int main(int argc, char *argv[]) {
   if (argc < 2) { printf("Usage: ./main <dir>\n"); exit(1); } // check n. of arguments
    initscr();      // Enter curses mode
    cbreak();       // Disable line buffering
    noecho();       // Disable echoing
    keypad(stdscr, TRUE);   // Enable function keys like F1
    curs_set(0);            // Hide cursor
    refresh();
    if (has_colors()) {
        start_color();
        init_pair(COLOR_OF_DIR, COLOR_YELLOW, COLOR_BLACK);
        init_pair(COLOR_OF_DEBUG_WIN, COLOR_WHITE, COLOR_BLUE);
    }

    char *dir = argv[1];
    int len = strlen(dir);
    if (dir[len-1] == '/') {dir[len-1] = '\0';} 
    //
    int h, w;
    getmaxyx(stdscr, h, w);
    int h_files = h - 1 - 6;
    int w_files = w * 0.3;
    create_menu_bar_window(1, w, 0, 0); // top 
    create_file_list_window(h_files, w_files, 1, 0); // left-center
    create_contents_window(h_files, w - w_files, 1, w_files); // right-center
    create_debug_window(6, w, 1 + h_files, 0); // bottom
    list_dir_in_file_list_window(dir);
    //
    int ch;
    while ((ch = getch()) != 'q') { 
        switch (ch)  {
            case 'h': {
                char dir[PATH_MAX];
                show_help_window(dir, PATH_MAX - 1);
                if (strlen(dir) > 0) {
                    list_dir_in_file_list_window(dir);
                }
                refresh_file_list_window();
                refresh_contents_window();
                break;
            }
           case '\t':
               next_window();                
               refresh_menu_bar_window();
               refresh_file_list_window();
               refresh_debug_window();
               refresh_contents_window();
               break;
            default:
                handle_key_in_current_window(ch);
                break;
       }
    }
    //
    delete_menu_bar_window();
    delete_file_list_window();
    delete_contents_window();
    delete_debug_window();
    endwin();       // Exit from curses mode
    return 0;
}
```
![ch10_demo](./ch10_demo.png)

### 11. Mouse
- Lab task
  - Click window using a mouse to activate it
  - Click an entry in the file list window to select a file or directory
  - Double-click on one directory to enter it
- How to enable a mouse
  - Make mouse events visible
    - mmask_t mousemask(mmask_t netwmask, mmask_t *oldmask);
    - ALL_MOUSE_EVENTS for reporting all mouse event
  - Mouse event is reported as KEY_MOUSE from getch()
  - Read the event data and pop the event off the queue
    - int getmouse(MEVENT *event);
    - typedef struct { short id; int x,y,z; mmask_t bstate; } EVENT;
- Mouse APIs
  - bool wenclose(const WINDOW *win, int y, in t x);
  - bool mouse_trafo(int* pY, int* pX, bool to_screen);
  - bool wmouse_trafo(const WINDOW* win, int* pY, int* pX, bool to_screen);
  - bool has_mouse(void);
```c
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <limits.h>
#include <curses.h>
#include <panel.h>
#define WINDOWS 4
#define COLOR_OF_DIR    1
#define COLOR_OF_DEBUG_WIN  2
static WINDOW *men_win = NULL;
static WINDOW *fil_win = NULL;
static WINDOW *con_win = NULL;
static WINDOW *dbg_win = NULL;
static WINDOW *hlp_win = NULL;
typedef void (*WIN_KEY_HANDLE)(int);
typedef void (*WIN_MOUSE_HANDLE)(MEVENT *event);
typedef void (*WIN_REFRESH)(); // defining a function pointer
static WINDOW *windows[WINDOWS];
static WIN_KEY_HANDLE key_handle_funcs[WINDOWS];
static WIN_MOUSE_HANDLE mouse_handle_funcs[WINDOWS];
static WIN_REFRESH refresh_funcs[WINDOWS];
static int count = 0;
static int current_window_index = -1;
void show_help_window(char *dir, int len) {
    int h = 20;
    int w = 60;
    WINDOW *hlp_win = newwin(h, w, (LINES - h) / 2, (COLS - w) / 2);
    box(hlp_win, 0, 0);
    mvwaddstr(hlp_win, 0, 1, "Help");
    int y = 3;
    int x = 3;
    mvwprintw(hlp_win, y++, x, "ncurses : %s", curses_version());
    mvwprintw(hlp_win, y++, x, "terminal: %s", longname());
    mvwprintw(hlp_win, y++, x, "terminal: %s", termname());
    // Dir: _____________________
    mvwprintw(hlp_win, y, x, "Dir: ");
    x += strlen("Dir: ");
    wbkgdset(hlp_win, A_UNDERLINE);
    waddstr(hlp_win, "                   ");
    wbkgdset(hlp_win, A_NORMAL);
    wmove(hlp_win, y, x);
    curs_set(1);
    echo(); // print back
    nocbreak();
    wgetnstr(hlp_win, dir, len);
    curs_set(0);
    noecho();
    cbreak();
    wrefresh(hlp_win);
    delwin(hlp_win);
}
void init_win_manager() {
    for (int i = 0; i < WINDOWS; i++) {
        windows[i] = NULL;
    }
}
void register_window_with_key_handle(WINDOW *win, 
    WIN_REFRESH refresh_func,
    WIN_KEY_HANDLE key_handle_func,
    WIN_MOUSE_HANDLE mouse_handle_func) {
    windows[count] = win;
    key_handle_funcs[count] = key_handle_func;
    mouse_handle_funcs[count] = mouse_handle_func;
    refresh_funcs[count] = refresh_func;
    count++;
    if (current_window_index == -1) {
        current_window_index = 0;
    }
}
void register_window(WINDOW *win, WIN_REFRESH refresh_func) {
    register_window_with_key_handle(win, refresh_func, NULL, NULL);
}
void next_window() {
    if (current_window_index < count - 1) {
        current_window_index++;
    } else {
        current_window_index = 0;
    }
}
bool is_current_window(WINDOW *win) {
    return windows[current_window_index] == win;
}
static void wm_handle_mouse(MEVENT *event) {
    // which window is under this mouse click
    int window_index = -1;
    if ((event->bstate & BUTTON1_CLICKED) ||
        (event->bstate & BUTTON1_DOUBLE_CLICKED)) {
        for (int i = 0; i < count; i++) {
            if (wenclose(windows[i], event->y, event->x)) {
                window_index = i;
                break;
            }
        }
        if (window_index == -1) {
            return;
        }
        // update current_window_index
        int previous_window_index = current_window_index;
        current_window_index = window_index;
        // refresh the previous current window
        if (refresh_funcs[previous_window_index]) {
            refresh_funcs[previous_window_index]();
        }
        // Forward mouse event
        if (mouse_handle_funcs[current_window_index]) {
            mouse_handle_funcs[current_window_index](event);
        }
        // refresh the current window
        if (refresh_funcs[current_window_index]) {
            refresh_funcs[current_window_index]();
        }
    }
}
void handle_key_in_current_window(int ch) {
    if (ch == KEY_MOUSE) {
        MEVENT event;
        if (getmouse(&event) == OK) {
            wm_handle_mouse(&event);
        }
        return;
    }
    if (0 <= current_window_index && current_window_index < count) {
        if (key_handle_funcs[current_window_index]) {
            key_handle_funcs[current_window_index](ch);
        }
    }
}
void refresh_menu_bar_window() {
    if (is_current_window(men_win)) {wattron(men_win, A_REVERSE);}
    mvwaddstr(men_win,0,1,"Menu");
    if (is_current_window(men_win)) {wattroff(men_win, A_REVERSE);}
    wrefresh(men_win);
}
void create_menu_bar_window(int h, int w, int y, int x) {
    men_win = newwin(h, w, y, x);
    register_window(men_win, refresh_menu_bar_window);
    refresh_menu_bar_window();
}
void delete_menu_bar_window() { delwin(men_win);     men_win = NULL; }
static int entry_count = 0;
static struct dirent **entry_list = NULL; // from <dirent.h>
static char current_path[PATH_MAX];
static int index_of_selected = -1;
static void handle_key(int ch); // see below for the function definition
void refresh_file_list_window();
static void fil_handle_mouse(MEVENT *event) {
    int y = event->y;
    int x = event->x;
    wmouse_trafo(fil_win, &y, &x, FALSE);
    if ((event->bstate & BUTTON1_CLICKED) ||
        (event->bstate & BUTTON1_DOUBLE_CLICKED)) {
        index_of_selected = y - 1;
    }
    if (event->bstate & BUTTON1_DOUBLE_CLICKED) {
        if (index_of_selected >= 0 && index_of_selected < entry_count) {
            struct dirent *entry = entry_list[index_of_selected];
            if (entry->d_type == DT_DIR) {
                enter_dir(entry);
            }
        }
    }
}
void create_file_list_window(int h, int w, int y, int x) {
    fil_win = newwin(h, w, y, x);    
    register_window_with_key_handle(fil_win, refresh_file_list_window, 
        handle_key, fil_handle_mouse);
    current_path[0] = '\0';
    wborder(fil_win, 0, 0, 0, 0, 0, 0, 0, 0);
    mvwaddstr(fil_win, 0, 1, "File List");
    wrefresh(fil_win);
}
void delete_file_list_window() {    
	delwin(fil_win);    
	fil_win = NULL; 
	free(entry_list); 
	entry_list = NULL; 
	entry_count = 0;
}
void refresh_file_list_window() {
    int h = getmaxy(fil_win);
    int w = getmaxx(fil_win);
    wclear(fil_win);
    wborder(fil_win,0,0,0,0,0,0,0,0);
    if (is_current_window(fil_win)) { wattron(fil_win, A_REVERSE); }
    if (strlen(current_path) > w-2) { // print title
       mvwaddnstr(fil_win, 0,1,current_path,w-2-3);
       waddstr(fil_win, "...");
    } else { mvwaddstr(fil_win,0,1,current_path); }
    if (is_current_window(fil_win)) { wattroff(fil_win, A_REVERSE); }
    int y = 1;// Print entries
    if (entry_count == 0) {  
         mvwaddstr(fil_win,1,1,"Nothing");
    } 
    else { 
         for (int row = 0; row < h - 2; row++) { 
            int i=row;
            if (i>=entry_count) {
                break;
            }
            struct dirent *entry = entry_list[i];
            if (i == index_of_selected) {
                wattron(fil_win, A_REVERSE);
            }
            if (entry->d_type == DT_DIR) {
                wattron(fil_win, COLOR_PAIR(COLOR_OF_DIR));
            }
            mvwaddstr(fil_win, y, 1, entry->d_name);
            if (entry->d_type == DT_DIR) {
                wattroff(fil_win, COLOR_PAIR(COLOR_OF_DIR));
            }
            if (i == index_of_selected) {
                wattroff(fil_win, A_REVERSE);
            }
            y++;
      }
    }
    y = h-1;
    mvwprintw(fil_win, y,1,"%d", entry_count); // print footer
    wrefresh(fil_win);
}
void list_dir_in_file_list_window(const char *dir) {
    if (entry_list != NULL) free(entry_list);
    entry_list = NULL;
    entry_count = scandir(dir, &entry_list, NULL, NULL);
    index_of_selected = 0;
    strcpy(current_path, dir);
    refresh_file_list_window();
}
void enter_dir(struct dirent *entry) {
    if (strcmp(entry->d_name, ".") == 0) {
        return;
    }
    char new_dir[PATH_MAX];
    if (strcmp(entry->d_name, "..") == 0) {
        strcpy(new_dir, current_path);
        char *end = strrchr(new_dir, '/');
        *end = '\0';        
    } else {
        sprintf(new_dir, "%s/%s", current_path, entry->d_name);
    }
    list_dir_in_file_list_window(new_dir);
}

static void handle_key(int ch) {
    switch (ch) {
        case KEY_DOWN:
            if (index_of_selected < entry_count - 1) {
                index_of_selected++;
                refresh_file_list_window();
            }
            break;
        case KEY_UP:
            if (index_of_selected > 0) {
                index_of_selected--;
                refresh_file_list_window();
            }
            break;
        case '\n':
            if (0 <= index_of_selected && index_of_selected < entry_count) {
                struct dirent *entry = entry_list[index_of_selected];
                if (entry->d_type == DT_DIR) {
                    enter_dir(entry);
                }
            }
            break;
    }
}
void refresh_contents_window() {
    box(con_win, 0, 0);
    if (is_current_window(con_win)) {wattron(con_win, A_REVERSE);}    
    mvwaddstr(con_win, 0, 1, "Contents");
    if (is_current_window(con_win)) {wattroff(con_win, A_REVERSE);}    
    wrefresh(con_win);
}
void create_contents_window(int h, int w, int y, int x) {
    con_win = newwin(h, w, y, x);
    register_window(con_win,refresh_contents_window);
    refresh_contents_window();
}
void delete_contents_window() { delwin(con_win); con_win = NULL;}
void refresh_debug_window() {    
    wbkgd(dbg_win, COLOR_PAIR(COLOR_OF_DEBUG_WIN));
    wborder(dbg_win, '|', '|', '-', '-', '+', '+', '+', '+');
    if (is_current_window(dbg_win)) {wattron(dbg_win, A_REVERSE);}    
    mvwaddstr(dbg_win, 0, 1, "Debug");
    if (is_current_window(dbg_win)) {wattroff(dbg_win, A_REVERSE);}        
    wrefresh(dbg_win);
}
void create_debug_window(int h, int w, int y, int x) {
    dbg_win = newwin(h, w, y, x);
    register_window(dbg_win,refresh_debug_window);    
    refresh_debug_window();
}
void delete_debug_window() {    delwin(dbg_win);    dbg_win = NULL;}
int main(int argc, char *argv[]) {
   if (argc < 2) { printf("Usage: ./main <dir>\n"); exit(1); } // check n. of arguments
    initscr();      // Enter curses mode
    cbreak();       // Disable line buffering
    noecho();       // Disable echoing
    keypad(stdscr, TRUE);   // Enable function keys like F1
    curs_set(0);            // Hide cursor
    refresh();
    if (has_colors()) {
        start_color();
        init_pair(COLOR_OF_DIR, COLOR_YELLOW, COLOR_BLACK);
        init_pair(COLOR_OF_DEBUG_WIN, COLOR_WHITE, COLOR_BLUE);
    }
    mousemask(ALL_MOUSE_EVENTS, NULL);
    char *dir = argv[1];
    int len = strlen(dir);
    if (dir[len-1] == '/') {dir[len-1] = '\0';} 
    //
    int h, w;
    getmaxyx(stdscr, h, w);
    int h_files = h - 1 - 6;
    int w_files = w * 0.3;
    init_win_manager();
    create_menu_bar_window(1, w, 0, 0); // top 
    create_file_list_window(h_files, w_files, 1, 0); // left-center
    create_contents_window(h_files, w - w_files, 1, w_files); // right-center
    create_debug_window(6, w, 1 + h_files, 0); // bottom
    list_dir_in_file_list_window(dir);
    //
    int ch;
    while ((ch = getch()) != 'q') { 
        switch (ch)  {
            case 'h': {
                char dir[PATH_MAX];
                show_help_window(dir, PATH_MAX - 1);
                if (strlen(dir) > 0) {
                    list_dir_in_file_list_window(dir);
                }
                refresh_file_list_window();
                refresh_contents_window();
                break;
            }
           case '\t':
               next_window();                
               refresh_menu_bar_window();
               refresh_file_list_window();
               refresh_debug_window();
               refresh_contents_window();
               break;
            default:
                handle_key_in_current_window(ch);
                break;
       }
    }
    //
    delete_menu_bar_window();
    delete_file_list_window();
    delete_contents_window();
    delete_debug_window();
    endwin();       // Exit from curses mode
    return 0;
}
```
![ch11_demo](./ch11_demo.png)

### 12. Scroll

### 13. Pad

### 14. Make file list window scroll-able

## Section 3: NCURSES Extensions

### 15. Panel
### 16. Menu
### 17. Menu - More
### 18. Form
### 19. Form - More
