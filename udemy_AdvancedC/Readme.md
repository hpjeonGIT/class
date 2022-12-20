## Summary
- Title: Advanced C Programming Masterclass: Pointers & Memory in C
- Instructor: Vlad Budnitski

## Section 1: Welcome Aboard!

1. About the course


## Section 2: Pointers - Introduction to Basics

2. Introduction to Pointers + General Visualization
- Simple swap code
```c
int a = 5, b=7, tmp;
tmp = a;
a = b;
b = tmp;
```

3. Why using Pointers?

4. Another reasons and motivation behind the usage of pointers
- Passing arguments by pointer
- Returnning more than just one thing from a function
- Dynamic memory allocation

5. Declaration & Usage of Pointers
- `<data type> *<variable name>`
- Dereferencing
```c
int a = 5;
int *p;
p = &a;
```
  - Adress of "a"
    - printf("%p",&a);
    - printf("%p",p);
  - Value of "a"
    - printf("%d",a);
    - printf("%d",*p);

6. Pointers Initialization
- `int *p = NULL;` avoids the crash from uninitialized pointer value

7. Short Dereference
```c
int a;
int *p;
p = &a;
```

8. Challenge #1 - Printing value & address of a variable
```c
#include <stdio.h>
int main() {
  int a = 100;
  printf("%d at %p\n", a, &a);
  return 0;
}
```

9. Challenge #2 - Guessing the output

10. Exclusive Pass By Reference Guide
- How to return min/max value simultaneously?
- Returning 2 integer values simultaneously
  - No tuple support at C
  - Let arguments be passed as references
  - No return value - void function
```c
#include <stdio.h>
void findMinMax(int x, int y, int *pMax, int *pMin){
  if (x > y) {
     pMax = x;
     pMin = y;
  } else {
     pMax = y;
     pMin = x;
  }
}
int main() {
  int a=5, b=7;
  int pMax, pMin;
  findMinMax(a,b,&pMax,&pMin);
  printf("max = %d  min = %d\n", pMax, pMin);
```
- This runs OK
```c
  int a=5, b=7;
  int *pMax, *pMin;
  findMinMax(a,b,pMax,pMin);
  printf("max = %d  min = %d\n", *pMax, *pMin);
```
- But this segfaults
  - Why?
- Ref: https://en.wikipedia.org/wiki/X86_calling_conventions

11. Quick Summary

## Section 3: Pointers Arithmetic & "sizeof" operator

12. Pointers Arithmetic Introduction
- Pointer Arithmetic -> addresses Arithmetic
```c
#include <stdio.h>
int main() {
  int *ptr;
  int grades[3] = {80,90,100};
  printf("grades are %d\n", *(grades+1)); /* prints 90 */
  printf("%p %p %p \n", &grades[0]+1, &grades[1],&grades[2]-1); /* three of them are same */
  printf("%p %p\n", &grades, &grades[0]); /* both addresses are same */
  return 0;
}
```
- Note `printf("%p %p\n", &grades+1, &grades[0]+1);` yields differences. Why?
  - `printf("%ld %ld\n", sizeof(int), sizeof(grades));`
  - Size of grades[0] is 4, as int. Size of grades is 4x3 = 12

13. Pointers Arithmetic Examples
- The lecture is wrong. double grades[3]  is 8byte*3 and the increment/decrement will be 24 per each of +1 or -1

14. Rules - Summary
- `printf("%ld \n", &grades[2] - &grades[1]);` yields 1

15. Pointers Arithmetic - DIY Exercises
```c
#include <stdio.h>
int main() {
  int num = 30;
  int *p;
  printf("%d\n",num);  /* 30 */
  p = &num;
  printf("%p\n", &num);/* 0x7fff73ce59ec */
  printf("%p\n",p);    /* 0x7fff73ce59ec */
  *p = 20;
  printf("%d\n",num);  /* 20 */
  return 0;
}
```

16. The "sizeof" Operator - Introduction

17. The "sizeof" Operator - Basic Practice

18. "sizeof" & Static Arrays
- Very low sound volume

19. "sizeof" & Pointers
- Regardless of the data type, sizeof(ptr) yields:
  - 32bit OS: 4
  - 64bit OS: 8
```c
#include <stdio.h>
int main() {
  int *p;
  double *x;
  printf("%ld %ld\n",sizeof(p),  sizeof(x));  /* 8 8 */
  printf("%ld %ld\n",sizeof(*p), sizeof(*x)); /* 4 8 */
  return 0;
}
```

## Section 4: Pointers Concept and Beyond

20. Exercise - Swap function

21. Milestone #1 - develop your real swap function!

22. Milestone #2 - Solution
```c
#include <stdio.h>
void swap(int *a, int *b) {
  int t;
  t = *a;
  *a = *b;
  *b = t; 
}
int main() {
  int x = 7, y = 30;
  swap(&x,&y);
  printf("%d %d \n ", x,y);
  return 0;
}
```

23. Multiple Indirection
- pointer to pointer to int
```c
#include <stdio.h>
int main() {
  int a = 5;
  int *ptr1 = &a;
  int **ptr2 = &ptr1;
  int ***ptr3 = &ptr2;
  printf("%d %d %d %d\n", a, *ptr1, **ptr2, ***ptr3); /* 5 5 5 5 */
  return 0;
}
```

24. Generic Pointer (void *) - what is "void *" ?
- Generic Universal Pointer
- Points to an address of ANY TYPE of data
- Why useful?
  - General purpose functions: integer, double, unknown type
  - Returned pointer type
- **casting/dereference is required** to access the value
```c
#include <stdio.h>
int main() {
  int a = 5;
  int *ptr1 = &a;
  void *ptr = &a;
  printf("%d %d %d\n", a, *ptr1, *(int*) ptr);
  return 0;
}
```

25. Generic swap function (including memcpy function)
- void pointer doesn't carry the size of data type, and we need 3rd argument for the generic swap function
```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
void generic_swap(void *a, void *b, unsigned size) {
  void *temp = malloc(size);
  memcpy(temp, a, size);
  memcpy(a,b,size);
  memcpy(b,temp,size);
  free(temp);
}
int main() {
  int x = 7, y = 30;
  void *a, *b;
  a = &x;
  b = &y;
  generic_swap(a,b,sizeof(x));
  printf("%d %d \n ", x,y);
  return 0;
}
```
- Compare with Lecture 22

## Section 5: Arrays & Pointers - Theory & Practical Exercises
