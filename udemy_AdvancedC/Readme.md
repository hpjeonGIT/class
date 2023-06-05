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
  - Yields following assembly:
  ```asm

  68:	48 8d 4d ec          	lea    -0x14(%rbp),%rcx
  6c:	48 8d 55 e8          	lea    -0x18(%rbp),%rdx
  70:	8b 75 f4             	mov    -0xc(%rbp),%esi
  73:	8b 45 f0             	mov    -0x10(%rbp),%eax
  ```
```c
  int a=5, b=7;
  int *pMax, *pMin;
  findMinMax(a,b,pMax,pMin);
  printf("max = %d  min = %d\n", *pMax, *pMin);
```
- But this segfaults
  - Why? TBD
  - The corresponding assembly block:
  ```asm
    69:	48 8b 4d f8          	mov    -0x8(%rbp),%rcx
  6d:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  71:	8b 75 ec             	mov    -0x14(%rbp),%esi
  74:	8b 45 e8             	mov    -0x18(%rbp),%eax
  ```
  - Not LEA but MOV
    - Note that pMax and pMin are passed by value
  - `int *pMax` assigns a place holder in the heap, not allocating any memory to store value yet. findMinMax() will crash as it cannot access/store value
  - `int pMax` allocates memory in stack, having memory space to store/access value
- Following code works OK
```c
#include <stdio.h>
#include <stdlib.h>
void findMinMax(int x, int y, int *pMax, int *pMin){
  if (x > y) {
     *pMax = x;
     *pMin = y;
  } else {
     *pMax = y;
     *pMin = x;
  }
}
int main() {
  int a=5, b=7;
  int *pMax=NULL, *pMin=NULL;
  pMax = malloc(sizeof(int));
  pMin = malloc(sizeof(int));
  findMinMax(a,b,pMax,pMin);
  printf("max = %d  min = %d\n", *pMax, *pMin);
  free(pMax);
  free(pMin);
  return 0;
}
```
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
- Or double pointer (**)
- Use indirection when 
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
  - Mis-casting yields wrong value
  - In C++, std::any would be an alternative
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
#include <string.h> /* memcpy() */
#include <stdlib.h> /* malloc() */
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

52. ATTENTION! Problem with returning static array from a function
```c
#define SIZE 5
int * createArray() {
    int myArr[SIZE] = { 5,4,3,2,1 };
    return myArr;
}
```
- This function returns a local variable, which is invalid in main()
  - C doesn't return static arrays
  - C can return dynamic arrays
  - Ref: https://stackoverflow.com/questions/11656532/returning-an-array-using-c

## Section 6: Arrays Swapping Exercises - Intermediate to Expert Questions!

70.SwapArray(01)
```c
#include <stdio.h>
#include <stdlib.h>
#define SIZE 3
void swap(int ** ptr1, int ** ptr2) {
  int * tmp;
  tmp = *ptr1;
  *ptr1 = *ptr2;
  *ptr2 = tmp;
}
int main() {
  /*
  int A1[SIZE] = {1,2,3};
  int A2[SIZE] = {5,4,3};
  */
  int *A1 = malloc(sizeof(int)*SIZE); A1[0] = 1; A1[1] = 2; A1[2] = 3;
  int *A2 = malloc(sizeof(int)*SIZE); A2[0] = 5; A2[1] = 4; A2[2] = 0;
  for (int i=0;i<SIZE;i++) printf ("%d %d \n", A1[i], A2[i]);
  swap(&A1,&A2);
  for (int i=0;i<SIZE;i++) printf ("%d %d \n", A1[i], A2[i]);
  free(A1);
  free(A2);
  return 0;
}
```
  - Static array will not be able to use the swap() function here
- Using void pointer
```c
#include <stdio.h>
#include <stdlib.h>
#define SIZE 3
void swap(void ** ptr1, void ** ptr2) {
  void * tmp;
  tmp = *ptr1;
  *ptr1 = *ptr2;
  *ptr2 = tmp;
}
int main() {
  int *A1 = malloc(sizeof(int)*SIZE); A1[0] = 1; A1[1] = 2; A1[2] = 3;
  int *A2 = malloc(sizeof(int)*SIZE); A2[0] = 5; A2[1] = 4; A2[2] = 0;
  for (int i=0;i<SIZE;i++) printf ("%d %d \n", A1[i], A2[i]);
  swap((void*) A1,(void *)A2);
  for (int i=0;i<SIZE;i++) printf ("%d %d \n", A1[i], A2[i]);
  free(A1);
  free(A2);
  return 0;
}
```

## Section 7: Dynamic Memory Allocation

71. What is DMA and why we may need it?
- Passing by pointer
- Returning more than just one thing from a function
- Passing Arrays and Strings to functions
- Allocating unknown memory at run time

72. malloc function

73. Challenge #1 - Creating and returning a dynamically allocated array from a function
```c
#include <stdio.h>
#include <stdlib.h>
#define SIZE 5
int * createArray() {
    int *myArr;
    myArr = (int*) malloc(SIZE*sizeof(int));
    myArr[0] = 100;
    return myArr;
}
int main() {
  int *arr;
  arr = createArray();
  for (int i=0;i<SIZE;i++) printf("%d\n", arr[i]);
  free(arr);
  return 0;
}
```
- Do we need casting like (int*) for malloc?
- Ref: https://stackoverflow.com/questions/605845/do-i-cast-the-result-of-malloc

74. Challenge #2 - Print dynamically allocated array

75. calloc function
- Allocates a sequence of bytes
- Returns the address of the sequence
- All elements are set to 0
- Both of calloc/malloc return the contiguous memory
  - Ref: https://stackoverflow.com/questions/21332449/between-malloc-and-calloc-which-allocates-contiguous-memory
```c
int *a = (int*) malloc(arraySize*sizeof(int));
int *b = (int*) calloc(arraySize, sizeof(int));
```

76. free function
- Why heap memory is favored over stack?
  - It is slower than stack but all threads can access
    - Global access is allowed
    - Stack memory can be accessed by local variables only
  - Can be re-sized

77. Dangling pointer

78. Finding Memory Leakages (valgrind)

79. realloc function
- Allocates a new sequence of elements
- Copies previous elements

80. realloc issues

81. realloc - practical code example
```c
#include<stdio.h>
#include<stdlib.h>
int main() {
  int *grades = NULL;
  int SIZE=5;
  grades = (int *) malloc(SIZE*sizeof(int));
  for (int i = 0;i<SIZE;i++) grades[i] = i*10+125;
  for (int i = 0;i<SIZE;i++) printf("%d\n", grades[i]);
  SIZE *= 2;
  grades = (int *) realloc(grades, SIZE*sizeof(int));
  for (int i = 0;i<SIZE;i++) printf("%d\n", grades[i]);
  SIZE /=4;
  grades = (int *) realloc(grades, SIZE*sizeof(int));
  for (int i = 0;i<SIZE;i++) printf("%d\n", grades[i]); /* 125 135 */
  free(grades);
  return 0;
}
```

82. Implementing your own universal realloc function - Question

83. Implementing your own universal realloc function - Solution #1
```c
#include<stdlib.h>
void* myRealloc(void* srcblock, unsigned oldsize, unsigned newsize) {
  int smallsize;
  if (oldsize < newsize)
    smallsize = oldsize;
  else
    smallsize = newsize;
  char * newarr = (char*) malloc(newsize);
  if (!newarr) return NULL;
  for(int i=0; i<smallsize; i++) newarr[i] = ((char*)srcblock)[i];
  free(srcblock);
  return newarr;
}
int main() {
  int *grades = NULL;
  int SIZE=5;
  grades = (int *) malloc(SIZE*sizeof(int));
  for (int i = 0;i<SIZE;i++) grades[i] = i*10+125;
  for (int i = 0;i<SIZE;i++) printf("%d\n", grades[i]);
  //grades = (int *) realloc(grades, SIZE*sizeof(int));
  grades = (int *) myRealloc(grades, SIZE*sizeof(int), SIZE*sizeof(int)*2);
  for (int i = 0;i<SIZE*2;i++) printf("%d\n", grades[i]);
  grades = (int *) myRealloc(grades, SIZE*sizeof(int)*2, SIZE*sizeof(int)/2);
  for (int i = 0;i<SIZE/2;i++) printf("%d\n", grades[i]);
  return 0;
}
```
- Compare with Lecture 81

84. Implementing your own universal realloc function - Solution #2 - using memcpy

85. Adjustable Reallocation + Performance - Question

86. Adjustable Reallocation + Performance - Explanation & Solution

87. IMPORTANT Question - Create and Pass 1D array using Pointer to Pointer!

88. IMPORTANT Solution - Create and Pass 1D array using Pointer to Pointer!
- Use double pointer or indirection (**) 
```c
#include<stdio.h>
#include<stdlib.h>
void * createArray(unsigned newsize) {
  void * arr;
  arr = malloc(newsize);
  return arr;
}
void createArray2(unsigned newsize, void ** arr) {
  *arr = malloc(newsize);
}
int main() {
  int *arr1 = (int *) createArray(sizeof(int)*5);
  printf("%d\n", arr1[3]);
  free(arr1);
  int *arr2;
  createArray2(sizeof(int)*5, (void*) &arr2);
  printf("%d\n", arr2[3]);
  free(arr2);
  return 0;
}
```
- Indirection example: 
```c
#include <stdio.h>
#include <stdlib.h>
void alloc2(int** p) {
    *p = (int*)malloc(sizeof(int));
    **p = 10;
}
void alloc1(int* p) {
    p = (int*)malloc(sizeof(int));
    *p = 10;
}
int main(){
    int *p = NULL;
    alloc1(p); // pass by value
    //printf("%d ",*p);//undefined
    alloc2(&p);
    printf("%d ",*p);//will print 10
    free(p);
    return 0;
}
```
  - Ref: https://stackoverflow.com/questions/5580761/why-use-double-indirection-or-why-use-pointers-to-pointers

## Section 8: Advanced Exercises - Pointers & DMA

90. Exercise #2 - Splitting source array into ODD and EVEN arrays
- Note that we don't handle indirection (**) pointer in calculations. Use a temporary pointer to handle all calculations and send the address to the indirection in the end
```c
#include <stdio.h>
#include <stdlib.h>
int * generateOddEvenArrays(int *source, int nsize, int **oddPtr) {
  int *evenR, *oddR, neven=0, nodd=0;
  evenR = (int *) malloc(nsize*sizeof(int));
  oddR  = (int *) malloc(nsize*sizeof(int));
  for (int i=0;i<nsize;i++) {
    if (source[i]%2 ==0) {
      evenR[neven] = source[i];
      neven ++;
    } else {
      oddR[nodd] = source[i];
      nodd ++;
    }
  }
  *oddPtr = oddR;
  return evenR;
}
int main()
{ 
  int srcArr[] = {2,7,3,4,8,10,1};
  int *oddA;
  int *evenA;
  evenA = generateOddEvenArrays(srcArr, 7, &oddA);
  for(int i = 0;i< 7; i++) printf("%d\n", evenA[i]);
  for(int i = 0;i< 7; i++) printf("%d\n", oddA[i]);
  free(oddA);
  free(evenA);
  return 0;
}
```

## Section 9: 2d Dynamically Allocated Arrays (Matrix)

99. Arrays of Pointers
```c
int* arr[5];
for (int i=0;i<5;i++) {
  arr[i] = (int*)calloc(3,sizeof(int));
}
```

101. Creating a Totally dynamically 2D array
```c
int** a;
int rows=10, columns=11;
a = (int**)calloc(rows, sizeof(int*));
for (int i=0;i<rows; i++) {
  a[i] = (int*)calloc(columns, sizeof(int));
}
```

## Section 10. String & Pointers

115. Extra lecture on strings & pointers
- Following code segfaults
```c
  char *strptr = "Hello"; // Read only. Cannot change strptr
  strptr[0] = 'G';
```
- Following is allowed
```c
char str[] = "Hello";
char *strptr;
strptr = str; 
strptr[0] = 'G';
```
- Ref: https://www.codingninjas.com/codestudio/library/whats-the-difference-between-char-s-and-char-s-in-c

## Section 11. String Library Functions Implementation - Using pointers

## Section 12: Debuggers & Debugging - let's find out the problems!

## Section 13: Structs - Basics & Beyond

128. Creating Arrays of Struct variables in a static manner
```c
#include<stdio.h>
#include<stdlib.h>
typedef struct point {
  int x, y;
} Point;
int main() {
  Point pArr[5];
  for (int i=0;i<5;i++) {
     printf("Enter x / y for %d\n", i);
     scanf("%d %d", &pArr[i].x, &pArr[i].y);
  }
  for (int i=0;i<5;i++) printf("%d %d\n", pArr[i].x, pArr[i].y);
  return 0;
}
```

129. Dynamically Allocated Array of Structs
```c
#include<stdio.h>
#include<stdlib.h>
typedef struct point {
  int x, y;
} Point;
int main() {
  Point *pArr;
  pArr = (Point *) malloc(5*sizeof(Point));
  for (int i=0;i<5;i++) { 
     printf("Enter x / y for %d\n", i); 
     scanf("%d %d", &pArr[i].x, &pArr[i].y);
  }
  for (int i=0;i<5;i++) printf("%d %d\n", pArr[i].x, pArr[i].y);
  free(pArr);
  return 0;
}
```

130. Passing structs to functions by value + Updating by Pointer
- Pass by value will not update the struct variable
```c
#include<stdio.h>
#include<stdlib.h>
typedef struct point {
  int x, y;
} Point;
void increaseXby1(Point *p) {
  p->x ++; /* equivalent to (*p).x++; */
}
int main() {
  Point pArr = {123,456};
  printf("%d %d\n", pArr.x, pArr.y);
  increaseXby1(&pArr);
  printf("%d %d\n", pArr.x, pArr.y);
  return 0;
}
```

131. Structs Composition

132. Exercise 2: functions to dynamically allocate an array of structs - Question

133. Exercise 2: functions to dynamically allocate an array of structs - Solution
```c
#include<stdio.h>
#include<stdlib.h>
typedef struct point {
  int x, y;
} Point;
Point * createPoints(unsigned nsize) {
   Point *p;
   p = (Point*) malloc(sizeof(Point)*nsize);
   for(int i=0;i<nsize;i++) {
      p[i].x = i*10;
      p[i].y = i*11;
   }
   return p;
}
void increaseXby1(Point *p, unsigned nsize) {
   for(int i=0;i<nsize;i++) {
     p[i].x ++;
   } 
}
int main() {
  Point *pArr = createPoints(5);
  for (int i=0;i<5;i++) printf("%d %d\n", pArr[i].x, pArr[i].y);
  increaseXby1(pArr,5);
  for (int i=0;i<5;i++) printf("%d %d\n", pArr[i].x, pArr[i].y);
  free(pArr);
  return 0;
}
```

## Section 14: Introduction to Computer Architecture & Data Alignment

134. Introduction to Memory, Architecture, and Alignment

135. Word & Architectures
- Basically the memory was designed to be accessible byte by byte (byte addressable)
- WORD: computer's natual unit for data
  - Defined by the computer's architecture
  - Computer address memory in WORD-SIZED chunks
  - 16/32/64bits

136. Word Addressable vs Byte addressable
- Byte addressable
  - Each address identifies a single byte of data
  - Unique address to every byte of data
- Word addressable
  - Each address identifies a single word of data
  - Unique address to every word of data
  - Word size = 8 bytes in 64bit OS

137. Variables Alignment
- Naturally aligned: better CPU performance

138. Pratical checking variables addresses in memory in IDE

## Section 15: Structs Alignment

139. Introduction to Padding (Data alignment in conjunction with Structs)

140. Practical Struct Variable Memory Utilization
```c
#include <stdio.h>
#include <stdlib.h>
typedef struct point {
   char x;  /* 1byte */
   short y; /* 2bytes */
   int z;   /* 4bytes */
} Point ;  /* 8bytes  - check through sizeof() */
int main () {
  Point p1;
  p1.x = 0xFF;
  p1.y = 0x1234;
  p1.z = 0x6789ABCD;
  return 0;
}
```
- gdb memory address:
```bash
(gdb) p &p1
$6 = (Point *) 0x7fffffffd568
(gdb) x/2xw 0x7fffffffd568
0x7fffffffd568: 0x123400ff      0x6789abcd
```
- Note that the first block `0x123400ff` is the p1.y and p1.x. Second block `0x6789abcd` is the p1.z
- Note that p1.x has 00, as padding
- This is why Point struct is 8 bytes

141. Example #1 - Struct Memory Utilization and Data Alignment (+padding)
```c
#include <stdio.h>
#include <stdlib.h>
typedef struct point {
   char x;  /* 1byte */
   double y; /* 8bytes */
   int z;   /* 4bytes */
} Point ;  /* 24 bytes due to the padding by the longest data type */
int main () {
  unsigned int bytesPoint = sizeof(Point);
  printf("%d %d %d %d \n", sizeof(char), sizeof(double), sizeof(int), bytesPoint);
  Point p1;
  p1.x = 0xFF;
  p1.y = 0x0123456789ABCDEF;
  p1.z = 0x6789ABCD;
  return 0;
}
```
- gdb snapshot:
```bash
(gdb) p &p1
$2 = (Point *) 0x7fffffffd550
(gdb) x/8xw 0x7fffffffd550
0x7fffffffd550: 0x555546ff      0x00005555      0x789abcdf      0x43723456
#---^^are padding ^^^^^^p1.x------^^^^^^^^------p1.y - why numbers are different? TBD
0x7fffffffd560: 0x6789abcd      0x00007fff      0x00000000      0x00000000
#-----------------p1.z
```

142. Example #2 - Reorganizing members order and its affect on memory utilization
```c
#include <stdio.h>
#include <stdlib.h>
typedef struct point {
   char x;   /* 1byte */
   int y;    /* 4bytes */
   double z; /* 8bytes */
} Point ;    /* now 16 bytes */
int main () {
  Point p1;
  p1.x = 0xFF;
  p1.z = 0x0123456789ABCDEF;
  p1.y = 0x6789ABCD;
  return 0;
}
```
- gdb snapshot:
```bash
(gdb) p &p1
$2 = (Point *) 0x7fffffffd560
(gdb) x/8xw 0x7fffffffd560
0x7fffffffd560: 0xffffd6ff      0x6789abcd      0x789abcdf      0x43723456
#-----------------^^^^^^p1.x------p1.y------------p1.z
0x7fffffffd570: 0x555546a0      0x00005555      0xf7a03c87      0x00007fff
```

143. Exercise #1 - Structs, Members Organization, Data Alignment and Memory - Question

144. Exercise #1 - Structs, Members Organization, Data Alignment and Memory - Solution 
- A sample misaligned struct
```c
typedef struct point {
  char x;
  double y;
  char z;
}
```
  - This yields 1 + 7 padding + 8 + 1 + 7 padding = 24bytes
- A sample aligned struct
```c
typedef struct point {
  char x;
  char z;
  double y;
}
```
  - This yields 1 + 1 + 6 padding + 8 = 16 bytes

145. Adding Data Member to Struct without increasing the size of a variable in memory
```c
typedef struct point {
  char x;
  short t; 
  char z;
  double y;
}
```
  - This yields 1 + 2 + 1 + 4 padding + 8 = 16 bytes
  - Note that this struct consumes the same memory of lecture 144 but has one more member data of short t

146. Exercise #2 - Structs, Members Organization, Data Alignment and Memory - Question

147. Exercise #2 - Structs, Members Organization, Data Alignment and Memory - Solution 
```c
typedef struct point {
  char x;
  int t; 
  char z;
  double y;
}
```
- This consumes 1 + 3padding + 4 + 1 + 7padding + 8 = 24 bytes
```c
typedef struct point {
  char x;
  char z;
  int t; 
  double y;
}
```
- This consumes 1 + 1 + 2padding + 4 + 8 = 16 bytes
- Same results in C++. No optimization or no re-ordering in C++?

148. Data alignment and Padding with structs composition

149. Tightly packing & Packing to UnAligned Data

## Section 16: Pointers to Functions

150. Function Pointers - Introduction
- Name of an array = the address of the first element
- Name of a function = initial address of the function code
  - `printf("%p\n", func);` prints the address of the func()
     - This doesn't launch func()
```c
#include <stdio.h>
void func() {
  printf("hello world\n");
}
int main() {
  func();
  printf("%p  %p\n", func, main); /* 0x5565d55f268a  0x5565d55f269d */
  return 0;
}
```

151. Problem Example - Movitation to use Functions Pointers
- Instead of function names, we use addresses to launch those functions

152. Declaration of Pointer to a Function

153. Coding Example with Pointers to Functions
```c
#include <stdio.h>
void ageFunc(int age) {
  printf("Age is %d\n", age);
}
int main() {
  ageFunc(3);
  void (*fptr) (int); /* creation */
  fptr = &ageFunc;
  printf("%p\n", fptr); /* same memory address below */ 
  fptr = ageFunc;
  printf("%p\n", fptr); /* same memory address above */
  (fptr)(5);  /* launch the function */
  return 0;
}
```

154. Common Mistakes when using a Pointer to a Function

155. An array of Pointers to Functions
```c
#include <stdio.h>
void add(int x, int y) {
  printf("%d + %d = %d\n", x,y,x+y);
}
void subtract(int x, int y) {
  printf("%d - %d = %d\n", x,y,x-y);
}
void multi(int x, int y) {
  printf("%d * %d = %d\n", x,y,x*y);
}
int main() {
  int x = 3, y = 5;
  void (*fp[]) (int, int) = {&add,&subtract,&multi};
  (*fp[0])(x,y);
  (*fp[1])(x,y);
  (*fp[2])(x,y);
  return 0;
}
```

## Section 17: Files - Advanced Content

156. Functions for working and finding File's location
- myFile.txt
```bash
Hello
Welcome
```
- File pointer : H/e/l/l/o/\n/W/e/l/c/o/m/e
- Working with File's location
  - tell: current position of the file pointer
  - seek: Moves the position to left or right
  - rewind: moves the position to the beginning of the file

157. Common things to note when working with Sequential (Textual) files
```txt
1 Mike US 21
2 Christina Germany 35
3 David Canada 28
```
- Change the coutry of Christina as US

158. Introduction to Structured Files
```c
struct Point {
  int x, y
}
```

159. Writing a Structure Record to a File
```c
#include <stdio.h>
typedef struct point {
  int x;
  int y;
} Point;
int main() {
  Point p = {3,5};
  FILE *fp = fopen("myfile.bin","wb");
  if (fp != NULL) {
    printf("File is opened\n");
    fwrite(&p, sizeof(Point), 1, fp);
    fclose(fp);
  }
  return 0;
}
```

160. Writing 2 Points read from the user to a file
```c
#include <stdio.h>
typedef struct point {
  int x;
  int y;
} Point;
int main() {
  Point p1, p2;
  FILE *fp = NULL;
  printf("Enter x1: "); scanf("%d", &p1.x);
  printf("Enter y1: "); scanf("%d", &p1.y);
  printf("Enter x2: "); scanf("%d", &p2.x);
  printf("Enter y2: "); scanf("%d", &p2.y);
  fp = fopen("myFile2.bin","wb");
  if (fp != NULL) {
    printf("File is opened\n");
    fwrite(&p1, sizeof(Point), 1, fp);
    fwrite(&p2, sizeof(Point), 1, fp);
    fclose(fp);
  }
  return 0;
}
```
- For 1,2,3,4 inputs, ghex shows as '01 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00'

161. Writing an array of points to a file
```c
#include <stdio.h>
typedef struct point {
  int x;
  int y;
} Point;
int main() {
  Point pArr[5] = {{1,2},{3,4},{5,6},{7,8},{9,10}};
  FILE *fp = NULL;
  fp = fopen("myFile3.bin","wb");
  if (fp != NULL) { 
    printf("File is opened\n");
    fwrite(&pArr, sizeof(Point), 5, fp);
    fclose(fp);
  }
  return 0; 
}
```

162. Search for a specific point in a file Reading
```c
#include <stdio.h>
typedef struct point {
  int x;
  int y;
} Point;
int main() {
  Point tP;
  Point pFile;
  FILE *fp = fopen("myFile3.bin","rb");
  printf("What is the point you're looking for?\n");
  printf("Enter x:");
  scanf("%d", &tP.x);
  printf("Enter y:");
  scanf("%d", &tP.y);
  if (fp != NULL) {
    while (!feof(fp)) {
      fread(&pFile, sizeof(Point), 1, fp);
      if ((pFile.x == tP.x) && (pFile.y == tP.y)) printf("Found\n");
    }
    fclose(fp);
  }
  return 0;
}
```

163. Edit a specific point in the file
```c
#include <stdio.h>
typedef struct point {
  int x;
  int y;
} Point;
int main() {
  Point m_p;
  int choice;
  printf("what is the point you would like to edit? (1-5)\n");
  scanf("%d", &choice);
  FILE *fp = fopen("myFile3.bin","rb+");
  if (fp != NULL) {
    fseek(fp, (choice-1)*sizeof(Point), SEEK_SET);
    printf("Enter new x: ");
    scanf("%d", &m_p.x);
    printf("Enter new y: ");
    scanf("%d", &m_p.y);
    fwrite(&m_p, sizeof(Point), 1, fp);
    fclose(fp);
  }
  return 0;
}
```

## Section 18: Enums

164. Enums - Part 1
```c
#include <stdio.h>
#define _CRT_SECURE_NO_WARNINGS
enum workingDays
{
  MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY
};
int main() {
  int userDay;
  printf("Please enter the day you worked:");
  scanf("%d", &userDay);
  if (userDay == MONDAY) // instead of 0
    printf("Monday ...\n");
  else printf("other days\n");
  return 0;
}
```

165. Enums - Part 2
```c
#include <stdio.h>
#define _CRT_SECURE_NO_WARNINGS
enum months
{
  JAN=1, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEPT, OCT, NOV, DEC
// now enum begins from 1, not zero
};
int main() {
  enum months month;
  char *monthsNames[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
  int salaries[12] = {51,52,53,54,55,56,57,58,59,60,61,62};
  for( month=JAN; month<=DEC; month++)
  {
    printf("%s%10d\n", monthsNames[month-1], salaries[month-1]);
  }
  return 0;
}
```
- In Enums, same value for multiple keywords is allowed
  - `enum months { JAN=0, FEB=0, MAR=1, APR=1, MAY=2, JUN=3, JUL=3, AUG=4, SEPT=4, OCT=5, NOV=6, DEC=6 };`

## Section 19: Constants & Pointers Masterclass

166. Constants - General Intro + Usage
- `const age = 35;` == `const int age = 35;`
- `const int *ptr;`
  - The value of ptr cannot be changed
- `int *const ptr;`
  - The pointer is constant while the value can be changed
- `const int *const ptr;`
  - The address/value cannot be changed

167. Constants - Why & When + Example
- Why do we need constants?
  - Performance
  - Defense from bug
  - Code readability
- When do we need constants?
  - Whenever possible
    - Compiler will optimize accordingly
  - When further change in value is not expected
- Why use constants, not literal numbers?
  - Best practice for readability
```c
#include<stdio.h>
void findAvg1(int *Arr, int size);
void findAvg2(const int *Arr, const int size);
int main()
{
  int Grades[3] = {80,90,85};
  int n = 3;
  findAvg1(Grades, n);
  findAvg2(Grades, n);
  return 0;
}
```
- findAvg1() and findAvg2() yield following assembly
```assembly
 43:	e8 00 00 00 00       	callq  48 <main+0x48>
  48:	8b 55 e8             	mov    -0x18(%rbp),%edx
  4b:	48 8d 45 ec          	lea    -0x14(%rbp),%rax
  4f:	89 d6                	mov    %edx,%esi
  51:	48 89 c7             	mov    %rax,%rdi
  54:	e8 00 00 00 00       	callq  59 <main+0x59>
  59:	b8 00 00 00 00       	mov    $0x0,%eax
  5e:	48 8b 4d f8          	mov    -0x8(%rbp),%rcx
```

## Section 20: Counting Arrays - Part #2 - Practice exercises functions

168. Function - FreqAsValue - Question
- Develop a function that receives an array of integers, its size, and some value k. The array consists of values b/w 0 and k (included)
- The function should check and return 1, if the number of occurrence of any value in the array equals to the value itself. Otherwise, return 0
- [1,2,2,1,4,3]: k=4, return 0
- [3,2,2,1,3,3]: k=3, return 1

169. Function - FreqAsValue - Solution
```c
int FreqAsValue(const int *Arr, const int N, const int k)
{
  int count[k]; // better to use dynamic memory as size k might be unknown at compile time
  int j;
  for (int i=0;i<k;i++) count[i] = 0;
  for (int i=0;i<N;i++)
  {
    j = Arr[i];
    count[j]++;
  }
  for (int i=0;i<k;i++)
  {
    if (count[i] != 0 && count[i] != i) return 0;
    // if (count[i] != i) return 0; this is better
  }
  return 1;
}
```

170. Function - FindDominantValue - Question
171. Function - FindDominantValue - Solution

172. Function - FindSpecificValueMaxSum - Question
173. Function - FindSpecificValueMaxSum - Solution

174. Function - FindLargestMissingValue - Question
175. Function - FindLargestMissingValue - Solution

176. Function - areAllCharactersIncluded - Question
177. Function - areAllCharactersIncluded - Solution

## Section 21: Optional: Introduction to bitwise operations

## Section 22: Optional: Basic Algorithms

## Section 23: Congratulations! You've made it! What's next?
