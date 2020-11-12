//gcc -no-pie main.c
#include <stdio.h>
int a=10;
int b=20;
int result;

int main() {
  __asm__(
    "movl a, %eax \n\t"
    "movl b, %ebx \n\t"
    "imul %ebx, %eax \n\t"
    "movl %eax, result \n\t"
    );
    printf("The result is %d\n", result);
}
