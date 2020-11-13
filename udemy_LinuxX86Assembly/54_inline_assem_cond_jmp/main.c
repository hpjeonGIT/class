#include <stdio.h>

int main() {
  int a = 20;
  int b = 30;
  int result;
  __asm__(
    "cmp %1, %2\n\t"
    "jge greater\n\t"
    "movl %1, %0\n\t"
    "jmp end\n"
    "greater: \n\t"
    "movl %2,%0\n"
    "end:"
    :"=r"(result)
    :"r"(a), "r"(b)
  );
  printf("The value is %d\n", result);
  return 0;
}
