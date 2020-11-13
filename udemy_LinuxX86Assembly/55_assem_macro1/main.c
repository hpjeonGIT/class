#include <stdio.h>
#define GREATER(a,b,result) ({\
  __asm__( \
          "cmp %1, %2\n\t" \
          "jge 0f\n\t" \
          "movl %1,%0\n\t" \
          "jmp 1f\n" \
          "0: \n\t" \
          "movl %2, %0\n\t" \
          "1: " \
          :"=r"(result) \
          :"r"(a), "r"(b) \
        ); \
})

// Sample C MACRO
#define MAX 100
#define MIN 0
#define SUM(a,b,result) ((result) = (a) + (b))
// Ex) answer = SUM(10+40)
int main() {
   int data1 = 10;
   int data2 = 20;
   int result;
   GREATER(data1,data2,result);
   printf("The greater is %d\n",result);
   return 0;
}
