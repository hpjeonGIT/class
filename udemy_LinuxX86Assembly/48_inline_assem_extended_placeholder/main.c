/*
__asm__("assembly code"
      : "=r"(result)
      : "r"(data1), "r"(data2))
      will produce:
      %0 will represent reg for result
      %1 will represent reg for data1
      %2 will represent reg for data2
*/

#include <stdio.h>
int main(){
  int data1 = 10;
  int data2 = 20;
  int result;
  __asm__("imull %1,%2 \n\t"
          "movl %2,%0\n\t"
          :"=r"(result)
          :"r"(data1),"r"(data2)
          );
  printf("The result is %d\n", result);
  return 0;
}
