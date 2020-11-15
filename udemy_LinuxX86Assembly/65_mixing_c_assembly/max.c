#include <stdio.h>
#include <inttypes.h>
// gcc -std=c99 max.c max.s ; ./a.out

int64_t getmax(int64_t a, int64_t b, int64_t c);

int main(){
  printf("%ld\n", getmax(40, -9, 67));
  printf("%ld\n", getmax( 0,  4, -7));
  printf("%ld\n", getmax(33, -99, 4));
  return 0;
}
