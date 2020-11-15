#include <stdio.h>
#include <inttypes.h>

uint64_t factorial(unsigned n);

int main() {
  for (int i=0; i<30; i++) {
    printf("factorial(%2u)=%lu\n",i, factorial(i));
  }
  return 0;
}

uint64_t factorial_c(unsigned n) {
  return (n<=1) ? 1 : n*factorial(n-1);
}
