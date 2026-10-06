// ref=https://coderefinery.github.io/cmake-workshop/testing/
#include<iostream>
#include "func01.h"
int main() {
  int x = 123;
  int z = ret_2x(x);
  if (z == x*2) {
    return 0;
  } else {
    return 1;
  }
}
