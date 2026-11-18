#include <iostream>
#include "common.h"
int main(){
  int x = 123;
  int z = ret_2x(x);
  int y = ret_3x(x);
  std::cout << "Input = " << x << " Answer = " << z << std::endl;
  std::cout << y << std::endl;
  return 0;
}
