#include <iostream>
constexpr int GetNumber() { return 123; }
constexpr int Max(int x,int y) {
  if (x>y) {return x;}
  else {return y;}
  // return x>y ? x :y; // for C++11
}
int main() {
  constexpr int i = GetNumber(); std::cout << i << std::endl;
  constexpr int j = Max(i, 3);std::cout << j << std::endl;
  constexpr int k = Max(4, 3);std::cout << k << std::endl;
  return 0;
}