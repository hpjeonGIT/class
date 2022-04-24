#include <iostream>
template<typename T>
T find_max(T a, T b) {
  return a>b ? a: b;
}
int main() {
  std::cout << find_max(1,3) << std::endl;
  std::cout << find_max(1.f,3.f) << std::endl;
  std::cout << find_max(1.,3.) << std::endl;
  std::cout << find_max<double>(1,3.f) << std::endl;
  return 0;
}
