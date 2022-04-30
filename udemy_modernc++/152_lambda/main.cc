#include <iostream>
int main() {
  []() { std::cout << "hello world 1\n"; } ();
  auto fnc = []() { std::cout << "hello world 2\n"; } ;
  fnc();
  std::cout << typeid(fnc).name() << std::endl;
  return 0;
}