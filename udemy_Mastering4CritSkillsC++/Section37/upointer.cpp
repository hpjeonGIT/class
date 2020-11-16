#include <iostream>
#include <memory>

int main(int argc, char ** argv) {
  std::unique_ptr<int> p1 {new int {20 }};
  //std::unique_ptr<int> p2 = new int {20} ; // not working
  std::unique_ptr<int> p2 {new int {20 }};
  int *p3;
  p3 = new int;
  *p3 = 20; // without memory allocation, it wil not work
  std::cout << *p1 << " " << *p2 << std::endl;
  *p1 = 50;
  std::cout << *p1 << " " << *p2 << std::endl;
  p1.reset(new int { 100});
  std::cout << *p1 << " " << *p2 << std::endl;

  return 0;
}
