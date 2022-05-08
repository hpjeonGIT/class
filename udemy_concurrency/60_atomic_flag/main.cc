#include<thread>
#include<iostream>
#include<atomic>
int main() {
  std::atomic_flag flag = ATOMIC_FLAG_INIT;
  std::cout << flag.test_and_set() << std::endl;
  std::cout << flag.test_and_set() << std::endl;
  std::cout << flag.test_and_set() << std::endl;
  flag.clear();
  std::cout << flag.test_and_set() << std::endl;
  std::cout << flag.test_and_set() << std::endl;
  return 0;
}