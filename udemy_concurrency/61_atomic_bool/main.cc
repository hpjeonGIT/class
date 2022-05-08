#include <iostream>
#include <thread>
#include <atomic>
int main() {
  std::atomic<bool> x(false);
  std::cout << "atomic boolean is lock free - " 
            << (x.is_lock_free() ? "yes" : "no") << std::endl;
  //std::atomic<bool> y(x); // not working. No copy constructor
  //std::atomic<bool> z=x;  // not working. No assign operator
  std::atomic<bool> y(true);
  x.store(false);
  x.store(y);
  std::cout << "y=" << y.load() << std::endl; // prints 1=true
  bool z = x.exchange(false);
  std::cout << "now x =" << x.load() << std::endl; // prints 0=false
  std::cout << "previous x =" << z << std::endl; // prints 1=true
  return 0;
}