#include <iostream>
#include <thread>
#include <atomic>
int main() {
  int val[10] {11,22,33,44,55,66,77,88,99,0};
  std::atomic<int*> x_ptr = val;
  std::cout << "atomic boolean is lock free - " 
            << (x_ptr.is_lock_free() ? "yes" : "no") << std::endl; // yes
  int* y_ptr = val + 3;
  x_ptr.store(y_ptr);
  std::cout << "val by the pointer = " << *(x_ptr.load()) << std::endl; // 44
  bool rv = x_ptr.compare_exchange_weak(y_ptr, val+5); 
  std::cout << "store operation was : " << (rv?"yes":"no") << std::endl; // yes
  std::cout << "new val of x_ptr: " << *x_ptr << std::endl; // 66
  int* pre_ptr1 = x_ptr.fetch_add(2); // adding 2 as offset (index)
  std::cout << *pre_ptr1 << " " << *x_ptr << std::endl; //66 & 88
  int* pre_ptr2 = x_ptr.fetch_sub(4); // -4 for offset
  std::cout << *pre_ptr2 << " " << *x_ptr << std::endl; // 88 & 44
  x_ptr++;  std::cout << *x_ptr << std::endl; // 55
  x_ptr--;  std::cout << *x_ptr << std::endl; // 44
  return 0;
}