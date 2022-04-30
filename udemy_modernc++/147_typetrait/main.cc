#include <iostream>
#include <type_traits>
template<typename T>
T Divide(T a, T b) {
  if(std::is_floating_point<T>::value == false){
    std::cout << "Not floating point. We stop here\n";
    return 0;
  }
  return a/b;
}
int main() {
  std::cout << Divide(5,2) << std::endl;
}