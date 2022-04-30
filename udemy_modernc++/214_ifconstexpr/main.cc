#include <iostream>
#include <type_traits>
template<typename T>
void Print(const T& value) {
  if  (std::is_pointer_v<T>) {
    std::cout << *value << std::endl;
  } else if  (std::is_array_v<T>) {
    for (auto v: value) std::cout << v<< ' ';
    std::cout << std::endl;
  } else {
    std::cout << value << std::endl;
  }
}
int main() {
  int value {5};
  //Print(value);
  //Print(&value);
  int arr[] = {4,3,2,1};
  Print(arr);
  return 0;
}