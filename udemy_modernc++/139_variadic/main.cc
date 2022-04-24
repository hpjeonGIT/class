#include <iostream>
// base case function
void Print() {
  std::cout <<"reached the base function\n";
}
template<typename T, typename...Params>
void Print(T a, Params... args){
  std::cout << "calling " << a << " while " << sizeof...(args) << " " << sizeof...(Params) << std::endl;
  Print(args...);
}
int main() {
  Print(1, 2.5f, 3.0, "4");
  return 0;
}