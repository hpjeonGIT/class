#include <iostream>
int Square(int x) { return x*x; }
int Add(int x, int y) { return x+y; }
template<typename T>
auto PrintResult(T func) {
  return [func](auto&&...args) {
    std::cout << "Result is: ";
    return func(args...);
  };
}
int main() {
  auto result = PrintResult(Square); //return value is a lamba expression
  std::cout << result(5) << std::endl;
  auto result2 = PrintResult(Add); //return value is a lamba expression
  std::cout << result2(5,1) << std::endl;

}