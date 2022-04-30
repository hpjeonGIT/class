#include <iostream>
// base function
auto Sum() {  return 0; }
// variadic template function
template<typename T, typename ...Args>
auto Sum(T a, Args...args) { return a + Sum(args...);}
//Unary right fold
template<typename...Args>
auto Sum_UR(Args...args) { return (args + ...);}
//Unary left fold
template<typename...Args>
auto Sum_UL(Args...args) { return (... + args);}
//Binary right fold
template<typename...Args>
auto Sum_BR(Args...args) { return (args + ... + 0);}
//Binary left fold
template<typename...Args>
auto Sum_BL(Args...args) { return (0+ ... + args);}
int main() {
  std::cout << "Variadic: " << Sum(5,4,3,2,1) << std::endl;
  std::cout << "Unary right fold: " << Sum_UR(5,4,3,2,1) << std::endl;
  std::cout << "Unary right fold: " << Sum_UR(5,4,3,2,1) << std::endl;
  std::cout << "Unary left fold: " << Sum_UL(5,4,3,2,1) << std::endl;
  std::cout << "Binary right fold: " << Sum_BR() << std::endl;
  std::cout << "Binary left fold: " << Sum_BL(5,4,3,2,1) << std::endl;
  return 0;
}