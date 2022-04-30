#include <iostream>
template<typename... Args>
bool AnyOfEven(Args...args) { return(... || (args%2 == 0));}
template<typename... Args>
bool AllOfEven(Args...args) { return(... && (args%2 == 0));}
template<typename... Args, typename Predicate>
bool AnyOfP(Predicate  p, Args...args) { return(... || p(args));}
int main() {
  std::cout << "Any even? " << AnyOfEven(7,8,9) << std::endl;
  std::cout << "All even? " << AllOfEven(7,8,9) << std::endl;
  std::cout << "Any even? " << AnyOfP([](int x){return x%2==0;}, 7,8,9) << std::endl;
  return 0;
}