#include <iostream>
template<typename T>
struct Anonym {
  T operator() (T x, T y) const {
    return x+y;
  }
};
int main() {
  auto fnc = []()->double { return 3.5; } ;
  std::cout << fnc() << std::endl;
  auto sumLambda = [] (auto x, auto y) { return x+y;};
  Anonym<int> myFobj;
  std::cout << myFobj(5,2) << std::endl;
  std::cout << sumLambda(5,2) << std::endl;
  return 0;
}