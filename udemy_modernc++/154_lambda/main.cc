#include <iostream>
#include <vector>
template<typename T, typename CB>
void ForEach(std::vector<T> v, CB ops) {
  for (auto &x: v) {
    std::cout << " " << ops(x);
  }
  std::cout << std::endl;
}
int main(){
  std::vector<int> myV = {1, -3, 2, -7, -9};
  ForEach(myV, [](auto x) { return -x;});
  ForEach(myV, [](auto x) { return std::abs(x);});
  int offset = 10;
  ForEach(myV, [offset](auto x) { return x+offset;});
  return 0;
}