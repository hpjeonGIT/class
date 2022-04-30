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
  int vsum {};
  int rsum {};
  ForEach(myV, [vsum, &rsum](auto x) { 
    //vsum+=x;  // compiler error
    rsum += x; 
    return -x;}
  );
  std::cout << vsum << " " << rsum << std::endl;
  return 0;
}