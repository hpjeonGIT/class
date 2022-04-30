#include <iostream>
int main(){
  [](int x) {
    x*=2;
    [](int x) {
      std::cout << x << std::endl;
    }(x);
  }(5);
  return 0;
}