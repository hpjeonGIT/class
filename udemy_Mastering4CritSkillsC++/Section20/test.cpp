#include <iostream>
#include <vector>

int main(int argc, char ** argv){
  int age {10};
  int &x = age;
  //int x = &age; WRONG
  //int &x = &age; WRONG
  std::cout << x <<  std::endl;

  std::vector<int> z {1,2,3,4};
  for(auto &v : z) {
    std::cout << v << std::endl;
    v ++; // updating
  }
  for(auto v : z) {
    std::cout << v << std::endl;
    v++; // no effect
  }
  for(auto v : z) {
    std::cout << v << std::endl;
  }

  return 0;

}
