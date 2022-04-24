#include<iostream>
template<int myInt>
void Print() {
  std::cout << myInt << std::endl;
}
int main() {  
  Print<3>();
  //int i=3; Print<i>();
  const int i=3; Print<i>();
  return 0;
}