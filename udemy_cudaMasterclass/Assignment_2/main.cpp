#include <iostream>
#include <cmath>
using std::cout;
using std::endl;
int main(int argc, char** argv) {

  int i = 1<< 22;
  int j = 2 << 21;
  double x = pow(2,22);
  cout << i <<" "<< j <<" " << x << endl;
  i = 512 >> 1;
  j = 256 >> 1;
  cout << i <<" "<< j << endl;
  return 0;
}
