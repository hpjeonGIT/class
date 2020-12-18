#include <iostream>
using std::cout ;

int main() {
   int a = 0;
   cout << (a++, ++a, a++, ++a, a++) << "\n";
   cout << (a, 1, 2, 7) << "\n";
   return 0;
}
