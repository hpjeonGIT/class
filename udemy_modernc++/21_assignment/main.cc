#include <iostream>
int Add(int *a, int *b) {
  return *a+*b;
}
void AddVal(int *a, int *b, int *result) {
  *result = *a+*b;
}
void Swap(int *a, int *b) {
  int c;
  c = *a;
  *a = *b;
  *b = c; 
  // int *c; this is not working
  // c = a;
  // a = b;
  // b = c;
}
void Factorial(int *a, int *result) {
  if (*a > 1) {
    int *c = new int {*a - 1};
    Factorial(c,result);
    *result *= *a;
    delete(c);
  } else {
    *result = 1;
  }
}
int main() {
  int a {10}, b{22};
  std::cout << Add(&a, &b) << std::endl;
  int c {};
  AddVal(&a,&b,&c);
  std::cout << c << std::endl;
  Swap(&a,&b);
  std::cout << a << " " << b << std::endl;
  a = 4;
  Factorial(&a,&c);
  std::cout << c << std::endl;
  //  
  int *x = new int {10};
  int *y = new int {22};
  std::cout << Add(x,y) << std::endl;
  int *z = new int{};
  AddVal(x,y,z);
  std::cout << *z << std::endl;
  Swap(x,y);
  std::cout << *x << " " << *y << std::endl;
  *x = 4;
  Factorial(x,z);
  std::cout << *z << std::endl;
  delete(x);
  delete(y);
  delete(z);
  return 0;
}