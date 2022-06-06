#include "basic.h"
int main() {
  AbstractProductA *pA = new ProductA1{};
  AbstractProductB *pB = new ProductB1{};
  pA->ProductA();
  pB->ProductB();
  delete pA;
  delete pB;
}