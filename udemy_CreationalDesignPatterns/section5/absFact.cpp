#include "absFact.h"
AbstractProductA* ConcreteFactory1::CreateProductA() {
  return new ProductA1{};
}
AbstractProductB* ConcreteFactory1::CreateProductB() {
  return new ProductB1{};
}
AbstractProductA* ConcreteFactory2::CreateProductA() {
  return new ProductA2{};
}
AbstractProductB* ConcreteFactory2::CreateProductB() {
  return new ProductB2{};
}
void UsePattern(AbstractFactory *pFactory) {
  AbstractProductA *pA = pFactory->CreateProductA();
  AbstractProductB *pB = pFactory->CreateProductB();
  pA->ProductA();
  pB->ProductB();
  delete pA;
  delete pB;
}
int main() {
  AbstractFactory *pFactory = new ConcreteFactory1{};
  UsePattern(pFactory);
  delete pFactory;
}