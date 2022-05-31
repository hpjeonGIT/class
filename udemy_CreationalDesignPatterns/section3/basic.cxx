#include "basic.h"
#include <iostream>
void ConcreteProduct::Operation() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
}
void ConcreteProduct2::Operation() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
}
void Creator::AnOperation() {
  //m_pProduct = new ConcreteProduct{} ; // not generalized
  m_pProduct = Create(); // now generalizees. Object name is followed from the Create() of derived classes
  m_pProduct->Operation();
}
Product * ConcreteCreator::Create() {
  return new ConcreteProduct{};
}
Product * ConcreteCreator2::Create() {
  return new ConcreteProduct2{};
}
int main() {
  //Creator ct;
  ConcreteCreator2 ct;
  ct.AnOperation();
  return 0;
}