#include "base.h"
#include <iostream>
void ConcreteImplementorA::OperationImpl() {
  std::cout << "[ConcreteImplementorA] Implmentation invoked\n";
}
void ConcreteImplementorB::OperationImpl() {
  std::cout << "[ConcreteImplementorB] Implmentation invoked\n";
}
void RefinedAbstraction::Operation() {
  std::cout << "[RefinedAbstraction] =>";
  m_pImplementor->OperationImpl();
}
int main() {
  ConcreteImplementorA impl;
  Abstraction *p = new RefinedAbstraction(&impl);
  p->Operation();
  return 0; 
}