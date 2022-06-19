#include "base.h"
#include <iostream>
void ConcreteComponent::Operation() {
  std::cout <<"[ConcreteComponent] Operation invoked\n";
}
void ConcreteDecoratorA::Operation() {
  std::cout << "[ConcreteDecoratorA] Operation invoked\n";
  m_ptr->Operation();
}
void ConcreteDecoratorB::Operation() {
  std::cout << "[ConcreteDecoratorB] Operation invoked\n";
  m_ptr->Operation();
}
int main() {
  ConcreteComponent component{};
  ConcreteDecoratorA decA{&component} ;
  ConcreteDecoratorB decB{&decA};
  decB.Operation();
}