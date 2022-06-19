#include "base.h"
#include <iostream>
void ConcreteComponent::Operation() {
  std::cout <<"[ConcreteComponent] Operation invoked\n";
}
void Decorator::Operation() {
  m_ptr->Operation();
}
int main() {
  ConcreteComponent component{};
  ConcreteDecoratorA decA{&component} ;
  ConcreteDecoratorB decB{&decA};
  decB.Operation();
}