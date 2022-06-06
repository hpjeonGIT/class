#include "basic.h"
#include <iostream>
Prototype* ConcretePrototype1::Clone() {
  std::cout << "[ConcretePrototype1] Cloning...\n";
  return new ConcretePrototype1{*this};
}
Prototype* ConcretePrototype2::Clone() {
  std::cout << "[ConcretePrototype2] Cloning...\n";
  return new ConcretePrototype2{*this};
}
void Client::Operation() {
  auto p = prototype->Clone();
}
int main() {
  Client c;
  c.SetPrototype(new ConcretePrototype1{});
  c.Operation();
}