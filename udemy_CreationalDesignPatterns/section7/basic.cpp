#include "basic.h"
#include <iostream>
ConcreteBuilder::ConcreteBuilder() {
  std::cout << "[ConcreteBuilder] Created\n";
}
void ConcreteBuilder::BuildPart() {
  std::cout << "[ConcreteBuilder] Building ...\n";
  std::cout << "\t Part A\n";
  std::cout << "\t Part B\n";
  std::cout << "\t Part C\n";
  product = new Product{};
}
Product* ConcreteBuilder::GetResult() {
  std::cout << "[ConcreteBuilder] Returning result\n";
  return product;
}
Director::Director(Builder* builder): builder{builder} {
  std::cout << "[Director] Created\n";
}
void Director::Construct() {
  std::cout << "[Director] Construction process started\n";
  builder->BuildPart();
}
int main() {
  ConcreteBuilder builder;
  Director director{&builder};
  director.Construct();
  Product *p = builder.GetResult();
  delete p;
}