#include "base.h"
#include <iostream>
void ConcreteFlyweight::Operation(int extrinsic) {
  std::cout << "Intrinsic state:" << * m_pIntrinsicState << std::endl;
  std::cout << "Extrinsic state:" << extrinsic << std::endl; 
}
void UnsharedConcreteFlyweight::Operation(int extrinsic) {
  std::cout << "Internal state:" << m_InternalState << std::endl;
  std::cout << "Extrinsic state:" << extrinsic << std::endl;
}
Flyweight* FlyweightFactory::GetFlyweight(int key) {
  auto found = m_Flyweights.find(key) != end(m_Flyweights);
  if (found) {
    return m_Flyweights[key];    
  }
  static int intrinsicState{100};
  Flyweight *p = new ConcreteFlyweight{&intrinsicState};
  m_Flyweights[key] = p;
  return p;
}
Flyweight* FlyweightFactory::GetUnsharedFlyweight(int value) {
  return new UnsharedConcreteFlyweight{value};
}
int main() {
  int extrinsicState = 1;
  FlyweightFactory factory;
  auto f1 = factory.GetFlyweight(1);
  auto f2 = factory.GetFlyweight(1);
  auto f3 = factory.GetFlyweight(1);
  f1->Operation(extrinsicState++);
  f2->Operation(extrinsicState++);
  f3->Operation(extrinsicState++);
  return 0;
}