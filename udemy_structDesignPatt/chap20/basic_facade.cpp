#include "basic_facade.h"
Client::Client() {
  m_pF = std::make_shared<Facade>();  
}
void Client::Invoke() {
  m_pF->Use();  
}
Facade::Facade() {
  m_pA = std::make_shared<A>();
  m_pB = std::make_shared<B>();
  m_pC = std::make_shared<C>();
}
void Facade::Use() {
  m_pA->CallA();
  m_pB->CallB();
  m_pC->CallC();
}
int main() {
  Client c;
  c.Invoke();
}