#include "basic.h"
Client::Client() {
  m_pA = std::make_shared<A>();
  m_pB = std::make_shared<B>();
  m_pC = std::make_shared<C>();
}
void Client::Invoke() {
  m_pA->CallA();
  m_pB->CallB();
  m_pC->CallC();
}
int main() {
  Client c;
  c.Invoke();
}