#include "base.h"
int depth{};
void Leaf::Add(Component* pComponent) {}
Component* Leaf::GetChild(int index) {return nullptr;}
void Leaf::Operation() {
  std::cout << "[Leaf] Operation\n";
}
void Leaf::Remove(Component* pComponent){}

void Composite::Add(Component* pComponent) {
  m_Children.push_back(pComponent);
}
Component* Composite::GetChild(int index) {
  return m_Children[index];
}
void Composite::Operation() {
  ++depth;
  std::cout << "[Composite] Operation\n";
  for(auto pChild: m_Children) {
    for(int i=0;i<depth; ++i) {std::cout << '\t';}
    std::cout << "|-";
    pChild->Operation();
  }
  --depth;
}
void Composite::Remove(Component* pComponent){
  auto newend = std::remove(m_Children.begin(), m_Children.end(), pComponent);
  m_Children.erase(newend, end(m_Children));
}
int main() {
  Leaf leaf1, leaf2, leaf3;
  Composite subroot;
  subroot.Add(&leaf3);
  Composite root;
  root.Add(&leaf1);
  root.Add(&leaf2);
  root.Add(&subroot);
  root.Operation();
  return 0;
}
