#include <iostream>
template<typename T, int S>
class Stack {
  T m_Buffer[S];  
  int m_Top {-1};
public:
  void Push(const T &elem) {
    m_Buffer[++m_Top] = elem;
  }
  void Pop();
  void Print() {
    for(size_t i=0;i<=m_Top;++i) {
      std::cout << m_Buffer[i] << std::endl;
    }
  }
};
template<typename T, int S>
void Stack<T,S>::Pop() {
  --m_Top;
}
int main() {
  Stack<float,10> s1;
  s1.Push(3.f);
  s1.Push(1.f);
  s1.Print();
}