#include <iostream>
#include <cstring>
// Primary template
template<typename T>
T find_max(T a, T b) {
  return a>b ? a: b;
}
// Explicit instantiation
template char find_max(char a, char b);
// Explicit specialization
template<> const char* find_max(const char* a, const char* b) {
  return strcmp(a,b) > 0 ? a : b;
}
int main() {
  std::cout << find_max(1,3) << std::endl;
  std::cout << find_max(1.f,3.f) << std::endl;
  std::cout << find_max(1.,3.) << std::endl;
  std::cout << find_max<double>(1,3.f) << std::endl;
  std::cout << find_max('a','b') << std::endl;
  char b {'B'}; char a {'A'};  
  std::cout << find_max(a,b) << std::endl;
  const char* pb {"B"}; const char* pa {"A"};  
  std::cout << find_max(pa,pb) << std::endl;
  return 0;
}
