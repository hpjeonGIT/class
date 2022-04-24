#include <iostream>
#include <string>
#include <cstdio>
std::string ToUpper(const std::string &str) {
  std::string upper;
  for (const auto &el: str) {
    upper += toupper(el);
  }
  return upper;
}
std::string ToLower(const std::string &str) {
  std::string upper;
  for (const auto &el: str) {
    upper += tolower(el);
  }
  return upper;
}
int main(){
  std::string s0 = "Hello World";
  std::cout << ToUpper(s0) << ToLower(s0) << std::endl;
}