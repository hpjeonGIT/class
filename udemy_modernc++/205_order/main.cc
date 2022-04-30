#include<iostream>
void Print() {
std::cout << "__func__:" << __func__ << std::endl
  << "__PRETTY_FUNCTION__: " << __PRETTY_FUNCTION__ << std::endl
  << "__DATE__:" << __DATE__  << std::endl
  << "__FILE__:" << __FILE__  << std::endl
  << "__LINE__:" << __LINE__  << std::endl
  << "__STDC__:" << __STDC__  << std::endl;
}
// << "__FUNCSIG__:" << __FUNCSIG__  << std::endl // for MSVC
int main() {
  Print();
  return 0;
}