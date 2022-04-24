#include<iostream>
#include<string>
int main() {
  std::cout << std::string("C:\temp\newfile.txt") << std::endl;
  std::cout << std::string("C:\\temp\\newfile.txt") << std::endl;
  std::cout << std::string(R"(C:\temp\newfile.txt)") << std::endl;
  std::cout << std::string(R"MSG(C:\temp\newfile.txt)MSG") << std::endl;  
  return 0;
}