#include <iostream>
#include <fstream>
#include <string>
#include <experimental/filesystem>
int main() {
  using namespace std::experimental::filesystem;
  path source(current_path());
  source /= "quiz8.cpp";
  path dest(current_path());
  dest /= "Copy.cpp";
  std::ifstream input { source };
  if (!input) {
    std::cout << " source file missing\n";
    return -1;
  }
  std::ofstream output { dest };
  std::string line;
  while(!std::getline(input,line).eof()) {
    output << line << std::endl;
  }
  return 0;
}