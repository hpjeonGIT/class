#include <iostream>
#include <fstream>
#include <string>
#include <experimental/filesystem>
int main() {
  using namespace std::experimental::filesystem;
  path source(current_path());
  source /= "abc.jpg";
  path dest(current_path());
  dest /= "Copy.jpg";
  std::ifstream input { source, std::ios::binary|std::ios::in };
  if (!input) {
    std::cout << " source file missing\n";
    return -1;
  }
  if (exists(dest)) {
    std::cout << "file : " << dest << " will be over-written\n";
  }
  std::ofstream output { dest, std::ios::binary|std::ios::out };
  output << input.rdbuf();
  return 0;
}