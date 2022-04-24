#include <iostream>
#include <vector>
int Process(int16_t count) {
  std::vector<bool> b1 {true, false, true, false, true, false};
  int ierr {};
  for (auto el : b1) {
    try {
      if (!el) {
        ++ierr;
        throw std::runtime_error("false found");
      }
    } catch (std::runtime_error &ex) {
      std::cout << "Error found: " << ex.what() << std::endl;
      if (ierr>2) {
        throw std::out_of_range("throwing to the first layer");
      }
    }
  }
  return 0;    
}
int main() {
  try {
    int rv = Process(3);
  }
  catch (std::exception &ex) {
    std::cout << "first layer error_message IS : " << ex.what() << std::endl;
  }
  return 0;
}