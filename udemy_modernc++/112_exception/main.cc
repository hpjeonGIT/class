#include <iostream>
#include <vector>
int Process(int16_t count) {
  int16_t* x = (int16_t*)malloc(count*sizeof(int16_t));
  if (x==nullptr) {
    throw std::runtime_error("cannot allocate");
  }
  free(x);
  return 0;    
}
int main() {
  try {
    int rv = Process(std::numeric_limits<int16_t>::max()+1);
  } catch (std::runtime_error &ex) {
    std::cout << "we have: " << ex.what() << std::endl;
  }
  return 0;
}