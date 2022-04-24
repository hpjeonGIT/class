#include <iostream>
#include <vector>
int Process(int16_t count) {
  if (count < 10 & count > 0) 
    throw std::out_of_range("Count should be >10");
  if (count == 100)
    throw std::runtime_error("Designed to throw at 100");
  int16_t* x = new int16_t[count];
  delete[] x;
  return 0;    
}
int main() {
  try {
    int rv = Process(33000);
  } catch (std::runtime_error &ex) {
    std::cout << "error_message is : " << ex.what() << std::endl;
  }
  catch (std::out_of_range &ex) {
    std::cout << "error_message is : " << ex.what() << std::endl;
  }
  catch (std::bad_alloc &ex) {
    std::cout << "error_message is : " << ex.what() << std::endl;
  }
  catch (std::exception &ex) {
    std::cout << "error_message IS : " << ex.what() << std::endl;
  }
  catch (...) {
    std::cout << "no identity of exception " << std::endl;
  }
  return 0;
}