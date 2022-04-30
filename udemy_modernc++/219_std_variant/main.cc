#include <iostream>
#include <variant>
int main() {
  try {
    std::variant<int, std::string> v{ "hello"};
    auto val = std::get<std::string> (v);
    val = std::get<1>(v);
    auto activeIndex = v.index();
    std::cout << val << " at " << activeIndex << std::endl;
    val = std::get<0>(v);
  } catch(std::exception &ex) {
    std::cout << "Exception: " << ex.what() << std::endl;
  }
  return 0;
}