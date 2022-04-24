#include <iostream>
#include <string>
enum class Case{SENSITIVE, INSENSITIVE};
size_t Find( const std::string &src,
             const std::string &search,
             Case scase = Case::INSENSITIVE,
             size_t offset=0) {
  std::string l_src, l_search;
  std::cout << static_cast<int>(scase) << std::endl;
  if (scase == Case::INSENSITIVE) {
    for(auto &el: src) l_src += toupper(el);
    for(auto &el: search) l_search += toupper(el);
    return l_src.find(l_search, offset);
  } else {
    return src.find(search, offset);
  }
}
int main(){
  std::string s0 = "abc Hello World";
  std::cout << Find(s0, "hello") <<std::endl;
  std::cout << Find(s0, "bye") <<std::endl;
  std::cout << Find(s0, "hell0", Case::SENSITIVE) << std::endl;  
}