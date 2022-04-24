#include <iostream>
#include <string>
#include <vector>
enum class Case{SENSITIVE, INSENSITIVE};
std::vector<int> FindAll(
  const std::string &src,
  const std::string &search,
  Case scase = Case::INSENSITIVE, size_t offset=0) {
  std::string l_src, l_search;
  std::vector<int> voffset;
  size_t loc = 0;
  if (scase == Case::INSENSITIVE) {
    for(auto &el: src) l_src += toupper(el);
    for(auto &el: search) l_search += toupper(el);    
  } else {
    l_src = src;
    l_search = search;    
  }
  while(true) {
    auto loc = l_src.find(l_search, offset);
    if (loc == std::string::npos) break;
    voffset.push_back(loc);
    offset = loc+l_search.size();
  }    
  return voffset;
}
int main() {
  auto x = FindAll("HelloHelloheLLo","hello");
  for (auto &el : x) std::cout << el << std::endl;
}
