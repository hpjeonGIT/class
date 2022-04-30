#include<iostream>
#include<fstream>
int main() {
  std::string txt{"hello world"};
  // std::ofstream out {"file.txt"};
  // if (! txt.empty() ) {
  //   out << txt;
  // } else {
  //   out << "no text";
  // }
  if (std::ofstream out {"file.txt"}; ! txt.empty() ) {
    out << txt;
  } else {
    out << "no text";
  }
  return 0;
}