#include<iostream>
#include<fstream>
#include<string>
void Write() {
  std::ofstream out{"data.txt"};
  out << "hello world\n";
  out << 123 << std::endl;
  out.close();
}
void Read() {
  std::ifstream inp { "data.txt"};
  if(!inp.is_open()) {
    std::cout << "Cannot open the file\n";
    return;
  }
  std::string msg;
  std::getline(inp,msg);
  int value {};
  inp >> value;inp >> value;
  if (inp.fail()) {std::cout << "cannot read\n";}
  if (inp.eof()) {std::cout << "EOF reached\n";}
  if (inp.good()) {
    std::cout << "IO successful\n";
  } else {
    std::cout << "IO failed\n";
  }
  inp.close();
  std::cout << msg << ":" << value << std::endl;
}
int main() {
  Write();
  Read();
  return 0;
}