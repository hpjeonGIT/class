#include <iostream>
//enum Aolor{RED=4, GREEN, BLUE};
enum Boor:char  {RED='f', GREEN, BLUE};

enum class Color{RED, GREEN, BLUE};
enum class Kolor : long {RED=3, GREEN, BLUE};
enum class Qolor : char {RED='r', GREEN, BLUE};

int main() {
  std::cout << RED << GREEN << BLUE << std::endl;
  
  std::cout << static_cast<int>(Color::RED) << static_cast<int>(Color::GREEN) << static_cast<int>(Color::BLUE) << std::endl;
  //std::cout << Kolor::RED << Kolor::GREEN << Kolor::BLUE << std::endl;
  //std::cout << Qolor::RED << Qolor::GREEN << Qolor::BLUE << std::endl;
}