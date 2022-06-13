#include "colorfont.h"
#include <iostream>
void Console::SetColor(Color color) {
  switch (color) {
    case Color::RED:
      std::cout << "\033[31m";
      break;
    case Color::GREEN:
      std::cout << "\033[32m";
      break;
    case Color::BLUE:
      std::cout << "\033[34m";
      break;
    case Color::WHITE:  
      std::cout << "\033[00m";
      break;
  }
}
void Console::Write(const std::string &text, Color color) {
  SetColor(color);
  std::cout << text;
  SetColor(Color::WHITE);
}
void Console::WriteLine(const std::string &text, Color color) {
  Write(text+'\n', color);
}
int main() {
  Console::WriteLine("Hello World", Color::RED);
  Console::Write("Different color", Color::GREEN);
  return 0;
}