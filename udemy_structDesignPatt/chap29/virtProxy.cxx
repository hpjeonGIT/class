#include "virtProxy.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
void Image::SetFileName(const std::string& fileName) {
  m_FileName = fileName;
}
Image::Image(const std::string& fileName): m_FileName{fileName} {}
const std::string& Image::GetFileName() const { return m_FileName; }
void Bitmap::Display() { std::cout << m_Buffer;}
void Bitmap::Load() {
  m_Buffer.clear();
  std::ifstream file {GetFileName()};
  if (!file) throw std::runtime_error{"Failed to open file"};
  std::string line{};
  std::cout << "Loading bitmap[";
  using namespace std::chrono_literals;
  while(std::getline(file,line)) {
    m_Buffer += line + '\n';
    std::this_thread::sleep_for(100ms);
    std::cout << '.' << std::flush;
  }
  std::cout << "] Done!\n";
}
void Bitmap::Load(const std::string& fileName) {
  SetFileName(fileName);
  Load();
}
int main() {
  std::shared_ptr<Image> p{new Bitmap {"Smiley.txt"}};
  p->Load();
  //p->Display();
}