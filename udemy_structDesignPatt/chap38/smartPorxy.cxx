#include "smartPorxy.h"
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
BitmapProxy::BitmapProxy():BitmapProxy{""} {}
BitmapProxy::BitmapProxy(const std::string &fileName) {
  m_pBitmap = std::make_shared<Bitmap>(fileName);
}
void BitmapProxy::Display() {
  if (!m_IsLoaded) { 
    std::cout << "[Proxy] Loading Bitmap\n";
    if(m_FileName.empty()) {
      m_pBitmap->Load(); 
    } else {
      m_pBitmap->Load(m_FileName);
    }
  }
  m_pBitmap->Display();
}
void BitmapProxy::Load() {  m_FileName.clear();}
void BitmapProxy::Load(const std::string & fileName) { m_FileName = fileName; }
int main() {
  //std::shared_ptr<Image> p{new Bitmap {"Smiley.txt"}};
  //std::shared_ptr<Image> p {new BitmapProxy {"Smiley.txt"}};
  //p->Load();
  //p->Display();
  Pointer<Image> p = new Bitmap{"Smiley.txt"};
  p.Get()->Load();
  p.Get()->Display();
}