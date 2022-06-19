#include "base.h"
#include <iostream>
FileInputStream::FileInputStream(const std::string& fileName) {
  m_Reader.open(fileName);
  if (!m_Reader) {
    throw std::runtime_error {"Could not open the file for reading"};
  }
}
bool FileInputStream::Read(std::string& text) {
  text.clear();
  std::getline(m_Reader,text);
  return !text.empty();
}
void FileInputStream::Close() {
  if(m_Reader.is_open()) {
    m_Reader.close();
  }
}
FileOutputStream::FileOutputStream(const std::string& fileName){
  m_Writer.open(fileName);
  if(!m_Writer) {
    throw std::runtime_error{"Could not open file for writing"};
  }
}
void FileOutputStream::Write(const std::string& text) {
  m_Writer << text;
}
void FileOutputStream::Close() {
  if (m_Writer.is_open()) {
    m_Writer.close();
  }
}
void Read() {
  FileInputStream Input{"test.txt"};
  std::string text{};
  while(Input.Read(text)) {
    std::cout << text << std::endl;
  }
}
void Write() {
  FileOutputStream output{"test.txt"};
  output.Write("First line\n");
  output.Write("Second line\n");
  output.Write("Third line\n");
}
int main() {
  Write();
  Read();
  return 0;
}