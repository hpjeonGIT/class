#include "buff.h"
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
  m_Writer << text << '\n';
}
void FileOutputStream::Close() {
  if (m_Writer.is_open()) {
    m_Writer.close();
  }
}
void Read() {
  BufferedInputStream Input{"test.txt"};
  std::string text{};
  while(Input.Read(text)) {
    std::cout << text << std::endl;
  }
}
void Write() {
  BufferedOutputStream output{"test.txt"};
  output.Write("First line");
  output.Write("Second line");
  output.Write("Third line");
}
void BufferedOutputStream::Write(const std::string& text) {
  std::cout << "Buffered Write\n";
  FileOutputStream::Write(text);
}
void BufferedOutputStream::Close() {
  FileOutputStream::Close();
}
bool BufferedInputStream::Read(std::string& text) {
  std::cout << "Buffered Read\n";
  auto result = FileInputStream::Read(text);
  return result; // bool type
}
void BufferedInputStream::Close() {
  FileInputStream::Close();
}

int main() {
  Write();
  Read();
  return 0;
}