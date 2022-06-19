#pragma once
#include <string>
#include <fstream>
class InputStream {
public:
  virtual bool Read(std::string &text) = 0;
  virtual void Close() = 0;
  virtual ~InputStream() = default;
};
class FileInputStream: public InputStream {
  std::ifstream m_Reader;
public:
FileInputStream() = default;
FileInputStream(const std::string& fileName);
  bool Read(std::string &text) override;
  void Close() override;
};
class OutputStream {
public:
  virtual void Write(const std::string &text) = 0;
  virtual void Close() = 0;
  virtual ~OutputStream() = default;
};
class FileOutputStream: public OutputStream {
  std::ofstream m_Writer{};
public:
  FileOutputStream() = default;
  FileOutputStream(const std::string& fileName);
  void Write(const std::string& text) override;
  void Close() override;
};