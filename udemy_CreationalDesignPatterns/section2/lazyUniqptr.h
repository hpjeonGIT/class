#pragma once
#include <cstdio>
#include <string>
#include <memory>
#include <iostream>
class Logger {
  FILE *m_pStream;
  std::string m_Tag;
  Logger();  
  inline static std::unique_ptr<Logger> m_pInstance{};
public:
  Logger(const Logger&) = delete; // disable copy constructor 
  Logger & operator = (const Logger &) = delete; // disable assign operator
  static Logger& Instance();
  ~Logger();
  void WriteLog(const char *pMessage);
  void SetTag(const char *pTag);
};