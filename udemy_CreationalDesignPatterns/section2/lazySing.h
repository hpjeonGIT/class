#pragma once
#include <cstdio>
#include <string>
class Logger {
  FILE *m_pStream;
  std::string m_Tag;
  Logger();  
  //static Logger m_Instance; // eager instance
  static Logger *m_pInstance; // lazy instance
public:
  Logger(const Logger&) = delete; // disable copy constructor 
  Logger & operator = (const Logger &) = delete; // disable assign operator
  static Logger& Instance();
  ~Logger();
  void WriteLog(const char *pMessage);
  void SetTag(const char *pTag);
};