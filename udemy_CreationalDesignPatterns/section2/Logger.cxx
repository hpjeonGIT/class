#include "Logger.h"
#include <iostream>
Logger Logger::m_Instance;
Logger::Logger() { m_pStream  = fopen("applog.txt","w");}
Logger& Logger::Instance() { return m_Instance; }
Logger::~Logger() { fclose(m_pStream);}
void Logger::WriteLog(const char* pMessage) {
  fprintf(m_pStream, "[%s] %s\n", m_Tag.c_str(), pMessage);
  fflush(m_pStream);
}
void Logger::SetTag(const char* pTag) { m_Tag = pTag; }
void OpenConnection() {
  Logger &lg2 = Logger::Instance();
  lg2.WriteLog("from OpenConn");
}
int main() {
  Logger &lg = Logger::Instance();;
  lg.SetTag("0530");
  lg.WriteLog("starts");
  OpenConnection();
  lg.WriteLog("Application is shutting down");
}