#include "lazyUniqptr.h"
Logger::Logger() { std::cout << "constructor\n"; m_pStream  = fopen("applog.txt","w");}
Logger& Logger::Instance() { 
  if (m_pInstance == nullptr)  m_pInstance.reset(new Logger{});
  return *m_pInstance; }
Logger::~Logger() { 
  fclose(m_pStream);
  std::cout << "destructor \n";
}
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