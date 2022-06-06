#include "mysql.h"
#include <iostream>
void MySqlConnection::Open() { std::cout << "[MySqlConnection] Connection opened\n";}
void MySqlCommand::ExeucteCommand() {
  std::cout << "[MySqlCommand] Executing command on" 
            << m_pConnection->GetConnectionString()
            << std::endl;
}
MySqlRecordSet* MySqlCommand::ExecuteQuery(){
  std::cout << "[MySqlCommand] Executing query\n";
  return new MySqlRecordSet();
}
MySqlRecordSet::MySqlRecordSet() {
  m_Cursor = m_Db.begin();
}
const std::string& MySqlRecordSet::Get() {
  return *m_Cursor++;
}
bool MySqlRecordSet::HasNext() {
  return m_Cursor != m_Db.end();
}
