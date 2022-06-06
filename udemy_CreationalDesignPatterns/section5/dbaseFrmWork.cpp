#include "dbaseFrmWork.h"
#include "mysql.h"
#include "dbfactory.h"
#include <iostream>
void SqlConnection::Open() { std::cout << "[SqlConnection] Connection opened\n";}
void SqlCommand::ExeucteCommand() {
  std::cout << "[SqlCommand] Executing command on" 
            << m_pConnection->GetConnectionString()
            << std::endl;
}
SqlRecordSet* SqlCommand::ExecuteQuery(){
  std::cout << "[SqlCommand] Executing query\n";
  return new SqlRecordSet();
}
SqlRecordSet::SqlRecordSet() {
  m_Cursor = m_Db.begin();
}
const std::string& SqlRecordSet::Get() {
  return *m_Cursor++;
}
bool SqlRecordSet::HasNext() {
  return m_Cursor != m_Db.end();
}
void UsingFactory(DbFactory *pFactory) {
  Connection* pCon = pFactory->CreateConnection();
  pCon->SetConnectionString("uid=umar;db=movies;table=actors");
  pCon->Open();
  Command* pCmd = pFactory->CreateCommand();
  pCmd->SetConnection(pCon);
  pCmd->SetCommand("select * from actors");
  RecordSet *pRec = pCmd->ExecuteQuery();
  while(pRec->HasNext()) {
    std::cout << pRec->Get() << std::endl;
  }
  delete pCon;
  delete pCmd;
  delete pRec;
}
int main() {
  MySqlFactory f;
  UsingFactory(&f);
}