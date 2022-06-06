#include "dbaseFrmWork.h"
class MySqlConnection: public Connection {
public:
  void Open() override;
};
class MySqlRecordSet: public RecordSet {
  const std::vector<std::string> m_Db{ "Rambo", "Rocky", "Cliff Hanger"};
  std::vector<std::string>::const_iterator m_Cursor;
public:
  MySqlRecordSet();
  const std::string& Get() override;
  bool HasNext() override;
};
class MySqlCommand: public Command {
public:
  void ExeucteCommand() override;
  MySqlRecordSet* ExecuteQuery() override;
};
