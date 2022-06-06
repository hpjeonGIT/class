#pragma once
#include <string>
#include <vector>
class Connection {
  std::string m_ConnectionString;
public:
  void SetConnectionString(const std::string &connStr) {
    m_ConnectionString = connStr;    
  }
  const std::string & GetConnectionString() const {
    return m_ConnectionString;
  }
  virtual void Open() = 0;
  virtual ~Connection() = default;
};
class RecordSet {
public:
  virtual const std::string & Get() = 0;
  virtual bool HasNext() = 0;
  virtual ~RecordSet() = default;
};
class Command {
  std::string m_CommandString;
protected:
  Connection *m_pConnection{};
public:
  Connection * GetConnection() const {
    return m_pConnection;
  }
  const std::string & GetCommandString() const {
    return m_CommandString;
  }
  void SetCommand(const std::string &commandStr) {
    m_CommandString = commandStr;    
  }
  void SetConnection(Connection *pConnection) {
    m_pConnection = pConnection;
  }
  virtual void ExeucteCommand() = 0;
  virtual RecordSet * ExecuteQuery() = 0;
  virtual ~Command() = default;
};
class SqlConnection: public Connection {
public:
  void Open() override;
};
class SqlRecordSet: public RecordSet {
  const std::vector<std::string> m_Db{ "Terminator", "Predator", "Eraser"};
  std::vector<std::string>::const_iterator m_Cursor;
public:
  SqlRecordSet();
  const std::string& Get() override;
  bool HasNext() override;
};
class SqlCommand: public Command {
public:
  //SqlCommand() {}
  void ExeucteCommand() override;
  SqlRecordSet* ExecuteQuery() override;
};
