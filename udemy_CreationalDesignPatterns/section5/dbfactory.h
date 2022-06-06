#pragma once
#include <string_view>
#include "dbaseFrmWork.h"
class DbFactory {
public:
  virtual Command* CreateCommand() = 0;
  virtual Connection* CreateConnection() = 0;
  virtual ~DbFactory() = default;
};
class SqlFactory: public DbFactory {
public:
  Command* CreateCommand() override;
  Connection* CreateConnection() override;  
};
class MySqlFactory: public DbFactory {
public:
  Command* CreateCommand() override;
  Connection* CreateConnection() override;  
};