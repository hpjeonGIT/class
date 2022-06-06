#include "dbfactory.h"
#include "dbaseFrmWork.h"
#include "mysql.h"
Command* SqlFactory::CreateCommand() {
  return new SqlCommand{};
}
Connection* SqlFactory::CreateConnection() {
  return new SqlConnection{};
}
Command* MySqlFactory::CreateCommand() {
  return new MySqlCommand{};
}
Connection* MySqlFactory::CreateConnection() {
  return new MySqlConnection{};
}