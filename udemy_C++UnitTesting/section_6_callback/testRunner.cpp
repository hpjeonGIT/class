#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <map>

#include "IDatabaseConnection.hpp"
#include "Employee.hpp"
#include "EmployeeManager.hpp"


class MockDatabaseConnection : public IDatabaseConnection
{
public:
    MockDatabaseConnection(std::string serverAddress);

    //MOCK_METHODn n=0,10
/*   // legacy format
    MOCK_METHOD0(connect, void());
    MOCK_METHOD0(disconnect, void());

    MOCK_CONST_METHOD1(getSalary, float(int));
    MOCK_METHOD2(updateSalary, void(int, float) );

    MOCK_CONST_METHOD1(getSalariesRange, std::vector<Employee>(float));
    MOCK_CONST_METHOD2(getSalariesRange, std::vector<Employee>(float, float));
*/

   MOCK_METHOD(void, connect, ());
   MOCK_METHOD(void, disconnect, ());

   MOCK_METHOD(float, getSalary, (int), (const)); // MOCKing also override the functions of parent class
   MOCK_METHOD(void, updateSalary, (int, float));
   MOCK_METHOD(std::vector<Employee>, getSalariesRange, (float), (const));
   MOCK_METHOD(std::vector<Employee>, getSalariesRange, (float,float), (const));
   MOCK_METHOD((std::map<std::string, int>), something, ());

};

void realCallback()
{
  std::cout << "Real callback invoked\n";
}


MockDatabaseConnection::MockDatabaseConnection(std::string serverAddress) : IDatabaseConnection(serverAddress)
{

}

TEST(TestEmployeeManager, CallbackTest)
{
  MockDatabaseConnection dbConnection("DummyAddress");
  testing::MockFunction<void()> mockfunction;
  dbConnection.setOnConnect(mockfunction.AsStdFunction());
  EXPECT_CALL(mockFunction, Call()); 

  dbConnection.connect();  
}

int main(int argc, char **argv)
{
 ::testing::InitGoogleTest(&argc, argv);
 return RUN_ALL_TESTS();
}
