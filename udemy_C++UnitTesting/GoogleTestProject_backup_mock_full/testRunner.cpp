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

MockDatabaseConnection::MockDatabaseConnection(std::string serverAddress) : IDatabaseConnection(serverAddress)
{

}


// Throwing an error
TEST(TestEmployeeManager, TestConnectionError)
{
    MockDatabaseConnection dbConnection("dummyConnection");
    EXPECT_CALL(dbConnection, connect()).WillOnce(testing::Throw(std::runtime_error("Dummy error")));
    EXPECT_CALL(dbConnection, disconnect());

    EmployeeManager employeeManager(&dbConnection);
}


// Action + throwing an error
ACTION(myThrow)
{
  std::cout <<"Throwing an error\n";
  throw std::runtime_error("Dummy error");
}

TEST(TestEmployeeManager, TestConnectionErrorAction)
{
    MockDatabaseConnection dbConnection("dummyAddress");
    EXPECT_CALL(dbConnection, connect()).WillOnce(myThrow());
    EXPECT_CALL(dbConnection, disconnect());
    ASSERT_THROW(EmployeeManager employeeManager(&dbConnection), std::runtime_error);
}

// Using Invoke
void someFreeFunction()
{
    std::cout <<"Free function\n";
    throw std::runtime_error("Dummy Exception");
}

TEST(TestEmployeeManager, TestConnectionErrorInvoke)
{
    MockDatabaseConnection dbConnection("dummyAddress");
    //EXPECT_CALL(dbConnection, connect()).WillOnce(testing::Invoke(someFreeFunction));
    EXPECT_CALL(dbConnection, connect()).WillOnce(testing::Invoke([]() {
        std::cout << "Lambda called\n"; throw std::runtime_error("Dummy error");
        }
    ));

    ASSERT_THROW(EmployeeManager employeeManager(&dbConnection), std::runtime_error);
}



TEST(TestEmployeeManager, TestConnection)
{
    MockDatabaseConnection dbConnection("dummyConnection");
    EXPECT_CALL(dbConnection, connect());
    EXPECT_CALL(dbConnection, disconnect());

    EmployeeManager employeeManager(&dbConnection);
}

TEST(TestEmployeeManager, TestUpdateSalary)
{
    MockDatabaseConnection dbConnection("dummyConnection");
    EXPECT_CALL(dbConnection, connect());
    EXPECT_CALL(dbConnection, disconnect());
    EXPECT_CALL(dbConnection, updateSalary(testing::_, testing::_)).Times(1);
    EmployeeManager employeeManager(&dbConnection);
    employeeManager.setSalary(50, 6000);
}

TEST(TestEmployeeManager, TestGetSalary)
{
    const int employeeId = 50;
    const float salary = 6100.0;
    MockDatabaseConnection dbConnection("dummyConnection");
    EXPECT_CALL(dbConnection, connect());
    EXPECT_CALL(dbConnection, disconnect());
    EXPECT_CALL(dbConnection, getSalary(testing::_)).WillOnce(testing::Return(salary));
    EmployeeManager employeeManager(&dbConnection);
    float returnedSalary = employeeManager.getSalary(employeeId);
    ASSERT_EQ(salary, returnedSalary);
}


TEST(TestEmployeeManager, TestGetSalaryRange)
{
    const int low = 5000, high = 8000;
  std::vector<Employee> returnedVector{Employee{1,  5600, "John"},
                                       Employee{2, 7000,"Jane"},
                                       Employee{3, 6000, "Alex"}};
    MockDatabaseConnection dbConnection("dummyConnection");
    EXPECT_CALL(dbConnection, connect());
    EXPECT_CALL(dbConnection, disconnect());
    EXPECT_CALL(dbConnection, getSalariesRange(low,high)).WillOnce(testing::Return(returnedVector));
    EmployeeManager employeeManager(&dbConnection);

    std::map<std::string, float> returnedMap = employeeManager.getSalariesBetween(low,high);
    for(auto it=returnedMap.begin(); it!=returnedMap.end() ; ++it) 
    {
        std::cout <<it->first <<" "<<it->second << std::endl;
        ASSERT_THAT(it->second, testing::AllOf(testing::Gt(low),testing::Lt(high)));
    }
}

int main(int argc, char **argv)
{
 ::testing::InitGoogleTest(&argc, argv);
 return RUN_ALL_TESTS();
}
