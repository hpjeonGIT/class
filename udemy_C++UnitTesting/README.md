## Udemy class:C++ Unit Testing: Google Test and Google Mock, Beginning Test Driven Development (TDD) in C++ with Googletest and Googlemock
- Created by Serban Stoenescu

## Objective
- Using cmake + google test for TDD
- Some features of c++14 like generator and tuple

## Mocking
- Mocking means that we fake methods of a class. Instead of connection to real DB, we may mock the method, pretending to connect and testing.

## Questions
- In section 24, static void SetUpTestCase() is used. As static data, account.deposit() is given in the constructor of class AccountTestFixture. Can the initialization be located in the inside of SetUpTestCase()?
- In section 27, parametric tests are studied while more study is required regarding TestWithParam
  - <int> or tuple might be used while if pointer is used, how to apply?

## Section 1: Course Intro

## Section 2: Setting Up Google test

11. Google Test CMake Sample
- Google test module configuration
```
##
proc ModulesHelp { } {        puts stderr "google test" }
module-whatis   "googltest"
# for Tcl script use only
set     topdir          /home/hpjeon/sw_local/gtest
prepend-path    LD_LIBRARY_PATH $topdir/lib
prepend-path    CPATH           $topdir/include
setenv GTEST_LIBRARY $topdir/lib
setenv GTEST_INCLUDE_DIRS $topdir/include
setenv GTEST_MAIN_LIBRARY $topdir/lib
```
  - module load gtest

## Section 3: Unit Testing Basics

13. Intro

14. What is a Unit test

15. Unit Test Characteristics
- Can execute independently
- Run fast (milliseconds)
  - There could be thousands of unit tests
- Do not rely on external input
  - Isolated entity

16. Types of Testing
- Unit test
  - Individual modle
  - White box type of testing
- Integration test
  - Integrated modules
- System testing
  - End to end testing
  - Black box type of testing

17. Unit Test Structure 
- Arrange
  - Test setup
  - Set all inputs and preconditions
- Act
  - Call the method under test
- Assert
  - Check that the results are correct

18. Unit Test Structure (code example)
- CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)
find_package(GTest REQUIRED)
message("GTest_INCLUDE_DIRS = ${GTest_INCLUDE_DIRS}")
add_library(commonLibrary LibraryCode.cpp)
add_executable(mainApp main.cpp)
target_link_libraries(mainApp commonLibrary)
add_executable(unitTestRunner testRunner.cpp)
target_link_libraries(unitTestRunner commonLibrary ${GTEST_LIBRARIES} pthread)
```
- LibraryCode.hpp
```cpp
#pragma once
#include <vector>
int countPositives(std::vector<int> const& inputVector);
```
- LibraryCode.cpp
```cpp
#include "LibraryCode.hpp"
#include <algorithm>
bool isPositive(int x)
{
 return x >= 0;
}
int countPositives(std::vector<int> const& inputVector)
{
    return std::count_if(inputVector.begin(), inputVector.end(), isPositive);
}
```
- main.cpp
```cpp
#include <iostream>
#include "LibraryCode.hpp"
int main(int argc, char **argv)
{
    std::cout << "Actual application code \n";
 //   std::cout << "2 + 3 = " << add(2, 3) << '\n';
    return 0;
}
```
- testRunner.cpp
```cpp
#include <iostream>
#include <gtest/gtest.h>
#include "LibraryCode.hpp"
TEST(TestCountPositives, BasicTest)
{
    //Arrange
    std::vector<int> inputVector{1, -2, 3, -4, 5, -6, -7};
    //Act
    int count = countPositives(inputVector);
    //Assert
    ASSERT_EQ(3, count);
}
TEST(TestCountPositives, EmptyVectorTest)
{
    //Arrange
    std::vector<int> inputVector{};
    //Act
    int count = countPositives(inputVector);
    //Assert
    ASSERT_EQ(0, count);
}
TEST(TestCountPositives, AllNegativesTest)
{
    //Arrange
    std::vector<int> inputVector{-1, -2, -3};
    //Act
    int count = countPositives(inputVector);
    //Assert
    ASSERT_EQ(0, count);
}
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```
- Command
```bash
$ cmake ..
-- The C compiler identification is GNU 7.5.0
-- The CXX compiler identification is GNU 7.5.0
...
$ ./unitTestRunner 
[==========] Running 3 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 3 tests from TestCountPositives
[ RUN      ] TestCountPositives.BasicTest
[       OK ] TestCountPositives.BasicTest (0 ms)
[ RUN      ] TestCountPositives.EmptyVectorTest
[       OK ] TestCountPositives.EmptyVectorTest (0 ms)
[ RUN      ] TestCountPositives.AllNegativesTest
[       OK ] TestCountPositives.AllNegativesTest (0 ms)
[----------] 3 tests from TestCountPositives (0 ms total)
[----------] Global test environment tear-down
[==========] 3 tests from 1 test suite ran. (1 ms total)
[  PASSED  ] 3 tests.
```

19. Assertions
- Success 
- Failure
  - Fatal: ASSERT_* 
  - Non-fatal: tests continue. EXPECT_*

20. Assertions (code example)

21. Assertions on Strings - wrong way to do it
- `ASSERT_EQ("HELLO WORLD", someString);` does not work as it will compare the address, not the strings
  - `if("HELLO WORLD" == someString)` will not work. Use strcmp()

22. Assertions on Strings
- ASSERT_STREQ() or EXPECT_STREQ()
- ASSERT_STRNE() or EXPECT_STRNE()
- ASSERT_STRCASEEQ() or EXPECT_STRCASEEQ() # ignores case
- ASSERT_STRCASENE() or EXPECT_STRCASENE() # ignores case

23. Assertion on Strings (code example)

24. Assertions on Exceptions
- ASSERT_THROW()/EXPECT_THROW() # throws exception of an exact type
- ASSERT_ANY_THROW()/EXPECT_ANY_THROW() # throws an exception of any type
- ASSERT_NO_THROW()/EXPECT_NO_THROW() # throws no exception

25. Assertions on Exceptions (code example)
```cpp
#include <iostream>
#include <gtest/gtest.h>
#include "LibraryCode.hpp"
TEST(SquareRootTest, NegativeArgumentTest)
{
    double inputValue = -9;
    //ASSERT_ANY_THROW(mySqrt(inputValue));
    ASSERT_THROW(mySqrt(inputValue), std::runtime_error);
}
TEST(SquareRootTest, PositiveArgumentTest)
{
    double inputValue = 9;
    ASSERT_NO_THROW(mySqrt(inputValue));
}
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```
- Using `ASSERT_THROW(mySqrt(inputValue), std::overflow_error);` will fail as mySqrt() will throw a runtime_error

26. Summary
- Unit tests are useful for regresssion testing, meaning that when you make a change, they help prevent the introduction of new bugs.
- Unit tests are short, independent and fast.
- Unit testing is part of functional testing, i. e. the correctness of the code is checked. Individual functions/methods are usually tested.
- A unit test is divided into three parts: Arrange (test setup), Act (call the method), Assert(check the result).
- An assertion is where the test condition is checked. They are fatal (ASSERT) or non-fatal(EXPECT).
- There are special assertions for strings.
- Assertions can be used to check if an exception was thrown, or what type of exception was thrown.

## Section 4: Fixtures: Remove Redundant Code

27. Intro

28. Introduction to Test Fixtures
- Reusing the code
- For tests with similar Setup and TearDown
- testing::Test
  - SetUp()
  - TearDown()
  - SetUpTestCase()
  - TearDownTestCase()

29. Test Fixtures (Code Example)
```cpp
#include <iostream>
#include <gtest/gtest.h>
#include <stdexcept>
#include "LibraryCode.hpp"
TEST(AccountTest, TestEmptyAccount)
{
  Account account;
  double balance = account.getBalance();
  ASSERT_EQ(0, balance);
}
class AccountTestFixture: public testing::Test
{
  public:
   AccountTestFixture();
   virtual ~AccountTestFixture();
   void SetUp() override;
   void TearDown() override;
   static void SetUpTestCase();
   static void TearDownTestCase();
  protected:
   Account account;
};
AccountTestFixture::AccountTestFixture() {   std::cout << "Constructor called\n"; }
AccountTestFixture::~AccountTestFixture() {  std::cout << "Destructor called\n"; }
void AccountTestFixture::SetUpTestCase() {   std::cout << "SetUpTestCase called\n"; }
void AccountTestFixture::TearDownTestCase() {   std::cout << "TearDownTestCase called\n"; }
void AccountTestFixture::SetUp() {
  std::cout << "SetUp called\n";
  account.deposit(10.5); 
}
void AccountTestFixture::TearDown() {
  std::cout << "TearDown called\n"; 
}
TEST_F(AccountTestFixture, TestDeposit)
{ 
  std::cout << "Test body\n";
  ASSERT_EQ(10.5, account.getBalance());
}
TEST_F(AccountTestFixture,  TestWithdrawOK)
{
  account.withdraw(3);
  ASSERT_EQ(7.5, account.getBalance());
}
TEST_F(AccountTestFixture,  TestWithdrawInsufficientFunds)
{
  ASSERT_THROW(account.withdraw(300), std::runtime_error);
}
TEST_F(AccountTestFixture,  TestTransferOK)
{
  Account to;
  account.transfer(to, 2);
  ASSERT_EQ(8.5, account.getBalance());
  ASSERT_EQ(2, to.getBalance());
}
TEST_F(AccountTestFixture,  TestTransferInsufficientFunds)
{
  Account to;
  ASSERT_THROW(account.transfer(to, 200), std::runtime_error);
}
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

30. Fixture Flow
- Fixture Construct
  - SetUp
    - Test Body
  - Tear Down
- Fixture Destruct
- SetUp/TearDown
  - Use for virtual methods
  - Do not use fatal assertions (ASSERT_*) in constructors. Uset in SetUp
  - Do not call code that can throw exceptions in the destructor. Use them in TearDown

31. Parameterized Tests
- Same code but different values
- Avoid code duplication 
- `testing::Test` => `testing::TestWithParam<T>`

32. GoogleTest update

33. Parameterized Tests (Code example)
```cpp
#include <iostream>
#include <gtest/gtest.h>
#include "LibraryCode.hpp"
// Validator(5, 10)
// 4, 5, 6, 7, 9, 10, 11 
class ValidatorFixture : public testing::TestWithParam<std::tuple<int, bool>>
{
public:
protected:
 Validator mValidator{5, 10};
};
TEST_P(ValidatorFixture, TestInRange)
{
  std::tuple<int, bool> tuple = GetParam();
  int param = std::get<0>(tuple);
  bool expectedValue = std::get<1>(tuple);
  std::cout << "param = " << param << " expected value = " << expectedValue << '\n';
  bool isInside = mValidator.inRange(param);
  ASSERT_EQ(expectedValue, isInside);
}

INSTANTIATE_TEST_CASE_P(InRangeTrue, ValidatorFixture, 
  testing::Values(
                  std::make_tuple(-50, false),
                  std::make_tuple(4, false),
                  std::make_tuple(5, true),
                  std::make_tuple(6, true),
                  std::make_tuple(7, true),
                  std::make_tuple(9, true),
                  std::make_tuple(10, true),
                  std::make_tuple(11, false),
                  std::make_tuple(100, false) ) );
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

34. Summary
- A fixture is useful for removing code duplication.
- It's used where the setup phase and cleanup phase are similar.
- It's a class where the test setup is written in the SetUp() method and the cleanup is in TearDown().
- A new fixture is created for each test.
- Parameterized tests can be used to generate tests that have the same body, but different input values.
- When you generate a test, the expected output values can be packed together with the input values using complex data structures.
- Generators can be used to generate input values for the test.

## Section 5: Setting Up Google Mock

35. Intro

36. Google Mock Visual Studio (Windows)

37. Google Mock CMake Sample (Linux)
- main.cpp
```cpp
#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
class BaseClass {
  public: 
    BaseClass() = default;
    ~BaseClass() = default;
    virtual void BaseMethod() { 
      std::cout << "Say Hello\n";
    }
};
class MockedClass: public BaseClass {
  public:
    MockedClass() = default;
    MOCK_METHOD0(BaseMethod, void()); // function name, signagture, argument ...
};
int add(int a, int b) {
  return a+b;
}
TEST(TestSample, TestMock) {
  MockedClass mc;
  EXPECT_CALL(mc, BaseMethod()).Times(1);
  mc.BaseMethod();
}
TEST(TestSample, TestAddition) {
  ASSERT_EQ(2, add(1,1));
}
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
```
- CMakeLists.txt
```
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)
find_package(GTest REQUIRED)
message("GTest_INCLUDE_DIRS = ${GTest_INCLUDE_DIRS}")
set(GMOCK_LIBRARIES ${GTEST_INCLUDE_DIR}/../lib/libgmock.a)
add_executable(mainApp main.cpp)
target_link_libraries(mainApp ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES} pthread)
```
  - Need to add GMOCK_LIBRARIES
- Command
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make 
$ ./mainApp 
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from TestSample
[ RUN      ] TestSample.TestMock
[       OK ] TestSample.TestMock (0 ms)
[ RUN      ] TestSample.TestAddition
[       OK ] TestSample.TestAddition (0 ms)
[----------] 2 tests from TestSample (0 ms total)
[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (0 ms total)
[  PASSED  ] 2 tests.
```

## Section 6: Google Mock

38. Intro

39. Mocking Project Resources

40. Introduction to Mocking
- What is Mocking
  - Mock = a type of test double
  - Real object -> fake
  - Useful for isolation and collaboration tests
- Test Doubles - Other Types
  - Fake
    - Working implementation
    - Takes a shortcut
    - Not suitable for production such as in-memory database
  - Stub
    - Responds with pre-defined data
  - Mock
    - Has set expectations
      - Throws/doesn't throw exceptions
      - Calls methods

41. Mocking Methods
- Current method
  - In class, use the method from a class
  - `MOCK_METHOD(ReturnType, MethodName, (Arguments...))`
  - `MOCK_METHOD(int, add, (int,int));`
- Legacy method
  - Method with `n` parameters
  - `MOCK_METHODn(name, returnType(paramType1,...));`
  - `MOCK_METHOD2(sum int(int,int));`
  - `MOCK_METHOD0(doSome, void());`  
  - Or `MOCK_CONST_METHOD2(sum, int(int,int))` when sum() function has const specification

42. Presentation of Our Project

43. Mocking Methods - Current Way (code example)
- Need to call `EXPECT_CALL()` prior to running the actual mocked method

44. Mocking Methods - Legacy (code example)

45. Setting Expectations and Behaviour
- Setting Expectations
  - ON_CALL vs EXPECT_CALL
    - ON_CALL: sets the behavior when a method gets called. Not often used
    - EXPECT_CALL = ON_CALL + expectations. Use this call mostly
      - EXPECT_CALL(someObject, someMethod).Times(2) # checks someMethod is called twice
      - EXPECT_CALL(someObject, someMethod(value_I_want)) # checks someMethod is called with value_I_want
- Setting Mock Behavior
```cpp
ACTION(myAction) { ... }
EXPECT_CALL(someObject, someMethod()).WillOnce(myAction());
```
  - WillOnce
  - WillRepeatedly
  - WillByDefault
  - Return
  - ReturnRef
  - `EXPECT_CALL(someObject, someMethod()).WillRepeatedly(Return(6))`
- Cardinality
  - AnyNumber()
  - AtLeast(n)
  - AtMost(n)
  - Between(m,n)
  - Exactly(n) or n

46. Mocking - Times (code example)
- `EXPECT_CALL(dbConnection, updateSalary(50,6000)).Times(1);` # checks updateSalary() function is called with (50,6000) once
- `EXPECT_CALL(dbConnection, updateSalary(testing::_, testing::_)).Times(1);` # regardless of arguments, checks updateSalary() is called once
  - Use `testing::_` as a wild card

47. Mocking - Returns (code example)
```cpp
const float salary = 6100.0;
...
EXPECT_CALL(dbConnection, getSalary(testing::_)).Times(1).WillOnce(testing::Return(salary));
```

48. Invoking Actions (code example)
```cpp
ACTION(myThrow)
{
    std::cout << "Throwing an error!\n";
    throw std::runtime_error("Dummy error");
}
TEST(TestEmployeeManager, TestConnectionErrorAction)
{
    MockDatabaseConnection dbConnection("DummyAddresss");
    EXPECT_CALL(dbConnection, connect()).WillOnce(myThrow());
    ASSERT_THROW(EmployeeManager employeeManager(&dbConnection), std::runtime_error);
}
```
- May use `testing::Invoke()` with a regular function or a lambda expression

49. Matchers
- EXPECT_CALL(someObject, someMethod(5,"hello")); # checks if someMethod() is called with 5 and "hello"
- EXPECT_CALL(someObject, someMethod(Gt(5))); # checks if someMethod() is called with with an argument greater than 5
  - Gt, Ge, Lt, Le, Eq
  - For any parameter, use `_`
- IsNull()/IsNotNull()
- HasSubstr("orld") # string matchers
- Combining Matchers
  - EXPECT_CALL(someObject, someMethod(AllOf(Gt(5),Le(100),Not(7))));
  - AnyOf, AllOfArray(), AnyOfArray(), Not()
  - ASSERT_THAT # for combinations of matchers. Useful for validation of vectors/arrays

50. Matchers (Code Example)

51. Assertions on Vectors (Code Example)
```cpp
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "LibraryCode.hpp"
TEST(VectorTests, ElementsAreTest)
{
  std::vector<int> v = generateNumbers(5, 3);
  // 0, 1, 2, 0, 1

  ASSERT_THAT(v, testing::ElementsAre(0, 1, 2, 0, 1));
}
TEST(VectorTests, RangeTest)
{
  using namespace testing;
  std::vector<int> v = generateNumbers(5, 3);
  
  ASSERT_THAT(v, Each(AllOf(Ge(0), Lt(3))));
}
int main(int argc, char **argv)
{
 ::testing::InitGoogleTest(&argc, argv);
 return RUN_ALL_TESTS();
}
```

52. Callbacks

53. Mocking Private and Static Methods

54. Summary
- Mocks can be used to isolate the test.
- Mocked methods have empty implementations.
- They can be used to control the behaviour of certain methods, like: returning a certain result, calling another method, throwing exceptions.
- Mocks can be used for collaboration tests. That means you can test that method A called method B, with what parameters, and so on
- Matchers can be used for matching parameters. Special matchers are "_" (anything) or the exact value ("Exactly").
- Other matchers usually found: Gt( greater than), Ge(greater or equal), Lt(lower than), Le(lower or equal).
- There are special matchers for strings.
- Matchers can be used in assertions on vectors.
