#include <iostream>
#include <vector>
#include "LibraryCode.hpp"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

class SomeClass
{
  public:
    SomeClass() = default;
    virtual void someMethod()
    {
      std::cout <<"Say something \n";
    };
};

class MockedClass: public SomeClass
{
  public:
    MockedClass() = default;
    MOCK_METHOD0(someMethod, void());
};

TEST(TestSample, TestMock)
{
  MockedClass mc;
  EXPECT_CALL(mc, someMethod()).Times(2);
  mc.someMethod();
}

TEST(TestCountPositives, BasicTest)
{
  std::vector<int> inputVector{1, -2, 3, -4, 5, -6, -7};
  int count = countPositives(inputVector);
  ASSERT_EQ(3, count);
}

TEST(TestCountPositives, EmptyVectorTest)
{
  std::vector<int> inputVector{};
  int count = countPositives(inputVector);
  EXPECT_EQ(0, count);
}

TEST(TestCountPositives, AllNegativesTest)
{
  std::vector<int> inputVector{-1,-2,-3};
  int count = countPositives(inputVector);
  EXPECT_EQ(0, count);
  std::cout << "After assertion \n";
}

TEST(ToUpperTest, BasicTest)
{
  char inputString[] = "Hello World";
  toUpper(inputString);
  ASSERT_STREQ("HELLO WORLD", inputString);
}

TEST(SquareRootTest, NegativeArgumentTest)
{
  double inputValue = -9;
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
