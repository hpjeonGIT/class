#include <iostream>
#include <stdexcept>
#include <vector>
#include "LibraryCode.hpp"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

//Validator (5,10)
// 4,5,7,10,11

class ValidatorFixture: public testing::TestWithParam<int>
{
  public:
  protected:
    Validator mValidator{5,10};
};

TEST_P(ValidatorFixture, TestInRange) // Parametric test
{
  int param = GetParam();
  std::cout << "Param = " <<param <<std::endl;
  bool isInside = mValidator.inRange(param);
  ASSERT_TRUE(isInside);
}

INSTANTIATE_TEST_CASE_P(InRangeTrue, ValidatorFixture, testing::Values(1,5,6,7,9,10,100));

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

