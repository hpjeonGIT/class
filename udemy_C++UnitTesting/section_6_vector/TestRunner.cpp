#include <iostream>
#include <vector>
#include "LibraryCode.hpp"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

TEST(VectorTests, ElementsAreTest)
{
  std::vector<int> v = generateNumbers(5,3);
  ASSERT_THAT(v, testing::ElementsAre(0,1,2,0));
}

TEST(VectorTests, RangeTest)
{
  std::vector<int> v = generateNumbers(5,3);
  ASSERT_THAT(v, testing::Each(testing::AllOf(testing::Ge(0), testing::Lt(3))));
}
int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
