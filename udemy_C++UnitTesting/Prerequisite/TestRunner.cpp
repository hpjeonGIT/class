#include <iostream>
#include "LibraryCode.hpp"
#include "gtest/gtest.h"

TEST(TestSample, TestAddition)
{
 ASSERT_EQ(2, add(1,1));
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
