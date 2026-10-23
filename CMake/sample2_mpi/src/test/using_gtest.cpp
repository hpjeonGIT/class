#include "func01.h"
#include "gtest/gtest.h"

class MyUnitTest : public ::testing::Test {
protected:
    MyUnitTest() {}
    virtual ~MyUnitTest() {}
    virtual void SetUp() {
    }
    virtual void TearDown() {
    }
};

TEST_F(MyUnitTest, withGtest1) {
     int x = 123;
     int z = ret_2x(x);
     EXPECT_EQ(x*2,z);
}

int main(int argc, char** argv) {
     testing::InitGoogleTest(&argc, argv);
     return RUN_ALL_TESTS();
}

