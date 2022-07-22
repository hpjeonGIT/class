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