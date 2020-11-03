#include <iostream>
#include <stdexcept>
#include <vector>
#include "LibraryCode.hpp"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

TEST(AccountTest, TestEmptyAccount)
{
  Account account;
  double balance = account.getBalance();
  ASSERT_EQ(0, balance);
}


class AccountTestFixture: public testing::Test
{
  public:
    void SetUp() override;
    void TearDown() override;
    //static void SetUpTestCase();
    //static void TearDownTestCase();
  protected:
    Account *account;
};

void AccountTestFixture::SetUp()
{
  std::cout <<"SetUp called\n";
  account = new Account();
  account->deposit(10.5);
}

void AccountTestFixture::TearDown()
{
  std::cout <<"TearDown called\n";
  delete account;
}
/*
void AccountTestFixture::SetUpTestCase()
{
  std::cout <<"SetUpTestCase called\n";
  account->deposit(10.5);
}

void AccountTestFixture::TearDownTestCase()
{
  std::cout <<"TearDownTestCase called\n";
}
*/

TEST_F(AccountTestFixture, TestDeposit)
{
  std::cout <<"first test\n";
  ASSERT_EQ(10.5, account->getBalance());
}

TEST_F(AccountTestFixture, TestTransferInsufficientFunds)
{
  Account to;
  ASSERT_THROW(account->transfer(to,200), std::runtime_error);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

