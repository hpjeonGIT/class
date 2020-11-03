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
    AccountTestFixture();
    virtual ~AccountTestFixture();
    //void SetUp() override;
    //void TearDown() override;
    static void SetUpTestCase();
    static void TearDownTestCase();
  protected:
    Account account;
};

AccountTestFixture::AccountTestFixture()
{
  std::cout <<"Constructor called\n";
  account.deposit(10.5);
}

AccountTestFixture::~AccountTestFixture()
{
  std::cout <<"Destructor called\n";
}

/*
void AccountTestFixture::SetUp()
{
  std::cout <<"SetUp called\n";
  account.deposit(10.5);
}

void AccountTestFixture::TearDown()
{
  std::cout <<"TearDown called\n";
}
*/
void AccountTestFixture::SetUpTestCase()
{
  std::cout <<"SetUpTestCase called\n";
}

void AccountTestFixture::TearDownTestCase()
{
  std::cout <<"TearDownTestCase called\n";
}


TEST_F(AccountTestFixture, TestDeposit)
{
  std::cout <<"first test\n";
  ASSERT_EQ(10.5, account.getBalance());
}

TEST_F(AccountTestFixture, TestTransferInsufficientFunds)
{
  Account to;
  ASSERT_THROW(account.transfer(to,200), std::runtime_error);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

