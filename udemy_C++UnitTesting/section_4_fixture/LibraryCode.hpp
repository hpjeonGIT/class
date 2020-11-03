#pragma once

class Account 
{
  public:
    Account();
    void deposit(double sum);
    void withdraw(double sum);
    const double getBalance();
    void transfer(Account& to, double sum);
  private:
    double mBalance;
};
