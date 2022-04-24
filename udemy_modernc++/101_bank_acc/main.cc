#include <iostream>
class Account {
private:
  std::string m_Name;
  int m_AccNo;
  static int s_ANGenerator;
protected:
  float m_Balance;  
public:
  //Account();
  Account(const std::string &name, float balance): m_Name{name}, m_Balance{balance} {
    m_AccNo = ++s_ANGenerator;
    }
  ~Account()=default;
  const std::string GetName() const {return m_Name;}
  float GetBalance() const {return m_Balance; }
  int GetAccountNo() const {return m_AccNo; }
  void Deposit(float balance) { m_Balance += balance; }
  void Withdraw(float balance) { m_Balance -= balance; }
};
class Savings: public Account {
  float m_Rate;
public:
  /*
  Savings(const std::string &name, float balance, float rate)
  : m_Name(name), m_Balance(balance), m_Rate(rate) { m_AccNo = ++s_ANGenerator;  }; => This doesn't work */
  Savings(const std::string &name, float balance, float rate)
  : Account{name, balance}, m_Rate{rate} {} // MUST use the constructor of the base class
  ~Savings() = default;
};
class Checking: public Account {
  static float s_CheckingLimit;
public:
  //using Account::Account;
  Checking(const std::string &name, float balance): Account{name, balance} {}
  ~Checking() = default;
  void Withdraw(float balance) {
    if (m_Balance-balance < s_CheckingLimit) {
      std::cout << "Reached checking Limit. Withdrawal is disabled\n";
    } else {
      Account::Withdraw(balance);
    }
  }
};
int Account::s_ANGenerator = 1000; // needs int in the beginning
float Checking::s_CheckingLimit = 50.00f; 
int main() {
  Account acc01("Bob",1000.00f);
  std::cout << "Initial balance " << acc01.GetBalance() << std::endl;
  acc01.Deposit(200.00f);
  acc01.Withdraw(380.00f);
  std::cout << "Current balance " << acc01.GetBalance() << std::endl;
  Savings acc02("Bob",1000.00f, 0.01f);
  std::cout << "Initial balance " << acc02.GetBalance() << std::endl;
  acc02.Deposit(200.00f);
  acc02.Withdraw(380.00f);
  std::cout << "Current balance " << acc02.GetBalance() << std::endl;
  Checking acc03("Bob",1000.00f);
  std::cout << "Initial balance " << acc03.GetBalance() << std::endl;
  acc03.Withdraw(700.00f);
  acc03.Withdraw(300.00f);
  std::cout << "Current balance " << acc03.GetBalance() << std::endl;
  return 0;
}

