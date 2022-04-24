#include <iostream>
#include <memory>
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
  virtual ~Account() {std::cout <<"destructor of Account\n";}
  const std::string GetName() const {return m_Name;}
  float GetBalance() const {return m_Balance; }
  int GetAccountNo() const {return m_AccNo; }
  virtual float GetInterestRate() const {return 0.0f;}
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
  ~Savings() override {std::cout <<"destructor of Savings\n";}
  float GetInterestRate() const override final {return m_Rate;} 
};
/*
class Ssavings: public Savings {
public:
  Ssavings(const std::string &name, float balance, float rate)
  : Savings{name, balance, rate} {}
  ~Ssavings() = default;
  float GetInterestRate() const override {return 0.0f;}  => cannot override due to final keyword above
}; */
class Checking: public Account {
  static float s_CheckingLimit;
public:
  //using Account::Account;
  Checking(const std::string &name, float balance): Account{name, balance} {}
  ~Checking() override {std::cout <<"destructor of Checking\n" ;}
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
void Transact(Account *pAcc) {
  pAcc->Deposit(100.00f);
  std::cout << "rate = " << pAcc->GetInterestRate() << std::endl;  
}
int main() {
  Checking acc03("Bob",1000.00f);
  Transact(&acc03);
  Savings acc02("Bob",1000.00f, 0.01f);
  Transact(&acc02);
  Account *acc04 = new Checking {"John", 10.};
  delete acc04;
  return 0;
}

