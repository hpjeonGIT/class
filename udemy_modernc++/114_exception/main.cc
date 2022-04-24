#include <iostream>
#include <memory>
class myObj {
public:
  myObj()  {std::cout << "created\n";}
  ~myObj() {std::cout << "removed\n";}
};
int Process(int16_t count) {
  myObj t1;
  myObj *t2 = new myObj;
  std::unique_ptr<myObj> t3 {new myObj};
  if (count <10 )
    throw std::out_of_range("too small");
  delete t2;
  return 0;    
}
int main() {
  try {
    int rv = Process(3);
  }
  catch (std::exception &ex) {
    std::cout << "error_message IS : " << ex.what() << std::endl;
  }
  return 0;
}