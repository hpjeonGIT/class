## Structural Design Patterns in Modern C++
- Instructor: Umar Lone

## Section 2: Adapter

10. Introduction
- Using a component written by someone else
- Purpose
  - Converts the interface of a class into another interface clients expect. Adapter lets classes work together that couldn't otherwise because of incompatible interfaces
  - Briefly, interface conversion
- Implementation
  - Composition
    - Adapter composes the adaptee and calls its method through a reference/pointer indirection
  - Or inheritance
    - Adapter inherits from the adaptee and calls the methods directly

11. Basic Example
- Composition style
```cpp
//header
#pragma once
class Target {
public:
  virtual void Request() = 0;
  virtual ~Target() = default;
};
class Adaptee {
public:
  void SpecificRequest();
};
class Adapter: public Target {
  Adaptee m_Adaptee;
public:
  void Request() override;
};
//cpp
#include "basic.h"
#include <iostream>
void Client(Target *pTarget) {
  pTarget -> Request();
}
void Adaptee::SpecificRequest(){
  std::cout << "[Adaptee] SpecificRequest\n"  ;
}
void Adapter::Request() {
  std::cout << "[Adapter] Calling SpeficRequest\n" ;
  m_Adaptee.SpecificRequest();
}
int main() {
  Adapter a;
  Client(&a);
}
```
  - Note that Adapter calls m_Adaptee.SpecificRequest()
- Inheritance style
```cpp
// header
#pragma once
class Target {
public:
  virtual void Request() = 0;
  virtual ~Target() = default;
};
class Adaptee {
public:
  void SpecificRequest();
};
class Adapter: public Target, public Adaptee {
  Adaptee m_Adaptee;
public:
  void Request() override;
};
// cpp
#include "basic.h"
#include <iostream>
void Client(Target *pTarget) {
  pTarget -> Request();
}
void Adaptee::SpecificRequest(){
  std::cout << "[Adaptee] SpecificRequest\n"  ;
}
void Adapter::Request() {
  std::cout << "[Adapter] Calling SpeficRequest\n" ;
  SpecificRequest();
}
int main() {
  Adapter a;
  Client(&a);
}
```
- Note that Adapter calls SpecificRequest() of Adaptee as it inherits

12. Game Input
- flight.h
```cpp
#pragma once
#include <random>
class Input {
public:
  virtual bool Up() = 0;
  virtual bool Down() = 0;
  virtual bool Left() = 0;
  virtual bool Right() = 0;
  virtual ~Input() = default;
};
class Keyboard: public Input {
  std::default_random_engine m_Engine{12345};
  bool SimulationInput();
public:
  bool Up() override;
  bool Down() override;
  bool Left() override;
  bool Right() override;
};
```
- flight.cpp
```cpp
#include "flight.h"
#include <thread>
#include <iostream>
#include <chrono>
using namespace std::chrono_literals;
bool Keyboard::SimulationInput() {
  std::bernoulli_distribution dist {.5};
  return dist(m_Engine);
}
bool Keyboard::Up() { return SimulationInput(); }
bool Keyboard::Down() { return SimulationInput(); }
bool Keyboard::Left() { return SimulationInput(); }
bool Keyboard::Right() { return SimulationInput(); }
void Game(Input *pInput) {
  int count{5};
  while(count != 0) {
    std::cout  << "================\n";
    if (pInput->Up()) { std::cout << "Pitch up\n"; }
    else if (pInput->Down()) { std::cout << "Pitch down\n"; }
    else { std::cout << "Plane is level\n"; }
    if (pInput->Left()) { std::cout << "Plane is turning left\n"; }
    else if (pInput->Right()) { std::cout << "Plane is turning right\n"; }
    else { std::cout << "Plane is flying straight\n"; }
    std::cout << std::endl;
    std::this_thread::sleep_for(1s);
    --count;
  }
}
int main() {
  Keyboard k;
  Game(&k);
}
```

13. Using Adapter
- How to map Keyboard in the above code to Accelerometer class?
  - Game() : client
  - Accelerometer: Adaptee
- Accel.h
```cpp
#pragma once
class Accelerometer {
public:
  double GetHorizontalAxis();
  double GetVerticalAxis();
}
```
- Accel.cpp
```cpp
#include "Accel.h"
double Accelerometer::GetHorizontalAxis() {
  std::uniform_int_distribution<> dist{-10,10};
  return dist(m_Engine);
}
double Accelerometer::GetVerticalAxis() {
  std::uniform_int_distribution<> dist{-10,10};
  return dist(m_Engine);
}
```

14. Adapter Implementation
- How to convert GetHorizontal/Vertical to up/down/left/right?
- We try compostion style here
- We are not mapping Accelerometer to Keyboard, but to Input
- adapter.h
```cpp
#pragma once
#include <random>
class Input {
public:
  virtual bool Up() = 0;
  virtual bool Down() = 0;
  virtual bool Left() = 0;
  virtual bool Right() = 0;
  virtual ~Input() = default;
};
class Keyboard: public Input {
  std::default_random_engine m_Engine{12345};
  bool SimulationInput();
public:
  bool Up() override;
  bool Down() override;
  bool Left() override;
  bool Right() override;
};
class Accelerometer {
  std::default_random_engine m_Engine{12345};  
public:
  double GetHorizontalAxis();
  double GetVerticalAxis();
};
class AccelAdapter: public Input {
  Accelerometer m_Accel;
public:
  bool Up() override;
  bool Down() override;
  bool Left() override;
  bool Right() override;
};
```
- adapter.cpp
```cpp
#include "adapter.h"
#include <thread>
#include <iostream>
#include <chrono>
using namespace std::chrono_literals;
bool Keyboard::SimulationInput() {
  std::bernoulli_distribution dist {.5};
  return dist(m_Engine);
}
bool Keyboard::Up() { return SimulationInput(); }
bool Keyboard::Down() { return SimulationInput(); }
bool Keyboard::Left() { return SimulationInput(); }
bool Keyboard::Right() { return SimulationInput(); }
double Accelerometer::GetHorizontalAxis() {
  std::uniform_int_distribution<> dist{-10,10};
  return dist(m_Engine);
}
double Accelerometer::GetVerticalAxis() {
  std::uniform_int_distribution<> dist{-10,10};
  return dist(m_Engine);
}
bool AccelAdapter::Up() {return m_Accel.GetVerticalAxis() > 0;}
bool AccelAdapter::Down() {return m_Accel.GetVerticalAxis() < 0;}
bool AccelAdapter::Left() {return m_Accel.GetHorizontalAxis() <0;}
bool AccelAdapter::Right() {return m_Accel.GetHorizontalAxis() >0;}
void Game(Input *pInput) {
  int count{5};
  while(count != 0) {
    std::cout  << "================\n";
    if (pInput->Up()) { std::cout << "Pitch up\n"; }
    else if (pInput->Down()) { std::cout << "Pitch down\n"; }
    else { std::cout << "Plane is level\n"; }
    if (pInput->Left()) { std::cout << "Plane is turning left\n"; }
    else if (pInput->Right()) { std::cout << "Plane is turning right\n"; }
    else { std::cout << "Plane is flying straight\n"; }
    std::cout << std::endl;
    std::this_thread::sleep_for(1s);
    --count;
  }
}
int main() {
  //Keyboard k;
  AccelAdapter k;
  Game(&k);
}
```

15. Class Adapter
- Using inheritance style
  - We override the functions of the adaptee
- adapter.h
```cpp
#pragma once
#include <random>
class Input {
public:
  virtual bool Up() = 0;
  virtual bool Down() = 0;
  virtual bool Left() = 0;
  virtual bool Right() = 0;
  virtual ~Input() = default;
};
class Keyboard: public Input {
  std::default_random_engine m_Engine{12345};
  bool SimulationInput();
public:
  bool Up() override;
  bool Down() override;
  bool Left() override;
  bool Right() override;
};
class Accelerometer {
  std::default_random_engine m_Engine{12345};  
public:
  double GetHorizontalAxis();
  double GetVerticalAxis();
};
class AccelAdapter: public Input, private Accelerometer {
  Accelerometer m_Accel;
public:
  bool Up() override;
  bool Down() override;
  bool Left() override;
  bool Right() override;
};
```
  - Note that when Accelerometer is inherited as private, the method of Adaptee will not be exposed
- adapter.cpp
```cpp
#include "adapter.h"
#include <thread>
#include <iostream>
#include <chrono>
using namespace std::chrono_literals;
bool Keyboard::SimulationInput() {
  std::bernoulli_distribution dist {.5};
  return dist(m_Engine);
}
bool Keyboard::Up() { return SimulationInput(); }
bool Keyboard::Down() { return SimulationInput(); }
bool Keyboard::Left() { return SimulationInput(); }
bool Keyboard::Right() { return SimulationInput(); }
double Accelerometer::GetHorizontalAxis() {
  std::uniform_int_distribution<> dist{-10,10};
  return dist(m_Engine);
}
double Accelerometer::GetVerticalAxis() {
  std::uniform_int_distribution<> dist{-10,10};
  return dist(m_Engine);
}
bool AccelAdapter::Up() {return GetVerticalAxis() > 0;}
bool AccelAdapter::Down() {return GetVerticalAxis() < 0;}
bool AccelAdapter::Left() {return GetHorizontalAxis() <0;}
bool AccelAdapter::Right() {return GetHorizontalAxis() >0;}
void Game(Input *pInput) {
  int count{5};
  while(count != 0) {
    std::cout  << "================\n";
    if (pInput->Up()) { std::cout << "Pitch up\n"; }
    else if (pInput->Down()) { std::cout << "Pitch down\n"; }
    else { std::cout << "Plane is level\n"; }
    if (pInput->Left()) { std::cout << "Plane is turning left\n"; }
    else if (pInput->Right()) { std::cout << "Plane is turning right\n"; }
    else { std::cout << "Plane is flying straight\n"; }
    std::cout << std::endl;
    std::this_thread::sleep_for(1s);
    --count;
  }
}
int main() {
  //Keyboard k;
  AccelAdapter k;
  Game(&k);
}
```

16. Pros & Cons
- Pros
  - One adapter can work with multiple classes
    - Even with subclasses of adaptee
  - Can always adapt to an existing class
- Cons
  - Cannot override adaptee's behavior
  - Methods are invoked through pointer indirection
- Class adapter
  - Pros
    - Method calls are direct as they are inherited
      - No pointer indirection
    - Can override adaptee's behavior
  - Cons
    - Won't work if the adaptee is final or sealed
    - Won't work with subclasses of adaptees
- Use when
  - You want to use an existing class, and it has an incompatible interface
  - You need to use classes from an existing hierarchy, but they have incompatible interface
  - You need to reuse an existing class with incompatible interface, but want to modify some behavior

## Section 3: Facade

18. Introduction
- A layer of the system
- Facilitates many/complex interface
- Advantage
  - Client is not tightly coupled with components
  - Complex interface is converted into simple interface
- Purpose
  - Provides an unified interface to a set of interfaces in a subsystem. Facade defines a higher-level interface that makes the subsystem easier to use
- Implementation
  - A function or a class or a set of classes
  - Multiple facades can be created for a system
  - Each facade may provide a simplified interface for a particular functionality
  - Such facades can inherit a common base class, that may be abstract
  - This can further reduce the coupling b/w the client and the classes in the system
  - Can be a singleton
  - Doesn't encapsulate the classes of the system
  - May hide the implementation classes to reduce coupling
  - Transparaent facade: Allows direct access to the underlying classes
  - Opaque facade: hides the underlying classes
- Briefly, facades pass the client request to the underlying classes and perform additional processing

19. Basic Example
- Converts complex interface into a simpler one
- basic.h
```cpp
#pragma once
#include <iostream>
#include <memory>
class A {
public:
  void CallA() { std::cout<<"Called A\n";}
};
class B {
public:
  void CallB() {std::cout<<"Called B\n";}
};
class C {
public:
  void CallC() {std::cout<<"Called C\n";}
};
class Client {
  std::shared_ptr<A> m_pA;
  std::shared_ptr<B> m_pB;
  std::shared_ptr<C> m_pC;
  
public:
  Client();
  ~Client() = default;
  void Invoke();
};
```
- basic.cpp
```cpp
#include "basic.h"
Client::Client() {
  m_pA = std::make_shared<A>();
  m_pB = std::make_shared<B>();
  m_pC = std::make_shared<C>();
}
void Client::Invoke() {
  m_pA->CallA();
  m_pB->CallB();
  m_pC->CallC();
}
int main() {
  Client c;
  c.Invoke();
}
```
- Let's add a class of Facade
- basic_facade.h
```cpp
#pragma once
#include <iostream>
#include <memory>
class A {
public:
  void CallA() { std::cout<<"Called A\n";}
};
class B {
public:
  void CallB() {std::cout<<"Called B\n";}
};
class C {
public:
  void CallC() {std::cout<<"Called C\n";}
};
class Facade {
  std::shared_ptr<A> m_pA;
  std::shared_ptr<B> m_pB;
  std::shared_ptr<C> m_pC;
public:
  Facade();
  ~Facade()=default;
  void Use();
};
class Client {
  std::shared_ptr<Facade> m_pF;
public:
  Client();
  ~Client() = default;
  void Invoke();
};
```
- basic_facade.cpp
```cpp
#include "basic_facade.h"
Client::Client() {
  m_pF = std::make_shared<Facade>();  
}
void Client::Invoke() {
  m_pF->Use();  
}
Facade::Facade() {
  m_pA = std::make_shared<A>();
  m_pB = std::make_shared<B>();
  m_pC = std::make_shared<C>();
}
void Facade::Use() {
  m_pA->CallA();
  m_pB->CallB();
  m_pC->CallC();
}
int main() {
  Client c;
  c.Invoke();
}
```
  - Client will have an attribute of Facade object
  - Facade object encapsulates the complex interface making/calling A, B, C

20. Console Project - I
- WIN32 API

21. Console Project - II
- How to port the code b/w Windows and Linux?

22. Console Project - III

23. Console Project - IV

24. Console Facade For Linux
- colorfont.h
```cpp
#pragma once
#include <string>
enum class Color{RED, GREEN, BLUE, WHITE };
class Console{
  Console() = default;
  static void SetColor(Color color);
public:
  static void Write(const std::string &text, Color color = Color::WHITE);
  static void WriteLine(const std::string &text, Color color = Color::WHITE);
};
```
- colorfont.cpp
```cpp
#include "colorfont.h"
#include <iostream>
void Console::SetColor(Color color) {
  switch (color) {
    case Color::RED:
      std::cout << "\033[31m";
      break;
    case Color::GREEN:
      std::cout << "\033[32m";
      break;
    case Color::BLUE:
      std::cout << "\033[34m";
      break;
    case Color::WHITE:  
      std::cout << "\033[00m";
      break;
  }
}
void Console::Write(const std::string &text, Color color) {
  SetColor(color);
  std::cout << text;
  SetColor(Color::WHITE);
}
void Console::WriteLine(const std::string &text, Color color) {
  Write(text+'\n', color);
}
int main() {
  Console::WriteLine("Hello World", Color::RED);
  Console::Write("Different color", Color::GREEN);
  return 0;
}
```
- Console class will take care of color setting and writing
  - WriteLine() and Write() functions are provided as static or monostate
  - When OS changes, remap Console class only. main() is not changed

25. Pros & Cons
- Pros
  - Facade isolates the clients from components with complex interface
  - Reduces the number of objects the clients interact with
  - Leads to weak coupling
  - Underlying components can change without impacting the client
  - Reduces compilation dependencies in large systems
  - Facades do not hide the underlying classes of the subsystem
  - Clients can still use the low-elver classes if necessary
- Cons
  - Overuse leads to too many layers
  - Performance of system may degrade
- When to use
  - You want to provide a simple interface to a complex system
    - This could be a default view for most clients
    - Other clients that need customization can use the underlying classes directly
  - A system has evolved and gets more complex
    - Early users might want to retain their views of the system
  - Your applicatin depends on low-level OS APIs
    - You want to avoid coupling with a specific OS
    - You want to provide an OO wrapper
  - Team members with different leve of experience use the system
  - Too many dependencies b/w clients and the imlementation classes of a subsystem

## Section 4: Proxy

27. Introduciton
- An alias
- When you cannot modify the original object
- The interface of proxy is same to the original object
- Purpose
  - Provides a surrogate or placeholder for another object to control access to it
- Implementation
  - Proxy should have the same interface as that of the real object
  - This is important because the client should not distinguish b/w the real object and the proxy
  - One way to achieve this is to inherit the proxy from the same class that the real object inherits from
  - This allows us to replace an object with proxy without significant changes
- Proxy types
  - virtual: creates expensive objects on demand
  - cache: caches expensive calls
  - remote: simplifies client implementation
  - protection: provides access management
  - smart: performs additional actions

28. Basic Example
- basic.h
```cpp
#pragma once
#include <memory>
class Subject {
public:
  virtual void Request() = 0;
  virtual ~Subject() = default;
};
class RealSubject: public Subject { 
public:
  void Request();
};
class Proxy : public Subject{
  std::shared_ptr<RealSubject> m_pSubject {}; //default is nullptr
public:
  void Request() override;
  Proxy() = default;
  ~Proxy() = default;
};
```
- basic.cxx
```cpp
#include "basic.h"
#include <iostream>
void RealSubject::Request() {
  std::cout << "[RealSubject] Request processed\n";
}
void Proxy::Request() {
  if (m_pSubject == nullptr) {
    std::cout << "[Proxy] Creating RealSubject\n";
    m_pSubject = std::make_shared<RealSubject>();
  }
  std::cout << "[Proxy] Additional behavior\n";
  m_pSubject -> Request();
}
void Operate (std::shared_ptr<Subject> s) {
  s->Request();
}
int main() {
  //auto sub = std::make_shared<RealSubject>();
  auto sub = std::make_shared<Proxy>(); // will do same as above
  Operate(sub);
  return 0;
}
```
- Proxy class is a derived class from Subject, like RealSubject
- It can do same Request() but can do extra process
- Note that using <RealSubject> or <Proxy> yields the same result

29. Virtual Proxy - I
- Creates objects on demand

30. Virtual Proxy - II
- virtProxy.h
```cpp
#pragma once
#include <string>
class Image {
  std::string m_FileName;
protected:
  void SetFileName(const std::string & fileName);
public:
  Image() = default;
  Image(const std::string & fileName);
  const std::string& GetFileName() const;
  virtual ~Image() = default;
  virtual void Display() = 0;
  virtual void Load() = 0;
  virtual void Load(const std::string & fileName) = 0;
};
class Bitmap: public Image {
  std::string m_Buffer{};
public:
  using Image::Image; // c++11 feature. Automatic constructor
  void Display() override;
  void Load() override;
  void Load(const std::string & fileName) override;
};
```
- virtProxy.cxx
```cpp
#include "virtProxy.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
void Image::SetFileName(const std::string& fileName) {
  m_FileName = fileName;
}
Image::Image(const std::string& fileName): m_FileName{fileName} {}
const std::string& Image::GetFileName() const { return m_FileName; }
void Bitmap::Display() { std::cout << m_Buffer;}
void Bitmap::Load() {
  m_Buffer.clear();
  std::ifstream file {GetFileName()};
  if (!file) throw std::runtime_error{"Failed to open file"};
  std::string line{};
  std::cout << "Loading bitmap[";
  using namespace std::chrono_literals;
  while(std::getline(file,line)) {
    m_Buffer += line + '\n';
    std::this_thread::sleep_for(100ms);
    std::cout << '.' << std::flush;
  }
  std::cout << "] Done!\n";
}
void Bitmap::Load(const std::string& fileName) {
  SetFileName(fileName);
  Load();
}
int main() {
  std::shared_ptr<Image> p{new Bitmap {"Smiley.txt"}};
  p->Load();
  p->Display();
}
```
- Issues: the current Bitmap class will 1) read the file and 2) allocate memory for m_Buffer, regardless of using Display()
  - Let's use Proxy

31. Virtual Proxy - III
- Addtional header file
```cpp
class BitmapProxy: public Image {
  std::shared_ptr<Bitmap> m_pBitmap{};
  std::string m_FileName;
  bool m_IsLoaded{false};
public:
  BitmapProxy();
  BitmapProxy(const std::string& fileName);
  ~BitmapProxy() = default;
  void Display() override;
  void Load() override;
  void Load(const std::string & fileName) override;
};
```
- Updated cxx file
```cpp
BitmapProxy::BitmapProxy():BitmapProxy{""} {}
BitmapProxy::BitmapProxy(const std::string &fileName) {
  m_pBitmap = std::make_shared<Bitmap>(fileName);
}
void BitmapProxy::Display() {
  if (!m_IsLoaded) { 
    std::cout << "[Proxy] Loading Bitmap\n";
    if(m_FileName.empty()) {
      m_pBitmap->Load(); 
    } else {
      m_pBitmap->Load(m_FileName);
    }
  }
  m_pBitmap->Display();
}
void BitmapProxy::Load() {  m_FileName.clear();}
void BitmapProxy::Load(const std::string & fileName) { m_FileName = fileName; }
int main() {
  std::shared_ptr<Image> p {new BitmapProxy {"Smiley.txt"}};
  p->Load();
  //p->Display();
}
```
- Note that the constructor of BitmapProxy must prodce the object of m_pBitmap when constructed
- Note that BitmapProxy object p is made from the base class Image
- BitmapProxy adjusted the Load/Display() function, loading files at Display() only. Still the interfaces are same and the same main code runs but the file is loaded when Display() is rquired

32. Protection Proxy - I
- Restricts the access to the real object

33. Protection Proxy - II
- basic.h
```cpp
#pragma once
#include <string>
#include <memory>
#include <experimental/filesystem>
class Employee {
  std::string m_Name;
  std::string m_Role;
public:
  Employee(const std::string& name, const std::string &role);
  const std::string& GetName() const;
  const std::string& GetRole() const;
  std::string GetInfo() const;
};
class Storage {
public:
  virtual void CreateFile(const std::string &fileName) = 0;
  virtual void DeleteFile(const std::string &fileName) = 0;
  virtual void UpdateFile(const std::string &fileName) = 0;
  virtual void ViewFile(const std::string &fileName) = 0;
  virtual ~Storage() = default;
};
class Repository : public Storage {
  std::shared_ptr<Employee> m_pEmp;
  std::experimental::filesystem::path m_CurrentPath{};
public:
  std::shared_ptr<Employee> GetUser() const {
    return m_pEmp;
  }
  void SetEmployee(std::shared_ptr<Employee> p) {
    m_pEmp = p;
  }
  Repository(const std::string &repoPath);
  void CreateFile(const std::string& fileName) override;
  void DeleteFile(const std::string& fileName) override;
  void UpdateFile(const std::string& fileName) override;
  void ViewFile(const std::string& fileName) override;
};
```
- basic.cxx
```cpp
#include "basic.h"
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <fstream>
Employee::Employee(const std::string& name, 
                   const std::string &role) : m_Name{name}, m_Role{role} {
                      
}
const std::string& Employee::GetName() const {
  return m_Name;  
}
const std::string& Employee::GetRole() const {
  return m_Role;
}
std::string Employee::GetInfo() const {
  std::ostringstream out;
  out << '[' << m_Role << ']' << m_Name << ' ';
  return out.str();
}
Repository::Repository(const std::string& repoPath) : m_CurrentPath{repoPath} {

}
void Repository::CreateFile(const std::string& fileName) {
  auto path = m_CurrentPath;
  path /= fileName;
  std::ofstream out{path};
  if(!out.is_open()) {
    throw std::runtime_error{"Could not create file"};    
  }
  std::cout << GetUser()->GetInfo() << " is creating a file\n";
  std::string fileData;
  std::cout << "[Create] Enter data:";
  getline(std::cin, fileData);
  out << fileData;
  std::cout << "File created successfully!";
}
void Repository::DeleteFile(const std::string& fileName) {
  auto path = m_CurrentPath;
  path /= fileName;
  if (!exists(path)) {
    throw std::runtime_error("Path does not exist");
  }
  std::cout << GetUser()->GetInfo() << " is deleting a file\n";
  if (remove(path)) {
    std::cout << "File deleted successfully";
  }  
}
void Repository::UpdateFile(const std::string& fileName) {
  auto path = m_CurrentPath;
  path /= fileName;
  std::ofstream out{path, out.app};
  if(!out.is_open()) {
    throw std::runtime_error{"Could not open file"};    
  }
  std::cout << GetUser()->GetInfo() << " is updating a file\n";
  std::string fileData;
  std::cout << "[Update] Enter data:";
  getline(std::cin, fileData);
  out << "\n### UPDATE ####\n" << fileData;
  std::cout << "File updated successfully!";
}
void Repository::ViewFile(const std::string& fileName) {
  auto path = m_CurrentPath;
  path /= fileName;
  std::ifstream in{path};
  if(!in.is_open()) {
    throw std::runtime_error{"Could not open file"};    
  }
  std::cout << GetUser()->GetInfo() << " is viewing a file\n";
  std::string line;
  while(getline(in, line)) {
    std::cout << line << std::endl;
  }
}
int main() {
  try{
    Repository repo{R"(./)"};
    std::shared_ptr<Employee> e1 (new Employee {"Umar", "Progerammer"});
    std::shared_ptr<Employee> e2 (new Employee {"Ayamm", "Manager"});
    repo.SetEmployee(e1);
    repo.CreateFile("data.txt");
  } catch (std::exception &ex) {
    std::cout <<"Exception:" << ex.what() << std::endl;
  }
}
```
- Compile command: `g++ -std=c++17 chap32/basic.cxx  -lstdc++fs`
- In the above code, everyone can access the repo data. How can we chage the code so only the manager can access?

34. Protection Proxy - III
- Proxy header file
```cpp
class RepoProxy: public Storage {
  std::shared_ptr<Repository> m_pRepo;
  bool IsAuthorized() const;
  std::vector<std::string> m_AuthorizedRoles;
public:
  std::shared_ptr<Employee> GetUser() const;
  void SetEmployee(std::shared_ptr<Employee> emp);
  void SetAuthorizedRoles(std::initializer_list<std::string> authorizedRoles);
  RepoProxy(const std::string & path);
  void CreateFile(const std::string& fileName) override;
  void DeleteFile(const std::string& fileName) override;
  void UpdateFile(const std::string& fileName) override;
  void ViewFile(const std::string& fileName) override;
  ~RepoProxy() = default;
};
```
- Related source file
```cpp
bool RepoProxy::IsAuthorized() const{
  if(m_AuthorizedRoles.empty()) {
    throw std::runtime_error{"Authorized roles not set"};
  }
  return std::any_of(begin(m_AuthorizedRoles), end(m_AuthorizedRoles),
  [this](const std::string &role) {
    return GetUser()->GetRole() == role;
  });
}
std::shared_ptr<Employee> RepoProxy::GetUser() const {
  return m_pRepo->GetUser();
}
void RepoProxy::SetEmployee(std::shared_ptr<Employee> emp) {
  m_pRepo->SetEmployee(emp);
}
void RepoProxy::SetAuthorizedRoles(std::initializer_list<std::string> authorizedRoles) {
  m_AuthorizedRoles.assign(authorizedRoles);
}
RepoProxy::RepoProxy(const std::string& path):
  m_pRepo{std::make_shared<Repository>(path)} {}
void RepoProxy::CreateFile(const std::string& fileName) {
  if(IsAuthorized()) {
    m_pRepo->CreateFile(fileName);
  } else {
    std::cout << GetUser()->GetInfo() << " is not auhorized to create a file\n";
  }
}
void RepoProxy::DeleteFile(const std::string& fileName) {
    if(IsAuthorized()) {
    m_pRepo->DeleteFile(fileName);
  } else {
    std::cout << GetUser()->GetInfo() << " is not auhorized to create a file\n";
  }
}
void RepoProxy::UpdateFile(const std::string& fileName) {  
  if(IsAuthorized()) {
    m_pRepo->UpdateFile(fileName);
  } else {
    std::cout << GetUser()->GetInfo() << " is not auhorized to create a file\n";
  }
}
void RepoProxy::ViewFile(const std::string& fileName) {
    if(IsAuthorized()) {
    m_pRepo->ViewFile(fileName);
  } else {
    std::cout << GetUser()->GetInfo() << " is not auhorized to create a file\n";
  }
}
int main() {
  try{
    //Repository repo{R"(./)"};
    RepoProxy repo{R"(./)"};
    repo.SetAuthorizedRoles({"Manager","Tech Lead"});
    std::shared_ptr<Employee> e1 (new Employee {"Umar", "Progerammer"});
    std::shared_ptr<Employee> e2 (new Employee {"Ayamm", "Manager"});
    repo.SetEmployee(e1);
    //repo.CreateFile("data.txt");
    repo.SetEmployee(e2);
    repo.ViewFile("data.txt");
  } catch (std::exception &ex) {
    std::cout <<"Exception:" << ex.what() << std::endl;
  }
}
```
- Now only registered employees can view files
- Using smart pointer
  - https://stackoverflow.com/questions/11820981/stdshared-ptr-and-initializer-lists
  - `std::shared_ptr<Foo> p(new Foo('a', true, Blue));` or `auto p = std::make_shared<Foo>('a', true, Blue);`

35. Remote Proxy - I
36. Remote Proxy - II
37. Remote Proxy - III
38. Smart Proxy
39. Pros & Cons
