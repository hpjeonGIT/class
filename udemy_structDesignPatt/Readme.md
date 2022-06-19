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

27. Introduction
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
- Remote proxy will connect to a class located in a separate server
  - RPC api is necessary
  - stub is the gate of SERVER

36. Remote Proxy - II
- The example uses WIN32 api

37. Remote Proxy - III

38. Smart Proxy
- Proxy with automatic deallocation like smart pointer
- Using virtualProxy above
- Extra header component
```cpp
template<typename T>
class Pointer {
  T * m_ptr;
public:
  Pointer(T* ptr) : m_ptr{ptr} {}
  ~Pointer() { delete m_ptr;}
  T *Get() { return m_ptr; }
};
```cpp
- main source
int main() {
  Pointer<Image> p = new Bitmap{"Smiley.txt"};
  p.Get()->Load();
  p.Get()->Display();
}
```
- When p is destructed, the object is deleted and memory leak is avoided
- As p is an object, not a pointer, needs Get() function. Or overload operator of `->`

39. Pros & Cons
- Pros
  - Creates a layer of indirection
  - Can hide the location of the real subject
  - Allows restricted access to the real subject
  - Provides matching interface
- Cons
  - Tight coupling b/w the proxy and the real subject
  - Adds only one new behavior
- When to use
  - when it is not feasible to use the real object directly

## Section 5: Decorator

41. Introduction
- Without modifying the object, we add new muliple behavior
- Purpose
  - Attach additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality
  - Additional features are added to an object, not a class
- Implementation
  - A decorator's interface must conform to the interface of the object it decorates
    - Must inherit from common base class
  - Abstract decorator is not recquired if only one responsibility has to be added
    - The decorator can itself forward the calls to the object
  - Abstract decorator & its base class should be lightweight
    - Should focus on defining interface, not storing data
    - Avoid adding concrete implementation as not all subclasses may require it
    - These objects are part of every decorator and will make the decorators bulky to use
  - Decorator changes the skin; alternative is to change the guts (Strategy Pattern)

42. Basic Example - I
- base.h
```cpp
#pragma once
class Component {
public:
  virtual void Operation() = 0;
  virtual ~Component() = default;
};
class ConcreteComponent: public Component {
public:
  void Operation() override;
};
class ConcreteDecoratorA: public Component {
    Component *m_ptr{};
public:
  ConcreteDecoratorA(Component * ptr) : m_ptr{ptr} {}
  void Operation() override;
};
class ConcreteDecoratorB: public Component {
    Component *m_ptr{};
public:
  ConcreteDecoratorB(Component * ptr) : m_ptr{ptr} {}
  void Operation() override;
};
```
- base.cxx
```cxx
#include "base.h"
#include <iostream>
void ConcreteComponent::Operation() {
  std::cout <<"[ConcreteComponent] Operation invoked\n";
}
void ConcreteDecoratorA::Operation() {
  std::cout << "[ConcreteDecoratorA] Operation invoked\n";
  m_ptr->Operation();
}
void ConcreteDecoratorB::Operation() {
  std::cout << "[ConcreteDecoratorB] Operation invoked\n";
  m_ptr->Operation();
}
int main() {
  ConcreteComponent component{};
  ConcreteDecoratorA decA{&component} ;
  ConcreteDecoratorB decB{&decA};
  decB.Operation();
}
```
- When runs, ConcreteaDecoratorB->ConcreteDecoratorA->ConcreteComponent are invoked sequentially
- Basically decorators are wrappers
- Similar to proxy but it adds more responsibility instead of facilitating controlled use/access 

43. Basic Example - II
- We may make a Decorator class to inherit Component class
- base.h
```cpp
#pragma once
class Component {
public:
  virtual void Operation() = 0;
  virtual ~Component() = default;
};
class ConcreteComponent: public Component {
public:
  void Operation() override;
};
class Decorator: public Component {
  Component * m_ptr{};
public:
  Decorator(Component * ptr) : m_ptr {ptr} {}
  void Operation() override;
};
class ConcreteDecoratorA: public Decorator {
  using Decorator::Decorator;
public:
};
class ConcreteDecoratorB: public Decorator {
  using Decorator::Decorator;
public:
};
```
- base.cxx
```cxx
#include "base.h"
#include <iostream>
void ConcreteComponent::Operation() {
  std::cout <<"[ConcreteComponent] Operation invoked\n";
}
void Decorator::Operation() {
  m_ptr->Operation();
}
int main() {
  ConcreteComponent component{};
  ConcreteDecoratorA decA{&component} ;
  ConcreteDecoratorB decB{&decA};
  decB.Operation();
}
```

44. Streams - I
- Basic stream for input/output file stream
- base.h
```cxx
#pragma once
#include <string>
#include <fstream>
class InputStream {
public:
  virtual bool Read(std::string &text) = 0;
  virtual void Close() = 0;
  virtual ~InputStream() = default;
};
class FileInputStream: public InputStream {
  std::ifstream m_Reader;
public:
FileInputStream() = default;
FileInputStream(const std::string& fileName);
  bool Read(std::string &text) override;
  void Close() override;
};
class OutputStream {
public:
  virtual void Write(const std::string &text) = 0;
  virtual void Close() = 0;
  virtual ~OutputStream() = default;
};
class FileOutputStream: public OutputStream {
  std::ofstream m_Writer{};
public:
  FileOutputStream() = default;
  FileOutputStream(const std::string& fileName);
  void Write(const std::string& text) override;
  void Close() override;
};
```
- base.cxx
```cxx
#include "base.h"
#include <iostream>
FileInputStream::FileInputStream(const std::string& fileName) {
  m_Reader.open(fileName);
  if (!m_Reader) {
    throw std::runtime_error {"Could not open the file for reading"};
  }
}
bool FileInputStream::Read(std::string& text) {
  text.clear();
  std::getline(m_Reader,text);
  return !text.empty();
}
void FileInputStream::Close() {
  if(m_Reader.is_open()) {
    m_Reader.close();
  }
}
FileOutputStream::FileOutputStream(const std::string& fileName){
  m_Writer.open(fileName);
  if(!m_Writer) {
    throw std::runtime_error{"Could not open file for writing"};
  }
}
void FileOutputStream::Write(const std::string& text) {
  m_Writer << text;
}
void FileOutputStream::Close() {
  if (m_Writer.is_open()) {
    m_Writer.close();
  }
}
void Read() {
  FileInputStream Input{"test.txt"};
  std::string text{};
  while(Input.Read(text)) {
    std::cout << text << std::endl;
  }
}
void Write() {
  FileOutputStream output{"test.txt"};
  output.Write("First line\n");
  output.Write("Second line\n");
  output.Write("Third line\n");
}
int main() {
  Write();
  Read();
  return 0;
}
```

45. Streams - II
- Let's implement buffering
- How to override Read/Write without modifying the base class member functions?

46. Streams - III
- We pretend buffering, without real buffering
- buff.h
```cxx
#pragma once
#include <string>
#include <fstream>
class InputStream {
public:
  virtual bool Read(std::string &text) = 0;
  virtual void Close() = 0;
  virtual ~InputStream() = default;
};
class FileInputStream: public InputStream {
  std::ifstream m_Reader;
public:
FileInputStream() = default;
FileInputStream(const std::string& fileName);
  bool Read(std::string &text) override;
  void Close() override;
};
class OutputStream {
public:
  virtual void Write(const std::string &text) = 0;
  virtual void Close() = 0;
  virtual ~OutputStream() = default;
};
class FileOutputStream: public OutputStream {
  std::ofstream m_Writer{};
public:
  FileOutputStream() = default;
  FileOutputStream(const std::string& fileName);
  void Write(const std::string& text) override;
  void Close() override;
};
class BufferedOutputStream: public FileOutputStream {
  char m_Buff[512]{};
  using FileOutputStream::FileOutputStream;
public:
  void Write(const std::string& text) override;
  void Close() override;
};
class BufferedInputStream: public FileInputStream {
  using FileInputStream::FileInputStream;
  char m_Buff[512]{};
public:
  bool Read(std::string& text) override;
  void Close() override;
};
```
- buff.cxx
```cxx
#include "buff.h"
#include <iostream>
FileInputStream::FileInputStream(const std::string& fileName) {
  m_Reader.open(fileName);
  if (!m_Reader) {
    throw std::runtime_error {"Could not open the file for reading"};
  }
}
bool FileInputStream::Read(std::string& text) {
  text.clear();
  std::getline(m_Reader,text);
  return !text.empty();
}
void FileInputStream::Close() {
  if(m_Reader.is_open()) {
    m_Reader.close();
  }
}
FileOutputStream::FileOutputStream(const std::string& fileName){
  m_Writer.open(fileName);
  if(!m_Writer) {
    throw std::runtime_error{"Could not open file for writing"};
  }
}
void FileOutputStream::Write(const std::string& text) {
  m_Writer << text << '\n';
}
void FileOutputStream::Close() {
  if (m_Writer.is_open()) {
    m_Writer.close();
  }
}
void Read() {
  BufferedInputStream Input{"test.txt"};
  std::string text{};
  while(Input.Read(text)) {
    std::cout << text << std::endl;
  }
}
void Write() {
  BufferedOutputStream output{"test.txt"};
  output.Write("First line");
  output.Write("Second line");
  output.Write("Third line");
}
void BufferedOutputStream::Write(const std::string& text) {
  std::cout << "Buffered Write\n";
  FileOutputStream::Write(text);
}
void BufferedOutputStream::Close() {
  FileOutputStream::Close();
}
bool BufferedInputStream::Read(std::string& text) {
  std::cout << "Buffered Read\n";
  auto result = FileInputStream::Read(text);
  return result; // bool type
}
void BufferedInputStream::Close() {
  FileInputStream::Close();
}
int main() {
  Write();
  Read();
  return 0;
}
```

47. Streams - IV
- Adding encryption/decryption

48. Streams - V
- Mixing encryption + buffering
- When multiple layers (encryption, compression, network, ...) are required
  - Inherting multiple classses will not be a good solution
  - Instead of inheriting, allocate an attribute using the base class

49. Streams - VI

50. Decorator Types
- Dynamic decorator
  - Behavior is added at runtime
- Static decorator
  - Decorator is chosen at compile-time
  - Cannot be changed at runtime
  - Can use the mixin clss
    - Mixin
      - Not from a base class but a class may use method from other classes
      - Not for C++/C#/java
- Functional decorator
  - Decorates a function instead of an object

51. Static decorator

52. Functional Decorator
- Accepts functions and callable objects as arguments
```cxx
#include <iostream>
int Square(int x) { return x*x; }
int Add(int x, int y) { return x+y; }
template<typename T>
auto PrintResult(T func) {
  return [func](auto&&...args) {
    std::cout << "Result is: ";
    return func(args...);
  };
}
int main() {
  auto result = PrintResult(Square); //return value is a lamba expression
  std::cout << result(5) << std::endl;
  auto result2 = PrintResult(Add); //return value is a lamba expression
  std::cout << result2(5,1) << std::endl;
  return 0;
}
```

53. Pros & Cons
- Pros
  - Flexible way of adding responsibilities to an object rather than inheritance
    - Uses Composition
    - Dynamic unlike inheritance
  - Features are added incrementally as the code progresses
  - You pay for the features only when you use
  - Easy to add a combination of capabilities
    - Same capability can be added twice
  - Components don't have to know about their decorators
- Cons
  - A decorated component is not identical to the component itself
  - Lots of small objects are created
    - Can make code messed up
- When to use
  - Responsibilities are added transparently and dynamicllay
  - Supports combination of behaviors
  - When legacy system needs new behavior
- Proxy vs decorator
  - Proxy allows one wrapper only
    - One behavior only
    - Tight coupling b/w classes
    - Adds restrictions
    - Compile-time
  - Decorator wraps multiple layer
    - Multiple behavior
    - Loose coupling b/w classes
    - Adds new behavior
    - Runtime

## Section 6: Composite

55. Introduction
- UI, drawing, ...
- Grouping in ppt. scaling up/down affects all boxes in the group

56. Composite intent and implementation overview
- Intent: Composes objects into tree structures to represent part-whole hierarchies. Composite lets clients treat individual objects and compositions of objects uniformly
- Implementation
  - The leaf and composite should be treated uniformly by the client
  - The Component class should define as many common operations possible for all subclasses
    - Clietns should not have to differentiate b/w leaf & composite
  - Type of data structure to use for storage inside composite depends on required efficiency

57. Basic Example
- base.h
```cxx
#pragma once
#include <vector>
#include <iostream>
#include <algorithm>
class Component {
public:
  virtual void Operation() = 0;
  virtual void Add(Component *pComponent) = 0;
  virtual void Remove(Component *pComponent) = 0;
  virtual Component * GetChild(int index) = 0;
  virtual ~Component() = default;
};
class Leaf: public Component {
public:
  void Add(Component* pComponent) override;
  Component* GetChild(int index) override;
  void Operation() override;
  void Remove(Component* pComponent) override;
};
class Composite : public Component {
  std::vector<Component*> m_Children {};
public:
  void Add(Component* pComponent) override;
  Component* GetChild(int index) override;
  void Operation() override;
  void Remove(Component* pComponent) override;
};
```
- base.cxx
```cxx
#include "base.h"
int depth{};
void Leaf::Add(Component* pComponent) {}
Component* Leaf::GetChild(int index) {return nullptr;}
void Leaf::Operation() {
  std::cout << "[Leaf] Operation\n";
}
void Leaf::Remove(Component* pComponent){}

void Composite::Add(Component* pComponent) {
  m_Children.push_back(pComponent);
}
Component* Composite::GetChild(int index) {
  return m_Children[index];
}
void Composite::Operation() {
  ++depth;
  std::cout << "[Composite] Operation\n";
  for(auto pChild: m_Children) {
    for(int i=0;i<depth; ++i) {std::cout << '\t';}
    std::cout << "|-";
    pChild->Operation();
  }
  --depth;
}
void Composite::Remove(Component* pComponent){
  auto newend = std::remove(m_Children.begin(), m_Children.end(), pComponent);
  m_Children.erase(newend, end(m_Children));
}
int main() {
  Leaf leaf1, leaf2, leaf3;
  Composite subroot;
  subroot.Add(&leaf3);
  Composite root;
  root.Add(&leaf1);
  root.Add(&leaf2);
  root.Add(&subroot);
  root.Operation();
  return 0;
}
```
- Results
```bash
$g++ -std=c++17 chap57/base.cxx 
$ ./a.out 
[Composite] Operation
	|-[Leaf] Operation
	|-[Leaf] Operation
	|-[Composite] Operation
		|-[Leaf] Operation
```  
- By running the root, the entire tree is executed
  - This is how graphic renderer draws

58. UI Example overview
- Win32 API

59. UI example - I

60. UI example - II

61. UI example - III

62. Pros & Cons

## Section 7: Bridge

65. Introduction
- An abstraction represents an idea of an entity with relevant details
- Irrelevant or unwanted details are ignored

66. Bridge intent and implementation Overview
- Abstraction and its implementation may vary. They are done in separate hierarchies
- Connection b/w abstraction and implementation hierarcy is called **bridge**
- The alternative name is handle-body
  - Handle: abstraction
  - Body: implementation
- Intent: decouples an abstraction from its implementation

67. Basic Example
- base.h
```cxx
#pragma once
class Implementor {
public:
  virtual void OperationImpl() = 0;
  virtual ~Implementor() = default;
};
class ConcreteImplementorA: public Implementor {
public:
  void OperationImpl() override;
};
class ConcreteImplementorB: public Implementor {
public:
  void OperationImpl() override;
};
class Abstraction {
protected:
  Implementor *m_pImplementor{};
public:
  explicit Abstraction(Implementor* m_p_implementor)
     : m_pImplementor(m_p_implementor) {}
  virtual void Operation() = 0;
  virtual ~Abstraction() = default;
};
class RefinedAbstraction: public Abstraction {
  using Abstraction::Abstraction;
public:
  void Operation() override;
};
```
- base.cxx
```cxx
#include "base.h"
#include <iostream>
void ConcreteImplementorA::OperationImpl() {
  std::cout << "[ConcreteImplementorA] Implmentation invoked\n";
}
void ConcreteImplementorB::OperationImpl() {
  std::cout << "[ConcreteImplementorB] Implmentation invoked\n";
}
void RefinedAbstraction::Operation() {
  std::cout << "[RefinedAbstraction] =>";
  m_pImplementor->OperationImpl();
}
int main() {
  ConcreteImplementorA impl;
  Abstraction *p = new RefinedAbstraction(&impl);
  p->Operation();
  return 0; 
}
```
- Abstraction is coupled with Implementor class, which is inherited to ConcreteImplementorA/B
- RefineAbstraction class can use the method of ConcreteImplementorA/B through Bridge design

68. Shapes Hierarchy - I
69. Shapes Hierarchy - II
70. Shapes Hierarchy - III
71. Shapes Hierarchy - IV
72. Shapes Hierarchy Issues
- Abstract: shape, line, circle, rectangle
- Implementation: GDI, DirectX, OpenGL
- Line method may use GDI or OpenGL through Bridge

73. Bridge Implementation
- When a new shape function is necessary, add more abstraction classes
- When new library is added, add more implementor classes

74. Handle-Body - I
- Handle-Body
  - A one to one relationship b/w Abstraction and its implementor is also called degenerate bridge
  - The alternative name for this pattern is Handle-Body
    - Abstraction -> Handle
    - Implementor -> Body
  - Aliases as Cheshire Cat, compilation firewall, Opaque pointer or D-Pointer
  - As known as Pointer to Implementation (PIMPL)
- PIMPL
  - Creates a wrapper around a class
  - Shields the clients from changes to the implementor
  - Prevents compilation changes from seeping into multiple source files
  - Hides the complex details of a class from the clients
  - Implements extra functionality for the implementation class

75. Handle-Body - II
76. Handle-Body - III
77. PIMPL - I
78. PIMPL - II
- More common in C++

79. Static Bridge
- Inherits abstract and implementor together

80. Pros & Cons

## Section 8: Flyweight

82. Introduction
- Can minimize memory usage
- Beneficial when we create large number of objects
- Instead of storing data individually in each object, it can be stored outside & shared b/w objects
- This will drastically reduce the overall memory requirement
- Intrinsic state
  - canbe shared b/w flyweights
- Extrinsic state
  - Not shareable
  - Some information is not part of the object
  - Computed outside of flyweight and passed to them when required

83. Intent & Implementation Overview
- Uses sharing to support large numbers of fine-grained objects efficiently
- Implementation
  - In most cases, the client does not create the flyweight itself
  - It is requested from a pool
  - Typically a factory that may use associative container to store the flyweights
  - A client requests a flyweight through its key
  - The pool will either create it with intrinsic state or supply an existing one
  - The extrinsic state should be computed separately
  - The interface of the flyweight does not enforce sharing
  - The pool can instantiate all flyweights and keep thme around permanently if their count is low
  - The flyweights are immutable, and their behavior depends on the extrinsic state

84. Basic Implementation
- base.h
```cxx
#pragma once
#include <unordered_map>
class Flyweight {
public:
  virtual void Operation(int extrinsic) = 0;
  virtual ~Flyweight() = default;
};
class ConcreteFlyweight: public Flyweight {
  int *m_pIntrinsicState{};
public:
  ConcreteFlyweight(int* mPIntrinsicState)
  : m_pIntrinsicState(mPIntrinsicState){}
  void Operation(int extrinsic) override;
};
class UnsharedConcreteFlyweight: public Flyweight {
  int m_InternalState{};
public:
  UnsharedConcreteFlyweight(int m_InternalState)
  : m_InternalState(m_InternalState){}
  void Operation(int extrinsic) override;
};
class FlyweightFactory {
  inline static std::unordered_map<int, Flyweight *> m_Flyweights{};
public:
  Flyweight* GetFlyweight(int key);
  Flyweight* GetUnsharedFlyweight(int value);
};
```
- base.cxx
```cxx
#include "base.h"
#include <iostream>
void ConcreteFlyweight::Operation(int extrinsic) {
  std::cout << "Intrinsic state:" << * m_pIntrinsicState << std::endl;
  std::cout << "Extrinsic state:" << extrinsic << std::endl; 
}
void UnsharedConcreteFlyweight::Operation(int extrinsic) {
  std::cout << "Internal state:" << m_InternalState << std::endl;
  std::cout << "Extrinsic state:" << extrinsic << std::endl;
}
Flyweight* FlyweightFactory::GetFlyweight(int key) {
  auto found = m_Flyweights.find(key) != end(m_Flyweights);
  if (found) {
    return m_Flyweights[key];    
  }
  static int intrinsicState{100};
  Flyweight *p = new ConcreteFlyweight{&intrinsicState};
  m_Flyweights[key] = p;
  return p;
}
Flyweight* FlyweightFactory::GetUnsharedFlyweight(int value) {
  return new UnsharedConcreteFlyweight{value};
}
int main() {
  int extrinsicState = 1;
  FlyweightFactory factory;
  auto f1 = factory.GetFlyweight(1);
  auto f2 = factory.GetFlyweight(1);
  auto f3 = factory.GetFlyweight(1);
  f1->Operation(extrinsicState++);
  f2->Operation(extrinsicState++);
  f3->Operation(extrinsicState++);
  return 0;
}
```
- Clients will use a factory function
- Intrinsic data is declared as static
  - Note that it is mutable, and cannot be modified

85. Game Implementation - I
- model.h
```cxx
#pragma once
#include <iostream>
#include <string_view>
#include <vector>
#include <memory>
struct Position3D {
  int x,y,z;
  friend std::ostream& operator<< (std::ostream&os, const Position3D& obj) {
    return os << "{" << obj.x <<","<<obj.y <<","<<obj.z <<")\n";
  }
};
class Model {
public:
  virtual void Render();
  virtual void Render(Position3D position);
};
class Vegetation: public Model {
  inline static int m_Count {};
  std::vector<int> m_MeshData{};
  const char *m_Texture{};
  std::string m_Tint{};
  Position3D m_Position{};
public:
  Vegetation(std::string_view tint, Position3D position);
  void Render() override;
  static void ShowCount();
};
```
- model.cxx
```cxx
#include "model.h"
void Model::Render() {}
void Model::Render(Position3D position){}
Vegetation::Vegetation(std::string_view tint, 
                       Position3D position) 
            : m_Tint{tint}, m_Position{position} {
  ++m_Count;
  m_MeshData.assign({5,1,2,8,2,9});
  m_Texture = R"(
    #
   ###
  #####
    #
    #
    #
)";
}
void Vegetation::Render() {
  std::cout << m_Texture;
  std::cout << "Mesh: " ;
  for (auto m: m_MeshData) {
    std::cout << m << " ";    
  }
  std::cout << "\nTint" << m_Tint << std::endl;
  std::cout << "Position: " << m_Position << std::endl;
}
//void Vegetation::Render(Position3D position){}
void Vegetation::ShowCount() {
  std::cout << "Total objects created: " << m_Count << std::endl;
}
int main() {
  std::vector<std::shared_ptr<Vegetation>> m_Trees{};
  for(int i=0;i<15;++i) {
    if (i<5) {
      m_Trees.push_back(std::make_shared<Vegetation>("Green", Position3D{i*10,i*10,i*10}));
    } else if (i>5 && i <= 10) {
      m_Trees.push_back(std::make_shared<Vegetation>("Dark Green", Position3D{i*10,i*10+10,i*10}));
    } else {
      m_Trees.push_back(std::make_shared<Vegetation>("Light Green", Position3D{i*10+10,i*10,i*10}));
    }
  }
  for (auto tree: m_Trees) {
    tree->Render();
  }
  Vegetation::ShowCount();
}
```
- Total 15 objects are made

86. Game Implementation - II
- model.h
```cxx
#pragma once
#include <iostream>
#include <string_view>
#include <vector>
#include <memory>
#include <unordered_map>
struct Position3D {
  int x,y,z;
  friend std::ostream& operator<< (std::ostream&os, const Position3D& obj) {
    return os << "{" << obj.x <<","<<obj.y <<","<<obj.z <<")\n";
  }
};
class Model {
public:
  virtual void Render();
  virtual void Render(Position3D position);
};
class VegetationData {
  std::vector<int> m_MeshData{};
  const char *m_Texture{};
public:
  VegetationData();
  const char *GetTexture() const;
  const std::vector<int> & GetMeshData() const;
};  
class Vegetation: public Model {
  inline static int m_Count {};
  VegetationData *m_pVegData{};
  std::string m_Tint{};
public:
  Vegetation(std::string_view tint, VegetationData *p);
  void Render(Position3D position) override;
  static void ShowCount();
};
using VegetationPtr = std::shared_ptr<Vegetation>;
class VegetationFactory {
  std::unordered_map<std::string_view, VegetationPtr> m_Flyweights{};
  VegetationData * m_pVegData{};
public:
  VegetationFactory(VegetationData* mPVegData) : m_pVegData{mPVegData}{}
  VegetationPtr GetVegetation(std::string_view tint);
};
```
- mode.cxx
```cxx
#include "model.h"
void Model::Render() {}
void Model::Render(Position3D position){}
VegetationData::VegetationData() {
m_MeshData.assign({5,1,2,8,2,9});
  m_Texture = R"(
    #
   ###
  #####
    #
    #
    #
)";
}
const char* VegetationData::GetTexture() const {
  return m_Texture;
}
const std::vector<int>& VegetationData::GetMeshData() const {
  return m_MeshData;
}
Vegetation::Vegetation(std::string_view tint, 
                       VegetationData *p) 
            : m_Tint{tint}, m_pVegData{p} {
  ++m_Count;
            }
void Vegetation::Render(Position3D position) {
  std::cout << m_pVegData->GetTexture();
  std::cout << "Mesh: " ;
  for (auto m: m_pVegData->GetMeshData()) {
    std::cout << m << " ";    
  }
  std::cout << "\nTint" << m_Tint << std::endl;
  std::cout << "Position: " << position << std::endl;
}
//void Vegetation::Render(Position3D position){}
void Vegetation::ShowCount() {
  std::cout << "Total objects created: " << m_Count << std::endl;
}
VegetationPtr VegetationFactory::GetVegetation(std::string_view tint) {
  auto found = m_Flyweights.find(tint) != end(m_Flyweights);
  if (!found) {
    m_Flyweights[tint] = std::make_shared<Vegetation>(tint, m_pVegData);
  }
  return m_Flyweights[tint];
}
int main() {
  std::vector<VegetationPtr> m_Trees{};
  VegetationData data{};
  VegetationFactory factory{&data};
  for(int i=0;i<15;++i) {
    if (i<5) {
      m_Trees.push_back(factory.GetVegetation("Green"));
      m_Trees[i]->Render({i*10,i*10,i*10});
    } else if (i>5 && i <= 10) {
      m_Trees.push_back(factory.GetVegetation("Dark Green"));
      m_Trees[i]->Render({i*10,i*10+10,i*10});
    } else {
      m_Trees.push_back(factory.GetVegetation("Light Green"));
      m_Trees[i]->Render({i*10+10,i*10,i*10});
    }
  }
  Vegetation::ShowCount();
  return 0;
}
```
- We make Vegetation data as intrinsic
- Positions are extrinsinc
- Total 3 objects are created
- Needs a factory to make VegetationData object
- For string_view type, do not use `&`

87. Game Implementation - III

88. String Interning - I
89. String Interning - II
90. String Interning - III

91. Boost.Flyweight
92. Pors & Cons
