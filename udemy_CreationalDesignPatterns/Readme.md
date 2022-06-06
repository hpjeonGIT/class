## Creational Design Patterns in Modern C++
- Instructor: Umar Lone

## Section 1: Introduction

2. Introduction to Patterns
- Catalogue

|     |       | Creational     | Structural      | Behavioral      |
|-----|-------|----------------|-----------------|-----------------|
|     | Class | Factory Method | Adapter (class) | Interpreter     |
|     |       |                |                 | Template Method |
|     | Object| Abstract Factory | Adapter(object)| Chain of Responsibility |
|     |       | Builder          | Bridge         | Command |
|     |       | Prototype        | Composite      | Iterator |
|scope|       | Singleton        | Decorator      | Mediator |
|     |       |                  | Facade         | Memento |
|     |       |                  | Flyweight      | Observer |
|     |       |                  | Proxy          | State |
|     |       |                  |                | Strategy |
|     |       |                  |                | Visitor |

3. Overview of UML Class Diagram
- UML (Unified Modeling Language)
  - Class is depicted with a rectangle
    - name
    - attribute/field/memberdata
    - method
  - Inheritance
    - An arrow points the base class
  - Composition
    - A diamond shows the relation of container
      - When the main container is destructued, the parts of the container also are destructed
  - Aggregation
    - An empty diamond
      - Even the container is destructed, the part is not destructed
  - Association
    - plain line: Uses/controls
    - Arrow line: 
  - Note
    - Dotted lines

4. S.O.L.I.D principles - I
- Single Responsibility Principle
  - A class should have only one reason to change
  - Should have only one responsibility
  - Classes with multiple responsibilities break when changed
  - Put each responsibility in a separate class
  - Ex) A class of Note - add/remove method. Do we need visualize()? We may make another class of View to visualize Note objects
- Open Closed Principle
  - Modules should be open for extension but closed for modification
  - Ex) A class of Note - Add() needs to be updated when a symbol leads paragraph - Instead of modifying the existing Add(), we make a new class and overrides Add() with such a feature

5. SOLID principles - II
- Liskov Substitution Principle
  - Subtypes must be substitutable for their base types
  - Applies to inheritance relationship

6. SOLID principles - III
- Interface Segregation Principle
  - Clients should not be forced to depend on methods they do not use
  - An interface with too many methods will be complex to use (aka fat interface)
  - Separate the interface and put methods based on the client usage
  - Ex) Instead of having both of read()/write() in file Class, make an Input class with read() and a Writer class with write() method
- Dependency Inversion Principle
  - Abstract should not depend on details. Detailas should depend on abstractions
  - Abstraction means an interface and details mean classes

7. Creational Pattern Overview
- Singleton: Ensure only one instance
- Factory Method: Create instance without depending on its concrete type
- Object Pool: Reuse existing instances
- Abstract Factory: Create instances from a specific family
- Prototype: Clone existing objects from a prototype
- Builder: Construct a complex object step by step

## Section 2: Singleton

10. Introduction
- Ensure a class only has one instance, and provide a global point of access to it
  - Note that this is GLOBAL
- Implementation
  - The class is made responsible for its own instance
  - It intercepts the call for construction and returns a single instance
  - Same instance is returned everytime
  - Direct construciton of object is disabled
  - The class creates its own instance which is provided to the clients
- Singleton class from GoF
  - Methods
    - static instance()
    - SingletonOperation()
    - Get SingletonData()
  - Attributes
    - static uniqueInstance
    - singletonData
  - This UML doesn't show destructor

11. Basic Example
- Set constructor as private
- Singleton.h
```cpp
#pragma once
class Singleton{
  Singleton() = default;
  static Singleton m_Instance;
public:
  static Singleton &Instance();
  void MethodA();
};
```
- basic.cxx
```cpp
#include <iostream>
#include "Singleton.h"
Singleton Singleton::m_Instance;
Singleton& Singleton::Instance() { return m_Instance; }
void Singleton::MethodA() { }
int main() {
  Singleton &s = Singleton::Instance();
  s.MethodA();
  //Singleton s2; // not compiled
}
```

12. Logger Class - I
- How to prevent users from creating additional Logger?

13. Logger Class - II
- Make the constructor as private
- Make a static instance of m_Instance
- Add a method of Instance() to return m_Instance
  - Returns the same instance  
- Logger.h
```cpp
#pragma once
#include <cstdio>
#include <string>
class Logger {
  FILE *m_pStream;
  std::string m_Tag;
  Logger();  
  static Logger m_Instance;
public:
  Logger(const Logger&) = delete;
  static Logger& Instance();
  ~Logger();
  void WriteLog(const char *pMessage);
  void SetTag(const char *pTag);
};
```
- Logger.cxx
```cpp
#include "Logger.h"
#include <iostream>
Logger Logger::m_Instance;
Logger::Logger() { m_pStream  = fopen("applog.txt","w");}
Logger& Logger::Instance() { return m_Instance; }
Logger::~Logger() { fclose(m_pStream);}
void Logger::WriteLog(const char* pMessage) {
  fprintf(m_pStream, "[%s] %s\n", m_Tag.c_str(), pMessage);
  fflush(m_pStream);
}
void Logger::SetTag(const char* pTag) { m_Tag = pTag; }
void OpenConnection() {
  Logger &lg2 = Logger::Instance();
  lg2.WriteLog("from OpenConn");
}
int main() {
  Logger &lg = Logger::Instance();;
  lg.SetTag("0530");
  lg.WriteLog("starts");
  OpenConnection();
  lg.WriteLog("Application is shutting down");
}
```
- What if `Logger lg = Logger::Instance()` in main() ?
  - Returning `Logger` is not defined and copy constructor from compiler is activated, doing shallow copy
  - Disable copy constructor by adding `Logger(const Logger&) = delete;`
  - Disable assign operator by adding `Logger & operator = (const Logger &) = delete;`  
  
14. Lazy instantiation
- In the above example, `m_Instance` is instantiated before main() begins (eager instantiation)
- Lazy instantiation: instantiated when Instance() call is made
  - Instantiates only when necessary
  - We need a pointer variable
- lazySing.h
```cpp
#pragma once
#include <cstdio>
#include <string>
class Logger {
  FILE *m_pStream;
  std::string m_Tag;
  Logger();  
  //static Logger m_Instance; // eager instance
  static Logger *m_pInstance; // lazy instance
public:
  Logger(const Logger&) = delete; // disable copy constructor 
  Logger & operator = (const Logger &) = delete; // disable assign operator
  static Logger& Instance();
  ~Logger();
  void WriteLog(const char *pMessage);
  void SetTag(const char *pTag);
};
```
- lazySing.cxx
```cpp
#include "lazySing.h"
#include <iostream>
Logger *Logger::m_pInstance;
Logger::Logger() { m_pStream  = fopen("applog.txt","w");}
Logger& Logger::Instance() { 
  if (m_pInstance == nullptr)  m_pInstance = new Logger();
  return *m_pInstance; }
Logger::~Logger() { 
  fclose(m_pStream);
  delete m_pInstance;
}
void Logger::WriteLog(const char* pMessage) {
  fprintf(m_pStream, "[%s] %s\n", m_Tag.c_str(), pMessage);
  fflush(m_pStream);
}
void Logger::SetTag(const char* pTag) { m_Tag = pTag; }
void OpenConnection() {
  Logger &lg2 = Logger::Instance();
  lg2.WriteLog("from OpenConn");
}
int main() {
  Logger &lg = Logger::Instance();;
  lg.SetTag("0530");
  lg.WriteLog("starts");
  OpenConnection();
  lg.WriteLog("Application is shutting down");
}
```
- Problem - constructur/destructor are not called
  - Memory leak from m_pInstance
  - How to call destructor?

15. Destruction policies
- How to ensure to invoke destructor?
  - Use smart pointer instead of raw pointer
  - lazyUniqptr.h
```cpp
#pragma once
#include <cstdio>
#include <string>
#include <memory>
#include <iostream>
class Logger {
  FILE *m_pStream;
  std::string m_Tag;
  Logger();  
  inline static std::unique_ptr<Logger> m_pInstance{};
public:
  Logger(const Logger&) = delete; // disable copy constructor 
  Logger & operator = (const Logger &) = delete; // disable assign operator
  static Logger& Instance();
  ~Logger();
  void WriteLog(const char *pMessage);
  void SetTag(const char *pTag);
};
```
  - lazyUniqptr.cxx
```cpp
#include "lazyUniqptr.h"
Logger::Logger() { std::cout << "constructor\n"; m_pStream  = fopen("applog.txt","w");}
Logger& Logger::Instance() { 
  if (m_pInstance == nullptr)  m_pInstance.reset(new Logger{});
  return *m_pInstance; }
Logger::~Logger() { 
  fclose(m_pStream);
  std::cout << "destructor \n";
}
void Logger::WriteLog(const char* pMessage) {
  fprintf(m_pStream, "[%s] %s\n", m_Tag.c_str(), pMessage);
  fflush(m_pStream);  
}
void Logger::SetTag(const char* pTag) { m_Tag = pTag; }
void OpenConnection() {
  Logger &lg2 = Logger::Instance();
  lg2.WriteLog("from OpenConn");
}
int main() {
  Logger &lg = Logger::Instance();;
  lg.SetTag("0530");
  lg.WriteLog("starts");
  OpenConnection();
  lg.WriteLog("Application is shutting down");
}
```  
  - Or add `atexit()` with lambda in the constructor
- Static initialization order fiasco
  - Sequence of global static is not decided
  - Potential memory leak

16. Multithreading Issues
- Use static mutex
- To avoid locking over multiple threads, apply double-checked locking pattern (DCLP)
  - This may not work always

17. Why DCLP fails
- Read/write is not thread-safe

18. Meyer's Singleton
- Use eager instance then it is thread-safe
- For multiple-threading, use Meyer's Singleton

19. Using std::call_once
- Multiple-thread safe
- In Windows, InitOnceExeucteOnce()
- In Linux, pthread_once()

20. CRTP idiom
- Curiously Recurring Template Pattern
- Static instance in a base class
- Derived class call itself as template argument
- Multiple-thread safe

21. Clock Class
- Sync with current time
- Creating a new object will sync with current time anyway
  - Singleton behavior
- Actually each thread has same data

22. Monostate Pattern
- Using `inline static` for attributes
- `static` for methods
- Provides singularity through behavior
- May not be instantiated
- Provides methods through static method

23. Singleton vs. Monostate
- Singleton
  - Enforces singlular instance through struncture
  - Only one instance can exist
  - Support lazy instantiation
  - Requires static instance method
  - Can support inheritance & polymorphism
  - Existing classes can be made singletons
  - Flexible
- Monostate
  - Enforces Singlular instance through behavior
  - Class may or may not be instantiated
  - No support for lazy instantiation
  - All attributes are static
  - Static methods cannot be overriden
  - Difficult to change existing classes to monostate
  - Inflexible

24. Singleton Issues
  - May not be usable in unit-test
    - May use a derived class, which is not used in the main code
  - Against Dependency Inversion Principle by using the name of class
  
25. Registry of Singletones - I
- Multiton
- In a class, multiples of singleton objects exist

26. Registry of Singletones - II
- Using lazy instantiation for dynamic management

27. Pros & Cons
- Pros
  - Class itself controls the instantiation process (eager or lazy)
  - Can allow multiple (but limited) instances
  - Better than global variable
    - But still GLOBAL
  - Can be subsclassed
- Cons
  - Making testsing (unit-test) difficult
  - DCLP is defective in multiple threads
  - Lazy instance destruction is complex

## Section 3: Factory Method

29. Introduction
- Popular in framework and library
  - Can be used without the name of class
  - Using `new` requires to know the name of class in advance
  - Factory method can create instances of objects
- Intent of Factory method
  - Define an interface for creating an object, but let subclass decide which class to instantiate. Factory method lets class defer instantiation to subclasses
- Implementation
  - An overridable method is provided that returns instance of a class
  - This method can be overriden to return instance of a subclass
  - Behaves like a constructor
    - Constructor returns the same instance while factory method can return the instance of any sub-type
    - Factory method is called as virtual constructor

30. Basic Implementation - I
- Base: Product class -> derived: ConcreteProduct class
- Creator class creates ConcreteProduct object using `new`
```cpp
void Creator::AnOperation() {
  m_pProduct = new ConcreteProduct{} ;
  m_pProduct->Operation();
}
```
  - What if we want to create different object than ConcreteProduct?
  - How to generalize?

31. Basic Implementation - II
  - Make
  - basic.h
```cxx
#pragma once
class Product {
public:
  virtual void Operation() = 0;
  virtual ~Product() = default;
};
class ConcreteProduct: public Product {
  void Operation();
};
class ConcreteProduct2: public Product {
  void Operation();
};
class Creator {
  Product *m_pProduct;
public:
  void AnOperation();
  virtual Product * Create() {return nullptr; }
};
class ConcreteCreator: public Creator {
public:
  Product * Create() override;  
};
class ConcreteCreator2: public Creator {
public:
  Product * Create() override;  
};
```
  - basic.cxx
```cxx
#include "basic.h"
#include <iostream>
void ConcreteProduct::Operation() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
}
void ConcreteProduct2::Operation() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
}
void Creator::AnOperation() {
  //m_pProduct = new ConcreteProduct{} ; // not generalized
  m_pProduct = Create(); // now generalizees. Object name is followed from the Create() of derived classes
  m_pProduct->Operation();
}
Product * ConcreteCreator::Create() {
  return new ConcreteProduct{};
}
Product * ConcreteCreator2::Create() {
  return new ConcreteProduct2{};
}
int main() {
  //Creator ct;
  ConcreteCreator2 ct;
  ct.AnOperation();
  return 0;
}
```
- Product -> ConcreteProduct or ConcreteProduct2
- Creator class creates Product object
  - ConcreteCreateor class creates ConcreteProduct object
  - AnOperation() is inherited into derived classes
  - When AnOperation() runs, it runs Create() method of each derived class - ConcreteCreator or ConcreteCreator2
  - Create() method creates derived object of ConcreteProduct or ConcreteProduct2
  - Created object m_pProduct is ConcreteProduct or ConcreteProduct2
  - When Operation() runs, the compiler will defer if this is a member function of ConcreteProduct or ConcreteProduct2

32. Application Framework Discussion
- Base class: Document
  - Base method: read/write
- Derived class: TextDocument
  - Will inherit read/write
- Application can use TextDocument through Document class

33. Application Framework Implementation
- AppFrameWork.h
```cpp
#include "AppFrameWork.h"
#include <iostream>
void Application::New() {  m_pDocument = new TextDocument {};}
void Application::Open() {
  m_pDocument = new TextDocument {};
  m_pDocument->Read();
}
void Application::Save() {  m_pDocument->Write();}
void TextDocument::Write() {  std::cout <<  __PRETTY_FUNCTION__  << std::endl;}
void TextDocument::Read() {  std::cout <<  __PRETTY_FUNCTION__  << std::endl;}
int main() {
  Application app;
  app.New();
  app.Open();
  app.Save();
}
```
- AppFrameWork.cxx
```cpp
#include "AppFrameWork.h"
#include <iostream>
void Application::New() {  m_pDocument = new TextDocument {};}
void Application::Open() {
  m_pDocument = new TextDocument {};
  m_pDocument->Read();
}
void Application::Save() {  m_pDocument->Write();}
void TextDocument::Write() {  std::cout <<  __PRETTY_FUNCTION__  << std::endl;}
void TextDocument::Read() {  std::cout <<  __PRETTY_FUNCTION__  << std::endl;}
int main() {
  Application app;
  app.New();
  app.Open();
  app.Save();
}
```
- New() in Application is hard-coded with TextDocument
  - How can we create an object of another derived class?
  - We use factory method

34. Application Framework with Factory Method
- We add SpreadsheetDocument class
  - Now we need SpreadsheetApplication class
- AppFrameWork.h
```cpp
#pragma once
class Document {
public:
  virtual void Write() = 0;
  virtual void Read() = 0;
  virtual ~Document() = default;
};
class TextDocument: public Document {
public:
  void Write() override;
  void Read() override;
};
class Application {
  Document *m_pDocument;
public:
  void New();
  void Open();
  void Save();
};
class SpreadsheetDocument: public Document {
public:
  void Write() override;
  void Read() override;
};
class SpreadsheetApplication {
  Document *m_pDocument;
public:
  void New();
  void Open();
  void Save();
};
```
- AppFrameWork.cpp
```cpp
#include "AppFrameWork.h"
#include <iostream>
void Application::New() {  m_pDocument = new TextDocument {};}
void Application::Open() {
  m_pDocument = new TextDocument {};
  m_pDocument->Read();
}
void Application::Save() {  m_pDocument->Write();}
void TextDocument::Write() {  std::cout <<  __PRETTY_FUNCTION__  << std::endl;}
void TextDocument::Read() {  std::cout <<  __PRETTY_FUNCTION__  << std::endl;}
void SpreadsheetApplication::New() {  m_pDocument = new SpreadsheetDocument {};}
void SpreadsheetApplication::Open() {
  m_pDocument = new SpreadsheetDocument {};
  m_pDocument->Read();
}
void SpreadsheetApplication::Save() {  m_pDocument->Write();}
void SpreadsheetDocument::Write() {  std::cout <<  __PRETTY_FUNCTION__  << std::endl;}
void SpreadsheetDocument::Read() {  std::cout <<  __PRETTY_FUNCTION__  << std::endl;}
int main() {
  Application app;
  app.New();
  app.Open();
  app.Save();
  SpreadsheetApplication shapp;
  shapp.New();
  shapp.Open();
  shapp.Save();
}
```
- Now how we deallocate memory?
  - Use smart pointer
- AppFrameWork.h
```cpp
#pragma once
#include <memory>
class Document {
public:
  virtual void Write() = 0;
  virtual void Read() = 0;
  virtual ~Document() = default;
};
class TextDocument: public Document {
public:
  void Write() override;
  void Read() override;
};
using DocumentPtr = std::unique_ptr<Document>;
class Application {
protected:
  DocumentPtr m_pDocument;
public:
  void New();
  void Open();
  void Save();
  virtual DocumentPtr Create() {return nullptr;}
};
class TextApplication: public Application {
public:
  void New();
  void Open();
  void Save();
  DocumentPtr Create();
};
class SpreadsheetDocument: public Document {
public:
  void Write() override;
  void Read() override;
};
class SpreadsheetApplication: public Application {
public:
  void New();
  void Open();
  void Save();
  DocumentPtr Create();
};
```
- AppFrameWork.cpp
```cpp
#include "AppFrameWork.h"
#include <iostream>
void Application::New() {  m_pDocument = Create();}
void Application::Open() {
  m_pDocument = Create();
  m_pDocument->Read();
}
void Application::Save() {  m_pDocument->Write();}
DocumentPtr TextApplication::Create() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  return std::make_unique<TextDocument>();
}
void TextApplication::New() {  m_pDocument = Create();}
void TextApplication::Open() {
  m_pDocument = Create();
  m_pDocument->Read();
}
void TextApplication::Save() {  m_pDocument->Write();}
void TextDocument::Write() {  std::cout <<  __PRETTY_FUNCTION__  << std::endl;}
void TextDocument::Read() {  std::cout <<  __PRETTY_FUNCTION__  << std::endl;}
DocumentPtr SpreadsheetApplication::Create() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  return std::make_unique<SpreadsheetDocument>();
}
void SpreadsheetApplication::New() {  m_pDocument = Create();}
void SpreadsheetApplication::Open() {
  m_pDocument = Create();
  m_pDocument->Read();
}
void SpreadsheetApplication::Save() {  m_pDocument->Write();}
void SpreadsheetDocument::Write() {  std::cout <<  __PRETTY_FUNCTION__  << std::endl;}
void SpreadsheetDocument::Read() {  std::cout <<  __PRETTY_FUNCTION__  << std::endl;}
int main() {
  TextApplication app;
  app.Create();
  app.Open();
  app.Save();
  SpreadsheetApplication shapp;
  shapp.Create();
  shapp.Open();
  shapp.Save();
}
```
- How the sample code from the instructor works? Private attributes are not inherited. Must be protected

35. Parameterised Factory Method
- Introducing a class of DocumentFactory
  - Reading an argument, we can map the appropriate derived class object
  - This create function can be static as it doesn't contain any state
  - Instead of TextApplication() or SpreadsheetApplication(), DocumentFactory may provide the creation of multiple different derived class objects
    - Instead of adding new derived application, add more conditional statements into the creation function of DocumentFactory

36. std::unique_ptr
- Instead of `std::unique_ptr<int> p2 {new int{5}};`
- Use `auto p3 = std::make_unique<int>(5);`
  - Actually this is a factory method
   
37. std::shared_ptr
- `make_*` in cpp is a factory method: make_pair, make_shared, ...

38. Pros & Cons
- Pros
  - Instances can be created at runtime
  - Promotes loose coupling 
  - Construction becomes simpler due to abstraction
  - Construction process is encapsulated
  - May not return a new instance every time
- Cons
  - Every new product class may require a corresponding factor class
    - Can be avoided if a paramerized factory method is used
- When to use?
  - A class does not know which instance it needs at runtime
  - A class does not want to depend on the concrete classes that it uses
  - You want to encapsulate the creation process
  - Clients implement the classes that you use - you need to create instances of such classes

## Section 4: Object Pool

40. Introduction
- Implemented through factory method but may not return new objects
- Useful when lots of object created/destructed repetitively - in games
- Purpose
  - Improve performance and memory use by re-using objects from a fixed pool instead of allocating and freeing them repetitively
- Implementation
  - ObjectPool maintains an array or a list of SharedObject instances
  - SharedObject instances are not created by the clients. Instead they use the ObjectPool
  - Objects are constructed when
    - the program starts
    - the pool is empty
    - an existing SharedObject is not available
  - For the last case, the pool can grow automatically
  - ObjectPool can be implemented as a Singleton or Monostate
  - The clients acquire a SharedObject instance by invoking a factory method in the pool
  - When the client gets a SharedObject instance, it is either removed from the ObjectPool or marked as 'used'
  - The client may manually return a SharedObject to the ObjectPool or it may be done automatically
  - This instance can be reused again
  - The pooled object instance can be reset
    - Before giving it to the client
    - After it is returned to the pool
  - The objectPool is responsible for deleting the pooled instances
  - These instances are usually deleted at the end of the program
  - To avoid tight coupling with concrete pooled objects, ObjectPool can use a factory to instantiate them

41. Basic Example
- ObjectPool is monostate, not singleton
- basic_pool.h
```cpp
#pragma once
#include <vector>
class SharedObject {
  bool m_IsUsed{true};
  int m_ID;
public:
  bool IsUsed() const {return m_IsUsed;}
  void SetUsedState(bool used) {m_IsUsed = used;}
  void MethodA();
  void MethodB();
  void Reset();
  void SetID(int i_) {m_ID = i_;}
  int GetID() {return m_ID;}
};
class ObjectPool {
  ObjectPool() = default;
  inline static std::vector<SharedObject*> m_PooledObjects{};
public:
  static SharedObject * AcquireObject();
  static void ReleaseObject(SharedObject *pSO);
  static int ncount;
};
```
- basic_pool.cpp
```cpp
#include "basic_pool.h"
#include <iostream>
void SharedObject::MethodA() { std::cout << "MethodA ID=" << GetID() << " \n"; }
void SharedObject::MethodB() { std::cout << "MethodB\n"; }
void SharedObject::Reset() { std::cout << "Resetting the state\n"; }
SharedObject* ObjectPool::AcquireObject() {
  for(auto & so: m_PooledObjects) {
    if (!so->IsUsed()) {
      std::cout << "[POOL] Returning an existing object\n";
      so->SetUsedState(true);
      so->Reset();
      return so;
    }
  }
  std::cout << "[POOL] Creating a new object\n";
  ncount ++;
  SharedObject *so = new SharedObject{};
  so->SetID(ncount);
  m_PooledObjects.push_back(so);
  return so;
}
void ObjectPool::ReleaseObject(SharedObject* pSO) {
  for (auto &so: m_PooledObjects) {
    if (so == pSO) {
      so->SetUsedState(false);
    }
  }
}
int ObjectPool::ncount = 0;
int main() {
  auto s1 = ObjectPool::AcquireObject();
  s1->MethodA();
  s1->MethodB();
  auto s2 = ObjectPool::AcquireObject();
  s2->MethodA();
  s2->MethodB();
  ObjectPool::ReleaseObject(s1);
  auto s3 = ObjectPool::AcquireObject();
  s3->MethodA();
  s3->MethodB();
  return 0;
}
```
- Assignment: Rebuild the code using smart pointers
  - basic_pool_sptr.h
```cpp
#pragma once
#include <vector>
#include <memory>
class SharedObject {
  bool m_IsUsed{true};
  int m_ID;
public:
  bool IsUsed() const {return m_IsUsed;}
  void SetUsedState(bool used) {m_IsUsed = used;}
  void MethodA();
  void MethodB();
  void Reset();
  void SetID(int i_) {m_ID = i_;}
  int GetID() {return m_ID;}
};
using SharedObjectUniqPtr = std::unique_ptr<SharedObject>;
using SharedObjectSharedPtr = std::shared_ptr<SharedObject>;
class ObjectPool {
  ObjectPool() = default;
  //inline static std::vector<SharedObjectUniqPtr> m_PooledObjects{};
  inline static std::vector<SharedObjectSharedPtr> m_PooledObjects{};
public:
  //static SharedObjectUniqPtr AcquireObject();
  //static void ReleaseObject(SharedObjectUniqPtr pSO);
  static SharedObjectSharedPtr AcquireObject();
  static void ReleaseObject(SharedObjectSharedPtr pSO);
  static int ncount;
};
```
  - basic_pool_sptr.cpp
```cpp
#include "basic_pool_sptr.h"
#include <iostream>
void SharedObject::MethodA() { std::cout << "MethodA ID=" << GetID() << " \n"; }
void SharedObject::MethodB() { std::cout << "MethodB\n"; }
void SharedObject::Reset() { std::cout << "Resetting the state\n"; }
//SharedObjectUniqPtr ObjectPool::AcquireObject() {
SharedObjectSharedPtr ObjectPool::AcquireObject() {
  for(auto & so: m_PooledObjects) {
    if (!so->IsUsed()) {
      std::cout << "[POOL] Returning an existing object\n";
      so->SetUsedState(true);
      so->Reset();
      return so;
    }
  }
  std::cout << "[POOL] Creating a new object\n";
  ncount ++;
  //auto so = std::make_unique<SharedObject> ();
  auto so = std::make_shared<SharedObject> ();
  so->SetID(ncount);
  m_PooledObjects.push_back(so);
  return so;
}
//void ObjectPool::ReleaseObject(SharedObjectUniqPtr pSO) {
void ObjectPool::ReleaseObject(SharedObjectSharedPtr pSO) {
  for (auto &so: m_PooledObjects) {
    if (so == pSO) {
      so->SetUsedState(false);
    }
  }
}
int ObjectPool::ncount = 0;
int main() {
  auto s1 = ObjectPool::AcquireObject();
  s1->MethodA();
  s1->MethodB();
  auto s2 = ObjectPool::AcquireObject();
  s2->MethodA();
  s2->MethodB();
  ObjectPool::ReleaseObject(s1);
  auto s3 = ObjectPool::AcquireObject();
  s3->MethodA();
  s3->MethodB();
  return 0;
}
```
  - Q: why uniq_ptr doesn't work?

42. Pooling Game Objects - I

43. Pooling Game Objects - II

44. Multple Actors - I

45. Multple Actors - II

46. Multple Actors - III

47. Generic Pool - I

48. Generic Pool - II
- Using template

49. Pros & Cons
- Pros
  - Reduces coupling with concrete classes
  - Behaves like operator new, but is more flexible
  - Caching existing instances improves performance of the application
  - Reduces the overhead of heap allocation and deallocation
  - Reduces heap fragmentation
- Cons 
  - Memory may be wasted on unused pooled objects
  - Pooled objects may remain in memory until the end of the program
  - Objects that are acquired from the pool must be reset prior to their use
  - Clients have to ensure that an unused object is returned to the pool
  - ObjectPool class may get tightly coupled with the classes of the pooled objects
- When to use
  - When you want to frequently create & destroy objects
  - Allocating heap objects is slow
  - Frequent allocation leads to heap fragmentation
  - Objects are expensive to create

## Section 5: Abstract Factory

51. Introduction
- Purpose
  - Provide an interface for creating families of related or dependent objects without specifying their concrete classes
- Implementation
  - An abstract factory defines an interface for creating differnt products (factory methods)
  - Factories are added for each context
  - All factories inherit from the abstract factory
  - Each factory will create instances of classes for the corresponding context
  - Only one factory will be used in the whole application through the base abstract factory reference

52. Basic Example
- basic.h
```cpp
#pragma once
#include <iostream>
class AbstractProductA {
public:
  virtual void ProductA() = 0;
  virtual ~AbstractProductA() = default;
};
class AbstractProductB {
public:
  virtual void ProductB() = 0;
  virtual ~AbstractProductB() = default;
};
class ProductA1 : public AbstractProductA {
public:
  void ProductA() override { std::cout << "[1] Product A\n"; }
};
class ProductB1 : public AbstractProductB {
public:
  void ProductB() override { std::cout << "[1] Product B\n"; }
};
class ProductA2 : public AbstractProductA {
public:
  void ProductA() override { std::cout << "[2] Product A\n"; }
};
class ProductB2 : public AbstractProductB {
public:
  void ProductB() override { std::cout << "[2] Product B\n"; }
};
```
- basic.cpp
```cpp
#include "basic.h"
int main() {
  AbstractProductA *pA = new ProductA1{};
  AbstractProductB *pB = new ProductB1{};
  pA->ProductA();
  pB->ProductB();
  delete pA;
  delete pB;
}
```
- How to create ProductA2 class object?
  - Change the code `new ProductA1{};` into `new ProductA2{};`
  - Or we use abstract factory method

53. Basic Example with Abstract Factory
- We add a class of AbstractFactory()
- absFact.h
```cpp
#pragma once
#include <iostream>
class AbstractProductA {
public:
  virtual void ProductA() = 0;
  virtual ~AbstractProductA() = default;
};
class AbstractProductB {
public:
  virtual void ProductB() = 0;
  virtual ~AbstractProductB() = default;
};
class ProductA1 : public AbstractProductA {
public:
  void ProductA() override { std::cout << "[1] Product A\n"; }
};
class ProductB1 : public AbstractProductB {
public:
  void ProductB() override { std::cout << "[1] Product B\n"; }
};
class ProductA2 : public AbstractProductA {
public:
  void ProductA() override { std::cout << "[2] Product A\n"; }
};
class ProductB2 : public AbstractProductB {
public:
  void ProductB() override { std::cout << "[2] Product B\n"; }
};
class AbstractFactory {
public:
  virtual AbstractProductA * CreateProductA() = 0;
  virtual AbstractProductB * CreateProductB() = 0;
  virtual ~AbstractFactory() = default;
};
class ConcreteFactory1 : public AbstractFactory {
public:
  AbstractProductA * CreateProductA() override;
  AbstractProductB * CreateProductB() override;
};
class ConcreteFactory2 : public AbstractFactory {
public:
  AbstractProductA * CreateProductA() override;
  AbstractProductB * CreateProductB() override;
};
```
- absFact.cpp
```cpp
#include "absFact.h"
AbstractProductA* ConcreteFactory1::CreateProductA() {
  return new ProductA1{};
}
AbstractProductB* ConcreteFactory1::CreateProductB() {
  return new ProductB1{};
}
AbstractProductA* ConcreteFactory2::CreateProductA() {
  return new ProductA2{};
}
AbstractProductB* ConcreteFactory2::CreateProductB() {
  return new ProductB2{};
}
void UsePattern(AbstractFactory *pFactory) {
  AbstractProductA *pA = pFactory->CreateProductA();
  AbstractProductB *pB = pFactory->CreateProductB();
  pA->ProductA();
  pB->ProductB();
  delete pA;
  delete pB;
}
int main() {
  AbstractFactory *pFactory = new ConcreteFactory1{};
  UsePattern(pFactory);
  delete pFactory;
}
```

54. Database Framework Introduction
- Needs 
  - Connection: uid, db, table, ...
  - Command: CRUD
  - Recordset:
  - Transaction: not included in this example
  - Parameters: not included in this example

55. Database Framework Implementation

56. SqlServer Database Classes
- dbaseFrmWork.h
```cpp
#pragma once
#include <string>
#include <vector>
class Connection {
  std::string m_ConnectionString;
public:
  void SetConnectionString(const std::string &connStr) {
    m_ConnectionString = connStr;    
  }
  const std::string & GetConnectionString() const {
    return m_ConnectionString;
  }
  virtual void Open() = 0;
  virtual ~Connection() = default;
};
class RecordSet {
public:
  virtual const std::string & Get() = 0;
  virtual bool HasNext() = 0;
  virtual ~RecordSet() = default;
};
class Command {
  std::string m_CommandString;
protected:
  Connection *m_pConnection{};
public:
  Connection * GetConnection() const {
    return m_pConnection;
  }
  const std::string & GetCommandString() const {
    return m_CommandString;
  }
  void SetCommand(const std::string &commandStr) {
    m_CommandString = commandStr;    
  }
  void SetConnection(Connection *pConnection) {
    m_pConnection = pConnection;
  }
  virtual void ExeucteCommand() = 0;
  virtual RecordSet * ExecuteQuery() = 0;
  virtual ~Command() = default;
};
class SqlConnection: public Connection {
public:
  void Open() override;
};
class SqlRecordSet: public RecordSet {
  const std::vector<std::string> m_Db{ "Terminator", "Predator", "Eraser"};
  std::vector<std::string>::const_iterator m_Cursor;
public:
  SqlRecordSet();
  const std::string& Get() override;
  bool HasNext() override;
};
class SqlCommand: public Command {
public:
  //SqlCommand() {}
  void ExeucteCommand() override;
  SqlRecordSet* ExecuteQuery() override;
};
```
- dbaseFrmWork.cpp
```cpp
#include "dbaseFrmWork.h"
#include <iostream>
void SqlConnection::Open() { std::cout << "[SqlConnection] Connection opened\n";}
void SqlCommand::ExeucteCommand() {
  std::cout << "[SqlCommand] Executing command on" 
            << m_pConnection->GetConnectionString()
            << std::endl;
}
SqlRecordSet* SqlCommand::ExecuteQuery(){
  std::cout << "[SqlCommand] Executing query\n";
  return new SqlRecordSet();
}
SqlRecordSet::SqlRecordSet() {
  m_Cursor = m_Db.begin();
}
const std::string& SqlRecordSet::Get() {
  return *m_Cursor++;
}
bool SqlRecordSet::HasNext() {
  return m_Cursor != m_Db.end();
}
int main() {
  SqlConnection *pCon = new SqlConnection{};
  pCon->SetConnectionString("uid=umar;db=movies;table=actors");
  pCon->Open();
  SqlCommand *pCmd = new SqlCommand{};
  pCmd->SetConnection(pCon);
  pCmd->SetCommand("select * from actors");
  SqlRecordSet *pRec = pCmd->ExecuteQuery();
  while(pRec->HasNext()) {
    std::cout << pRec->Get() << std::endl;
  }
  delete pCon;
  delete pCmd;
  delete pRec;
}
```

57. MySql Database Classes
- The above main works with base class as
```cpp
  Connection *pCon = new SqlConnection{};
...
  Command *pCmd = new SqlCommand{};
...
  RecordSet *pRec = pCmd->ExecuteQuery();
```
- For MySqlConnection, may use `Connection *pCon = new MySqlConnection{};`
  - This is error-prone
  - How to avoid?
    - May use MACRO

58. Database Framework Usage
- Using macro
```cpp
#ifdef SQL
  Connection *pCon = new SqlConnection{};
#else defined MYSQL
  Connection *pCon = new MySqlConnection{};
#endif
```
- Not recommended
- Using conditional statement has similar issues

59. Using Factory Method
- How to instantiate without concrete class name?
- Provide a DbFactory class
- dbfactory.h
```cpp
#pragma once
#include <string_view>
#include "dbaseFrmWork.h"
class DbFactory {
public:
  static Connection* CreateConnection(std::string_view type);
  static Command* CreateCommand(std::string_view type);
};
```
- dbfactory.cpp
```cpp
#include "dbfactory.h"
#include "dbaseFrmWork.h"
#include "mysql.h"
Command* DbFactory::CreateCommand(std::string_view type) {
  if (type=="sql") {
    return new SqlCommand{};
  } else if (type=="mysql") {
    return new MySqlCommand{};
  } 
  return nullptr;
}
Connection* DbFactory::CreateConnection(std::string_view type) {
  if (type=="sql") {
    return new SqlConnection{};
  } else if (type=="mysql") {
    return new MySqlConnection{};
  } 
  return nullptr;
}  
```
- main.cpp
```cpp
void FactoryMethod() {
  int dbtype = 0;
  Connection* pCon{DbFactory::CreateConnection("sql")};
  pCon->SetConnectionString("uid=umar;db=movies;table=actors");
  pCon->Open();
  Command* pCmd{DbFactory::CreateCommand("sql")};
  pCmd->SetConnection(pCon);
  pCmd->SetCommand("select * from actors");
  RecordSet *pRec = pCmd->ExecuteQuery();
  while(pRec->HasNext()) {
    std::cout << pRec->Get() << std::endl;
  }
  delete pCon;
  delete pCmd;
  delete pRec;
}
int main() {
  FactoryMethod();
}
```
- Q:
  - what if Command is created as sql while Connection is created as mysql?
  - How to couple them as a single type?
    - Needs to be specific set
    - absract factory can enforce as a specific set

60. Using Abstract Factory
- dbfactory.h
```cpp
#pragma once
#include <string_view>
#include "dbaseFrmWork.h"
class DbFactory {
public:
  virtual Command* CreateCommand() = 0;
  virtual Connection* CreateConnection() = 0;
  virtual ~DbFactory() = default;
};
class SqlFactory: public DbFactory {
public:
  Command* CreateCommand() override;
  Connection* CreateConnection() override;  
};
class MySqlFactory: public DbFactory {
public:
  Command* CreateCommand() override;
  Connection* CreateConnection() override;  
};
```
- dbfactory.cpp
```cpp
#include "dbfactory.h"
#include "dbaseFrmWork.h"
#include "mysql.h"
Command* SqlFactory::CreateCommand() {
  return new SqlCommand{};
}
Connection* SqlFactory::CreateConnection() {
  return new SqlConnection{};
}
Command* MySqlFactory::CreateCommand() {
  return new MySqlCommand{};
}
Connection* MySqlFactory::CreateConnection() {
  return new MySqlConnection{};
}
```
- main.cpp
```cpp
void UsingFactory(DbFactory *pFactory) {
  Connection* pCon = pFactory->CreateConnection();
  pCon->SetConnectionString("uid=umar;db=movies;table=actors");
  pCon->Open();
  Command* pCmd = pFactory->CreateCommand();
  pCmd->SetConnection(pCon);
  pCmd->SetCommand("select * from actors");
  RecordSet *pRec = pCmd->ExecuteQuery();
  while(pRec->HasNext()) {
    std::cout << pRec->Get() << std::endl;
  }
  delete pCon;
  delete pCmd;
  delete pRec;
}
int main() {
  MySqlFactory f;
  UsingFactory(&f);
}
```
61. Pros & Cons
- Pros
  - Promotes loose coupling
  - To support more configurations in future, you need to
    - Add the classes for a new set
    - Add corresponding factor class
  - Enforces consistency among products as the application can get instances of classes only from one set at a time  
- Cons
  - Adding new products is difficult
  - Adding a new configuration causes class explosion
- Factory method vs. Abstract factory
  - Subclasses manage the creation of the concrete type vs creation depends on the type of factory used
  - Easy to extend the factory class to support new products vs difficult to extend the factories to support new products
  - Many factories can be used simultaneously vs only one factory is used at a time
- When to use
  - When provide instances to clients without exposing concrete classes
  - Configure a system with one of multiple product configurations
  - A system must be able to use classes only from one family at a time and you want to enforce that

## Section 6: Prototype

63. Introduction
- Purpose
  - Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype
- Implementation
  - The classes whose objects need to be cloned inherit from a common base class
  - This class provides an override method called clone
  - This method can be overriden by the sub-classes to create a copy of themselves
  - The client can then call this method to create copies/clones of existing objects
  - After cloning, the client may change some state of the cloned objects
  - Consequently, the classes may have to provide initialize/setter methods

64. Cloning Types
- Shallow copy
  - Not actual copy of the internal data
  - Copies the reference
- Deep copy
  - This has to be implemented manually

65. Basic Example
- basic.h
```cpp
#pragma
class Prototype {
public:
  virtual Prototype * Clone() = 0;
  virtual ~Prototype() = default;
};
class ConcretePrototype1: public Prototype {
public:
  Prototype* Clone() override;
};
class ConcretePrototype2: public Prototype {
public:
  Prototype* Clone() override;
};
class Client {
  Prototype *prototype;
public:
  void SetPrototype(Prototype *p) {
    prototype = p;
  }
  void Operation();
};
```
- basic.cpp
```cpp
#include "basic.h"
#include <iostream>
Prototype* ConcretePrototype1::Clone() {
  std::cout << "[ConcretePrototype1] Cloning...\n";
  return new ConcretePrototype1{*this};
}
Prototype* ConcretePrototype2::Clone() {
  std::cout << "[ConcretePrototype2] Cloning...\n";
  return new ConcretePrototype2{*this};
}
void Client::Operation() {
  auto p = prototype->Clone();
}
int main() {
  Client c;
  c.SetPrototype(new ConcretePrototype1{});
  c.Operation();
}
```

66. Game Introduction
- A class of car for blue/red car, blue/yellow bus, ...

67. Game Implementation - I

68. Game Implementation - II

69. Game Implementation - III
- anim.h
```cpp
#pragma once
#include <string>
#include <iostream>
#include <random>
class Animation {
  std::string m_AnimationData{} ;
public:
  Animation()=default;
  Animation(std::string_view animFile);
  const std::string & GetAnimationData() const {
    return m_AnimationData;    
  }
  void SetAnimationData(const std::string &animationData)  {
    m_AnimationData = animationData;
  }
};
struct Position {
  int x;
  int y;
  friend std::ostream & operator<< (std::ostream &out, Position p) {
    return out << "(" << p.x << ',' << p.y << ')';
  }
};
class Vehicle {
  int m_Speed{};
  int m_HitPoints{};
  std::string m_Name{};
  Animation *m_pAnimation{};
  Position m_Position{};
public:
  Vehicle(int mSpeed, int mHitPoints, const std::string& mName, 
          std::string_view animFile, const Position& mPosition);
  virtual ~Vehicle();
  int GetSpeed() const {    return m_Speed;  }
  int GetHitPoints() const { return m_HitPoints; }
  const std::string& GetName() const {    return m_Name;  }
  Position GetPosition() const { return m_Position; }
  const std::string & GetAnimation() const;
  void SetSpeed(int speed) {m_Speed = speed; }
  void SetPosition(Position position) { m_Position = position; }
  void SetName(const std::string &name) { m_Name = name; }
  void SetHitPoitns(int hitPoints) { m_HitPoints = hitPoints; }
  void SetAnimationData(const std::string &animData);
  virtual void Update()=0;
};
class GreenCar : public Vehicle {
  using Vehicle::Vehicle;
public:
  void Update() override;
};
class RedCar : public Vehicle {
  using Vehicle::Vehicle;
  float m_SpeedFactor{1.5f};
  std::default_random_engine m_Engine{100};
  std::bernoulli_distribution m_Dist{.5};
public:
  void SetSpeedFactory(float factor) { m_SpeedFactor = factor;}
  void Update() override;
};
class BlueBus : public Vehicle {
  using Vehicle::Vehicle;
  std::default_random_engine m_Engine{500};
  std::bernoulli_distribution m_Dist{0.5};
public:
  void Update() override;
};
class YellowBus : public Vehicle {
  using Vehicle::Vehicle;
  std::default_random_engine m_Engine{500};
  std::bernoulli_distribution m_Dist{0.5};
public:
  void Update() override;
};
class GameManager{
  std::vector<Vehicle*> m_Vehicles{};
public:
  void Run();
  ~GameManager();
};
Vehicle * Create(std::string_view type, int mSpeed, int mHitPoints, const std::string& mName,
          std::string_view animFile, const Position& mPosition)  {
  if (type == "redcar") {
    return new RedCar{mSpeed, mHitPoints, mName, animFile, mPosition};
  } else if (type == "greencar") {
    return new GreenCar{mSpeed, mHitPoints, mName, animFile, mPosition};
  } else if (type == "yellowbus") {
    return new YellowBus{mSpeed, mHitPoints, mName, animFile, mPosition};
  } else if (type == "bluebus") {
    return new BlueBus{mSpeed, mHitPoints, mName, animFile, mPosition};
  } 
  return nullptr;
}  
```
- anim.cpp
```cpp
#include "anim.h"
#include <iostream>
#include <thread>
#include <cstdlib>
using namespace std::literals::chrono_literals;
Animation::Animation(std::string_view animFile) {
  std::cout << "[Animation] Loading " << animFile << ' ';
  for (int i=0;i<10;++i) {
    std::cout << ".";
    std::this_thread::sleep_for(200ms);
  }
  std::cout << '\n';
  m_AnimationData.assign("^^^^^");
}
Vehicle::Vehicle(int mSpeed, int mHitPoints, const std::string& mName,
          std::string_view animFile, const Position& mPosition) 
          : m_Speed{mSpeed}, m_HitPoints{mHitPoints},
            m_Name{mName}, m_Position{mPosition} {
            m_pAnimation = new Animation{animFile};
}
Vehicle::~Vehicle() { 
  delete m_pAnimation;
}
void Vehicle::SetAnimationData(const std::string& animData) { 
  m_pAnimation->SetAnimationData(animData);
}
const std::string& Vehicle::GetAnimation() const {
  return m_pAnimation->GetAnimationData();
}
void Vehicle::Update() { }
void GreenCar::Update() {
  std::cout << "[" << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n"
    << "\tSpeed:" << GetSpeed() << "\n"
    << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
void RedCar::Update() {
  std::cout << "[" << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n";
  if (m_Dist(m_Engine)) {
    std::cout << "\tIncrease speed temporarily:" << GetSpeed() * m_SpeedFactor << "\n";
  } else {
    std::cout << "\tSpeed:" << GetSpeed() << "\n";
  }
  std::cout << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
void BlueBus::Update() {
  std::cout << "[" << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n";
  if (m_Dist(m_Engine)) {
    std::cout << "\tMoving out of the way\n";
  } 
  std::cout << "\tSpeed:" << GetSpeed() << "\n";
  std::cout << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
void YellowBus::Update() {
  std::cout << "[" << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n";
  if (m_Dist(m_Engine)) {
    std::cout << "\tMoving out of the way\n";
  } 
  std::cout << "\tSpeed:" << GetSpeed() << "\n";
  std::cout << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
void GameManager::Run() {
  m_Vehicles.push_back(Create("redcar",   30,10, "RedCar",    "red.anim",   {0,0}));
  m_Vehicles.push_back(Create("greencar",  30,10, "GreenCar",  "green.anim", {100,0}));
  m_Vehicles.push_back(Create("yellowbus", 30,10, "YellowBus", "rbus.anim",  {100,200}));
  m_Vehicles.push_back(Create("bluebus",   30,10, "BlueBus",   "bbus.anim",  {100,200}));
  int count{5};
  while(count !=0) {
    std::this_thread::sleep_for(1s);
    // system("cls"); // for windows
    std::system("clear");
    for (auto vehicle: m_Vehicles) {
      vehicle->Update();
    }
    if (count ==2) {
        m_Vehicles.push_back(Create("redcar", 30,15, "RedCar", "red.anim", {0,0}));
    }
    if (count ==3) {
      m_Vehicles.push_back(Create("yellowbus", 20,20, "YellowBus", "rbus.anim", {0,0}));
    }
    --count;
  }
}
GameManager::~GameManager() {
  for (auto vehicle:m_Vehicles) {
    delete vehicle;
  }
}
int main() {
  GameManager mgr;
  mgr.Run();
}
```
- Screen log
```bash
$ ./a.out 
[Animation] Loading red.anim ..........
[Animation] Loading green.anim ..........
[Animation] Loading rbus.anim ..........
```
- How to avoid the loading of the animation?
  - This doesn't really load animation in the code but we may assume that loading animation is a very expensive process like loading from a disk

70. Cloning Example
- Why we need to copy the existing object?

71. Prototype Implementation - I
- We add a Clone() method
```cpp
Vehicle* RedCar::Clone() {
  std::cout << "Cloning-> "<< GetName() << "\n";
  return new RedCar{*this};
}
...
if (count ==2) {
        //m_Vehicles.push_back(Create("redcar", 30,15, "RedCar", "red.anim", {0,0}));
        auto vehicle = m_Vehicles[0]->Clone();
        vehicle->SetPosition({50,50});
        vehicle->SetHitPoints(15);
        m_Vehicles.push_back(vehicle);
    }
...
```
- Issues of memory leak

72. Prototype Implementation - II
- We used `return new RedCar{*this};` and this uses the constructor. This enables us to shallow copy of the `m_PAnmiation` but this causes double free

73. Prototype Implementation - III
- Provide copy/move constructor and assign/move operator
  - Rule of 5
- anim.h
```cpp
#pragma once
#include <string>
#include <iostream>
#include <random>
class Animation {
  std::string m_AnimationData{} ;
public:
  Animation()=default;
  Animation(std::string_view animFile);
  const std::string & GetAnimationData() const {
    return m_AnimationData;    
  }
  void SetAnimationData(const std::string &animationData)  {
    m_AnimationData = animationData;
  }
};
struct Position {
  int x;
  int y;
  friend std::ostream & operator<< (std::ostream &out, Position p) {
    return out << "(" << p.x << ',' << p.y << ')';
  }
};
class Vehicle {
  int m_Speed{};
  int m_HitPoints{};
  std::string m_Name{};
  Animation *m_pAnimation{};
  Position m_Position{};
public:
  Vehicle(int mSpeed, int mHitPoints, const std::string& mName, 
          std::string_view animFile, const Position& mPosition);
  virtual ~Vehicle();
  Vehicle(const Vehicle &other); // copy constructor
  Vehicle &operator=(const Vehicle &other);  // assign operator
  Vehicle(Vehicle &&other) noexcept; // Move constructor
  Vehicle & operator=(Vehicle &&other) noexcept; // move operator
  int GetSpeed() const {    return m_Speed;  }
  int GetHitPoints() const { return m_HitPoints; }
  const std::string& GetName() const {    return m_Name;  }
  Position GetPosition() const { return m_Position; }
  const std::string & GetAnimation() const;
  void SetSpeed(int speed) {m_Speed = speed; }
  void SetPosition(Position position) { m_Position = position; }
  void SetName(const std::string &name) { m_Name = name; }
  void SetHitPoints(int hitPoints) { m_HitPoints = hitPoints; }
  void SetAnimationData(const std::string &animData);
  virtual void Update() = 0;
  virtual Vehicle * Clone() = 0;
};
class GreenCar : public Vehicle {
  using Vehicle::Vehicle;
public:
  void Update() override;
  Vehicle * Clone() override;
};
class RedCar : public Vehicle {
  using Vehicle::Vehicle;
  float m_SpeedFactor{1.5f};
  std::default_random_engine m_Engine{100};
  std::bernoulli_distribution m_Dist{.5};
public:
  void SetSpeedFactory(float factor) { m_SpeedFactor = factor;}
  void Update() override;
  Vehicle * Clone() override;
};
class BlueBus : public Vehicle {
  using Vehicle::Vehicle;
  std::default_random_engine m_Engine{500};
  std::bernoulli_distribution m_Dist{0.5};
public:
  void Update() override;
  Vehicle * Clone() override;
};
class YellowBus : public Vehicle {
  using Vehicle::Vehicle;
  std::default_random_engine m_Engine{500};
  std::bernoulli_distribution m_Dist{0.5};
public:
  void Update() override;
  Vehicle * Clone() override;
};
class GameManager{
  std::vector<Vehicle*> m_Vehicles{};
public:
  void Run();
  ~GameManager();
};
Vehicle * Create(std::string_view type, int mSpeed, int mHitPoints, const std::string& mName,
          std::string_view animFile, const Position& mPosition)  {
  if (type == "redcar") {
    return new RedCar{mSpeed, mHitPoints, mName, animFile, mPosition};
  } else if (type == "greencar") {
    return new GreenCar{mSpeed, mHitPoints, mName, animFile, mPosition};
  } else if (type == "yellowbus") {
    return new YellowBus{mSpeed, mHitPoints, mName, animFile, mPosition};
  } else if (type == "bluebus") {
    return new BlueBus{mSpeed, mHitPoints, mName, animFile, mPosition};
  } 
  return nullptr;
}  
```
- anim.cpp
```cpp
#include "anim.h"
#include <iostream>
#include <thread>
#include <cstdlib>
using namespace std::literals::chrono_literals;
Animation::Animation(std::string_view animFile) {
  std::cout << "[Animation] Loading " << animFile << ' ';
  for (int i=0;i<10;++i) {
    std::cout << ".";
    std::this_thread::sleep_for(200ms);
  }
  std::cout << '\n';
  m_AnimationData.assign("^^^^^");
}
Vehicle::Vehicle(int mSpeed, int mHitPoints, const std::string& mName,
          std::string_view animFile, const Position& mPosition) 
          : m_Speed{mSpeed}, m_HitPoints{mHitPoints},
            m_Name{mName}, m_Position{mPosition} {
            m_pAnimation = new Animation{animFile};
}
Vehicle::~Vehicle() { 
  delete m_pAnimation;
}
void Vehicle::SetAnimationData(const std::string& animData) { 
  m_pAnimation->SetAnimationData(animData);
}
const std::string& Vehicle::GetAnimation() const {
  return m_pAnimation->GetAnimationData();
}
Vehicle::Vehicle(const Vehicle &other):  // copy constructor
  m_Speed{other.m_Speed}, m_Name{other.m_Name}, 
  m_HitPoints{other.m_HitPoints}, m_Position{other.m_Position} {
    m_pAnimation = new Animation();
    m_pAnimation->SetAnimationData(other.GetAnimation());
}
Vehicle & Vehicle::operator=(const Vehicle &other) {  // assign operator
  if (this != &other) {
    m_Speed = other.m_Speed;
    m_Name = other.m_Name;
    m_HitPoints = other.m_HitPoints;
    m_Position = other.m_Position;
    m_pAnimation->SetAnimationData(other.GetAnimation());
  }
  return *this;
}
Vehicle::Vehicle(Vehicle &&other) noexcept : // Move constructor
    m_Speed{other.m_Speed}, m_Name{other.m_Name}, 
    m_HitPoints{other.m_HitPoints}, m_Position{other.m_Position} {
    m_pAnimation = other.m_pAnimation;
    other.m_pAnimation = nullptr;
    other.m_Position = {0,0};
    other.m_HitPoints = 0;
    other.m_Speed = 0;
    other.m_Name.clear();
}
Vehicle & Vehicle::operator=(Vehicle &&other) noexcept { // move operator
  if (this != &other) {
    m_Speed = other.m_Speed;
    m_Name = other.m_Name;
    m_HitPoints = other.m_HitPoints;
    m_Position = other.m_Position;
    delete m_pAnimation;
    m_pAnimation  = other.m_pAnimation;
    other.m_pAnimation = nullptr;
    other.m_Position = {0,0};
    other.m_HitPoints = 0;
    other.m_Speed = 0;
    other.m_Name.clear();
  }
  return *this;
}  
void Vehicle::Update() { }
void GreenCar::Update() {
  std::cout << "[" << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n"
    << "\tSpeed:" << GetSpeed() << "\n"
    << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
void RedCar::Update() {
  std::cout << "[" << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n";
  if (m_Dist(m_Engine)) {
    std::cout << "\tIncrease speed temporarily:" << GetSpeed() * m_SpeedFactor << "\n";
  } else {
    std::cout << "\tSpeed:" << GetSpeed() << "\n";
  }
  std::cout << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
void BlueBus::Update() {
  std::cout << "[" << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n";
  if (m_Dist(m_Engine)) {
    std::cout << "\tMoving out of the way\n";
  } 
  std::cout << "\tSpeed:" << GetSpeed() << "\n";
  std::cout << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
void YellowBus::Update() {
  std::cout << "[" << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n";
  if (m_Dist(m_Engine)) {
    std::cout << "\tMoving out of the way\n";
  } 
  std::cout << "\tSpeed:" << GetSpeed() << "\n";
  std::cout << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
Vehicle* RedCar::Clone() {
  std::cout << "Cloning-> "<< GetName() << "\n";
  return new RedCar{*this};
}
Vehicle* GreenCar::Clone() {
  std::cout << "Cloning-> "<< GetName() << "\n";
  return new GreenCar{*this};
}
Vehicle* YellowBus::Clone() {
  std::cout << "Cloning-> "<< GetName() << "\n";
  return new YellowBus{*this};
}
Vehicle* BlueBus::Clone() {
  std::cout << "Cloning-> "<< GetName() << "\n";
  return new BlueBus{*this};
}
void GameManager::Run() {
  m_Vehicles.push_back(Create("redcar",   30,10, "RedCar",    "red.anim",   {0,0}));
  m_Vehicles.push_back(Create("greencar",  30,10, "GreenCar",  "green.anim", {100,0}));
  m_Vehicles.push_back(Create("yellowbus", 30,10, "YellowBus", "rbus.anim",  {100,200}));
  m_Vehicles.push_back(Create("bluebus",   30,10, "BlueBus",   "bbus.anim",  {100,200}));
  int count{5};
  while(count !=0) {
    std::this_thread::sleep_for(1s);
    // system("cls"); // for windows
    std::system("clear");
    for (auto vehicle: m_Vehicles) {
      vehicle->Update();
    }
    if (count ==2) {
        //m_Vehicles.push_back(Create("redcar", 30,15, "RedCar", "red.anim", {0,0}));
        auto vehicle = m_Vehicles[0]->Clone();
        vehicle->SetPosition({50,50});
        vehicle->SetHitPoints(15);
        m_Vehicles.push_back(vehicle);
    }
    if (count ==3) {
      //m_Vehicles.push_back(Create("yellowbus", 20,20, "YellowBus", "rbus.anim", {0,0}));
      auto vehicle = m_Vehicles[2]->Clone();
      vehicle->SetPosition({150,150});
      vehicle->SetSpeed(10);
      m_Vehicles.push_back(vehicle);
    }
    --count;
  }
}
GameManager::~GameManager() {
  for (auto vehicle:m_Vehicles) {
    delete vehicle;
  }
}
int main() {
  GameManager mgr;
  mgr.Run();
}
```

74. Class vs Object
- Instead of having the classes of RedCar and GreenCar, make use of RedCar classs to make an object of GreenCar then change the color

75. Varying State
- How to change the state of existing objects
  - Can reduce the number of classes
- Classes of car and bus only

76. Prototype Manager - I
- Instead of making RedCar -> GreenCar, GreenCar can be cloned from RedCar
- Let's add a prototype manager

77. Prototype Manager - II
- reduced.h
```cpp
#pragma once
#include <string>
#include <iostream>
#include <random>
#include <unordered_map>
class Animation {
  std::string m_AnimationData{} ;
public:
  Animation()=default;
  Animation(std::string_view animFile);
  const std::string & GetAnimationData() const {
    return m_AnimationData;    
  }
  void SetAnimationData(const std::string &animationData)  {
    m_AnimationData = animationData;
  }
};
struct Position {
  int x;
  int y;
  friend std::ostream & operator<< (std::ostream &out, Position p) {
    return out << "(" << p.x << ',' << p.y << ')';
  }
};
class Vehicle {
  int m_Speed{};
  int m_HitPoints{};
  std::string m_Name{};
  Animation *m_pAnimation{};
  Position m_Position{};
  std::string m_Color{};
public:
  Vehicle();
  Vehicle(int mSpeed, int mHitPoints, const std::string& mName, 
          std::string_view animFile, const Position& mPosition,
          const std::string& mColor);
  virtual ~Vehicle();
  Vehicle(const Vehicle &other); // copy constructor
  Vehicle &operator=(const Vehicle &other);  // assign operator
  Vehicle(Vehicle &&other) noexcept; // Move constructor
  Vehicle & operator=(Vehicle &&other) noexcept; // move operator
  int GetSpeed() const {    return m_Speed;  }
  int GetHitPoints() const { return m_HitPoints; }
  const std::string& GetName() const {    return m_Name;  }
  Position GetPosition() const { return m_Position; }
  const std::string & GetAnimation() const;
  const std::string& GetColor() const { return m_Color; }
  void SetSpeed(int speed) {m_Speed = speed; }
  void SetPosition(Position position) { m_Position = position; }
  void SetName(const std::string &name) { m_Name = name; }
  void SetHitPoints(int hitPoints) { m_HitPoints = hitPoints; }
  void SetAnimationData(const std::string &animData);
  void SetColor(const std::string& color) { m_Color = color; }
  virtual void Update() = 0;
  virtual Vehicle * Clone() = 0;
};
class Car : public Vehicle {
  using Vehicle::Vehicle;
  float m_SpeedFactor{1.5f};
  std::default_random_engine m_Engine{100};
  std::bernoulli_distribution m_Dist{.5};
public:
  void SetSpeedFactory(float factor) { m_SpeedFactor = factor;}
  void Update() override;
  Vehicle * Clone() override;
};
class Bus : public Vehicle {
  using Vehicle::Vehicle;
  std::default_random_engine m_Engine{500};
  std::bernoulli_distribution m_Dist{0.5};
public:
  void Update() override;
  Vehicle * Clone() override;
};
class GameManager{
  std::vector<Vehicle*> m_Vehicles{};
public:
  void Run();
  ~GameManager();
};
Vehicle * Create(std::string_view type, int mSpeed, int mHitPoints, const std::string& mName,
          std::string_view animFile, const Position& mPosition)  {
  if (type == "redcar") {
    return new Car{mSpeed, mHitPoints, mName, animFile, mPosition, "Red"};
  } else if (type == "greencar") {
    return new Car{mSpeed, mHitPoints, mName, animFile, mPosition, "Green"};
  } else if (type == "yellowbus") {
    return new Bus{mSpeed, mHitPoints, mName, animFile, mPosition, "Yellow"};
  } else if (type == "bluebus") {
    return new Bus{mSpeed, mHitPoints, mName, animFile, mPosition, "Blue"};
  } 
  return nullptr;
}
class VehiclePrototypes {
  inline static std::unordered_map<std::string, Vehicle*> m_Prototypes{};
  VehiclePrototypes() = default;
public:
  static std::vector<std::string> GetKeys();
  static void RegisterPrototype(const std::string &key, Vehicle *prototype);
  static Vehicle * DeregisterPrototype(const std::string &key);
  static Vehicle * GetPrototype(const std::string &key);
};
```
- reduced.cpp
```cpp
#include "reduced.h"
#include <iostream>
#include <thread>
#include <cstdlib>
using namespace std::literals::chrono_literals;
Animation::Animation(std::string_view animFile) {
  std::cout << "[Animation] Loading " << animFile << ' ';
  for (int i=0;i<10;++i) {
    std::cout << ".";
    std::this_thread::sleep_for(200ms);
  }
  std::cout << '\n';
  m_AnimationData.assign("^^^^^");
}
Vehicle::Vehicle() {
  m_pAnimation = new Animation{};
}
Vehicle::Vehicle(int mSpeed, int mHitPoints, const std::string& mName,
          std::string_view animFile, const Position& mPosition, 
          const std::string& mColor) 
          : m_Speed{mSpeed}, m_HitPoints{mHitPoints},
            m_Name{mName}, m_Position{mPosition}, m_Color{mColor} {
            m_pAnimation = new Animation{animFile};
}
Vehicle::~Vehicle() { 
  delete m_pAnimation;
}
void Vehicle::SetAnimationData(const std::string& animData) { 
  m_pAnimation->SetAnimationData(animData);
}
const std::string& Vehicle::GetAnimation() const {
  return m_pAnimation->GetAnimationData();
}
Vehicle::Vehicle(const Vehicle &other):  // copy constructor
  m_Speed{other.m_Speed}, m_Name{other.m_Name}, 
  m_HitPoints{other.m_HitPoints}, m_Position{other.m_Position},
  m_Color{other.m_Color} {
    m_pAnimation = new Animation();
    m_pAnimation->SetAnimationData(other.GetAnimation());
}
Vehicle & Vehicle::operator=(const Vehicle &other) {  // assign operator
  if (this != &other) {
    m_Speed = other.m_Speed;
    m_Name = other.m_Name;
    m_HitPoints = other.m_HitPoints;
    m_Position = other.m_Position;
    m_Color = other.m_Color;
    m_pAnimation->SetAnimationData(other.GetAnimation());
  }
  return *this;
}
Vehicle::Vehicle(Vehicle &&other) noexcept : // Move constructor
    m_Speed{other.m_Speed}, m_Name{other.m_Name}, 
    m_HitPoints{other.m_HitPoints}, m_Position{other.m_Position},
    m_Color{other.m_Color} {
    m_pAnimation = other.m_pAnimation;
    other.m_pAnimation = nullptr;
    other.m_Position = {0,0};
    other.m_HitPoints = 0;
    other.m_Speed = 0;
    other.m_Name.clear();
    other.m_Color.clear();
}
Vehicle & Vehicle::operator=(Vehicle &&other) noexcept { // move operator
  if (this != &other) {
    m_Speed = other.m_Speed;
    m_Name = other.m_Name;
    m_HitPoints = other.m_HitPoints;
    m_Position = other.m_Position;
    delete m_pAnimation;
    m_pAnimation  = other.m_pAnimation;
    other.m_pAnimation = nullptr;
    other.m_Position = {0,0};
    other.m_HitPoints = 0;
    other.m_Speed = 0;
    other.m_Name.clear();
  }
  return *this;
}  
void Vehicle::Update() { }
void Car::Update() {
  std::cout << "[" << GetColor() << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n";
  if (m_Dist(m_Engine) && GetColor() == "Red") {
    std::cout << "\tIncrease speed temporarily:" << GetSpeed() * m_SpeedFactor << "\n";
  } else {
    std::cout << "\tSpeed:" << GetSpeed() << "\n";
  }
  std::cout << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
void Bus::Update() {
  std::cout << "[" << GetColor() << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n";
  if (m_Dist(m_Engine)) {
    std::cout << "\tMoving out of the way\n";
  } 
  std::cout << "\tSpeed:" << GetSpeed() << "\n";
  std::cout << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
Vehicle* Car::Clone() {
  std::cout << "Cloning-> "<< GetName() << "\n";
  return new Car{*this};
}
Vehicle* Bus::Clone() {
  std::cout << "Cloning-> "<< GetName() << "\n";
  return new Bus{*this};
}
GameManager::~GameManager() {
  for (auto vehicle:m_Vehicles) {
    delete vehicle;
  }
}
std::vector<std::string> VehiclePrototypes::GetKeys() {
  std::vector<std::string> keys {} ;
  keys.reserve(m_Prototypes.size());
  for(const auto &kv : m_Prototypes) {
    keys.push_back(kv.first);
  }
  return keys;
}
void VehiclePrototypes::RegisterPrototype(const std::string &key, Vehicle *prototype) {
  if (auto it = m_Prototypes.find(key); it == end(m_Prototypes)) {
    m_Prototypes[key] = prototype;
  } else {
    std::cout << "Key already exists\n";
  }
}
Vehicle * VehiclePrototypes::DeregisterPrototype(const std::string &key) {
  if (auto it= m_Prototypes.find(key); it !=end(m_Prototypes)) {
    auto vehicle = m_Prototypes[key];
    m_Prototypes.erase(key);
    return vehicle;
  }
  return nullptr;
}
Vehicle * VehiclePrototypes::GetPrototype(const std::string &key) {
  if (auto it = m_Prototypes.find(key); it!=end(m_Prototypes)) {
    return m_Prototypes[key]->Clone();
  }
  return nullptr;
}
Vehicle *GetRedCar() {
  auto vehicle = VehiclePrototypes::GetPrototype("car");
  vehicle->SetColor("Red");
  vehicle->SetHitPoints(10);
  vehicle->SetSpeed(30);
  vehicle->SetPosition({0,0});
  Animation anim{"red.anim"};
  vehicle->SetAnimationData(anim.GetAnimationData());
  return vehicle;
}
Vehicle *GetGreenCar() {
  auto vehicle = VehiclePrototypes::GetPrototype("car");
  vehicle->SetColor("Green");
  vehicle->SetHitPoints(5);
  vehicle->SetSpeed(30);
  vehicle->SetPosition({100,0});
  Animation anim{"green.anim"};
  vehicle->SetAnimationData(anim.GetAnimationData());
  return vehicle;
}
Vehicle* GetYellowBus() {
  auto vehicle = VehiclePrototypes::GetPrototype("bus");
  vehicle->SetColor("Yellow");
  vehicle->SetHitPoints(20);
  vehicle->SetSpeed(25);
  vehicle->SetPosition({100,200});
  Animation anim{"ybus.anim"};
  vehicle->SetAnimationData(anim.GetAnimationData());
  return vehicle;
}
Vehicle* GetBlueBus() {
  auto vehicle = VehiclePrototypes::GetPrototype("bus");
  vehicle->SetColor("Blue");
  vehicle->SetHitPoints(20);
  vehicle->SetSpeed(25);
  vehicle->SetPosition({200,200});
  Animation anim{"bbus.anim"};
  vehicle->SetAnimationData(anim.GetAnimationData());
  return vehicle;
}
void GameManager::Run() {
  m_Vehicles.push_back(GetRedCar());
  m_Vehicles.push_back(GetGreenCar());
  m_Vehicles.push_back(GetYellowBus());
  m_Vehicles.push_back(GetBlueBus());
  int count{5};
  while(count !=0) {
    std::this_thread::sleep_for(1s);
    // system("cls"); // for windows
    std::system("clear");
    for (auto vehicle: m_Vehicles) {
      vehicle->Update();
    }
    if (count ==2) {
        //m_Vehicles.push_back(Create("redcar", 30,15, "RedCar", "red.anim", {0,0}));
        auto vehicle = m_Vehicles[0]->Clone();
        vehicle->SetPosition({50,50});
        vehicle->SetHitPoints(15);
        m_Vehicles.push_back(vehicle);
    }
    if (count ==3) {
      //m_Vehicles.push_back(Create("yellowbus", 20,20, "YellowBus", "rbus.anim", {0,0}));
      auto vehicle = m_Vehicles[2]->Clone();
      vehicle->SetPosition({150,150});
      vehicle->SetSpeed(10);
      m_Vehicles.push_back(vehicle);
    }
    --count;
  }
}
int main() {
  VehiclePrototypes::RegisterPrototype("car", new Car{});
  VehiclePrototypes::RegisterPrototype("bus", new Bus{});
  GameManager mgr;
  mgr.Run();
}
```

78. Memory Management
- Issue of memory leak
- Let's use smart pointer
- sptr.h
```cpp
#pragma once
#include <string>
#include <iostream>
#include <random>
#include <unordered_map>
#include <memory>
class Animation {
  std::string m_AnimationData{} ;
public:
  Animation()=default;
  Animation(std::string_view animFile);
  const std::string & GetAnimationData() const {
    return m_AnimationData;    
  }
  void SetAnimationData(const std::string &animationData)  {
    m_AnimationData = animationData;
  }
};
struct Position {
  int x;
  int y;
  friend std::ostream & operator<< (std::ostream &out, Position p) {
    return out << "(" << p.x << ',' << p.y << ')';
  }
};
class Vehicle;
using VehiclePtr = std::shared_ptr<Vehicle>;
class Vehicle {
  int m_Speed{};
  int m_HitPoints{};
  std::string m_Name{};
  Animation *m_pAnimation{};
  Position m_Position{};
  std::string m_Color{};
public:
  Vehicle();
  Vehicle(int mSpeed, int mHitPoints, const std::string& mName, 
          std::string_view animFile, const Position& mPosition,
          const std::string& mColor);
  virtual ~Vehicle();
  Vehicle(const Vehicle &other); // copy constructor
  Vehicle &operator=(const Vehicle &other);  // assign operator
  Vehicle(Vehicle &&other) noexcept; // Move constructor
  Vehicle & operator=(Vehicle &&other) noexcept; // move operator
  int GetSpeed() const {    return m_Speed;  }
  int GetHitPoints() const { return m_HitPoints; }
  const std::string& GetName() const {    return m_Name;  }
  Position GetPosition() const { return m_Position; }
  const std::string & GetAnimation() const;
  const std::string& GetColor() const { return m_Color; }
  void SetSpeed(int speed) {m_Speed = speed; }
  void SetPosition(Position position) { m_Position = position; }
  void SetName(const std::string &name) { m_Name = name; }
  void SetHitPoints(int hitPoints) { m_HitPoints = hitPoints; }
  void SetAnimationData(const std::string &animData);
  void SetColor(const std::string& color) { m_Color = color; }
  virtual void Update() = 0;
  virtual VehiclePtr Clone() = 0;
};
class Car : public Vehicle {
  using Vehicle::Vehicle;
  float m_SpeedFactor{1.5f};
  std::default_random_engine m_Engine{100};
  std::bernoulli_distribution m_Dist{.5};
public:
  void SetSpeedFactory(float factor) { m_SpeedFactor = factor;}
  void Update() override;
  VehiclePtr Clone() override;
};
class Bus : public Vehicle {
  using Vehicle::Vehicle;
  std::default_random_engine m_Engine{500};
  std::bernoulli_distribution m_Dist{0.5};
public:
  void Update() override;
  VehiclePtr Clone() override;
};
class GameManager{
  std::vector<VehiclePtr> m_Vehicles{};
public:
  void Run();
  ~GameManager() = default;
};
class VehiclePrototypes {
  inline static std::unordered_map<std::string, VehiclePtr> m_Prototypes{};
  VehiclePrototypes() = default;
public:
  static std::vector<std::string> GetKeys();
  static void RegisterPrototype(const std::string &key, VehiclePtr prototype);
  static VehiclePtr DeregisterPrototype(const std::string &key);
  static VehiclePtr GetPrototype(const std::string &key);
};
```
- sptr.cpp
```cpp
#include "sptr.h"
#include <iostream>
#include <thread>
#include <cstdlib>
using namespace std::literals::chrono_literals;
Animation::Animation(std::string_view animFile) {
  std::cout << "[Animation] Loading " << animFile << ' ';
  for (int i=0;i<10;++i) {
    std::cout << ".";
    std::this_thread::sleep_for(200ms);
  }
  std::cout << '\n';
  m_AnimationData.assign("^^^^^");
}
Vehicle::Vehicle() {
  m_pAnimation = new Animation{};
}
Vehicle::Vehicle(int mSpeed, int mHitPoints, const std::string& mName,
          std::string_view animFile, const Position& mPosition, 
          const std::string& mColor) 
          : m_Speed{mSpeed}, m_HitPoints{mHitPoints},
            m_Name{mName}, m_Position{mPosition}, m_Color{mColor} {
            m_pAnimation = new Animation{animFile};
}
Vehicle::~Vehicle() { 
  delete m_pAnimation;
}
void Vehicle::SetAnimationData(const std::string& animData) { 
  m_pAnimation->SetAnimationData(animData);
}
const std::string& Vehicle::GetAnimation() const {
  return m_pAnimation->GetAnimationData();
}
Vehicle::Vehicle(const Vehicle &other):  // copy constructor
  m_Speed{other.m_Speed}, m_Name{other.m_Name}, 
  m_HitPoints{other.m_HitPoints}, m_Position{other.m_Position},
  m_Color{other.m_Color} {
    m_pAnimation = new Animation();
    m_pAnimation->SetAnimationData(other.GetAnimation());
}
Vehicle & Vehicle::operator=(const Vehicle &other) {  // assign operator
  if (this != &other) {
    m_Speed = other.m_Speed;
    m_Name = other.m_Name;
    m_HitPoints = other.m_HitPoints;
    m_Position = other.m_Position;
    m_Color = other.m_Color;
    m_pAnimation->SetAnimationData(other.GetAnimation());
  }
  return *this;
}
Vehicle::Vehicle(Vehicle &&other) noexcept : // Move constructor
    m_Speed{other.m_Speed}, m_Name{other.m_Name}, 
    m_HitPoints{other.m_HitPoints}, m_Position{other.m_Position},
    m_Color{other.m_Color} {
    m_pAnimation = other.m_pAnimation;
    other.m_pAnimation = nullptr;
    other.m_Position = {0,0};
    other.m_HitPoints = 0;
    other.m_Speed = 0;
    other.m_Name.clear();
    other.m_Color.clear();
}
Vehicle & Vehicle::operator=(Vehicle &&other) noexcept { // move operator
  if (this != &other) {
    m_Speed = other.m_Speed;
    m_Name = other.m_Name;
    m_HitPoints = other.m_HitPoints;
    m_Position = other.m_Position;
    delete m_pAnimation;
    m_pAnimation  = other.m_pAnimation;
    other.m_pAnimation = nullptr;
    other.m_Position = {0,0};
    other.m_HitPoints = 0;
    other.m_Speed = 0;
    other.m_Name.clear();
  }
  return *this;
}  
void Vehicle::Update() { }
void Car::Update() {
  std::cout << "[" << GetColor() << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n";
  if (m_Dist(m_Engine) && GetColor() == "Red") {
    std::cout << "\tIncrease speed temporarily:" << GetSpeed() * m_SpeedFactor << "\n";
  } else {
    std::cout << "\tSpeed:" << GetSpeed() << "\n";
  }
  std::cout << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
void Bus::Update() {
  std::cout << "[" << GetColor() << GetName() << "]\n"
    << "\tAnimation:" << GetAnimation() << "\n";
  if (m_Dist(m_Engine)) {
    std::cout << "\tMoving out of the way\n";
  } 
  std::cout << "\tSpeed:" << GetSpeed() << "\n";
  std::cout << "\tHitPoints:" << GetHitPoints() << "\n"
    << "\tPosition:" << GetPosition() << "\n";
}
VehiclePtr Car::Clone() {
  std::cout << "Cloning-> "<< GetName() << "\n";
  return std::shared_ptr<Vehicle> { new Car{*this} };
}
VehiclePtr Bus::Clone() {
  std::cout << "Cloning-> "<< GetName() << "\n";
  return std::shared_ptr<Vehicle> { new Bus{*this}};
}
std::vector<std::string> VehiclePrototypes::GetKeys() {
  std::vector<std::string> keys {} ;
  keys.reserve(m_Prototypes.size());
  for(const auto &kv : m_Prototypes) {
    keys.push_back(kv.first);
  }
  return keys;
}
void VehiclePrototypes::RegisterPrototype(const std::string &key, VehiclePtr prototype) {
  if (auto it = m_Prototypes.find(key); it == end(m_Prototypes)) {
    m_Prototypes[key] = prototype;
  } else {
    std::cout << "Key already exists\n";
  }
}
VehiclePtr VehiclePrototypes::DeregisterPrototype(const std::string &key) {
  if (auto it= m_Prototypes.find(key); it !=end(m_Prototypes)) {
    auto vehicle = m_Prototypes[key];
    m_Prototypes.erase(key);
    return vehicle;
  }
  return nullptr;
}
VehiclePtr VehiclePrototypes::GetPrototype(const std::string &key) {
  if (auto it = m_Prototypes.find(key); it!=end(m_Prototypes)) {
    return m_Prototypes[key]->Clone();
  }
  return nullptr;
}
VehiclePtr GetRedCar() {
  auto vehicle = VehiclePrototypes::GetPrototype("car");
  vehicle->SetColor("Red");
  vehicle->SetHitPoints(10);
  vehicle->SetSpeed(30);
  vehicle->SetPosition({0,0});
  Animation anim{"red.anim"};
  vehicle->SetAnimationData(anim.GetAnimationData());
  return vehicle;
}
VehiclePtr GetGreenCar() {
  auto vehicle = VehiclePrototypes::GetPrototype("car");
  vehicle->SetColor("Green");
  vehicle->SetHitPoints(5);
  vehicle->SetSpeed(30);
  vehicle->SetPosition({100,0});
  Animation anim{"green.anim"};
  vehicle->SetAnimationData(anim.GetAnimationData());
  return vehicle;
}
VehiclePtr GetYellowBus() {
  auto vehicle = VehiclePrototypes::GetPrototype("bus");
  vehicle->SetColor("Yellow");
  vehicle->SetHitPoints(20);
  vehicle->SetSpeed(25);
  vehicle->SetPosition({100,200});
  Animation anim{"ybus.anim"};
  vehicle->SetAnimationData(anim.GetAnimationData());
  return vehicle;
}
VehiclePtr GetBlueBus() {
  auto vehicle = VehiclePrototypes::GetPrototype("bus");
  vehicle->SetColor("Blue");
  vehicle->SetHitPoints(20);
  vehicle->SetSpeed(25);
  vehicle->SetPosition({200,200});
  Animation anim{"bbus.anim"};
  vehicle->SetAnimationData(anim.GetAnimationData());
  return vehicle;
}
void GameManager::Run() {
  m_Vehicles.push_back(GetRedCar());
  m_Vehicles.push_back(GetGreenCar());
  m_Vehicles.push_back(GetYellowBus());
  m_Vehicles.push_back(GetBlueBus());
  int count{5};
  while(count !=0) {
    std::this_thread::sleep_for(1s);
    // system("cls"); // for windows
    std::system("clear");
    for (auto vehicle: m_Vehicles) {
      vehicle->Update();
    }
    if (count ==2) {
      auto vehicle = m_Vehicles[0]->Clone();
      vehicle->SetPosition({50,50});
      vehicle->SetHitPoints(15);
      m_Vehicles.push_back(vehicle);
    }
    if (count ==3) {
      auto vehicle = m_Vehicles[2]->Clone();
      vehicle->SetPosition({150,150});
      vehicle->SetSpeed(10);
      m_Vehicles.push_back(vehicle);
    }
    --count;
  }
}
int main() {
  VehiclePrototypes::RegisterPrototype("car", std::make_shared<Car>({}));
  VehiclePrototypes::RegisterPrototype("bus", std::make_shared<Bus>({}));
  GameManager mgr;
  mgr.Run();
}
```

79. Pros & Cons
- Alternative implementation
  - Serialization can be used for cloning
- Pros
  - Similar to factory method
  - Concrete classes are hidden from clients
    - Loose coupling
  - Products can be added or removed at runtime
    - Useful for languages without reflection
  - New objects can be specified by varying values
    - Reduces the number of classes
- Cons
  - Each class must support Clone() 
    - Difficult to add if the classes already exist
  - If the member attributes don't support copying, then it is difficult to implement
    - Stream, thread, ...
- When to use
  - For loose coupling
  - Avoids parallel hierarchy of factories

## Section 7: Builder

81. Introduction
- Builder Pattern
  - During development, a class may become complex as it may encapsulate more functionality
  - The class structure also becomes complex
  - This requires class objects to have different representations at runtime
  - Consequently we may have to instantiate the class with different structures or different internal states
  - This may require a multitude of constructors with lots of parameters
  - Finally the object construction becomes compelx and error prone
- Builder
  - Construction of a complex object can be simplified through builder
  - Builder encapsulates the construction logic in a different class
  - This allows creation of different representations of the object
  - The object is constructed step-by-step; other creation patterns construct objects in one shot
- Purpose
  - Separate the construction of a complex object from its representation so that the same construction process can create different representations
- Implementation
  - Construction process of an object is directed by a class called **director**
  - The director will use builder class to assembly an object in steps
  - To create different kinds of objects, the director will use different builders
  - All the builder class may inherit from a common base class
  - The base class will ahve the appropriate interface to allow creation of different kind of objects in steps

82. Basic Implementation
- basic.h
```cpp
#pragma once
class Product {
};
class Builder {
public:
  virtual void BuildPart() = 0;
  virtual ~Builder() = default;
};
class ConcreteBuilder: public Builder {
  Product *product;
public:
  ConcreteBuilder();
  void BuildPart() override;
  Product * GetResult();
};
class Director {
  Builder *builder;
public:
  Director(Builder* builder);
  void Construct();
};
```
- basic.cpp
```cpp
#include "basic.h"
#include <iostream>
ConcreteBuilder::ConcreteBuilder() {
  std::cout << "[ConcreteBuilder] Created\n";
}
void ConcreteBuilder::BuildPart() {
  std::cout << "[ConcreteBuilder] Building ...\n";
  std::cout << "\t Part A\n";
  std::cout << "\t Part B\n";
  std::cout << "\t Part C\n";
  product = new Product{};
}
Product* ConcreteBuilder::GetResult() {
  std::cout << "[ConcreteBuilder] Returning result\n";
  return product;
}
Director::Director(Builder* builder): builder{builder} {
  std::cout << "[Director] Created\n";
}
void Director::Construct() {
  std::cout << "[Director] Construction process started\n";
  builder->BuildPart();
}
int main() {
  ConcreteBuilder builder;
  Director director{&builder};
  director.Construct();
  Product *p = builder.GetResult();
  delete p;
}
```

83. File Example Introduction
- Windows API?

84. Issues

85. Builder Implementation

86. Construction Using Builder
- The same director object but many different builder objects can be defined

87. Modern Implementation
- No director class

88. Fluent Builder
- Fluent interface using chained methods

89. Pros and cons
- Pros
  - Hides the internal representation and structure of the product
  - Hides how the product gets assembled
  - To get a new product with a different internal represenation, you just define a new builder
  - Separation b/w code that constructrs the object and the one that represents it
  - Give more control over the construction process of the object
- Cons
  - May increase the overall complexity of the code
- When to use
  - Get rid of too many constructors in a class
  - Get rid of delegating constructors
  - Separate the algorithm of object construction