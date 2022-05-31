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

33. Application Framework Implementation

34. Application Framework with Factory Method

35. Parameterised Factory Method

36. std::unique_ptr

37. std::shared_ptr

38. Pros and Cons
