## Instructor: Stephen Ulibarri
# course name: Learn C++ for Game Development

39. Enums
```Cpp
#include <iostream>
#include <string>
enum PlayerStatus 
{
    PS_Crouched, // as default, 0
    PS_Standing,
    PS_Walking,
    PS_Running
};
int main() 
{
    PlayerStatus status;
    status = PS_Walking; // Or use PlayerStatus::PS_Walking
    if (status == PlayerStatus::PS_Walking) {
        std::cout << "it is walking" << std::endl;
    }    
    system("ls -alrt");
}
```
- may use `PS_Crouched=34` to assign non-default value
- Basically we don't care the actual size or number. Just use the enum variable names as declared

44. Pointers
- `int *ptr` actually means `int* ptr`
    - ptr is an address which stores integer data type
    - data at ptr is retrieved using `*ptr`

45. Pointers in practice
```cpp
#include <iostream>
#include <string>
struct Player
{
    std::string name;
    int x, y, z;
};
int main()
{
    int a = 100;
    float b = 100.f;
    int* aptr;
    aptr = &a;
    std::cout << aptr << std::endl;
    std::cout << *aptr << std::endl;
    int numbs[] = { 11, 22, 33};
    aptr = numbs;
    std::cout << *aptr << std::endl;
    aptr ++;
    std::cout << *aptr << std::endl;
    aptr ++;
    std::cout << *aptr << std::endl;
    //
    Player jenny = { "Jenny", 11, 22, 33};
    Player* transition = &jenny;
    std::cout << (*transition).name << std::endl;
    std::cout << transition->name << std::endl;
}
```
- Command:
```
$ g++ -std=c++14 code45.cpp 
$ ./a.out 
0x7ffc7e04fd78
100
11
22
33
Jenny
Jenny
```
- Syntactic sugar for struct pointer
    - Instead of `(*transition).name`, use `transition->name`

52. Inheritance in Practice
- Member data of base class might not be used in initialization list of derived classes
    - Use the setter/getter method of the base class to access the member data of the base class
- Default constructor (no argument) of the base class will be called when a derived class constructor is called:
```cpp
class Animal
{
public:
    Animal(){}
    Animal(std::string name, int age, int nlimbs) {}
    ...
class Dog: public Animal 
{
public:
    Dog(){}
    Dog(std::string name, int age, int nlimbs) {}
    ...
```
- `Dog dog1("shiva",12,4)` will call `Animal()->Dog("shiva",12,4)`
- In order to have `Animal("shiva",12,4)->Dog("shiva",12,4)`, use the initialization list as:
```
Dog(std::string name, int age, int nlimbs)
:Animal(std::string name, int age, int nlimbs)
{}
```

54. Access modifiers (Encapsulation):
- Ref: https://www.w3schools.com/cpp/cpp_access_specifiers.asp
- public - members are accessible from outside the class
- private - members cannot be accessed (or viewed) from outside the class. Derived class are not allowed to access the private member of the base class
    - Use the public method (getter/setter) of the base class to access the private member of the base class
- protected - members cannot be accessed from outside the class, however, they can be accessed in inherited classes


56. Stack and heap
- Stack memory
    - Ref: https://www.geeksforgeeks.org/stack-data-structure-introduction-program/
    - stack memory is where local variables get stored/constructed. When certain variable/functions are not used anymore, they are **popped**
    - **Linear structure**. No fragmentation at all. Faster than heap
    - This is why static array in fortran is fast
    - Ex: `myclass myinst;`
- Heap
    - is used for dynamic memory
    - **Tree structure**. Possibility of memory leak.
- `int *p = new int`;
    - The p is created at stack while its dynamic memory is allocated in heap
    - Need to delete the allocated memory to prevent memory leak
    - Ex: `myclass* myinst = new myclass(); ... ; delete myinst;`

61. Static
- static variables/instances exist out of the common scope
```Cpp
#include <iostream>
#include <string>
void update_count()
{
    static int count = 0; // After initialization, this will be skipped
    count++;
    std::cout << count << std::endl;
}
int main()
{
    update_count();
    update_count();
    update_count();
}
$ g++ -std=c++14 code61.cpp 
$ ./a.out 
1
2
3
```
- Static member datat in a class
``` Cpp
#include <iostream>
#include <string>
class Creature
{
public:
    static int count; // static int count=0; will not compile
    Creature() 
    {
        count++;
    }
    void print_count(){
        std::cout << count << std::endl;
    }
};
int Creature::count = 0; // Need to initialize outside of the class definition
int main()
{
    Creature doggy;
    doggy.print_count(); // this will be 1
    Creature kitty;
    kitty.print_count(); // this will be 2
    doggy.print_count(); // this will be 2 again
    return 0;
}  
$ g++ -std=c++14 code61_class.cpp
$ ./a.out 
1
2
2
```
- static member function in a class
    - The function can be called without instances
```Cpp
#include <iostream>
#include <string>
class Creature
{
public:
    static int count;
    Creature() 
    {
        count++;
    }
    static void bark() {
        std::cout << "Woof at " << count << std::endl;
    }
};
int Creature::count = 0;
int main()
{
    Creature::bark();    // Use ::, not '.'. No instance
    Creature doggy;
    doggy.bark();        // bark() is a member function of the instance. So use '.'
    Creature kitty;
    kitty.bark();
    doggy.bark();
    return 0;
} 
$ g++ -std=c++14 code61_class.cpp
$ ./a.out 
Woof at 0
Woof at 1
Woof at 2
Woof at 2
```

62. Virtual function
- To over-ride in the derived classes
- Doesn't need to write virtual or override but can write for decoration
```Cpp
#include <iostream>
class Parent
{
public:
    virtual void Greet()
    { std::cout << "Hello\n"; }
};
class Child : public Parent
{
public:
    void Greet()  // virtual can be omitted. Already virtual as inherited
    { std::cout << "Morning\n"; }
};
class GChild : public Child
{
public:
    virtual void Greet() override // virtual & override for decoration
    { std::cout << "Hola\n"; }
};
int main()
{
    Parent p1;
    p1.Greet();
    Child c1;
    c1.Greet();
    GChild g1;
    g1.Greet();
    return 0;
}  
$ g++ -std=c++14 code62.cpp
$ ./a.out 
Hello
Morning
Hola
```
- Pure virtual:
    - Ref: https://www.learncpp.com/cpp-tutorial/pure-virtual-functions-abstract-base-classes-and-interface-classes/
    - In Parent class, `virtual void Greet() = 0;`
    - Parent class **cannot be instantiated**
    - Now all of derived classes must re-define `Greet()`
        - Enforcing mechanism

64. Polymorphism
- Creates a pointer pointing Parent class
- Allocate child (or grand-child) class to the pointer
- Now the pointer can use the method/member data of the child (or grand-child) class
```Cpp
#include <iostream>
class Parent
{
public:
    virtual void Greet()
    { std::cout << "Hello\n"; }
};
class Child : public Parent
{
public:
    void Greet()
    { std::cout << "Morning\n"; }
};
class GChild : public Child
{
public:
    virtual void Greet() override
    { std::cout << "Hola\n"; }
};
int main()
{
    Parent p1;
    Child c1;
    GChild g1;
    Parent* hook;
    hook = &p1; hook->Greet();
    hook = &c1; hook->Greet();
    hook = &g1; hook->Greet();
    Parent* hookArray[]={&p1, &c1, &g1};
    for (int i=0;i<3; i++) hookArray[i]->Greet();
    return 0;
}  
$ g++ -std=c++14 code64.cpp
$ ./a.out 
Hello
Morning
Hola
Hello
Morning
Hola
```

66. Diamond inheritance
- When P->A->C while P->B->C
- C has the base classes of A & B
- A method(f) of P will conflict in C as A::f() vs B::f()
- May access C.A::f() or C.B::f()
```Cpp
#include <iostream>
class P
{
public:
    virtual void Greet()    { std::cout << "I am P\n"; }
};
class A : public P
{
public:
    void Greet()    { std::cout << "I am A\n"; }
};
class B : public P
{
public:
    void Greet()    { std::cout << "I am B\n"; }
};
class C : public A, public B
{
public:
    
};

int main()
{
    C* ptr = new C();
    //ptr->Greet(); //Compile error: request for member ‘Greet’ is ambiguous
    ptr->A::Greet();
    ptr->B::Greet();
    delete ptr;
    return 0;
}  
$ g++ code66.cpp 
$ ./a.out 
I am A
I am B
```
- Using virtual keyword will prevent diamond inheritance
```Cpp
class A: virtual public P
...
class B: virtual public P
...
class C: public A, public B
```
- this will yield compilation error message 

67. Casting
- `dynamic_cast` for inherited classes at RTE
- May be used as a type checker as it will return null when `dynamic_cast` fails
```Cpp
#include <iostream>
class P
{
public:
    virtual void Greet()    { std::cout << "I am P\n"; }
    void fP() {std::cout << "function at P\n";}
};
class A : public P
{
public:
    void Greet()    { std::cout << "I am A\n"; }
    void fA() {std::cout << "function at A\n";}
};
class B : public P
{
public:
    void Greet()    { std::cout << "I am B\n"; }
    void fB() {std::cout << "function at B\n";}
};
class C : public A
{
public:
  void Greet()    { std::cout << "I am C\n"; }
  void fC() {std::cout << "function at C\n";}
};

int main()
{
    P* p = new P;
    A* a = new A;
    B* b = new B;
    C* c = new C;
    P* arr[] = {p, a, b, c};
    for (int i = 0; i< 4; i++)
    {
        A* tmp = dynamic_cast<A*>(arr[i]);
        if (tmp) 
        {
            tmp->fA();
        } else {
            std::cout << "null found\n";
        }
    }    
    delete c;
    delete b;
    delete a;
    delete p;
    return 0;
}  
$ g++ -std=c++14 code67.cpp
$ ./a.out 
null found     // P is above A
function at A  // A is found
null found     // B is sibling
function at A  // C is below A 
```
- Using `static_cast` instead of `dynamic_cast` will yield `funciton at A` for every loop
