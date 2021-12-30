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
```
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
