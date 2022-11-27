## Java 17: Learn and dive deep into Java
- Instructor: Ryan Gonzalez

## Section 1: Welcome to the course

1. Intro to the course

## Section 2: Introduction to Java

2. Download and install JDK

3. Where to write Java code
- Java file extension is `.java`
- File name must be the name of the class
```java
class FirstProgram {
  public static void main(String [] args) {
    System.out.println("Hello World");
  }
}
```
- javac FirstProgram.java
- java FirstProgram
  - `Hello World`

4. Download and Install IntelliJ IDEA

5. IntelliJ IDEA first look
- Class name begins as a Capital letter
- Method name as small letter

6. Organizing the code in java
- Folders (packages)
  - Files (classes)
    - Methods (block of code)
      - Statements (code)
- Main class may be located in the top, without Folders or Packages

7. What is compiled code?
- javac compiles the java code
- java asks JVM to run the byte code 

8. Compiled code in intellij IDEA
```bash
$ javac FirstProgram.java 
$ javap -p FirstProgram
Compiled from "FirstProgram.java"
class FirstProgram {
  FirstProgram();
  public static void main(java.lang.String[]);
}
```

9. JDK, JRE, and JVM in Depth
- JDK converts the code into byte code
  - Developers need this. Not customers.
- JVM is a part of JRE, and runs the byte code
  - Customers need JRE on their system

10. Is Java slow?
- JIT (Just-In-Time) Compiler optimizes byte code into machine code

11. print() and println() methods in Java
- System.out.println() vs System.out.print()
  - Per line or not

12. Introduction to Comments in Java
- Can generate documentation (like doxygen?)
- `//TODO` is recognized in IntelliJ IDEA

## Section 3: Primitive data types

13. Introduction to Variables
- Heap/stack
- Local variables: inside of a method
- Instance variables: inside of a class but outside of a method
- Class variables: static variable in a class
```java
public class Main {
  int myAge = 29; //instance variable
  static int salary = 12345; // static class variable
  public static void main(String[] args) {
    int age; // local variable
    age = 26;
    System.out.println(age);
  }
}
```
  - Main.java
```bash
FirstProject/
└── src
    ├── Main.class
    └── Main.java
```

14. Variables Naming Conventions
- Allowed variable name: age3, _age, $age
  - Avoid _ or $ anyway
- Not allowed variable name: 5age, #age, ...
- Local variable may over-ride instance/static class variables

15. Introduction To Primitive data types
- 8 primitive types
  - boolean
  - Numeric
    - char
    - Integral
      - Integer
        - byte
        - short
        - int
        - long
      - Floating point
        - float
        - double
- Reference types
  - TBD

16. Integral data types
- Integer
  - byte: 8bit, [-128,127]
  - short: 16bit, [-32768,32767]
  - int: 32bit, [-2147483648,2147483647]
  - long: 64bit, [-9223372036854775808,9223372036854775808]
    - Ex) `long myLong = 123456L;`
    - Ex) `long myLong = 123_456L;`
    - `_` is ignored
- Floating point
  - float: 32bit, 4bytes, 7 decimal digits
    - Ex) `float myFloat = 2.123456F;`
  - double: 64bit, 8bytes, 16 decimal digits
    - Ex) `double myDouble = 2.123456789D;`
    - Ex) `double myD = 1.234_567_890_1;`
 
17. Arithmetic operators part 1
- Type casting: `int x = (int) (y+z)`;
- +, -, *, /

18. Arithmetic operators part 2
- `%`: modulo operator or division remainder
- `++`: increment operator
- `--`: decrement operator

19. Assignment Operators
```java
int x=4;
int y=x;
x++;
```
- `y` is still 4, as a deep copy
- `+=`, `-=`, `*=`, `/=`

20. Booleans and relational operators
- `==`, `!=`, `>`, `>=`, `<`, `<=`, true, false

21. char Data type
- 16-bit unicode character with range [0-65,535]
```java
    /* Letter A
     decimal: 65
     unicode: U+0041
     */
    System.out.println( (char) 65);
    System.out.println( "\u0041");
```
  - Prints `A` 

22. Type inference
- `var` is similar to `auto` in cpp
```java
    var myD = 1.234_567_890_1;
    System.out.println(myD);  
```
- If `var` variable is not assigned, compiler yields error

23. Escape Sequences and printf method
- `''` for char and `""` for string
- "" inside of ""
  - Use `\"`
  - Ex) `System.out.println("I love \"food\"");`
- `\n` for the new line
- `\t` for the tab
- `\\` for the back slash
- `printf` provides formatted string
```java
    String name = "John";
    int year = 2023;
    double salary = 123.456;
    System.out.printf("My name is %s\nThe year is %d\nThe salary is %.2f\n", name, year, salary);
```

24. User Input
```java
import java.util.Scanner;
public class Main {
  public static void main(String[] args) {
    Scanner input = new Scanner(System.in);
    String name = input.next(); // will wait for an input at CLI
    System.out.println("Welcome " + name);
    input.close();
  }
}
```
- `Scanner input` at stack memory
- `Scanner object` created at heap memory

25. Scanner methods and more examples
- next(): up to white space
- nextLine(): up to the end of line
- nextInt(): Reads integer. If an integer is not found, yields java.util.InputMismatchException
- nextDouble(): Reads double. If a double is not found, yields java.util.InputMismatchException

26. Wrapper classes

| Primitive data types | Wrapper class|
|----------------------|--------------|
| int                  | Integer      |
| char                 | Character    |
| byte                 | Byte         |
| short                | Short        |
| long                 | Long         |
| float                | Float        |
| double               | Double       |
| boolean              | Boolean      |

```java
    String s1 = "123";
    int n1 = Integer.parseInt(s1);
    int n2 = Integer.parseInt("456");
    System.out.println(n1+n2);
    // boxing
    Integer x = 5;
    System.out.println(x);
    // unboxing
    int d = x;
    System.out.println(d);
```

## Section 4: Conditionals

27. if-else statement
```java
if () { 
  ...
} else if () {
  ...
} else {
  ...
}
```

28. Nested-if example
```java
import java.util.Scanner;
public class Main {
  public static void main(String[] args) {
    Scanner in = new Scanner(System.in);
    if (in.hasNextInt()) {
      int num = in.nextInt();
      if (num <0) {
        System.out.println("negative number");        
      } else {
        System.out.println("positive number");
      }
    } else {
      System.out.println("Enter integer");
    }
  }
}
```

29. if-else Statement (example)

30. Exercise - even or odd

31. Logical Operators (AND & OR)
- `&&`: AND
- `||`: OR

32. Exercise - fizz-buzz

33. Logical Not (negating boolean values)
- `!`: NOT

34. Ternary Operators (Elvis Operator)
- expression ? expression1: expression2
```java
import java.util.Scanner;
public class Main {
  public static void main(String[] args) {
    int grade = 7;
    var status = (grade > 7) ?  "succeed" : "failed";
    System.out.println(status);
  }
}
```

35. Exercise - insurance rate

36. Switch Statement
```java
switch(grade) {
  case 'A':
    System.out.println("Excellent");
    break;
  case 'B': // B or C is "Good"    
  case 'C':
    System.out.println("Good");
    break;
  default:
    System.out.println("Unknown");
}
```
- To return value:
```java
String result = switch (grade) {
    case 'A' -> "Excellent";
    case 'B','C'-> "Good";    
    default -> "Unknown";
};  
System.out.println(result);
```
- This is not supported in java 11
- Installing openjdk 17
  - wget https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed50d7138ee7f/8/GPL/openjdk-17.0.2_linux-x64_bin.tar.gz
  - Unpack and setup module environment for `JAVA_HOME`

37. Introduciton to debugging

## Section 5: Iterations in Java

38. While loop
```java
while(condition) {
  ...
}
```

39. do-while loop

40. for loop

41. Jump statements and conditional debugging

42. Exercise on iterations in Java part 1

43. Exercise on iterations in Java part 2

44. Nested loops

45. Nested loops exercise part 1

46. Nested loops exercise part 2

## Section 6: Arrays and Strings

## Section 7: OOP part 1 (classes, methods, static keyword)

## Section 8: OOP part 2 (inheritance and record class)

## Section 9: OOP part 3 (abstract classes and interfaces)

## Section 10: Exception Handling

## Section 11: Lambda expressions

## Section 12: Generics

## Section 13: Collections Framework

## Section 14: Stream API

## Section 15: Date time and math APIs

## Sectino 16: File manipulation (I/O and NIO)

## Section 17: Unit testing using Junit 5
