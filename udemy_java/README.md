## Summary
- Title: Java 17: Learn and dive deep into Java
- Instructor: Ryan Gonzalez

## Section 1: Welcome to the course

1. Intro to the course

## Section 2: Introduction to Java

2. Download and install JDK

3. Where to write Java code
- Java file extension is `.java`
- File name must be the name of the class
  - A single Java program can contain multiple classes, and one of those classes should be declared as public
  - The name of the Java file should be the same as the name of the public class in that file.
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
- Class variables: Local variable in a class
  - static or not
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
- Local variable may over-ride instance/(static) class variables

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

38. while loop
```java
while(condition) {
  ...
}
```

39. do-while loop
```java
do  {
  ...
} while (condition);
```

40. for loop
```java
for (int i=0;i<10;i++) {
  ...
}
```
- multiple indices are allowed
```java
import java.util.Scanner;
public class Main {
  public static void main(String[] args) {
  for (int i=0,j=0; i<10 && j< 5; i++,j++) 
    System.out.printf("i=%d, j=%d\n",i,j);
  }
}
```
- Prints up to i=4, j=4

41. Jump statements and conditional debugging
- break: terminates the loop
- continue: continue the loop while skipping the current

42. Exercise on iterations in Java part 1

43. Exercise on iterations in Java part 2

44. Nested loops

45. Nested loops exercise part 1

46. Nested loops exercise part 2

## Section 6: Arrays and Strings

47. Introduction to Arrays part 1
- Array
  - Homogeneous data
  - Zero-based index
  - Fixed size
  - Fast
  - Ex) `String[] students = {"John", "Tom", "Mary"};`
```java
import java.util.Arrays;
public class Main {
  public static void main(String[] args) {
    int[] myA1 = new int [4];
    int[] myA2 = {1, 4, 6, 7};
    System.out.println(Arrays.toString(myA2));
  }
}
```

48. Introduction to Arrays part 2
- Default array values are:
  - int: 0
  - double: 0.0
  - boolean: false
  - char: ''
  - String: null
- Or use Arrays.fill() to initialize
```java
import java.util.Arrays;
public class Main {
  public static void main(String[] args) {
    int[] myA1 = new int [4];
    Arrays.fill(myA1, 123);
    System.out.println(Arrays.toString(myA1));
  }
}
```

49. Traversing Arrays
- `myArray.length` for the size of an array
```java
import java.util.Arrays;
public class Main {
  public static void main(String[] args) {
    String[] myS = {"John", "Tom", "Mary"};
    // indexed for
    for (int i=0; i< myS.length; i++) {
      System.out.println(myS[i]);
    }
    // for-each
    for (var a: myS) {
      System.out.println(a);
    }
  }
}
```

50. Exercise on Arrays

51. Two-dimensional Arrays
```java
import java.util.Arrays;
public class Main {
  public static void main(String[] args) {
    int[][] my2D = new int[2][4];
    my2D[0][1] = 1;
    my2D[1][3] = 3;
    for(int i=0; i < my2D.length; i++) {
      for (int j=0; j< my2D[i].length;j++) {
        System.out.println(my2D[i][j]);
      }
    }
    for(var a: my2D) {
      for (var b: a) {
        System.out.println(b);
      }
    }
  }
}
```
- May have different sizes at 2nd array
```java
import java.util.Arrays;
public class Main {
  public static void main(String[] args) {
    int[][] my2D = { {1,2,3}, {4,5,6,7}};
    for(int i=0; i < my2D.length; i++) {
      for (int j=0; j< my2D[i].length;j++) {
        System.out.println(my2D[i][j]);
      }
    }
    for(var a: my2D) {
      for (var b: a) {
        System.out.println(b);
      }
    }
  }
}
```

52. Two-dimensional arrays exercise

53. Strings Immutability
- How Java handles string: **String pool**
```java
import java.util.Arrays;
public class Main {
  public static void main(String[] args) {
    String s1 = "hello";
    String s2 = "hello";
    String s3 = "hello";
    System.out.println(s1);
  }
}
```
- s1 points the object "hello" in the string pool of heap
- s2 points the same object of s1
- s3 points the same object of s1
- Can save lots of heap memory

54. new String
- Static String => string pool
- new String() => class object
``` java
import java.util.Arrays;
public class Main {
  public static void main(String[] args) {
    String s1 = "hello"; // -> string pool
    String s2 = "hello"; // -> string pool
    System.out.println(s1 == s2); //true
    String s3 = new String("hello"); // -> new class object
    String s4 = new String("hello"); // -> new class object
    System.out.println(s1 == s3); // false
    System.out.println(s3 == s4); // false
    System.out.println(s1.equals(s2)); // true
    System.out.println(s1.equals(s3)); // true
    System.out.println(s3.equals(s1)); // true
    System.out.println(s3.equals(s4)); // true
  }
}
```
- In order to compare the string value, not the heap address, use `.equals()` method

55. Strings are arrays of characters

56. Strings Traversal
- split(): Similar to python split()
- substring(): in terms of array index
- toCharArray
```java
import java.util.Arrays;
public class Main {
  public static void main(String[] args) {
    String s1 = "java is awesome";
    //           0123456789
    System.out.println(s1.substring(8)); //[8,]
    System.out.println(s1.substring(5,9)); // [5,8]
    var words = s1.split(" ");
    System.out.println(words.length);
    char[] myC = s1.toCharArray();
    System.out.println(myC.length);
  }
}
```

57. Strings Methods
- Empty string ("") is not same to null string (null)
  - Ref: https://stackoverflow.com/questions/4802015/difference-between-null-and-empty-java-string
  - emptyS == nullS will return false
  - emptyS.equas(nullS) will return false
  - Comparison of emptyS.length() or nullS.length() may work
    - Or use substring(0,1)
- length()
- isEmpty()
- isBlank()
- trim(): only space in the head/tail. Spaces within b/w char are not removed
- concat()
- toUpperCase()
- toLowerCase()
- startsWith()
- endsWith()
- replaceFirst()
- contains()
```java
import java.util.Arrays;
public class Main {
  public static void main(String[] args) {
    String s1 = null; 
    String s2 = "";
    System.out.println(s1 == s2); // false
    //System.out.println(s1.isEmpty()); // cannot compile as null doesn't have a method of isEmpty()
    System.out.println(s2.isEmpty()); // true
    System.out.println(s2.isBlank()); // true
    String s3 = "  j a va ";
    String s4 = " is awesome";
    System.out.println(s3.trim().concat(s4)); // j a va is awesome
    System.out.println(s4.contains("awe")); // true
  }
}
```

58. StringBuilder
- For the long string, we may use Stringbuilder class, which is mutable

## Section 7: OOP part 1 (classes, methods, static keyword)

59. Classes and Objects
- Each object has
  - Identity: id, hash code, memory address
  - State: fields, variables
  - Behavior: methods
- Person.java
```java
public class Person {
  String name;
  public void walk() {
    System.out.println(name + " walking");
  }
}
```
- Main.java
```java
public class Main {
  public static void main(String[] args) {
    Person tom = new Person();
    tom.name = "Tom";
    System.out.println(tom.hashCode());
    tom.walk();
    Person john = new Person();
    john.name = "John";
    john.walk();
  }
}
```
- javac Person.java
- javac Main.java
- java Main

60. Methods
- A reusable block of code that performs a specific task and only runs when called/invoked
- Can return data type or void

61. Method Signature and Method overload
- Method signature: method name, parameters count, parameter type
- A class CANNOT have two methods with the same signature
- Methods may have the same name while they have:
  - different number of parameters
  - different types of parameters
  - Method overloading (static polymorphism)
- Arbitrary Number of Arguments
  - Using three dots
  - `public void add(int... numbers){...}`
  - When calling the function, we may send many of int arguments
    - Ex) `myObject.add(1,4,7,8,2,3);`

62. Pass by Value vs. Pass by Reference
- Main.java:
```java
import java.util.Arrays;
public class Main {
  public static void main(String[] args) {
    int x = 123;
    int[] A = {1, 10, 100};
    Pass myP = new Pass();
    myP.byValue(x); // int as by value
    System.out.println("In main, x = " + x);
    myP.byRef(A); // Array is sent as by reference
    System.out.println("In main, A = " + Arrays.toString(A));
  }
}
```
- Pass.java
```java
import java.util.Arrays;
public class Pass {
  public static void byValue(int x) {
    x ++;
    System.out.println("In byValue: x=" +x);
  }
  public static void byRef(int[] x) {
    for(int i=0;i<x.length;i++) {
      x[i]++;
    }
    System.out.println("In byRef: A=" + Arrays.toString(x));
  }
}
```

63. Class Constructor
- Constructors have the same name of the containing class and there is no return type
- Default constructor is made by compiler when there is no constructor defined

64. Access Modifiers
- Organizing the code
  - Folders (packages)
    - Files (classes)
    - Methods (blocks of code)
      - Statements (code)
- Default access modifier (package private): members are accessible within the same package
- Private access modifier: Members are accessible only within the class
- Public access modifier: Members are accesible anywhere in the program
- Protected access modifier: Members are accessible within the class/derived classes

65. this keyword
```java
public class Rectangle {
  private double length, width;
  public Rectangle() {// default constructor when no argument is given
    this(2,1); 
  }
  public Rectangle (double newLength, double newWidth) {
    this.length = newLength;
    this.width = newWidth;
  }
  double getArea() {
    return this.length*this.width; // length*width works as class variables but using this.* is more clear
  }
}
```

66. Static variable
- For a given class, the static variable exists only once among many class objects
- Default class variable is package-private
- Static class variable is public

67. Static Block
- `static { ...}` runs only once when the class is called
  - Regardless of calling constructor or not
  - Static initialization
- Person.java
```java
public class Person {
  static int counter;
  static {
    counter = 5; // Runs only once when the class is called (regardless of construction or not)
  }
}
```
- Main.java
```java
public class Main {
  public static void main(String[] args) {
    System.out.println(Person.counter); // 5 as it is called first time
    Person.counter++;
    System.out.println(Person.counter); // 6 as it is ++ above
    Person p = new Person();
    System.out.println(Person.counter); // 6 as counter is not changed
  }
}
```

68. Static Method
- Non-static variable cannot be used in static methods
- `this` cannot be used in static methods

69. Static Nested Classes and Inner-classes
- Local class: new class inside of a method
- Inner class: new class inside of a class
- Anonymous class: Overrides a class method once or within a scope
  - `@override`
  - `super`

## Section 8: OOP part 2 (inheritance and record class)

70. Introduction to inheritance

71. Protected access modifier
- Protected variables in a base class can be accessible in derived classes
- `private` variables are not inherited

72. Method overriding (runtime polymorphism)
- Child constructor will call the parent constructor first
- If there is a difference in arguments of the constructor, `super` may be used to inherit
- `@Override` to over-ride the method of Parent class
- Main.java:
```java
public class Main {
  public static void main(String[] args) {
    Child child = new Child();
    child.print();
  }
}
```
- Parent.java:
```java
public class Parent {
  public Parent(String name) {
    System.out.println("Parent constructor: "+name);
  }
  void print() {
    System.out.println("print in Parent");
  }
}
```
- Child.java:
```java
public class Child extends Parent {
  public Child() {
    super("John"); // without this, compile fails
    System.out.println("Child constructor");
  }
  @Override
  void print() {
    //super.print(); this will run print() of the Parent
    System.out.println("print in Child");
  }
}
```
- Result:
```bash
$ java Main
Parent constructor: John
Child constructor
print in Child
```

73. final and sealed Keywords
- `final` in variable declares the variable as CONSTANT
  - Use CAPITAL for the variable name
  - Initialize in the constructor
  - Cannot be changed in the run-time
- `final class` cannot inherited
- `sealed class Parent permits Child1` allows Child1 class only to get inheritance

74. Encapsulation (getters and setters)
- Encapsulation in OOP means the ability to hide data and behavior from users
- Enable private attribute for member data
- Enable public methods to set/get data
- Benefit of encapsulation over public member data: https://stackoverflow.com/questions/18845658/what-is-the-advantage-of-having-a-private-attribute-with-getters-and-setters

75. Object Methods (getClass)
- In java, all classes inherits from Object class
```java
import java.lang.reflect.Method;
public class Main {
  public static void main(String[] args) {
    Child child = new Child();
    child.print();
    System.out.println(child.getClass().getSimpleName()); // prints Child
    for (var method: child.getClass().getMethods()) {
      System.out.println(method); // prints all method names
    }
  }
}
```
- Result:
```bash
$ java Main
Parent constructor: John
Child constructor
print in Parent
print in Child
Child
public final void java.lang.Object.wait(long,int) throws java.lang.InterruptedException
public final void java.lang.Object.wait() throws java.lang.InterruptedException
public final native void java.lang.Object.wait(long) throws java.lang.InterruptedException
public boolean java.lang.Object.equals(java.lang.Object)
public java.lang.String java.lang.Object.toString()
public native int java.lang.Object.hashCode()
public final native java.lang.Class java.lang.Object.getClass()
public final native void java.lang.Object.notify()
public final native void java.lang.Object.notifyAll()
```

76. Object Methods (hashCode, equals, toString)

77. Record class (data carrier)
- Java doesn't support multiple inheritances
- `record` class is final
  - Can implement interface

## Section 9: OOP part 3 (abstract classes and interfaces)

78. The Diamond problem

79. Abstract classes and Methods
- Abstract class: can be inherited but cannot be instantiated
- Abstract method: a method using abstract keywords inside of an abstract class
  - Inheriting class must over-ride the existing abstract methods

80. Interfaces in Java
- Enables multiple inheritances in Java, using `implements` keyword
  - Not `extends`
- Types of interfaces
  - Empty
  - One abstract method
  - More than one abstract method
- No constructor
- Cannot be instantiated
- All fields are public/static/final
- All field data (member data) must be initialized

81. Multiple inheritance using interfaces
- A derived classs uses `implements` interface
- The interface `extends` multiple classes

82. Interface vs. abstract class

| Abstract class | Interface |
|----------------|-----------|
| A single inheritance only but multiple interfaces | Multiple inheritances. Extends one interface only|
| Can contain field data and constants with all access modifiers| Only constants (public static final) |
| Can have a constructor | no constructor allowed|
| Methods can be public/private/protected| Methods can be public or private|
| Provides `Is A` relation | Provides `Has A` relation|

83. OOP Principles
- Abstraction: abstract class and interface
- Encapsulation: setter & getter
- Inheritance: recycle code
- Polymorphism: method overloading/overriding

84. Object Oriented Principles

85. instanceof operator
```java
public class Main{
  public static void main(String[] args) {
    System.out.println("Hello" instanceof String); // true
    Parent p1 = new Parent();
    Child  c1 = new Child();
    System.out.println(p1 instanceof Child); // false
    System.out.println(c1 instanceof Parent); // true
    System.out.println(c1 instanceof Child); // true
  }
}
```

86. Enumerations
- Make a new file with the name of `public enum`
- ENUM can have a constructor
  - Each enum item will run the constructor when called
- Day.java:
```java
public enum Day {
  MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY; 
}
```
- Main.java:
```java
import java.util.Arrays;
public class Main{
  public static void main(String[] args) {
    System.out.println(Day.FRIDAY); // FRIDAY
    System.out.println(Day.FRIDAY.ordinal()); // 4 
    System.out.println(Arrays.toString(Day.values()));
  }
}
```

## Section 10: Exception Handling

87. Introduction to Exception Handling
- Some methods from the throwable class
  - getMessage()
  - getCause()
  - printStackTrace()
- Type of exception
  - Runtime exception
    - NullPointer Exception
    - NumberFormatException
    - IndexOutOfBoundException
    - ...
  - Other exceptions
    - IOException
    - SQLException

88. try-catch and Checked Exceptions
```java
public class Main {
  public static void main(String[] args) {
    try {
      System.out.println("before division by zero");
      System.out.println(5/0);
      System.out.println("after division by zero"); // not executed as exception happened above
    } catch (ArithmeticException e) {
      System.out.println("Exception found");
    } finally {
      System.out.println("final block"); // exception or not, executed always
    }
  }
}
```
89. try-with-resource and Exception Propagation

90. Custom Exception
- Using Exception class

## Section 11: Lambda expressions

91. Introduction to Lambda expressions
- Anonymous method
  - An implementation for the only abstract method in the functional interface
  - A position holder for an empty function, which can be defined as ad-hoc along the code
- Form
  - parameter -> expression
  - (parameter) -> expression
  - (parameter, parameter) -> expression
  - Ex)
    - `String sayHello() { return "Hello"; }` == `()->"Hello";`
    - `double divByTwo(double x) { return x/2; }` == `(num) -> num/2;`

92. Examples on Lambda expressions
- Passing Lambda as parameter
```java
interface Calculator {
  double calc(double x, double y);
}
public class Main {
  static void myPrint(Calculator calc1, double x, double y) {
    double res = calc1.calc(x,y);
    System.out.println(res);
  }
  public static void main(String[] args) {
    Calculator ftn1 = (x,y) -> x+y;
    myPrint(ftn1, 4, 5);
    myPrint((x,y)->x+y, 4,5);
    Calculator ftn2 = (x,y)-> x-y;
    myPrint(ftn2, 10, 5);
    myPrint((x,y)->x-y,10,5);
  }
}
```

93. Advangtage of Lambda expressions and variable capturing
- Only final local variables can be used inside of Lambda
  - Regular local variable cannot be used

94. Method Reference
- `MyInterface myInt = Class_name::method_name;`
- `MyInterface myInt = Class_object_name::method_name;`

## Section 12: Generics

95. Introduction (Why Generics)
- Resolves type safety problem
- MyObjectArray.java:
```java
public class MyObjectArray<T>{
  T[] array;
  public MyObjectArray(T[] array) {
    this.array = array;
  }
  public T[] getArray() {
    return array;
  }
  public T getElement(int idx) {
    return array[idx];
  }
  public void setArray(T[] array) {
    this.array = array;
  }
}
```
- Main.java:
```java
import java.util.Arrays;
public class Main {
  public static void main(String[] args) {
    Integer[] numbers = {1,2,3};
    MyObjectArray<Integer> myInt = new MyObjectArray<>(numbers);
    System.out.println(Arrays.toString(myInt.getArray()));
  }
}
```
  - Will work for Double/String as well
  - No primitive data type like int, double
  - Ref: https://stackoverflow.com/questions/5000521/java-generics-int-to-integer

96. Introduction to Generics

97. Generic Method and Generic Interface
- `<T extends Number>`: any type from Number only. String data will not be compiled
- MinMax.java:
```java
public interface MinMax<T extends Number> {
  T min(T x, T y);
  T max(T x, T y);
}
class MyClass implements MinMax<Double> {
  @Override
  public Double min(Double x, Double y)  {
    return x>y ? y: x;
  }
  @Override
  public Double max(Double x, Double y)  {
    return x>y ? x: y;
  }
}
/* Not compiled due to extends Number above
class MyClass2 implements MinMax<String> {
  @Override
  public String min(String x, String y)  {
    return x>y ? y: x;
  }
  public String max(String x, String y)  {
    return x>y ? x: y;
  }
}
*/
```

98. Wildcards
- ?:Presents unknown type
- Unbounded type : `MyArray<?>`
- Lowerbound wild card: `<? Super ...>`
- Upperbound wild card: `<? extends ...>`

## Section 13: Collections Framework

99. Introduction to Collections
- Containers for data
- Iterable
  - Collection
    - List
      - ArrayList
      - LinkedList
      - Vector - Stack
    - Queue
      - Deque
        - ArrayDeque
    - Set - SortedSet - Tree Set
      - HashSet
      - LinkedHashSet
- Map
  - HashTable
  - HashMap - LinkedHashMap
  - SortedMap - NavigableMap - TreeMap

100. Introduction to ArrayList
- Iterable - Collection - List - ArrayList
```java
import java.util.ArrayList;
import java.util.List;
public class Main {
  public static void main(String[] args) {
    List<Integer> numbers = new  ArrayList<>();
    numbers.add(99);
    numbers.add(null);
    numbers.add(5);
    System.out.println(numbers);
  }
}
```
- ArrayList
  - Maintains elements insertion order
  - Stores null and duplicate elements
  - Can insert elements at any location
  - Zero-based index
  - Allwos fast random access
  - ArrayList doesn't store primitive data types
- ArrayList under the hood
  - Default initial size = 10
  - Threshold/minCapacity/loadFactor = 0.75
    - If data is larger than 75% of the array, java will increase the array
    - Insertion/removal of mid-elements is very expensive due to re-arrangement of contiguous memory

101. Collection interface methods
```java
import java.util.ArrayList;
import java.util.List;
import java.util.Collection;
public class Main {
  public static void main(String[] args) {
    Collection<Integer> c1 = new ArrayList<>();
    c1.add(1);
    Collection<Integer> c2 = List.of(2,7,1,3,0);
    Collection<Integer> c3 = new ArrayList<>() {{ add(4); add(5); add(6);}};
    c1.addAll(c2); // Concatenating Lists
    System.out.println(c1);
  }
}
```
- remove(): Removes the corresponding elements
- removeAll(): Removes elements within the argument collection
- retainAll(): Removes all except the elements of argument collection
- isEmpty(): true or false
- size(): returns the size

102. List interface methods
```java
import java.util.ArrayList;
import java.util.List;
public class Main {
  public static void main(String[] args) {
    List<String> l1 = new ArrayList<>();
    l1.add("first");
    l1.add("second");
    l1.add(null);
    System.out.println(l1);
    System.out.println(l1.isEmpty());
    System.out.println(l1.size());
    System.out.println(l1.indexOf("second"));
    System.out.println(l1.indexOf("Second")); // returns -1
  }
}
```
- sort() with null crashes
  - To sort with null, use Comparator.nullsLast or nullsFirst such as: `l1.sort(Comparator.nullsLast(Comparator.naturalOrder()));`

103. Sorting List
```java
import java.util.ArrayList;
import java.util.List;
import java.util.Comparator;
public class SortingStrings  {
  public static void main(String[] args) {
    List<String> l1 = new ArrayList<>(List.of("B123", "Daw", "aZZ"));
    System.out.println("Before sorting: " + l1);
    l1.sort(Comparator.naturalOrder()); // sort by ASCII number
    l1.sort(String.CASE_INSENSITIVE_ORDER); // case insensitive order
    Comparator<String> lc = Comparator.comparingInt(String::length); // comparator by String length
    Comparator<String> lc2 = (s1,s2)->Integer.compare(s1.charAt(1), s2.charAt(1)); // sort by 2nd Char
    l1.sort(lc2);
    System.out.println("After sorting" + l1);
  }
}
```

104. Example on Using Array List

105. Introduction to LinkedList
- Fast insertion/removal than ArrayList
- Queue
   - FIFO

106. LinkedList
- Queue
  - offer(), poll(), peek(): doesn't yield error even when empty
  - add(), remove(), element()
- Deque
  - offerFirst(), offerLast(), pollFirst(), pollLast(), peekFirst(), peekLast()
  - addFirst(), addLast(), removeFirst(), removeLast(), getFirst(), getLast()
  
107. ArrayDeque (pronounced Array Deck)
- Fast add/remove from both sides
- ArrayDequeAsStack.java:
```java
import java.util.ArrayDeque;
public class ArrayDequeAsStack {
  // Insertion from Head:  push offerFirst  addFirst
  // remove from head:     pop  poll pollFirst  removeFirst
  // Examine from head:    peek  peekFirst  getFirst
  public static void main(String[] args) {
    ArrayDeque<String> arrayDeque = new ArrayDeque<>();
    arrayDeque.push("Book1");
    arrayDeque.push("Book2");
    arrayDeque.offerFirst("Book3");
    arrayDeque.addFirst("Book4");
    String element = arrayDeque.getFirst();
    System.out.println(element);
    System.out.println(arrayDeque);
  }
}
```
- ArrayDequeAsQueue.java:
```java
import java.util.ArrayDeque;
public class ArrayDequeAsQueue {
  // Insertion from tail:  offer offerLast  addLast
  // remove from head:     poll  pollFirst  removeFirst
  // Examine from head:    peek  peekFirst  getFirst
  public static void main(String[] args) {
    ArrayDeque<String> arrayDeque = new ArrayDeque<>();
    arrayDeque.offer("customer1");
    arrayDeque.offerLast("customer2");
    arrayDeque.addLast("customer3");
    arrayDeque.offer("customer4");

    String element = arrayDeque.removeFirst();
    System.out.println(element);

    System.out.println(arrayDeque);
  }
}
```

108. PriorityQueue
- Main.java:
```java
import java.util.PriorityQueue;
public class Main {
  // retrieval: poll, remove, peek, element
  public static void main(String[] args) {
    PriorityQueue<Integer> numbers = new PriorityQueue<>();
    numbers.add(99);
    numbers.add(45);
    numbers.add(1);
    System.out.println(numbers.poll());
    System.out.println(numbers.poll());
    System.out.println(numbers.poll());
    PriorityQueue<String> letters = new PriorityQueue<>();
    letters.add("z");
    letters.add("s");
    letters.add("a");
    System.out.println(letters.poll());
    System.out.println(letters.poll());
    System.out.println(letters.poll());
  }
}
```

109. Introduction to HashMap
- key and value
- put(), putAll(), putIfAbsent(), computeIfAbsent(), computeIfPresent(), compute()
  - If the same key is used, put() will overwrite the value
  - In order to avoid overwriting, may use putIfAbsent()
- Example1.java:
```java
import java.util.*;
public class Example1 {
  // adding / updating entry to map entry set:
  // put, putAll, putIfAbsent, computeIfAbsent, computeIfPresent, compute
  public static void main(String[] args) {
    Map<Integer, String> students = new HashMap<>();
    students.put(1, "John");
    students.put(2, "Tom");
   Map<Integer, String> otherMap = new HashMap<>(Map.of(3, "Samantha", 4, "Olivia"));
    students.putAll(otherMap);
    students.putIfAbsent(1, "Ryan");
    students.computeIfAbsent(1, k-> {
      System.out.println(k);
      return  "Ryan";
    });
    students.computeIfPresent(8, (k,v) -> v.toUpperCase()+"!!");
    students.compute(4, (k, v)-> {
      if(students.containsKey(k)){
        return v.toUpperCase()+"**";
      }
      return "Ryan";
    });
    System.out.println(students);
  }
}
```

110. HashMap

111. Why is it called HashMap and Example on HashMap
- Array of nodes/buckets/bins
  - Initial capacity of nodes = 16
  - key->hashing mechanism -> hashkey
- Hash collision: two different keys generate the same hash key

112. LinkedHashMap
- Doubly linked list to maintain insertion order
- Contains unique keys
- May have one or more null values
- Maintain insertion order or access order (LRU)
- Max entries option

113. TreeMap
- Key/value in sorted order

114. HashSet
- Set: unique elements only
- Internally uses HashMap
  - Repeated values are stored with different keys

115. LinkedHashSet
- Unique elements only in insertion order

116. TreeSet
- Unique elements in natural order
- No null

## Section 14: Stream API

117. Introduction to Java Stream API
- A stream is a sequence of data that supports multiple operations like filtering and mapping
- Streams don't change the original structure
- Streams don't hold data - streams process data
- Stream API pipeline
  - Stream source: Array, ArrayList, HashMap, ...
  - Stream operations: map(), distinct(), peek(), limit(), reduce(), filter(), ...
  - Results: Number, Array, ArrayList, HashMap, ...
- All streams are lazy
  - Intermediate operations are not evaluated until terminal operation is invoked.
- parallelStream can take benefit from multiple-cores

118. Introduction to Terminal Operations and Optional Keyword
- Optional keyword: may not contain values
  - OptionalInt, OptionalLong, OptionalDouble, Optional<Integer>, ...
- Terminal operations
  - findAny(), findFirst(), allMatch(), anyMatch(), noneMatch(), ...
  - `intList.parallelStream().findAny()` may return different values every time due to paralle operations

119. More on Terminal operations methods (reduce)
- Using reduce()
- ReduceExample1.java:
```java
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
public class ReduceExample1 {
  public static void main(String[] args) {
    List<Integer> list = new ArrayList<>(List.of(1, 2, 3, 4));
   Optional<Integer> sum = list.stream().reduce((acc, n)-> {
     System.out.println("acc: " + acc + ", n: " + n);
    return acc * n;
   });
    System.out.println(sum);
  Integer result =   list.stream().reduce(10, Integer::sum);
  }
}
```
- String concatenation works in the similar way

120. Introduction to Collectors
- Reduction operations using Collectors
```java
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
public class AccumulatorsExample {
  public static void main(String[] args) {
    List<String> list = new ArrayList<>(List.of("java", "is", "awesome", "ai", "abab", "python"));
    String joinedString = list.stream().collect(Collectors.joining(" ", ">>> ", " <<<"));
   Map<String, Integer> map = list.stream().collect(Collectors.toMap(s1-> s1, s1-> s1.length()));
   List<Integer> list1 = Stream.of(1, 2, 3).collect(Collectors.toUnmodifiableList());
    Map<Integer, Set<String>> collect = list.stream().collect(Collectors.groupingBy(String::length, Collectors.toSet()));
    Map<Boolean, List<String>> partitioned = list.stream().collect(Collectors.partitioningBy(s-> s.contains("a")));
    System.out.println(partitioned);
  }
}
```

121. Intermediate Operations
- Chaining methods

## Section 15: Date time and math APIs

122. Introduction to Date Time APIs Part 1
- Before Java8
  - SimpleDateFormat class (java.text)
    - Concrete class for formatting (date to text) and parsing (text to date) dates
  - Calendar class (java.util)
    - Provides many methods to get detailed information about a specific date
  - Date class (java.util)
    - Creates objects represents a specific instant in time with millisecond precision
- Month notation
  - MM: 12
  - MMM: Dec
  - MMMM: December
- Date
  - E: Tue
  - EEEE: Tuesday
- Java date starts from 1900
- Unix epoch: 1970-01-01
```java
import java.text.SimpleDateFormat;
import java.util.Date;
public class JavaSE7APIPart1 {
    public static void main(String[] args){
        SimpleDateFormat simpleDateFormat =
            new SimpleDateFormat("EEEE, MM/dd/yyyy HH:mm:ss.SS Z");
        Date today = new Date();
        System.out.println(simpleDateFormat.format(today));

    }
}
```
- Results: `Saturday, 12/03/2022 15:47:15.789 -0500`

123. Introduction to Date Time APIs Part 2
- Date::getDate(), getYear(), getDay() are deprecated
- Date::getTime(): milliseconds since 1970:01:01:00:00
```java
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;
/*
    Date starts from 1900
    Unix epoch:  Time starts from 1970-1-1
 */
public class DateCalendarExamples{
    public static void main(String[] args) throws ParseException {
        SimpleDateFormat simpleDateFormat =
            new SimpleDateFormat("dd/MM/yyyy");
        Date today = new Date();
        System.out.println(simpleDateFormat.format(today));
        System.out.println(today.getDate());
        System.out.println(today.getYear());
        System.out.println(today.getDay());
        System.out.println(today.getMonth());
        System.out.println(today.getTime());
        Date date = simpleDateFormat.parse("29/12/1979");
        System.out.println(date);
        //-------------------------------------------
        System.out.println(new Date());
        Calendar calendar = new GregorianCalendar();
        System.out.println(calendar.get(Calendar.ERA));
        System.out.println(calendar.get(Calendar.YEAR));
        System.out.println(calendar.get(Calendar.MONTH)); // index
        System.out.println(calendar.get(Calendar.WEEK_OF_YEAR));
        System.out.println(calendar.get(Calendar.DATE));
        System.out.println(calendar.get(Calendar.DAY_OF_MONTH));
        System.out.println(calendar.get(Calendar.DAY_OF_YEAR));
        System.out.println(calendar.get(Calendar.DAY_OF_WEEK)); // not index
        System.out.println(calendar.get(Calendar.HOUR));
        System.out.println(calendar.get(Calendar.HOUR_OF_DAY));
        System.out.println(calendar.get(Calendar.MINUTE));
        System.out.println(calendar.get(Calendar.SECOND));
        System.out.println(calendar.get(Calendar.MILLISECOND));
        System.out.println(calendar.get(Calendar.AM_PM)); // 0: AM, 1: PM
    }
}
```

124. LocalDate Part 1
```java
import java.time.*;
import java.time.temporal.ChronoField;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalAdjusters;
public class LocalDateExample {
    public static void main(String[] args) {
        LocalDate today = LocalDate.now();
        System.out.println("today: " + today);
        LocalDate hiringDate = LocalDate.of(2017, Month.DECEMBER, 29);
        System.out.println("hiringDate: " + hiringDate);
        LocalDate futureDate = LocalDate.parse("2027-12-27");
        System.out.println("futureDate: " + futureDate);
        System.out.println(today.getYear());
        System.out.println(today.lengthOfYear());
        System.out.println(today.get(ChronoField.YEAR));
        System.out.println(today.getMonth());
        System.out.println(today.getMonthValue());
        System.out.println(today.lengthOfMonth());
        System.out.println(today.get(ChronoField.MONTH_OF_YEAR));
        System.out.println(today.getDayOfWeek());
        System.out.println(today.get(ChronoField.DAY_OF_WEEK));
        System.out.println(today.getDayOfMonth());
        System.out.println(today.get(ChronoField.DAY_OF_MONTH));
        System.out.println(today.getDayOfYear());
        System.out.println(today.get(ChronoField.DAY_OF_YEAR));
        System.out.println(today.isLeapYear());
        boolean before = today.isBefore(hiringDate);
        System.out.println(before);
        boolean after = today.isAfter(hiringDate);
        System.out.println(after);
        LocalDateTime atStartOfDay = today.atTime(9, 15);
        atStartOfDay = today.atStartOfDay();
        System.out.println( atStartOfDay);
        LocalDate with = today.with(TemporalAdjusters.lastDayOfYear());
        System.out.println(with);
        LocalDate with2 = today.with(TemporalAdjusters.firstInMonth(DayOfWeek.MONDAY));
        System.out.println( with2);
        //// plus and minus
        today.minusDays(10);
        today.minusMonths(2);
        today.minusYears(2);
        //// number and temporal unit
        today.minus(12, ChronoUnit.MONTHS);
        //// temporal amount
        today.minus(Period.ofDays(12));
        //// years of experience
        Period period = Period.between(hiringDate, today);
        System.out.println("period " + period);
        System.out.println(period.getYears());
        //// years of experience
        LocalDate experience = today
            .minusYears(hiringDate.getYear())
            .minusMonths(hiringDate.getMonthValue())
            .minusDays(hiringDate.getDayOfMonth());
        System.out.println("experience: " + experience);

        System.out.println(LocalDate.MIN);
        System.out.println(LocalDate.MAX);
    }
}
```

125. LocalDate Part 2
```java
import java.time.*;
import java.time.temporal.ChronoField;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalAdjusters;

public class LocalDateExample {
    public static void main(String[] args) {
        LocalDate today = LocalDate.now();
        System.out.println("today: " + today);
        LocalDate hiringDate = LocalDate.of(2017, Month.DECEMBER, 29);
        System.out.println("hiringDate: " + hiringDate);
        LocalDate futureDate = LocalDate.parse("2027-12-27");
        System.out.println("futureDate: " + futureDate);
        System.out.println(today.getYear());
        System.out.println(today.lengthOfYear());
        System.out.println(today.get(ChronoField.YEAR));
        System.out.println(today.getMonth());
        System.out.println(today.getMonthValue());
        System.out.println(today.lengthOfMonth());
        System.out.println(today.get(ChronoField.MONTH_OF_YEAR));
        System.out.println(today.getDayOfWeek());
        System.out.println(today.get(ChronoField.DAY_OF_WEEK));
        System.out.println(today.getDayOfMonth());
        System.out.println(today.get(ChronoField.DAY_OF_MONTH));
        System.out.println(today.getDayOfYear());
        System.out.println(today.get(ChronoField.DAY_OF_YEAR));
        System.out.println(today.isLeapYear());
        boolean before = today.isBefore(hiringDate);
        System.out.println(before);
        boolean after = today.isAfter(hiringDate);
        System.out.println(after);
        LocalDateTime atStartOfDay = today.atTime(9, 15);
        atStartOfDay = today.atStartOfDay();
        System.out.println( atStartOfDay);
        LocalDate with = today.with(TemporalAdjusters.lastDayOfYear());
        System.out.println(with);
        LocalDate with2 = today.with(TemporalAdjusters.firstInMonth(DayOfWeek.MONDAY));
        System.out.println( with2);
        //// plus and minus
        today.minusDays(10);
        today.minusMonths(2);
        today.minusYears(2);
        //// number and temporal unit
        today.minus(12, ChronoUnit.MONTHS);
        //// temporal amount
        today.minus(Period.ofDays(12));
        //// years of experience
        Period period = Period.between(hiringDate, today);
        System.out.println("period " + period);
        System.out.println(period.getYears());
        //// years of experience
        LocalDate experience = today
            .minusYears(hiringDate.getYear())
            .minusMonths(hiringDate.getMonthValue())
            .minusDays(hiringDate.getDayOfMonth());
        System.out.println("experience: " + experience);
        System.out.println(LocalDate.MIN);
        System.out.println(LocalDate.MAX);
    }
}
```

126. LocalTime and LocalDateTime
- LocalTime example:
```java
import java.time.Duration;
import java.time.LocalTime;
import java.time.temporal.ChronoField;
import java.time.temporal.ChronoUnit;
public class LocalTimeExample {
    public static void main(String[] args) {
        LocalTime now = LocalTime.now();
        System.out.println("now: " + now);
        LocalTime workStart = LocalTime.parse("08:00");
        System.out.println("work start: " + workStart);
        LocalTime workEnd = LocalTime.of(16, 0);
        System.out.println("work end: " + workEnd);
        System.out.println(now.getHour());
        System.out.println(now.getMinute());
        System.out.println(now.getSecond());
        System.out.println(now.get(ChronoField.AMPM_OF_DAY)); // 0: AM, 1: PM
        System.out.println(LocalTime.MIN);
        System.out.println(LocalTime.MAX);
        // calculating difference between times
        System.out.println("Working hours: " + (workStart.until(workEnd, ChronoUnit.HOURS)));
        System.out.println("Working hours: " + ChronoUnit.HOURS.between(workStart, workEnd));
        System.out.println("Working hours: " + Duration.between(workStart, workEnd));
        // plus and minus methods
        now.plus(1, ChronoUnit.HOURS);
        now.plusHours(1);
        now.plusMinutes(5);
        // before and after methods
        boolean isBefore = LocalTime.parse("09:40").isBefore(LocalTime.parse("09:35"));
        System.out.println(isBefore);
        System.out.println(now.truncatedTo(ChronoUnit.HOURS));
    }
}
```
- LocalDateTime example:
```java
mport java.time.LocalDate;
import java.time.LocalTime;
import java.time.Month;
import java.time.format.DateTimeFormatter;
import java.time.format.FormatStyle;
import java.util.Locale;
import java.time.LocalDateTime;
public class LocalDateTimeExample {
    public static void main(String[] args) {
        LocalDateTime now = LocalDateTime.now();
        System.out.println(now);
        //String isoFormat = now.format(DateTimeFormatter.ISO_TIME);
        //System.out.println(isoFormat);
        //
        //String patternFormat = now.format(DateTimeFormatter.ofPattern("E, MMM dd hh:mm a"));
        //System.out.println(patternFormat);
        //String italianLocale = now.format(DateTimeFormatter.ofLocalizedDateTime(FormatStyle.MEDIUM).withLocale(Locale.ITALIAN));
        //System.out.println( italianLocale);
        String chineseLocale = now.format(DateTimeFormatter.ofLocalizedDateTime(FormatStyle.MEDIUM).withLocale(Locale.CHINA));
        System.out.println(chineseLocale);
        LocalDateTime lastVisit = LocalDateTime.of(2021, Month.JANUARY, 25, 6, 30);
        LocalDateTime lastUpdate = LocalDateTime.parse("2023-12-30T13:45:50.63", DateTimeFormatter.ISO_LOCAL_DATE_TIME);
        //System.out.println(lastUpdate);
        LocalDate date = LocalDate.now();
        LocalTime time = LocalTime.now();
        LocalDateTime dateTime = date.atTime(time);
        System.out.println(dateTime);
    }
}
```

127. Zoneid, ZoneOffset, and OffsetDateTime classes
```java
import java.time.*;
import java.util.Map;
import java.util.Set;
public class TimeZones {
    /*
        ZoneId Formats:
             20:30Z
             GMT+02:00
             America/Los_Angeles (Area/City)
     */
    public static void main(String[] args) {
        //System.out.println(LocalDateTime.now());
        //System.out.println(Instant.now()); // Zulu time = GMT
        //UTC: the primary time standard by which the world regulates clocks and time
        Instant nowUtc = Instant.now();
        //System.out.println(nowUtc.getEpochSecond()); // Returns: the seconds from the epoch of 1970-01-01T00:00:00Z
        //System.out.println(System.currentTimeMillis());
        //long startTime = System.currentTimeMillis();
        //// some code
        //System.out.println(System.currentTimeMillis() - startTime);
        // list of zones available
        //Set<String> availableZoneIds = ZoneId.getAvailableZoneIds();
        //availableZoneIds.forEach(System.out::println);
        // list of zone shortIDs
        Map<String, String> shortIds = ZoneId.SHORT_IDS;
        shortIds.forEach((k, v)-> System.out.println(k + " -> " + v));
        // Get Your Current ZoneId
        //System.out.println(ZoneId.systemDefault());
        // Other places ZoneId
        ZoneId tokyoTimeZone = ZoneId.of("Asia/Tokyo");
        ZoneId calcuttaTimeZone = ZoneId.of("Asia/Calcutta");
        ZoneId asiaSingapore = ZoneId.of("Asia/Singapore");
        // Date and Time now in my Zone
        //System.out.println(ZonedDateTime.now());
        // Date and Time now in different time zones
        //System.out.println(LocalDateTime.now(tokyoTimeZone));
        //System.out.println(ZonedDateTime.now(tokyoTimeZone));
        //System.out.println(nowUtc.atZone(tokyoTimeZone));
        //System.out.println(ZonedDateTime.now(calcuttaTimeZone));
        //System.out.println(Instant.now().atZone(calcuttaTimeZone));
        //ZonedDateTime zonedDateTime = ZonedDateTime.of(LocalDateTime.now(), tokyoTimeZone);
        //System.out.println(zonedDateTime);
        // If you don't know the time zone, but you know the offset
        ZoneOffset offset = ZoneOffset.of("-02:00");
        OffsetDateTime offsetDateTime = OffsetDateTime.of(LocalDateTime.now(), offset);
        System.out.println(offsetDateTime);
        System.out.println(ZoneOffset.MIN);
        System.out.println(ZoneOffset.MAX);
    }
}
```

128. Introduction to Math classs
- Math.random(): returns random number [0,1)
- Ref: https://www.javatpoint.com/java-math

129. Generate random numbers
- Different ways of generating random numbers
```java
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
public class RandomNumbers {
    public static void main(String[] args) {
        for (int i = 0; i < 100; i++) {
            System.out.println(getRandomNumber_4(4, 9));
        }
    }
    static int getRandomNumber_1(int min, int max){
        Random random = new Random();
        int range = max - min;
        return random.nextInt(range) + min;
    }
    static int getRandomNumber_2(int min, int max){
        Random random = new Random();
        return random.ints(min, max)
            .findFirst()
            .orElse(0);
    }
    static int getRandomNumber_3(int min, int max){
        int range = max - min;
            int rand = (int)(Math.random() * range) + min;
            return rand;
    }
    static int getRandomNumber_4(int min, int max){
      return   ThreadLocalRandom
            .current()
            .nextInt(min, max);
    }
}
```

130. Guess the number game

## Sectino 16: File manipulation (I/O and NIO)

131. Introduction to Java IO
- Java uses streams to perform IO operations
- Two types of IO streams
  - Byte stream is convenient for handling the input/output of bytes (PDF, videos, MP3, images)
  - Character stream uses Unicode to handle the input/output of characters

132. FileInputStream, FileOutputStream, FileReader, FileWriter Classes
- FileInputStream: reads byte streams
- FileOutputStream: writes byte streams
- FileReader: reads streams of characters
- FileWriter: writes streams of characters
```java
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
public class Main {
  static String filePath = "/tmp/udemy_java/132/file.txt";
  public static void main(String[] args) throws FileNotFoundException, IOException {
    FileOutputStream fos = new FileOutputStream(filePath);
    String sent = "java writes";
    fos.write(sent.getBytes());
    try(FileInputStream fis = new FileInputStream(filePath)) {
      byte[] bytes = fis.readAllBytes();
      String readText = new String(bytes);
      System.out.println(readText);
    }
  }
}
```

133. ByteArrayInputStream, ByteArrayOutputStream, CharArrayReadr/Writer Classes
```java
import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
public class ByteArrayInputStreamByteArrayOutputStreamExample {
   static String filePath = "/Users/ryan/Desktop/file.txt";
    public static void main(String[] args) throws IOException {
        String word = "java";
        byte[] bytes = word.getBytes();
        //try(ByteArrayInputStream bais = new ByteArrayInputStream(bytes)){
        //    System.out.println(bais.available());
        //    for (int i = 0; i < bytes.length; i++) {
        //        System.out.println((char) bais.read());
        //    }
        //}
       try(ByteArrayOutputStream baos = new ByteArrayOutputStream();
           FileOutputStream fos = new FileOutputStream(filePath, true)){
           //System.out.println(baos.size());
           baos.write('a');
           baos.write('b');
           baos.write('c');
           baos.write('d');
           //System.out.println(baos.size());
           //System.out.println(baos);
           baos.writeTo(fos);
            baos.flush();
       }
    }
}
```
```java
import java.io.CharArrayReader;
import java.io.CharArrayWriter;
import java.io.FileWriter;
import java.io.IOException;
public class CharArrayReaderWriter {
    static String filePath = "/Users/ryan/Desktop/file.txt";
    public static void main(String[] args) throws IOException {
        String word = "java";
        char[] chars = word.toCharArray();
        //try(CharArrayReader car = new CharArrayReader(chars)){
        //    int i;
        //    while ((i = car.read()) != -1){
        //        System.out.println((char) i);
        //    }
        //}
        try(CharArrayWriter caw = new CharArrayWriter();
            FileWriter fw = new FileWriter(filePath)){
            caw.write("hello Java");
            caw.write("hello Java2");
            System.out.println(caw);
            caw.writeTo(fw);
        }
    }
}
```

134. BufferedInputStream BufferedOutputStream, Buffered Reader, BufferedWriter Classes
- When reading/writing lage files in chunks
- mark(): marks the starting point for buffer
- reset(): goes back to the marking point
```java
import java.io.*;
public class BufferedInputStreamBufferedOutputStreamExample {
    // BufferedInputStream and BufferedOutputStream are efficient
    // when reading and writing large files in chunks
    static String filePath = "/Users/ryan/Desktop/file.txt";
    public static void main(String[] args) throws IOException {
        String word = "hello java";
        byte[] bytes = word.getBytes();
      //try(  FileOutputStream fos = new FileOutputStream(filePath);
      //      BufferedOutputStream bos = new BufferedOutputStream(fos)){
      //    bos.write(bytes);
      //}
        try(FileInputStream fis = new FileInputStream(filePath);
             BufferedInputStream bis = new BufferedInputStream(fis))    {
            System.out.println((char)bis.read()); // h
            System.out.println((char)bis.read()); // e
            bis.mark(200);
            System.out.println((char)bis.read()); // l
            System.out.println((char)bis.read()); // l
            System.out.println((char)bis.read()); // o
            System.out.println((char)bis.read()); // 
            bis.reset();
            System.out.println("after reset");
            System.out.println((char)bis.read()); // l
            System.out.println((char)bis.read()); // l
            System.out.println((char)bis.read()); // o
            //System.out.println(new String(bis.readAllBytes()));
        }
    }
}
```
- After reset(), note that `llo` is re-read
```java
import java.io.*;
public class BufferedReaderBufferedWriterExample {
    static String filePath = "/Users/ryan/Desktop/file.txt";
    public static void main(String[] args) throws IOException {
        //String word = "hello java";
        //char[] chars = word.toCharArray();
        //try(FileWriter fw = new FileWriter(filePath);
        //BufferedWriter bw = new BufferedWriter(fw)){
        //    bw.write(word);
        //    bw.newLine();
        //    bw.write(chars);
        //}
        try(FileReader fr = new FileReader(filePath);
        BufferedReader br = new BufferedReader(fr)){
            //List<String> list = br.lines().filter(l -> l.contains("3")).toList();
            //System.out.println(list);
            //System.out.println(br.readLine());
            //System.out.println(br.readLine());
            //br.mark(200);
            //System.out.println(br.readLine());
            //System.out.println(br.readLine());
            //br.reset();
            //System.out.println(br.readLine());
            //System.out.println(br.readLine());
            String line;
            while ((line = br.readLine()) != null){
                System.out.println(line);
            }
        }
    }
}
```

135. Introduction to Serialization
- Converts an object into a byte stream
- Advantages of serialization
  - Marshaling
  - JVM indepedent
  - Saves the object's state
- how to serialize an object?
  - Classes must implement directly or indirectly the serializable or externalizable interfaces
  - ObjectInputStream and ObjectOutputStream classes used to serialization and deserialization
  - Optional: add the `SerialVersionUID` field, a unique identifier used by JVM to compare the version of a class
  - Static and transient fields will not be serialized

136. Serialization Code using ObjectInputStream, ObjectOutputStream Classes
```java
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
public class Main {
    static String filePath = "/Users/ryan/Desktop/employee.txt";
    public static void main(String[] args) throws Exception {
        serialize();
        deserialize();
    }
    static void serialize() throws Exception{
        Employee employee = new Employee(101, "John Doe", "1234");
        try (FileOutputStream fos = new FileOutputStream(filePath);
             ObjectOutputStream out = new ObjectOutputStream(fos)){
            out.writeObject(employee);
        }
    }
    static void deserialize() throws Exception{
        try (FileInputStream fis = new FileInputStream(filePath);
             ObjectInputStream in = new ObjectInputStream(fis)){
            Employee employee =(Employee) in.readObject();
            System.out.println("id: " + employee.id);
            System.out.println("name: " + employee.name);
            System.out.println("password: " + employee.password);
        }
    }
}
```

137. File/Folder handling (File class)
- Creating folders/changing permissions
```java
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.Instant;
import java.time.ZoneId;
import java.time.ZonedDateTime;
public class Main {
    // File:
    //      classes are responsible for file I/O operation
    //      representation of a file or directory pathname.
    //      contains several methods for handling files and folders
    public static void main(String[] args) throws IOException {
        File directory = new File( "folder1/folder2/folder3"); // relative path
        if(directory.mkdirs()){
            System.out.println("directory created...");
        }
        File file = new File("folder1/folder2/folder3/text.txt");
        if(file.createNewFile()){
            System.out.println("file created...");
        }
        // get information
        System.out.println("file path: " + file.getPath()); // relative path
        System.out.println("absolute path: " + file.getAbsolutePath());
        System.out.println("file exists: " + file.exists());
        System.out.println("is file: " + directory.isFile());
        System.out.println("is directory: " + directory.isDirectory());
        System.out.println("is hidden: " + file.isHidden());
        System.out.println("last modified in millis: " + file.lastModified());
        ZonedDateTime lastModified = Instant.ofEpochMilli(file.lastModified()).atZone(ZoneId.systemDefault());
        System.out.println("last modified date time: " + lastModified);
        // check permissions
        System.out.println("can read: " + file.canRead());
        System.out.println("can write: " + file.canWrite());
        System.out.println("can execute: " + file.canExecute());
        file.setWritable(true);
        try(FileWriter fw = new FileWriter(file)){
            fw.write("java is awesome 2222");
        }
    }
}
```

138. Listing Files and Directories (recursively using File class)
```java
import java.io.File;
import java.util.Arrays;
public class Main {
    public static void main(String[] args) {
        File dir = new File("folder1/folder2");
        //String[] list = dir.list();
        File[] list = dir.listFiles(((dir1, name) -> name.contains("txt")));
        for (File file: list){
            System.out.println(file.getName());
        }
    }
    static void traverseDirectory(File dir){
        File[] list = dir.listFiles();
        for(File file: list){
            if(file.isDirectory()){
                System.out.println("directory: " + file.getName());
                traverseDirectory(file);
            }else {
                System.out.println("file found: " + file.getName());
            }
        }
    }
}
```

139. Introduction to NIO Path interface
- The Path interface along Java7, repreesenting a path in the file system
- Disadvantages of the File class
  - No support for symbolic links
  - Methods return false instead of throwing
  - Doesn't support Accessl Control List
  - Causes problems with large directories
  - Limited to the current OS
  - No file copy/move capability

140. Create Files/Directories using NIO Path interface and File Class
```java
import java.io.IOException;
import java.nio.file.*;
public class Main {
    public static void main(String[] args) throws IOException {
        //Path path = Paths.get("Users", "ryan", "Desktop", "folder1");
        Path path = Paths.get("/Users/ryan/Desktop/folder1");
        if(Files.notExists(path)){
            Files.createDirectories(path);
        }
        Path path1 = FileSystems.getDefault().getPath("");
        Path filePath = Paths.get(path.toString(),"text.txt");
        if(Files.notExists(filePath)){
            Files.createFile(filePath);
        }
    }
}
```

141. Read File Atgtributes(NIO Path and Files) 
```java
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.DosFileAttributes;
import java.nio.file.attribute.PosixFileAttributes;
public class Main {
    public static void main(String[] args) throws IOException {
        Path absPath = Paths.get("/Users/ryan/Desktop/code/FileAttributesList");
        Path directoryPath = Paths.get("Folder1");
        Path filePath = Paths.get("Folder1", "text.txt");
        System.out.println("directoryPath: " + directoryPath);
        System.out.println("isAbsolute: " + absPath.isAbsolute());
        System.out.println("Absolute Path: " + directoryPath.toAbsolutePath());
        System.out.println("URI: " + directoryPath.toUri());
        System.out.println("Name: " + directoryPath.getFileName());
        System.out.println("Parent: " + absPath.getParent());
        System.out.println("Root: " + directoryPath.toAbsolutePath().getRoot());
        System.out.println("Exist: " + Files.exists(directoryPath));
        System.out.println("Not Exist: " + Files.notExists(directoryPath));
        System.out.println("Hidden: " + Files.isHidden(directoryPath));
        System.out.println("Is Readable: " + Files.isReadable(filePath));
        BasicFileAttributes bfa = Files.readAttributes(filePath, BasicFileAttributes.class);
        //PosixFileAttributes posixFileAttributes = Files.readAttributes(filePath, PosixFileAttributes.class); // Posix ONLY
        //DosFileAttributes dosFileAttributes = Files.readAttributes(filePath, DosFileAttributes.class); // DOS system only
        //= Files.readAttributes(filePath, PosixFileAttributes.class);
        System.out.println(bfa.creationTime());
        System.out.println(bfa.isDirectory());
        System.out.println(bfa.lastAccessTime());
        System.out.println(bfa.lastModifiedTime());
        System.out.println(bfa.isRegularFile());
        System.out.println(bfa.size());
    }
}
```

142. Write Read/Write Files using NIO Files class
```java
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
public class Main {
    static String str = "Hello Java\n";
    static byte[] bytes = "some bytes\n".getBytes();
    static List<String> list = new ArrayList<>() {{
        add("list item 1");
        add("list item 2");
    }};
    public static void main(String[] args) throws IOException {
        Path path = Path.of("myFolder", "myFile.txt");
        Files.write(path, bytes, StandardOpenOption.APPEND);
        Files.write(path, list, StandardOpenOption.APPEND);
        Files.writeString(path, str, StandardOpenOption.APPEND);
        try (OutputStream outputStream
                 = Files.newOutputStream(path, StandardOpenOption.APPEND)) {
            outputStream.write(65);
        }
        try (BufferedWriter bw =
               Files.newBufferedWriter(path, StandardOpenOption.APPEND) ){
            bw.write("hello java again");
        }
        // reading data
        String s = Files.readString(path);
        System.out.println(Files.readAllLines(path));
        byte[] bytes1 = Files.readAllBytes(path);
        InputStream in = Files.newInputStream(path);
        BufferedReader br = Files.newBufferedReader(path);
    }
}
```
- File.readString(): reads the entire lines
- File.readAllLines(): reads all lines into List

143. Normalizing, Relativizing, Joining, and Comparing Paths
- normalize(): removes redundancy from path
- relativize(): creates a path b/w two paths. How to reach argument path from the object path
- resolve(): concatenates two paths (object + argument)
- equals(): compares two paths. true or false
- startsWith(): checks if a path starts with another path. true or false
- endsWith(): checks if a path ends with another path. true or false

144. Listing Files and Directories using NIO File Class
```java
import java.io.File;
import java.io.IOException;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
public class WalkExample {
    public static void main(String[] args) throws IOException {
        Path path = Path.of("my_folder");
        // ******** old method
        //File file = new File("my_folder");
        //System.out.println(Arrays.toString(file.list()));
        // ******** new method 1
        //try(Stream<Path> paths = Files.list(path)){
        //    //paths.filter(f-> f.endsWith("txt"));
        //    paths.forEach(System.out::println);
        //}
        // ******** new method 2
        try(Stream<Path> paths = Files.walk(path, 3, FileVisitOption.FOLLOW_LINKS)){
            //paths.forEach(System.out::println);
            List<Path> PDFs = paths.filter(p-> p.toString().endsWith("pdf")).toList();
            System.out.println(PDFs);
        }
    }
}
```
- Files.walk(): commonly used
- Files.walkFileTree(): when action is requred
  - FileVisitor object is necessary
```java
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
public class WalkTreeExample {
    public static void main(String[] args) throws IOException {
        MyFileVisitor visitor = new MyFileVisitor();
        Path path = Path.of("my_folder");
        Files.walkFileTree(path, visitor);
    }
}
class MyFileVisitor extends SimpleFileVisitor<Path>{
    @Override
    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
        System.out.println("At file: " + file);
        return FileVisitResult.CONTINUE;
    }
    @Override
    public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException {
        System.out.println("Before visiting: " + dir);
        return FileVisitResult.CONTINUE;
    }
}
```

145. Copy, Move, Delete files using NIO Files Class
```java
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
public class Main {
    public static void main(String[] args) throws IOException {
        Path srcPath = Path.of("my_folder/A/B/file.txt");
        Path destPath = Path.of("copied_files/file2.txt");
        //Files.copy(srcPath, destPath, StandardCopyOption.REPLACE_EXISTING);
        //Files.deleteIfExists(destPath);
        //Files.move(srcPath, destPath);
        Path srdDir = Path.of("my_folder");
        Path destDir = Path.of("new_folder");
        Files.copy(srdDir, destDir);
        CopyDirectory.copy(srdDir, destDir);
    }
}
class CopyDirectory extends SimpleFileVisitor<Path> {
    private Path src,  dest;

    private CopyDirectory(Path srcDirectory, Path desDirectory) {
        this.src = srcDirectory;
        this.dest = desDirectory;
    }
    public static boolean copy(Path srcPath, Path destPath){
        CopyDirectory visitor = new CopyDirectory(srcPath, destPath);
        try {
            Files.walkFileTree(srcPath, visitor);
            return true;
        } catch (IOException e) {
            System.out.println(e.getMessage());
            return false;
        }
    }
    @Override
    public FileVisitResult visitFile(Path file, BasicFileAttributes attributes) {
        try {
            Path targetFile = dest.resolve(src.relativize(file));
            Files.copy(file, targetFile);
        } catch (IOException e) {
            System.err.println(e.getMessage());
            return FileVisitResult.TERMINATE;
        }
        return FileVisitResult.CONTINUE;
    }
    @Override
    public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attributes) {
        try {
            Path newDir = dest.resolve(src.relativize(dir));
            if(Files.notExists(newDir)) {
                Files.createDirectory(newDir);
            }
        } catch (IOException e) {
            System.err.println(e.getMessage());
            return FileVisitResult.TERMINATE;
        }
        return FileVisitResult.CONTINUE;
    }
}
```

146. Watch Service API
- File system change tracker
```java
import java.io.IOException;
import java.nio.file.*;
public class Main {
    public static void main(String[] args) throws IOException, InterruptedException {
            try(WatchService watchService = FileSystems.getDefault().newWatchService()){
                Path dir = Paths.get("/Users/ryan/Desktop/myFolder");
                dir.register(watchService, StandardWatchEventKinds.ENTRY_DELETE,
                    StandardWatchEventKinds.ENTRY_CREATE,
                    StandardWatchEventKinds.ENTRY_MODIFY);
                WatchKey watchKey = watchService.take();
                do{
                    for(WatchEvent<?> event: watchKey.pollEvents()){
                        System.out.println("kind: " + event.kind());
                        System.out.println("context: " + event.context());
                        System.out.println("count: " + event.count());
                    }
                }while (watchKey.reset());
            }
    }
}
```

## Section 17: Unit testing using Junit 5

147. Introduction to JUnit5
- Testing levels
  - Unit
  - Integration
  - Acceptance
- Testing methods
  - Black-box: uni testers are not aware of the internal functionality of the system
  - While-box: the functional behavior of the SW is tested by the developers to validate their execution
  - Grey-box
- Test project
  - MathUtils

148. @Test @assertEquals
- assertEquals(a,b)

149. @DisplayName
- Description of the test

150. assertThrows @TestMethodOrder @Order
- Executable object
  - Not executed when defined
  - Lambda expression

151. Junit 5 Test LifeCycle
- @BeforeAll
  - @BeforeEach
    - @Test
  - @AfterEach
  - @BeforeEach
    - @Test
  - @AfterEach
- @AfterAll

152. assertAll assertTrue assertFalse

153. assertNull assertNotNull

154. @ParameterizedTest @CsvSource

155. @ValueSource

156. @CsvFileSource @Disabled fail

157. @RepeatedTest assertTimeout

158. assertArrayEquals assertIterableEquals
```java
package com.JUnitExample;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
public class MathUtil {
    public int add(int x, int y) {
        return x + y;
    }
    public int subtract(int x, int y) {
        return x - y;
    }
    public Integer divide(int x, int y) throws Exception {
        if (y == 0) {
            throw new Exception("zeros aren't allowed");
        } else if (x == 0) {
            return null;
        }
        return x / y;
    }
    public boolean isEven(int number) {
        return number % 2 == 0;
    }
    public int generateRandom(int limit) throws InterruptedException {
        Thread.sleep(500);
        return new Random().nextInt(limit);
    }
    public int[] duplicate(Integer[] numbers) {
        return Arrays.stream(numbers).mapToInt(e -> e * 2).toArray();
    }
    public List<Integer> duplicate(List<Integer> numbers) {
        return numbers.stream().map(e -> e * 2).toList();
    }
}
```
```java
package com.JUnitExample;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
public class MathUtil {
    public int add(int x, int y) {
        return x + y;
    }
    public int subtract(int x, int y) {
        return x - y;
    }
    public Integer divide(int x, int y) throws Exception {
        if (y == 0) {
            throw new Exception("zeros aren't allowed");
        } else if (x == 0) {
            return null;
        }
        return x / y;
    }
    public boolean isEven(int number) {
        return number % 2 == 0;
    }
    public int generateRandom(int limit) throws InterruptedException {
        Thread.sleep(500);
        return new Random().nextInt(limit);
    }
    public int[] duplicate(Integer[] numbers) {
        return Arrays.stream(numbers).mapToInt(e -> e * 2).toArray();
    }
    public List<Integer> duplicate(List<Integer> numbers) {
        return numbers.stream().map(e -> e * 2).toList();
    }
}
```

## Section 18: Students Questions and Answers

159. Widening and Narrowing of Primitive types
- Widening (upcasting)
  - Converts small data type to large data type
  - byte(8) -> short(16) -> int(32) -> long(64)
  - float(32) -> double (64)
- Narrowing (downcasting)
  - Converts big data type into smal data type
    - Extra casting like `byte b = 1; long x = (byte) b;`
  - Dangerous due to data loss possibility
  - byte(8) <- short(16) <- int(32) <- long(64)
  - float(32) <- double (64)
