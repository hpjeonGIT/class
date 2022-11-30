## Java 17: Learn and dive deep into Java
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

98. Wildcards

## Section 13: Collections Framework

## Section 14: Stream API

## Section 15: Date time and math APIs

## Sectino 16: File manipulation (I/O and NIO)

## Section 17: Unit testing using Junit 5
