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

## Section 4: Conditionals

## Section 5: Iterations in Java

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
