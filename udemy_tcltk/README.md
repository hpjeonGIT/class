## The Complete Course of TCL Programming 2024
- Instructor: The Tech Courses

## Section 1: Basic TCL

### 1. Welcome to TCL

### 2. Installing TCL
- At Ubuntu 20, `sudo apt install tcl`

### 3. Introduction to programming

### 4. Working with TCL shell
```bash
$ tclsh
% pwd
/home/hpjeon/hw/class/udemy_tcl
% puts Hello
Hello
% puts "Hello world"
Hello world
% 
```
- Coupling with readline: `rlwrap tclsh`
- Comment with #
  - In the beginning of line (no command exists)
  - When command exists, terminate with ;, then add # 
    - Ex: `puts $x; # comment here`

### 5. Using variables
```tcl
% set day "Saturday"
Saturday
% puts $day
Saturday
% puts "today is $day"
today is Saturday
```

### 6. Command substitution
- Using [...]
```bash
% expr 3+2
5
% set result [expr 3+2]
5
% puts $result
5
% set greeting [concat "hello" $name, ", it is " [expr 2+1] "pm"]
hello Joe, , it is 3 pm
% puts $greeting
hello Joe, , it is 3 pm
```

### 7. Basic Mathematical operations
- +,-,*,/,%
- sqrt(),abs(), pow(),rand(), round()
  - Functions cannot be used as a standalone. Must be within [...] as expr

### 8. Solutions to mathematical excercises
- 1/1000 will be zero as this is integer operation
  - Must be written as 1./1000.
```bash
% set t 3.0;
3.0
% set v 70.0;
70.0
% set d [expr $v*(1./1000)*60*60*$t]
756.0
```
### 9. Conditionals
```bash
% set x 1
1
% if {$x == 0 } { set y 0 } elseif {$x == 1} { set y 1 } else { set y 2 }
1
```

### 10. Example of conditionals

### 11. Ternary operators
```bash
% set y [expr $x >1 ? true: false]
false
```

### 12. Logical and bitwise operations
- TRUE, true, FALSE, false
- AND: && 
- OR: ||
- NOT: !
- Bitwise AND: &
- Bitwise OR: |
- Bitwise XOR: ^
- Bitwise shift left: <<
- Bitwise shift right: >>

### 13. Solutions to Logical and bitwise operations excercise

### 14. Operation precedence

### 15. Strings
- \n: new line
- \r: carrage return
- \t: horizontal tab
- \v: vertical tab
- \a: **alert or bell (audible)**
- \b: backspace
- \f: form feed (page break)
```tcl
% string compare "hello" "helloworld"
-1
% string first "hello" "helloworld"
0
% string index "hello" 1
e
% string length "hello"
5
% string range "hello world" 2 8
llo wor
```

### 16. Solution to strings excercise

### 17. What is a script

### 18. Text editors

### 19. Coding guidelines

### 20. Section 1 Quiz

### 21. Scripts Section 1

## Section 2: Intermediate TCL

### 22. Lists
```tcl
% set animals [list cat dog turtle]
cat dog turtle
% set animals2 {cat dog turtle}
cat dog turtle
% puts $animals
cat dog turtle
% puts $animals2
cat dog turtle
% puts [lindex $animals 0]
cat
% puts [lindex $animals 2]
turtle
% set text "Hello_world_good_morning"
Hello_world_good_morning
% set newtext [split $text "_"]
Hello world good morning
```

### 23. Adding elements to lists
- lappend: in the end
- linsert: in the middle but not modify the original list
```tcl
% puts $animals
cat dog turtle
% llength $animals
3
% lappend animals horse
cat dog turtle horse
% llength $animals
4
% set animals [linsert $animals 2 rabbit cow]
cat dog rabbit cow turtle horse
% puts $animals
cat dog rabbit cow turtle horse
```

### 24. Changing list elements
- lset: Sets a single element
- lreplace: replaces elements but must be used in expr
```tcl
% puts $animals ; # comment
cat dog rabbit cow turtle horse
% lset animals 4 hen
cat dog rabbit cow hen horse
% puts $animals ; # comment
cat dog rabbit cow hen horse
% set animals [lreplace $animals 0 1 lion tiger]
lion tiger rabbit cow hen horse
```

### 25. Extracting ranges, sorting lists, and iterating over lists
- lrange: elements in list range
- lsort: list sorting but does not modify the original list
- foreach x y {...}: loop over elements in y
```tcl
% puts $animals
lion tiger rabbit cow hen horse
% lrange $animals 2 4
rabbit cow hen
% lsort $animals
cow hen horse lion rabbit tiger
% foreach el $animals { puts "Animal: $el" }
Animal: lion
Animal: tiger
Animal: rabbit
Animal: cow
Animal: hen
Animal: horse
```

### 26. List cheatsheets and practice
### 27. Lists solution
### 28. Creating arrays
### 29. Iterating over arrays
### 30. Multidimesnsional arrays
### 31. Sorting lists and arrays
### 32. Arrays practice
### 33. Arrays solution
### 34. For and While loops
### 35. Nested loops
### 36. Breaking loops
### 37. Loops practice
### 38. Loops solution
### 39. Procedures
### 40. Recursive procedures
### 41. Procedures practice
### 42. Procedures solution
### 43. File IO
### 44. File IO practice
### 45. File IO solution
### 46. Various Query commands
### 47. Various Query commands solution
### 48. Special variables and command line
### 49. Special variables and command line solution
### 50. More math
### 51. More math solution
### 52. Regular expressions
### 53. Regexp examples
### 54. Regexp solution
### 55. Debug techniques
### 56. Debug session
### 57. Section II Quiz
### 58. Scripts Section II

## Section 3: Advanced TCL

## Section 4: Extra section: TK(Took Kit) in TCL
