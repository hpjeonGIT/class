## Title: Complete Ruby Programmer - Master Ruby
- Instructor: Mashrur Hossain and Evgeny Rahman
## Section 1:Introduction
1. Introduction to Programming
2. The Role of Programming in the Modern World
3. Importance of Programming Skills for all Professionals
4. Preview of Final Student Enrollment App
5. Preview of Web Scraping Automation Project
6. Preview of Data Engineering Project
7. Installation of Ruby on MacOS
8. Installation of Ruby on Windows
9. Installation of Ruby on Linux
10. Troubleshooting and Looking for Answers Online
11. Introudction to Code Editors
12. Installing Atom
## Section 2: Programming in Ruby
13. Introduction to the Terminal
14. Getting Around in the Terminal
15. Creating and Changing Directories
16. Creating and Editing Files from the command line
17. Where to find the code
18. The first ruby file - "hello world"
19. Running Ruby Files from the command line
- hello.rb:
```ruby
puts "hello world"  # prints string with \n
print "hello world" # prints string
p "hello world"     # prints literal
```
- Running on CLI
```bash
$ ruby hello.rb
hello world
hello world"hello world"
```
 
20. Printing Output - puts/print/p
21. Code Commenting
- Use #
 
22. Introduction to IRB
- Interactive RuBy: irb
```ruby
$ irb
irb(main):001:0> "hello"
=> "hello"
irb(main):002:0> puts "hello"
hello
=> nil
irb(main):003:0> p "hello"
"hello"
=> "hello"
```
 
23. Atom and Terminal Setup
24. Codealong exercise - Name and Food
25. Assignment - Print your own age
- Q: How to load Date package? Need to run Date.today
 
## Section 3: Strings and Numbers
26. Introduction to Variables and Why We Use Them
 
27. Variables - Changing Values
28. Variables - Changing Values Continued
```ruby
irb(main):008:0> a = "hello"
=> "hello"
irb(main):009:0> b = a
=> "hello"
irb(main):010:0> puts b
hello
=> nil
```
 
29. Getting Input from the Terminal
```ruby
irb(main):017:0> b = gets
hello
=> "hello\n"
irb(main):018:0> b
=> "hello\n"
```
 
30. Introduction to Strings
31. String Interpolation
```ruby
irb(main):028:0> a = "hello"
=> "hello"
irb(main):029:0> b = "world"
=> "world"
irb(main):030:0> c = "#{a} and #{b}"
=> "hello and world"
irb(main):031:0> c
=> "hello and world"
irb(main):032:0> x = 1.14
=> 1.14
irb(main):033:0> print "x = #{x}"  #<--- works for non-string variablles like int/float
x = 1.14=> nil
```
 
32. Changing a String
33. String Concatenation
```ruby
irb(main):035:0> a.concat(b)
=> "helloworld"
irb(main):036:0> a #<-------- a is changed
=> "helloworld"
```
34. Manipulating Strings
```ruby
irb(main):038:0> a.reverse
=> "dlrowolleh"
irb(main):039:0> a # <------ a is NOT changed
=> "helloworld"```
irb(main):040:0> a.reverse!  #<--- Adding bang(!) in the end
=> "dlrowolleh"
irb(main):041:0> a # <------ a ISchanged
=> "dlrowolleh"
```
 
35. Introduction to Comparisons
```ruby
irb(main):070:0> print true==true
true=> nil
irb(main):071:0> print true==false
false=> nil
```
 
36. Logic Comparisons
```ruby
irb(main):084:0> true && true
=> true
irb(main):085:0> true && false
=> false
irb(main):086:0> true || true
=> true
irb(main):087:0> true || false
=> true
irb(main):088:0> true or false
=> true
irb(main):089:0> true and false
=> false
```
- &&/|| is preferred than and/or
 
37. Numeric Comparisons
```ruby
irb(main):090:0> a = 5
=> 5
irb(main):091:0> b = 6
=> 6
irb(main):093:0> a == b
=> false
irb(main):094:0> a > b
=> false
irb(main):095:0> a < b
=> true
irb(main):092:0> a && b # returns RHS
=> 6
irb(main):096:0> a || b # returns LHS
=> 5
irb(main):097:0> b && a # returns RHS
=> 5
irb(main):098:0> b || a # returns LHS
=> 6
```
 
38. Introduction to Branching Logic - if
- code block of if:
```ruby
if a==b
  puts "Equal value found"
end
```
 
39. Branching Logic - else
```ruby
if a==b
  puts "Equal value found"
else
  puts "Unequal value found"
end
```
 
40. Branching Logic - elsif
```
if a==b
  puts "Equal value found"
elsif (a+1) == b
  puts "Special case"
else
  puts "Unequal value found"
end
```
 
41. Branching Logic - case
```ruby
case a
when 1
  print "one"
when 2
  print "two"
else
  print "Something else" 
end
```
 
42. Finding Text in a String
```ruby
irb(main):169:0> a = "hello world"
=> "hello world"
irb(main):170:0> a.include? "wor"
=> true
```
 
43. Finding Text - index
```ruby
irb(main):175:0> a.index('l')
=> 2
irb(main):177:0> a.index("world")
=> 6
```
- When not found, `nil` is returned
 
44. Introduction to Numbers - Integers and Floats
45. Numeric Operations - Integers
46. Numeric Operations - Floats
```ruby
irb(main):195:0> a = 3
=> 3
irb(main):196:0> Float(a)
=> 3.0
```
 
47. Casting Strings to Integers
```ruby
 
irb(main):197:0> a = "1"
=> "1"
irb(main):200:0> Integer(a)
=> 1
irb(main):201:0> a.to_i
=> 1
irb(main):202:0> "hello".to_i # all string is to zero
=> 0
```
 
48. Casting Strings to Floats
```ruby
irb(main):203:0> a.to_f
=> 1.0
irb(main):204:0> "hello".to_f
=> 0.0
irb(main):205:0> 3.to_f
=> 3.0
```
 
49. Getting Numbers from Input
```ruby
a = gets.chomp  # .chomp cleans the input
```
 
50. Random Numbers
```ruby
irb(main):206:0> a = rand
=> 0.6015031177473047
irb(main):207:0> a = rand
=> 0.15893068840958613
irb(main):208:0> a = rand
=> 0.37959579621009476
irb(main):216:0> a = rand 1000 # from zero to less than 1000. [0:1000)
=> 274
irb(main):217:0> a = rand 1000
=> 834
irb(main):218:0> a = rand 1000
=> 63
```
 
51. Combining Nubmers with Strings
52. Codealong Exercise - Introduction
53. Codealong Exercise - Guess the Number Game
54. Assignment - Build a User Input Validator
## Section 4: Methods and Data Structures
55. Changing the Terminal Prompt
56. Introduction to Methods
- Method: a code block surround with def ... end
```ruby
def hello
  "hello"
end
```
- This is a code block of definition and doesn't print "hello"
- When the block is executed, it prints "hello"
 
57. Method Arguments
```ruby
irb(main):244:0> def hello(message)
irb(main):245:1>   puts message
irb(main):246:1> end
=> :hello
irb(main):247:0> hello("world")
world
=> nil
irb(main):248:0> hello "world"
world
=> nil
```
 
58. Optional Arguments and Default Values
```ruby
irb(main):249:0> def hello(message = "blank message")
irb(main):250:1>   puts message
irb(main):251:1> end
```
 
59. The Different Styles of if
```ruby
irb(main):254:0> puts "True" if 5>4
True
=> nil
irb(main):255:0> puts "True" if 5<4
=> nil
irb(main):256:0> puts "Executed" unless 5< 4
Executed
=> nil
irb(main):257:0> puts "Executed" unless 5 > 4
=> nil
```
 
60. Method Returns
```ruby
def simple(number)
  if number > 5
    return "greater than 5"
  else
    return "less than 5"
  end
end
```
 
61. Calling One Method from Another
62. Introduction to the Concept of Data Structures
- A combination of data and functionality
- Arrays/Hashes
 
63. Introduction to Arrays
```py
 
irb(main):342:0> a = [123, 456,789]
=> [123, 456, 789]
irb(main):343:0> a.first
=> 123
irb(main):345:0> a.last
=> 789
irb(main):347:0> a.length
=> 3
```
 
64. Array Creation
```ruby
irb(main):348:0> a = Array.new(4)
=> [nil, nil, nil, nil]
irb(main):349:0> p a
[nil, nil, nil, nil]
=> [nil, nil, nil, nil]
irb(main):354:0> b = Array.new(4, rand(100)) # rand(100) is executed only once and distributed into 4 elements
=> [96, 96, 96, 96]
irb(main):355:0> b = Array.new(4, rand(100))
=> [65, 65, 65, 65]
irb(main):356:0> c = %w(a b c d e)
=> ["a", "b", "c", "d", "e"]
```
 
65. Array Manipulation
```ruby
irb(main):362:0> c.reverse
=> ["e", "d", "c", "b", "a"]
irb(main):363:0> c
=> ["a", "b", "c", "d", "e"]
irb(main):364:0> c + ['1','2']
=> ["a", "b", "c", "d", "e", "1", "2"]
irb(main):366:0> c.concat(['1','2'])
=> ["a", "b", "c", "d", "e", "1", "2"]
irb(main):367:0> c
=> ["a", "b", "c", "d", "e", "1", "2"]
```
 
66. Arryas - Push & Pop
```ruby
irb(main):368:0> c.pop  # returns the removed element (from the last of the array)
=> "2"
irb(main):369:0> c
=> ["a", "b", "c", "d", "e", "1"]
irb(main):370:0> x = c.pop
=> "1"
irb(main):371:0> x
=> "1"
irb(main):373:0> c
=> ["a", "b", "c", "d", "e"]
irb(main):374:0> x = c.push("abc") # returns the entire array (which is updated)
=> ["a", "b", "c", "d", "e", "abc"]
irb(main):375:0> x
=> ["a", "b", "c", "d", "e", "abc"]
irb(main):376:0> c
=> ["a", "b", "c", "d", "e", "abc"]
```
 
67. Retrieving Data from Arrays
68. Retrieving Data from Arrays - Continued
- Syntactic sugar
```ruby
irb(main):377:0> c.first
=> "a"
irb(main):378:0> c.last
=> "abc"
irb(main):379:0> c.delete("abc")
=> "abc"
irb(main):380:0> c
=> ["a", "b", "c", "d", "e"]
irb(main):381:0> c.take(3) # return first 3 elements
=> ["a", "b", "c"]
irb(main):382:0> c
=> ["a", "b", "c", "d", "e"] # take doesn't affect the base array
irb(main):385:0> c.length
=> 5
irb(main):386:0> c[5] # yields nil as index size violated
=> nil
irb(main):387:0> c.fetch(5) # Raised IndexError. Might be used in try/catch
Traceback (most recent call last):
        3: from /usr/bin/irb:11:in `<main>'
        2: from (irb):387
        1: from (irb):387:in `fetch'
IndexError (index 5 outside of array bounds: -5...5)
```
 
69. Introduction to Hashes
```ruby
irb(main):388:0> a = {}
=> {}
irb(main):389:0> b = Hash.new()
=> {}
irb(main):390:0> c = {"John"=> 88, "Alice"=> 70}
=> {"John"=>88, "Alice"=>70}
```
 
70. Keys, Values
```ruby
irb(main):391:0> c.keys()
=> ["John", "Alice"]
irb(main):392:0> c.values()
=> [88, 70]
irb(main):393:0> c["John"]
=> 88
irb(main):394:0> c["James"] == 44 # non-existing key
=> false
irb(main):395:0> c["James"]
=> nil
```
 
71. Symbols
- Colon (:) implies that this is a symbol (same as Julia)
```ruby
irb(main):396:0> myh = { name: "James", prof: "Programmer"}  # new syntax for has
=> {:name=>"James", :prof=>"Programmer"}
irb(main):397:0> myh2 = {:name => "James", :prof => "Programmer"} # The above is equivalent to this
=> {:name=>"James", :prof=>"Programmer"}
irb(main):398:0> myh[:name]
=> "James"
irb(main):399:0> myh["name"]  # string key doesn't work for the symbolic keys
=> nil
```
 
72. Retrieving Data from Hashes
- Non-existing key will produce nil
- Using .fetch() will raise error for non existing keys
 
73. Retrieving Data from Hashes - Continued
```ruby
irb(main):403:0> myh.has_key?(:name)
=> true
irb(main):404:0> puts c[:name] if c.has_key?(:name)
=> nil
irb(main):405:0> puts myh[:name] if myh.has_key?(:name)
James
=> nil
```
 
74. Hash Manipulation
```ruby
irb(main):414:0> myh
=> {:name=>"James", :prof=>"Programmer"}
irb(main):415:0> c
=> {"John"=>88, "Alice"=>70}
irb(main):416:0> c2 = myh.merge(c)  # concat or +  doesn't work for Hash
=> {:name=>"James", :prof=>"Programmer", "John"=>88, "Alice"=>70}
irb(main):417:0> myh
=> {:name=>"James", :prof=>"Programmer"} #  myh is not changed.
irb(main):418:0> c2
=> {:name=>"James", :prof=>"Programmer", "John"=>88, "Alice"=>70} # return value is the merged hash
irb(main):421:0> c2.delete(:prof)
=> "Programmer"
irb(main):422:0> c2
=> {:name=>"James", "John"=>88, "Alice"=>70}
```
 
75. Introduction to Loops
```ruby
loop do
  puts "hello"  # this is an infinite loop
end
```
 
76. While Loops
```ruby
irb(main):426:0> a = 0
=> 0
irb(main):427:0> while a < 5 do
irb(main):428:1*  puts a
irb(main):429:1>  a = a + 1
irb(main):430:1> end
0
1
2
3
4
=> nil
```
 
77. While Loops with User Input
```ruby
irb(main):435:0> begin
irb(main):436:1>   puts "Enter value other than zero"
irb(main):437:1>   choice = gets.chomp
irb(main):438:1> end while choice != "0" # must be string, not an integer
Enter value other than zero
123
Enter value other than zero
abc
Enter value other than zero
0
=> nil
```
 
78. Breaking Out of Loops
- The above code is not recommended as while condition is located in the end
```ruby
irb(main):439:0> loop do
irb(main):440:1*   puts "Enter value"
irb(main):441:1>   choice = gets.chomp
irb(main):442:1>   if choice == "0"
irb(main):443:2>     break
irb(main):444:2>   end
irb(main):445:1> end
Enter value
123
Enter value
abc
Enter value
0
```
 
79. Codealong Exercise - Introduction
80. Codealong Exercise - Contacts Directory
81. Assignment - Build a Crendentials Collection
82. Optional - Introduction to Algorithms and Sorting
83. Bubble sort demo and complexity analysis
84. optional - Implementing an Array Sort
 
## Section 5: Deep Dive Into Iteration and Blocks
85. Loops = Using Next
- next: `continue` in Python/Julia
    - Goes to the next iteration of the loop without evaluating any of the rest of the code
```ruby
irb(main):459:0> loop do
irb(main):460:1*  i = i + 1
irb(main):461:1>  if (i%2) == 0
irb(main):462:2>    puts "Found even number #{i}"
irb(main):463:2>    next
irb(main):464:2>  end
irb(main):465:1>  puts i
irb(main):466:1>  break if i> 5
irb(main):467:1> end
1
Found even number 2
3
Found even number 4
5
Found even number 6
7
=> nil
```
 
86. Until Loops
```ruby
irb(main):468:0> i=0
=> 0
irb(main):469:0> until i==5 do
irb(main):470:1*   puts i
irb(main):471:1>   i = i+1
irb(main):472:1> end
0
1
2
3
4
=> nil
```
 
87. Loops as Modifiers
```ruby
irb(main):473:0> i=0; i+=1 while i< 10; puts i
10
=> nil
```
 
88. For Loops
```ruby
irb(main):474:0> for i in [1,2,3,4]
irb(main):475:1>   puts i
irb(main):476:1> end
1
2
3
4
=> [1, 2, 3, 4]
```
 
89. Introduction to Iteration
90. Introduction to Blocks
91. Iterating Over a Range
92. Iterating Using Steps
93. Ruby Enumerate
94. Enumerators in Ruby
95. Hash Iteration
96. Operations Inside Itereation
97. Arrays - Map, Select, and Reject
98. Hashes - Map, Select, and Reject
99. Introduction to Recursion
100. Codealong Exercise - Introduction
101. Codealong Exercise - Filters on Product Catalog
102. Assignment - Find Students by Name or Age
