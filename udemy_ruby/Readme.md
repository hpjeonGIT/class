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
```ruby 
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
```ruby
irb(main):001:0> arr = [1,2,3,4]
=> [1, 2, 3, 4]
irb(main):003:0> arr.each {|el| puts el}
1
2
3
4
=> [1, 2, 3, 4]
```
 
90. Introduction to Blocks
- The above code is equivalent to following code segment:
```ruby
def puts_element(element)
  yield element
end
for i in arr
  puts_element(i) {|j| puts j}
end 
```
 
91. Iterating Over a Range
```ruby
irb(main):034:0> (1..5).each do |element|
irb(main):035:1*   puts element
irb(main):036:1> end
1
2
3
4
5
=> 1..5
irb(main):037:0> 1.upto(5) do |element|
irb(main):038:1*  puts element
irb(main):039:1> end
1
2
3
4
5
=> 1
irb(main):040:0> 5.downto(1) do |element|
irb(main):041:1*   puts element
irb(main):042:1> end
5
4
3
2
1
=> 5
irb(main):043:0> 5.times do |element| #<----- array begins from zero
irb(main):044:1*   puts element
irb(main):045:1> end
0
1
2
3
4
=> 5
```
 
92. Iterating Using Steps
```ruby
irb(main):046:0> (1..10).step(2) do |i|
irb(main):047:1*  puts i
irb(main):048:1> end
1
3
5
7
9
=> 1..10
```
 
93. Ruby Enumerate
94. Enumerators in Ruby
```ruby
irb(main):064:0> arr = ["a","b","c"]
=> ["a", "b", "c"]
irb(main):065:0> array_enumerator = arr.each
=> #<Enumerator: ["a", "b", "c"]:each>
irb(main):069:0> array_enumerator.each_with_index do  |i,j|  #<---- index is j !!!
irb(main):070:1*  puts "The index is #{j} and the item is #{i}"
irb(main):071:1> end
The index is 0 and the item is a
The index is 1 and the item is b
The index is 2 and the item is c
=> ["a", "b", "c"]
irb(main):073:0> arr.each_with_index do |i,j|
irb(main):074:1*   puts j.to_s + " " + i
irb(main):075:1> end
0 a
1 b
2 c
=> ["a", "b", "c"]
```
 
95. Hash Iteration
```ruby
irb(main):086:0> collection = { :animal=> "dog", sound: "bark"}
=> {:animal=>"dog", :sound=>"bark"}
irb(main):087:0> collection.each do |k, v|
irb(main):088:1*   puts "key =  #{k}   val =  #{v}"
irb(main):089:1> end
key =  animal   val =  dog
key =  sound   val =  bark
=> {:animal=>"dog", :sound=>"bark"}
```
 
96. Operations Inside Iteration
```ruby
irb(main):118:0> arr
=> ["a", "b", "c"]
irb(main):119:0> arr.each_with_index do |v,i|
irb(main):120:1*   arr[i] = "was " + v
irb(main):121:1> end
=> ["was a", "was b", "was c"]
irb(main):122:0> arr
=> ["was a", "was b", "was c"]
```
 
97. Arrays - Map, Select, and Reject
```ruby
irb(main):123:0> arr = ['red','green','yellow']
=> ["red", "green", "yellow"]
irb(main):129:0> arr2 = arr.select do |el|   # copying array except the target element
irb(main):130:1*            el != 'yellow' 
irb(main):131:1>        end
=> ["red", "green"]
irb(main):132:0> arr2
=> ["red", "green"]
irb(main):133:0> arr2 = arr.select { |el| el != "yellow"} #<--- same as above. One liner
=> ["red", "green"]
irb(main):134:0> arr2 = arr.reject { |el| el == "yellow"} #<--- Using reject
=> ["red", "green"]
irb(main):136:0> arr2 = arr.reject! { |el| el == "yellow"} #<--- In place modification using !
=> ["red", "green"]
irb(main):137:0> arr #<---- Now the original array is changed as well
=> ["red", "green"]
#
irb(main):138:0> arr = [1,2,3]
=> [1, 2, 3]
irb(main):139:0> arr2 = arr.map{ |i| i*2}
=> [2, 4, 6]
irb(main):140:0> arr2
=> [2, 4, 6]
irb(main):141:0> arr2 = arr.map!{ |i| i*2}  #<--- In place modification using !
=> [2, 4, 6]
irb(main):142:0> arr
=> [2, 4, 6]```
irb(main):143:0> arr = [1,2,3]
=> [1, 2, 3]
irb(main):144:0> arr2 = arr.map{ |i| i%2 == 0 ? i*2 : i*3}  # Using ternary operation
=> [3, 4, 9]
```
 
98. Hashes - Map, Select, and Reject
```ruby
irb(main):145:0> h1 = {John: 100, Alice: 88, Tom: 79}
=> {:John=>100, :Alice=>88, :Tom=>79}
irb(main):146:0> h2 = h1.select {|k,v| v > 80}
=> {:John=>100, :Alice=>88}
irb(main):147:0> h2
=> {:John=>100, :Alice=>88}
irb(main):148:0> h2 = h1.map{|k,v| v *100}
=> [10000, 8800, 7900]
```
 
99. Introduction to Recursion
```ruby
irb(main):149:0> def recf(i)
irb(main):150:1>   return i if i > 5
irb(main):151:1>   i += 1
irb(main):152:1>   puts i
irb(main):153:1>   recf(i)
irb(main):154:1> end
=> :recf
irb(main):155:0> puts(recf(0))
1
2
3
4
5
6
6
=> nil
```
 
100. Codealong Exercise - Introduction
101. Codealong Exercise - Filters on Product Catalog
```ruby
products = [
              {category: :shoes, name: "Special Sandals", brand: "EZ", price: 10.0},
              {category: :clothes, name: "Hatty's Hat", brand: "Hatty's", price: 20.0},
              {category: :electronics, name: "Magnasound", brand: "Maximum", price: 100.0},
              {category: :shoes, name: "High Heels", brand: "Pricey", price: 30.0}
            ]

def filter(products_arr, filter_type)

  return nil unless filter_type.is_a?(Hash)

  filter_value = filter_type.values[0]
  products_arr.select do |product|
    case filter_type.keys[0]
    when :category
      product[:category] == filter_value
    when :name
      product[:name] == filter_value
    when :brand
      product[:brand] == filter_value
    when :price
      product[:price] <= filter_value
    else
      nil
    end
  end
end
p filter(products, { :category => :shoes })
puts "-----------"
puts "-----------"
p filter(products, { :name => "Magnasound" })
puts "-----------"
puts "-----------"
p filter(products, { :brand => "EZ" })
puts "-----------"
puts "-----------"
p filter(products, { :price => 25.0 })
puts "-----------"
puts "-----------"
```
102. Assignment - Find Students by Name or Age
 
## Section 6: Working with Files
103. Files in Ruby
104. Opening a File
105. File Modes
106. Reading the Contents of a File
```ruby
irb(main):004:0> f = File.open("foo.txt","r")
=> #<File:foo.txt>
irb(main):005:0> puts f.read
hello world
=> nil
irb(main):006:0> f.close()
=> nil
```
 
107. Reading a File Line by Line
```ruby
irb(main):003:0> File.open("foo.txt","r").each do |line|
irb(main):004:1*   puts line
irb(main):005:1> end
hello world
Bye
end
=> #<File:foo.txt>
irb(main):006:0> f = File.open("foo.txt","r")
=> #<File:foo.txt>
irb(main):007:0> f.readline()
=> "hello world\n"
irb(main):008:0> f.readline()
=> "Bye\n"
irb(main):009:0> f.readline()
=> "end\n"
irb(main):010:0> f.readline()
Traceback (most recent call last):
        3: from /usr/bin/irb:11:in `<main>'
        2: from (irb):10
        1: from (irb):10:in `readline'
EOFError (end of file reached)
```
 
108. Closing Files
109. Writing to an Existing File
```ruby
irb(main):012:0> File.open("bar.txt","w") do |f|
irb(main):013:1*   f.write("Ruby programming")
irb(main):014:1>   f.write("Testing")  #<---- no new line. "\n" must be used in write()
irb(main):015:1> end
```
110. Writing to a New File
111. Writing user Input to Files
```ruby
irb(main):005:0> f = File.open("bar.txt", "a")
=> #<File:bar.txt>
irb(main):011:0> 5.times do |i|
irb(main):012:1*   puts "Enter text"
irb(main):013:1>   inp = gets.chomp
irb(main):014:1>   f.write("${inp}\n")
irb(main):015:1> end
Enter text
hello
Enter text
world
Enter text
3rd
Enter text
4th
Enter text
5th
=> 5
irb(main):016:0> f.close()
=> nil
```
 
112. Introduction to CSV Files
113. Working with CSV Files
```ruby
irb(main):002:0> require 'csv'
=> true
irb(main):003:0> f = CSV.read('data.csv')
=> [["apple", "hello", "morning", "kitchen"], ["banana", "math", "phone"]]
irb(main):004:0> print f
[["apple", "hello", "morning", "kitchen"], ["banana", "math", "phone"]]=> nil
irb(main):006:0> f[0][0]
=> "apple"
irb(main):007:0> f.first.first
=> "apple"
irb(main):008:0> s1 = "another,string,csv,test"
=> "another,string,csv,test"
irb(main):009:0> p1 = CSV.parse(s1)
=> [["another", "string", "csv", "test"]]
irb(main):010:0> puts p1
another
string
csv
test
=> nil
irb(main):014:0> CSV.open("data.csv","a") do |csv|
irb(main):015:1*   csv << p1  # must be a form of an array
irb(main):016:1> end
```
 
114. Check if a File Exists
```ruby
irb(main):017:0> File.exists?('some.txt')
=> false
```
- File/Dir.exists?() deprecates. Use File/Dir.exist?()
 
115. Working with Directories
```ruby
irb(main):020:0> Dir.pwd
=> "/home/foo/HW"
irb(main):021:0> Dir.chdir("./ruby")
=> 0
irb(main):022:0> Dir.exists?("./tmp")
=> false
irb(main):025:0> Dir.entries(".")
=> [".", "..", "hello.rb", "note.md", "foo.txt", "bar.txt", "data.csv"]
```
 
116. Opening Other Ruby Files
- Use `require`
    - Those *.rb files must be in path
   
117. Require Local Ruby Files
- Use `require_relative`
    - Objects are loaded but not local variables
```bash
$ cat local_ruby.rb
t1 = "sample text"
def say_hello(message)
  puts message
end
$ irb
irb(main):001:0> require_relative 'local_ruby'
=> true
irb(main):002:0> say_hello("world")
world
=> nil
irb(main):003:0> t1 #<----- not recognized
Traceback (most recent call last):
        2: from /usr/bin/irb:11:in `<main>'
        1: from (irb):3
NameError (undefined local variable or method `t1' for main:Object)
irb(main):004:0>
```
 
118. Codealong Exercise - Introduction
119. Codealong Exercise - Collecting Todo Notes
```ruby
not_exit = true

while not_exit do
  print "Please enter your todo or exit: "
  input = gets.chomp

  if input == 'exit'
    puts "Thank you for using the app"
    not_exit = false
  else
    todo = "TODO: #{input}\n"

    File.open('todos.txt','a') do |file|
      file.write(todo)
    end
  end
end
```

120. Assignment - Write Student Information to CSV
 
## Section 7: Object Oriented Programming
 
121. Introduction to Classes
```ruby
irb(main):004:0> puts Array.class
Class
=> nil
irb(main):005:0> puts 1.class
Integer
=> nil
irb(main):006:0> puts "Hello world".class
String
=> nil
irb(main):012:0> puts Integer.class
Class
=> nil
irb(main):013:0> puts Class.class
Class
=> nil
irb(main):015:0> class Person
irb(main):016:1>   def walk
irb(main):017:2>     puts "walking now"
irb(main):018:2>   end
irb(main):019:1> end
=> :walk
irb(main):020:0> p1 = Person.new  # Instantiating Person class
=> #<Person:0x0000556930c83220>
irb(main):021:0> p1.walk
walking now
=> nil
```
 
122. Introduction to Objects
- Instances of classes
 
123. Instantiating Objects
```ruby
irb(main):026:0> puts p1.class.superclass
Object
=> nil
irb(main):027:0> puts p1.class.superclass.superclass
BasicObject
=> nil
irb(main):065:0> class Person
irb(main):066:1>   def initialize(fname, lname)
irb(main):067:2>     @firstname = fname  #<--- instance variables
irb(main):068:2>     @lastname = lname
irb(main):069:2>   end
irb(main):070:1>   def say_my_name
irb(main):071:2>     puts "my name is #{@firstname} #{@lastname}"
irb(main):072:2>   end
irb(main):073:1> end
=> :say_my_name
irb(main):074:0> p1 = Person.new("James", "Black")
=> #<Person:0x0000556930c4daf8 @firstname="James", @lastname="Black">
irb(main):075:0> p1.say_my_name()
my name is James Black
=> nil
```
 
124. Displaying Objects
```ruby
irb(main):078:0> p p1.object_id;
46955139067260
=> 46955139067260
irb(main):079:0> p "hello world".object_id;
46955138996400
=> 46955138996400
```
 
125. Comparing Objects
```ruby
irb(main):080:0> p2 = Person.new("James", "Black")
=> #<Person:0x0000556930c21a98 @firstname="James", @lastname="Black">
irb(main):081:0> p1 == p2
=> false
irb(main):082:0> p1 === p2
=> false
irb(main):083:0> puts p1.object_id, p2.object_id
46955139067260
46955138977100
=> nil
```
- Over-writing operation ==
```ruby
irb(main):135:1>   def say_my_name
irb(main):136:2>     puts "my name is #{@firstname} #{@lastname}"
irb(main):137:2>   end
irb(main):138:1>   def first_name
irb(main):139:2>     return @firstname
irb(main):140:2>   end
irb(main):141:1>   def last_name
irb(main):142:2>     return @lastname
irb(main):143:2>   end
irb(main):144:1>   def ==(other)
irb(main):145:2>     (@firstname == other.first_name) && (@lastname == other.last_name)
irb(main):146:2>   end
irb(main):147:1> end
=> :==
irb(main):148:0> a = Person.new("foo","bar")
=> #<Person:0x00005569306fe020 @firstname="foo", @lastname="bar">
irb(main):149:0> b = Person.new("foo","bar")
=> #<Person:0x0000556930c833d8 @firstname="foo", @lastname="bar">
irb(main):150:0> a == b  # == operation is overrided and true is generated
=> true
```
 
126. Duck-typing in Ruby
- An application of duck test: if it walks like a duck and it quacks like a duck, then it must be a duck.
```ruby
irb(main):152:0> puts Person.is_a?(Array)
false
=> nil
irb(main):153:0> puts p1.is_a?(Person)
true
=> nil
```
- object.respond_to?(:METHOD) : can check if the object has a method named METHOD (: in the head)
```ruby
irb(main):154:0> class Person
irb(main):155:1>   def reverse
irb(main):156:2>     puts "reverse message"
irb(main):157:2>   end
irb(main):158:1> end
=> :reverse
irb(main):159:0> def check_it(obj)
irb(main):160:1>   obj.respond_to?(:reverse)
irb(main):161:1> end
=> :check_it
irb(main):162:0> puts check_it(Person.new)
true
=> nil
irb(main):163:0> puts check_it("string sample")
true
=> nil
```
 
127. Methods on Objects
```ruby
irb(main):164:0> puts self
main
=> nil
irb(main):165:0> puts self.class
Object
=> nil
irb(main):178:0> p1 = Person.new()
irb(main):180:0> p1.methods
=> [:walk, :describe_self, :reverse, :say_my_name, :==, :first_name, :last_name, :check_it, :instance_variable_set, :instance_variable_defined?, :remove_instance_variable, :instance_of?, :kind_of?, :is_a?, :tap, :instance_variable_get, :instance_variables, :method, :public_method, :singleton_method, :define_singleton_method, :public_send, :extend, :to_enum, :enum_for, :pp, :<=>, :===, :=~, :!~, :eql?, :respond_to?, :freeze, :inspect, :object_id, :send, :to_s, :display, :nil?, :hash, :class, :singleton_class, :clone, :dup, :itself, :yield_self, :taint, :tainted?, :untrust, :untaint, :trust, :untrusted?, :methods, :frozen?, :protected_methods, :singleton_methods, :public_methods, :private_methods, :!, :equal?, :instance_eval, :instance_exec, :!=, :__send__, :__id__]
```
 
128. Inheritance
```ruby
irb(main):182:0> class Foo
irb(main):183:1> end
=> nil
irb(main):184:0> class Bar < Foo
irb(main):185:1> end
=> nil
irb(main):186:0> p Foo.superclass
Object
=> Object
irb(main):187:0> p Bar.superclass
Foo
=> Foo
```
 
129. Overriding Methods
```ruby
irb(main):188:0> class Animal
irb(main):189:1>   def breathe
irb(main):190:2>     puts "Breathe in and out"
irb(main):191:2>   end
irb(main):192:1>  end
=> :breathe
irb(main):193:0> class Bear < Animal
irb(main):194:1>    def breathe
irb(main):195:2>     puts "Bear breathes"
irb(main):196:2>    end
irb(main):197:1> end
=> :breathe
irb(main):198:0> b1 = Bear.new
=> #<Bear:0x0000556930c63240>
irb(main):199:0> b1.breathe
Bear breathes
=> nil
irb(main):205:0> class Bear < Animal
irb(main):206:1>    def breathe
irb(main):207:2>     puts "Bear breathes"
irb(main):208:2>     super        # <--- superkeyword
irb(main):209:2>    end
irb(main):210:1> end
=> :breathe
irb(main):211:0> b1 = Bear.new
.breathe=> #<Bear:0x0000556930c31a88>
irb(main):212:0> b1.breathe
Bear breathes        #<-- Method of the child object
Breathe in and out   #<-- Method of the parent object
=> nil
```
 
130. Attributes - Getting and Setting
```ruby 
irb(main):225:0> class Person
irb(main):226:1>   attr_accessor :first_name, :last_name # attr_accessor: read and write, # attr_reader: read only
initialirb(main):227:1>   def initialize(fname, lname)
irb(main):228:2>     @first_name = fname
irb(main):229:2>     @last_name = lname
irb(main):230:2>   end
irb(main):231:1> end
=> :initialize
irb(main):232:0> p1 = Person.new("John", "Smith")
=> #<Person:0x0000556930acf230 @first_name="John", @last_name="Smith">
irb(main):233:0> p1.first_name
=> "John"
irb(main):234:0> p1.first_name = "Alice"
=> "Alice"
irb(main):235:0> p1
=> #<Person:0x0000556930acf230 @first_name="Alice", @last_name="Smith">
``` 
 
131. Class Methods
```ruby
irb(main):280:0> class Person
irb(main):281:1>   def self.is_human? # works like static in C++ class
irb(main):282:2>   true
irb(main):283:2>   end
irb(main):284:1>   def self.is_fish?
irb(main):285:2>   false
irb(main):286:2>   end
irb(main):287:1> end
irb(main):290:0> Person.is_human?
=> true
```
- Class method cannot be called from instance
  - Imagine static method in C++ class
- Instance method cannot be called from class

132. Modules
133. Requiring Modules
- Mixin: a class that contains methods for use by other classes without having to be the parent class of those other classes
  - Ref: https://en.wikipedia.org/wiki/Mixin

134. Composition vs Inheritance
- Inheritance: Inherit methods from parent class
- Composition: include methods from module. Something irrelevant to class  characteristics
```ruby
irb(main):316:0> module Flight
irb(main):317:1>   def can_fly?; true ; end
irb(main):318:1>   def has_wings?; true; end
irb(main):319:1> end
=> :has_wings?
irb(main):320:0> class Mammal
irb(main):321:1>   def warm_blood?; true; end
irb(main):322:1> end
=> :warm_blood?
irb(main):323:0> class Bat <Mammal
irb(main):324:1>   include Flight
irb(main):325:1> end
=> Bat
irb(main):326:0> b1 = Bat.new
=> #<Bat:0x0000556930aad360>
irb(main):327:0> b1.can_fly?
=> true
irb(main):328:0> b1.warm_blood?
=> true
```

135. Codealong Exercise - Introduction
136. Codealong Exercise - Calculate Area of Different Shapes
```ruby
class Shape
  def is_shape?
    true
  end
end

class Quadrilateral
  attr_accessor :side

  def initialize(side = 0)
    @side = side
  end

  def calculate_area
    side * side
  end
end

class Circle < Shape
  attr_accessor :radius

  def initialize(radius = 0)
    @radius = radius
  end

  def calculate_area
    3.14 * radius * radius
  end
end

class Square < Quadrilateral
end

class Rectangle < Quadrilateral
  attr_accessor :other_side

  def initialize(side, other_side)
    @other_side = other_side
    super(side)
  end

  def calculate_area
    other_side * side
  end
end

p Square.new(5).calculate_area
p Rectangle.new(5,6).calculate_area
p Circle.new(5).calculate_area
```

137. Assignment - Create a Student Object
138. Optional - Introduction to Binary Trees
138. Optional - Implementing a Binary Tree
```ruby
class TreeNode
  # Setting up the attributes used in the node
  attr_accessor :left, :right, :value

  # Set the node value when creating the node
  # The left and right branches start out empty
  def initialize(value)
    @value = value
    @left = @right = nil
  end

  # This method will insert a new node as long
  # as the input is of type TreeNode and
  # the node value is not already somewhere on the tree
  def insert(node)
    raise 'This is not a valid node' unless node.is_a? TreeNode

    if(node.value < @value)
      if @left.nil?
        @left = node
      else
        @left.insert(node)
      end
    elsif(node.value > @value)
      if @right.nil?
        @right = node
      else
        @right.insert(node)
      end
    end
  end

  # This method searches for a value on the tree
  # by looking at the root node, and then either
  # on the left or right side of the tree
  # depending on whether the value is greater or
  # lesser than the root node value
  def search(value)
    return true if value == @value
    if(value < @value)
      return false if left.nil?
      left.search(value)
    elsif(value > @value)
      return false if right.nil?
      right.search(value)
    end
  end
end

# This is the creation of the tree
tree = TreeNode.new(6)

tree.insert(2)
tree.insert(10)
tree.insert(8)
tree.insert(4)

# We can then search the tree for
# values that do and do not exist
# in the tree and print the results

puts tree.search(6)
puts tree.search(3)
puts tree.search(8)
```

## Section 8: Object Mapping

140. Model Classes
141. Domain Modeling
142. Validator Methods
```ruby
class Person
  attr_accessor :first_name, :last_name
  def valid?
  (!first_name.nil? && first_name.length > 0) &&
  (!last_name.nil? && last_name.length > 0)
  end
  def walk; end
  def talk; end
end
p1 = Person.new()
p1.valid?
p1.first_name = "Bob"
p1.last_name = "Hanson"
p1.valid?
```

143. Serialization
144. Converting an Object to CSV
```ruby
require 'csv'
class Person
  attr_accessor :first_name, :last_name
  def valid?
    (!first_name.nil? && first_name.length > 0) &&
    (!last_name.nil? && last_name.length > 0)
  end
  def to_csv
    CSV.generate do |csv|
     # csv << self  # not working
      csv << [first_name, last_name]
    end
  end 
end
p1 = Person.new()
p1.first_name = "Bob"
p1.last_name = "Hanson"
p p1.to_csv
```

145. CRUD
- Create/Read/Update/Delete

146. Writing Objects to File
147. Generating Unique IDs and Filenames
148. Reading Objects from File
149. Updating Objects in File
- Read data for an instance: instance method (read)
- Read ID to bring up an instance: class method (self.read)

150. Deleting Objects in Files
151. Validations and Review
152. Codealong Exercise Intro
153. Codealong Exercise - Corporate Directory in CSV
```ruby
require 'csv'

class DirectoryEntry
  attr_accessor :name, :position, :active

  def initialize(name, position, active = true)
    @name     = name
    @position = position
    @active   = active
  end

  def save
    File.open('directory.csv','a') do |entry|
      entry.write(self.to_csv)
    end
  end

  def to_csv
    CSV.generate do |csv|
      csv << [name, position, active]
    end
  end

  def self.list
    CSV.open('directory.csv','r') do |csv|
      display(csv.read)
    end
  end

  def self.display(list_of_entries)
    list_of_entries.each do |entry|
      p "#{entry[0]} - #{entry[1]} active: #{entry[2]}"
    end
  end
end

entry = DirectoryEntry.new("evgeny","instructor")
entry.save

entry2 = DirectoryEntry.new("alice","ceo",false)
entry2.save

DirectoryEntry.list
```

154. Assignment - Use Student Object to Write to File

## Section 9: Web Scraping Automation with Ruby

155. Using Programming for Daily Life Tasks
156. Problem Description - Car Shopping
157. Solution Preview
- https://code.evgenyrahman.com/rubycourse/carlist.html

158. Program Setup
159. Resources for This Section
160. Introduction to Gems
- https://rubygems.org/
- We use httparty
  - `gem install httparty`

161. Getting the Content from the Website
162. Parsing Response HTML
- Another gem: nokogiri

163. Using the CSS Selector
164. Pretty Printing Ruby Objects
165. Extracting Data
166. Extracting Data - Continued
167. Saving to jSON File
168. JSON Formatting
-- Atom beautify package

169. Converting Price to a Number
170. Filtering Data
171. Optional - Using Bundler
- Keeps tracking gem installation
- https://bundler.io/
- Edit `Genfile` in the code folder
```ruby
source 'https://rubygems.org'
gemspec
gem 'rake'
gem 'mongrel',  '1.2.0.pre2'
gem 'json'
```
- Run `bundle install` to install required Gems

172. Optional - Deep Dive into Ruby Gem

## Section 10: Data Engineering with Ruby

173. Solution Overview - User Analytics Pipeline
174. Working with Access Logs
- Data pipeline engineer

175. Reading the Log File
176. Parsing the Log File
177. Extracting User Data
178. Determining Browsers
```ruby
def determine_browser(user_agent)
  return "Firefox" if user_agent.incldue?("Firefox") && user_agent.include?("Gecko")
  return "Chrome" if user_agent.include?("Chrome")
  return "Safari" if user_agent.include?("Gecko") && user_agent.include?("Safari")
  "Other"
end
```
179. Extracting Email
- match: regex search in ruby
```ruby
def extract_email(log_line)
  email = log_line.match(/signup\?email\=([a-zA-Z0-9@.]*) HTTP\//)
  #puts email
  email.captures
end
```

180. Cross Referencing Users
181. Cross Referencing Users - Continued
182. Users Data Structure
183. File Name Patterns

## Section 11: Final Project

184. Final Project Overview
185. Final Project Requirements
186. Thank You!!!

## Section 12: Appendix

187. Acknowledgements
