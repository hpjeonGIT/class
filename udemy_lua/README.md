## Lua Scripting: Master complete Lua Programming from scratch
- Instructor: Gustavo Pezzi

## Section 1: Introduction

### 1. Introduction and Motivations

### 2. A Message for Roblox Users

### 3. History and Language Evolution
- Lua can be an additional script language on top of your current programming languages
- Runs on any embedded device which has C-compiler
- Very light
  - v5.3 has 9000 lines of C
- Syntax is easy and simple
- Kernel system is kept safe

### 4. Why do you want to learn Lua?

## Section 2: Installing and Using Lua

### 5. A Quick Message About Installing Lua

### 6. Using Lua with REPL
- https://replit.com/languages/lua
  - No install of Lua at local machine. Use through web-page

### 7. Installing Lua on Linux
- At Ubunt20: `sudo apt install lua5.3`
- Demo:
```bash
$ lua
Lua 5.3.3  Copyright (C) 1994-2016 Lua.org, PUC-Rio
> os.exit()
```

### 8. Installing Lua on Mac OS

### 9. Installing Lua on Windows

## Section 3: Course Source Code

### 10. Source Code (Download)

## Section 4: Variables and Expressions

### 11. Out First Lua Script
```lua
print("hello world!")
```
- Demo:
```bash
$ lua ch11.lua 
hello world
```
- Lua is case-sensitive

### 12. Variables
```lua
print("Examples of variables:")
score = 0
lives = 3
pname = "James Martin"
print(type(score))
```
- `type()` prints the data type of the variable
- Data type
  - nil
  - number
  - string
  - function
  - CFunction
  - userdata
  - table
- To join a number with a string, use `..`
```bash
> x = 10/2
> print("result is "..x)
result is 5.0
```

### 13. Older Lua Versions and REPL
- REPL version might be very old
- `//`: integer division. May not work at ver 5.1 or lower
  - May use math.floor()
```bash  
> math.floor(25/2)
12
```

### 14. Proposed Activity: Variables
- Comment in Lua: `--` in the first column
- Multi-line comments using `--[[ ... ]]--`
- Ref: https://www.codecademy.com/resources/docs/lua/comments

### 15. Variables Acitivity Solution
```bash
> name = "james"
> print(nam)
nil
```
- Lua will not complain or error out for typo - undefined varialbe. Only 'nil' is printed
```bash
> print("Name is "..nam)
stdin:1: attempt to concatenate a nil value (global 'nam')
stack traceback:
	stdin:1: in main chunk
	[C]: in ?
```
  - Concatenating mark `..` will find typo or undefined variable
- `++` or `+=` not working. Use `n = n+1` or `n = n-1`

### 16. Exercises: Variables and Expressions

### 17. Incrementing and Decrementing Variables

## Section 5: Conditionals and Logical Operators

### 18. Conditionals
```lua
level = 1
score = 0
if score >=1000 then
  level = level + 1
end
```
- In conditional statement, () can be applied
```lua
level = 1
score = 0
time_elapsed = 0
if (score >=1000) then
  level = level + 1
else
  time_elapsed = time_elapsed + 1 
end
print("Level: " .. level)
print("Time:  " .. time_elapsed)
```

### 19. Elseif
- Equal: `==`
- Not equal: `~=`
- \>, \>=, \<, \<=
```lua
if ... then
...
elseif ... then
...
else
...
end
```

### 20. Proposed Activity: Conditional Statements

### 21. Conditionals Acivity Solution
```lua
> string.lower("HELLO")
hello
```

### 22. Logical Operators
- `and`
- `or`
- `not`: negate

### 23. Popular Logical Operators

### 24. Exercises: Conditionals

## Section 6: Strings and Standard Library Module

### 25. String Manipulation
```lua
> email = "ABC@email.com"
> string.lower(email)
abc@email.com
> string.upper(email)
ABC@EMAIL.COM
> color = "#ce10e3"
> pure_color = string.gsub(color,"#","")
> pure_color
ce10e3
> print(string.sub(color,2,4)) -- from 2 to 4th index
ce1
> print(#color) -- #color returns the size of string in color
7
> print(string.sub(color,2,#color))
ce10e3
> string.find(email,'@') -- returns the first index of @ in the string
4	4
```
- gsub: global substitution
- Lua string has 1 based index like Fortran and R
- Adding `#` to the variable shows the length of the string
  - Only the string variable 

### 26. Muilti-line Strings
- For the method of objects, use `:`
```lua
> abc = "hello"
> abc:upper()
HELLO
> file = io.open("./ch19.lua","r") -- local file = ... yields nil
> text = file:read("all") -- local text = ... yields nil
> file:close()
true
> text
...
```

### 27. Reading the Contents of a File
- Using arguments when running Lua file
```lua
print(arg[0],arg[1])
file = io.open(arg[1],"r")
text = file:read("all")
file:close()
print(text)
```
- Arguments are 0 based index
- Demo:
```bash
$ lua ch27.lua text.csv
ch27.lua	text.csv
1 hello
2 world
3 bye
4 April
```

### 28. Multiple Assignment
```lua
> x = y = 0 -- not working
stdin:1: unexpected symbol near '='
> x,y=0,0 -- works OK
> x,y,z = 0,0,0
> print("x="..x.." y="..y.." z="..z)
x=0 y=0 z=0
```

### 29. Standard Library Modules
- os.date()
- os.difftime(t2,t1)
- os.remove(filename)
- os.rename(oldname, newname)
- os.time()
- os.exit()
- math.sqrt(x)
- math.abs(x)
- math.cos(x)
- math.sin(x)
- math.tan(x)
- math.atan(x)
- math.floor(x)
- math.ceil(x)
- math.random(a,b) -- a,b are integers of the range: [a,b]
- math.randomseed(seed) -- may use os.time() as the seed
```lua
> math.pi
3.1415926535898
> math.random() -- [0,1.0)
0.78309922339395
> math.random()
0.79844003310427
> math.random(100,200)
192
> math.randomseed(os.time())
```

### 30. Exercises: Strings and Random

### 31. Patterns in String Find
- string.find() uses regex
  - Escape character in Lua is `%`
```
> email = "abc@hotmail.com"
> string.find(email,".")
1	1
> string.find(email,"%.")
12	12
```

### 32. Special Characters in Lua Patterns
- `.`: any character
- `%a`: A-Z and a-z
- `%c`: all control characters like null, tab, return, linefeed, ...
- `%d`: 0-9
- `%l`: lower case letters a-z
- `%p`: all punctuation characters .,?!;;@[]_{}~
- `%s`: all white space characters tab, return, linefeed, space, ...
- `%u`: upper case letters A-Z
- `%w`: alphanumeric characters A-Za-z0-9
- `%x`: Hexadecimal digits 0-9A-Fa-f
- `%.`: dot
- `[set]`: characters inside of []. `[%w~]` means all alphanumeric + `~`
- `[^set]`: complementary of []

## Section 7: Loops and Functions

### 33. The For Loop
```lua
for i=1,3 do
  print(i)
end
```
- Or `for i=start,end,increment do ... end`
```lua
> for count=1,10,3 do print(count) end
1
4
7
10
```

### 34. The While Loop
```lua
math.randomseed(os.time())
x = 0
while x < 5 do
  x = math.random(0,10)
  print("X="..x)
end
```

### 35. Loop Options
- for loop vs while loop

### 36. Variable Scope
- `{ ... }`
- Variables within while loop are global
  - To localize, use `local`
```lua
math.randomseed(os.time())
x = 0
while x < 3 do
  local y = math.random(0,10)
  print("X="..x.." Y="..y)
  if x < y then
    x = x + 1
  end
end
print(y) -- this will print nil
```

### 37. Local Scope and Lua Chunks
- Lua chunk: each lua file
- Each local variable is local within each lua chunk

### 38. Syntax & Semantic Errors

### 39. Solving Logical Mistakes

### 40. Input Variables from the Keyboard
- How to receive inputs from the screen
  - `io.read()`
```lua
print("+---------------+")
print("| Welcome, "..os.date())
print("| Select menu ")
print("+---------------+")
--
print("Select option:")
user_option = io.read("*n")
if user_option == 1 then
  print(" one ")
elseif user_option == 2 then
  print(" two ")
else
  print("others")
end
```

### 41. Handling Input Options

### 42. Different Input Options
- `txt = io.read("a")`: reads all input
- `txt = io.read("*a")`: reads all input
- `txt = io.read("*n")`: reads a number
- `txt = io.read("*l")`: reads one line
- `txt = io.read()`: reads one line
- `txt = io.read(4)`: reads 4 characters
- `a,b = io.read(4,6)`: reads 4 and 6 characters, storing into a and b
- `a,b = io.read("*n","*n")`: reads two numbers, storing into a and b
- `x = tonumber(io.read())`: read the input and convert to a number

### 43. Finding Distance B/W Points

### 44. Finding Angle B/W Points
- math.atan2(): an optimized version of math.atan()
  - This will be deprecated in the newer version
  - Just use math.atan()

### 45. Arctangent Function in New Lua Versions

### 46. Loop Activity Example
```lua
x = -1
sum = 0
ncount = 0
while x ~= 0 do
  x = io.read("*n")
  sum = sum + x
  ncount = ncount + 1
end
print ("Sum ="..sum)
print ("Average = "..sum/ncount)
```
- Demo:
```bash
$ lua ch46.lua 
3
2
1
0
Sum =6
Average = 1.5
```

### 47. Loop Activity Solution
- Using `repeat ... until ...`
  - Reverse of `while ... do ... end`
```lua
num = 1
repeat
  print(num)
  num = num + 1
until num > 3
```

### 48. Exercises: Loops

### 49. An Introduction to Functions
```lua
function display_menu(x)
  print(" This is menu ----"..x)
end
function diff(x,y)
  return (x - y)
end
display_menu(1)
display_menu(2)
print(diff(3,1))
```

### 50. Exercise: Functions

## Section 8: Tables

### 51. Tables in Lua
```lua
> a = {1,2,3}
> print(a[1])
1
> b= {["jim"] = 4.13, ["john"] = 111.1, ["alice"] = 92.2}
> print(b["john"])
111.1
> c = { jim=1.12, john=4.1, alice = 22.3}
> c.jim
1.12
> print(c[alice])
nil
> print(c["alice"])
22.3
> for k,v in pairs(c) do
>> print(k..v)
>> end
john4.1
jim1.12
alice22.3
> c = {1,2,3}
> for k,v in ipairs(c) do print(k.."th = "..v) end
1th = 1
2th = 2
3th = 3
> x = {[1] = 123, [2] = 456}
> for k,v in ipairs(x) do print(k.."th = "..v) end
1th = 123
2th = 456
```
- Lua table uses 1-based index
- Key-value table can be defined as shown above
  - When defining, quotation marks might not be necessary but calling key requires the quotation marks
  - pairs() returns key-value pairs, **without order**
    - Works for key-value table
  - ipairs() returns index-value pairs, in order
    - Works for array table (no-key) or key-value where keys are integers {[1]=123, [3]=233}
- Key element can be accessed using dot (Ex: c.jim)

### 52. Looping Key-Value Pairs in Lua Entries

### 53. Tables Example
- Multiple keys

### 54. Proposed Activity: Reading Table Entries
- Nested table
```lua
> codec = { {page=1,line=1,word=1,code="W"}, {page=1,line=1,word=2,code="O"}, 
>> {page=1,line=2,word=1, code="X"},{page=2,line=1,word=2, code="C"}}
> codec[1]["page"]
1
> codec[1]["word"]
1
> codec[1]["code"]
W
```

### 55. Table Activity Solution
```lua
> codec[1].page
1
> for i,entry in ipairs(codec) do 
>>   if entry.page == 1 and entry.word==2 then
>>      print(entry.code)
>>   end
>> end
O
```

### 56. Table as Configuration Files
- Use index as key to clarify the index
```lua
game_conf = {
  [1] = {
    name = "army1",
    layer = 2,
    initial_pos = math.random()
  },
  [2] = {
    name = "army2",
    layer = 2,
    initial_pos = math.random()
  },
  ...
  [99] = {
    name = "background",
    layer = 100,
    initial_pos = math.random()
  },
}
```
- Json or xml can do similar purpose but they cannot embed math functions as values

### 57. Tables as Matrices
```lua
M = { 
  {3.4, 1.2, 3.3},
  {1.0, 1.8, 0.0},
  {1.0, 2.0, 3.0}
}
```

### 58. Proposed Formative Project
```lua
board = {}
function clear_board()
  for i=1,3 do
    board[i] = {}
    for j =1,3 do
      board[i][j] = " "
    end
  end
end
function display_board()
   print("   [1][2][3]")
   print("[1]["..board[1][1].."]["..board[1][2].."]["..board[1][3].."]")
   print("[2]["..board[2][1].."]["..board[2][2].."]["..board[2][3].."]")
   print("[3]["..board[3][1].."]["..board[3][2].."]["..board[3][3].."]")
end
function board_full()
  isfull=true
  for i=1,3 do
    for j=1,3 do
      if board[i][j] == " " then
          isfull = false
      end
    end
  end
  return isfull
end
function check_winner()
  winner = nil
  if board[1][1] ~= " " then
    if ( (board[1][1] == board[1][2]) and (board[1][2] == board[1][3]) )
    or ( (board[1][1] == board[2][2]) and (board[2][2] == board[3][3]) ) 
    or ( (board[1][1] == board[2][1]) and (board[2][1] == board[3][1]) ) then
      winner = board[1][1]
    end
  end
  if board[3][1] ~= " " then
    if (board[3][1] == board[3][2]) and (board[3][2] == board[3][3])  then
      winner = board[3][1] 
    end
  end
  if board[1][3] ~= " " then
    if (board[1][3] == board[2][3]) and (board[2][3] == board[3][3])  then
      winner = board[1][3]
    end
  end
  return winner
end
player = "X"
move = 0
```
- Demo:
```bash
$ lua ch58.lua 
   [1][2][3]
[1][ ][ ][ ]
[2][ ][ ][ ]
[3][ ][ ][ ]
Enter the row/col for X: 
3 1
   [1][2][3]
[1][ ][ ][ ]
[2][ ][ ][ ]
[3][X][ ][ ]
Enter the row/col for O: 
2 2
   [1][2][3]
[1][ ][ ][ ]
[2][ ][O][ ]
[3][X][ ][ ]
Enter the row/col for X: 
3 3
   [1][2][3]
[1][ ][ ][ ]
[2][ ][O][ ]
[3][X][ ][X]
Enter the row/col for O: 
2 1
   [1][2][3]
[1][ ][ ][ ]
[2][O][O][ ]
[3][X][ ][X]
Enter the row/col for X: 
3 2
   [1][2][3]
[1][ ][ ][ ]
[2][O][O][ ]
[3][X][X][X]
Winner is X
```

## Section 9: Metatables and Object-Oriented Programming

### 59. Moving Forward

### 60. Metatables
- Operator overloading for tables
- Meta methods
  - __index
  - __newindex
  - __add
  - __sub
  - __mul
  - __concat
  - __call
  - __tostring
```lua
local meta = {}
local v3d = {}
-- constructor
function v3d.new(x,y,z)
  local v = {x=x,y=y,z=z}
  setmetatable(v,meta)
  return v
end
-- operator overloading
function v3d.add(v1,v2)
  return v3d.new(v1.x+v2.x,v1.y+v2.y,v1.z+v2.z)
end
meta.__add = v3d.add
function v3d.tostring(v)
  return "("..v.x..","..v.y..","..v.z..")"
end
meta.__tostring = v3d.tostring
-- Sample
pos = v3d.new(1.0, 2.0, 3.3)
vel = v3d.new(10.0, -2.1, 0.0)
local result = vel  + pos
print(pos)
print(vel)
print(result)
```


### 61. Exercises: Metatables
- Adding sub and mul
```lua
local meta = {}
local v3d = {}
-- constructor
function v3d.new(x,y,z)
  local v = {x=x,y=y,z=z}
  setmetatable(v,meta)
  return v
end
function v3d.add(v1,v2)
  return v3d.new(v1.x+v2.x,v1.y+v2.y,v1.z+v2.z)
end
meta.__add = v3d.add
function v3d.sub(v1,v2)
  return v3d.new(v1.x-v2.x,v1.y-v2.y,v1.z-v2.z)
end
meta.__sub = v3d.sub
function v3d.mul(v1,v2)
  return v3d.new(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z)
end
meta.__mul = v3d.mul
function v3d.tostring(v)
  return "("..v.x..","..v.y..","..v.z..")"
end
meta.__tostring = v3d.tostring
--
pos = v3d.new(1.0, 2.0, 3.3)
vel = v3d.new(10.0, -2.1, 0.0)
local result = vel  + pos
print(pos)
print(vel)
print(result)
print(vel-pos)
print(vel*pos)
```

### 62. Object-Oriented Programming in Lua
- Object: collection of data and functions

### 63. Creating Classes and Objects
```lua
-- Attributes or member data
BankAccount = {
  account_number = 0,
  holder_name = "",
  balance = 0.0
} 
-- Methods
function BankAccount:deposit(amount)
  self.balance = self.balance + amount
end
function BankAccount:withdraw(amount)
  self.balance = self.balance - amount
end
-- Constructor
function BankAccount:new(t)
  t = t or {}  -- given table or empty
  setmetatable(t,self)
  self.__index = self
  return t
end
-- Demo
johns_account = BankAccount:new({
   account_number = 12345,
   holder_name = "John Kent",
   balance = 0.0
})
johns_account:deposit(500)
print("For John:"..johns_account.balance)
```

### 64. Exercises: Classes and Objects

## Section 10: More on Lua functions

### 65. Higher-order functions and closures
- Functions as first-class values
- Functional programming
- Higher-order functions
  - Takes one or more functions as arguments
  - Returns a function as its result
  - Lambda function
- Closures
  - Local data in a function are available to the function that are defined in the current function

### 66. Variadic Functions
- `...` allows function to handle an unknown number of arguments
```lua
function find_max(...)
  local n = 0
  local arg = {...}
  local max = arg[1]
  for i, el in ipairs(arg) do
    if el > max then
     max = el
    end
    n = n + 1
  end
  return n, max
end
local n,max = find_max(1,2,3,99,7)
print("Max = "..max.." out of "..n.." elements")
```

### 67. Coroutines
- Not parallel programming but multi-thread run

## Section 11: Integrating Lua with C

### 68. Working with Lua and C

### 69. C Project Folder Structure
- lua-c
  - Makefile
  - lib
    - lua
      - src: Lua 5.3 src code
  - scripts
    - myscript.lua
  - src
    - main.c

### 70. Executing Lua File From C

### 71. Get Lua Global Values in C

### 72. The Stack

### 73. Push Pop and Peak

### 74. Calling Lua Functions From C

### 75. Checking and Handling Script Errors

### 76. Calling C Functions in Lua

### 77. Userdata

### 78. Sending and Receiving Userdata

### 79. Reading Lua Tables in C

### 80. Installing SDL

### 81. Creating a SDL Window

### 82. The Game Loop

### 83. SDL Rendering

### 84. Fixing Our Game Loop Timestep

### 85. Delta Time

### 86. Controlling the Player Movement with Lua

### 87. Proposed Exercise: Creating the function rect()

### 88. Final Considerations on Integrating Lua and C

## Section 12: Conclusion and Next Steps

## Section 13: Bonus Section: Using Lua with Roblox Studio

## Section 14: Bonus Section: Building Lua 5.4 from Source

## Section 15: Bonus Lecture
