## Erlang: The Complete Beginner's Guide
- Instructor: Catalin Stefan

## Section 1: Introduction

### 1. What will we talk about

### 2. How to take this course

### 3. Important message about Udemy reviews

### 4. Engage on social media

## Section 2: Development Environment

### 5. Install Erlang on Mac

### 6. Install IntelliJ IDE on Mac

### 7. Install Erlang on a PC
- At Ubuntu, `sudo apt-get install erlang`

### 8. Install the IntelliJ IDE on a PC

### 9. How to get the code
- Ref: https://github.com/CatalinStefan/LearnErlang

## Section 3: Hello World

### 10. Let's write some code
- hw10.erl:
```erl
-module(hw10).
%%API
-export([helloworld/0]).
helloworld() -> "Hello World".
```
- Running from erlang console
```bash
$ erl
Erlang/OTP 22 [erts-10.6.4] [source] [64-bit] [smp:4:4] [ds:4:4:10] [async-threads:1]
Eshell V10.6.4  (abort with ^G)
1> c(hw10).
{ok,hw10}
2> hw10:helloworld().
"Hello World"
```
- To exit the console, `^G` then enter `q`

### 11. Hello World explanation
- `-module()` is a container or a functionality. Contains the name
- `.` indicates the end point
- `%%` for comments
- `-export` tells what functions this module has. Then the list of functions is addressed. `/0` means 0 argument
```erl
-module(hw11).
%%API
-export([helloworld/0, hi/0]).
helloworld() -> "Hello World".
hi() -> "Hi world".
```

## Section 4: Language Basics

### 12. Functions and Recursive Functions
- Sample factorial function code:
```erl
-module(hw12).
%% Comment
-export([factorial/1]).
factorial(1) ->
 1;
factorial(N)-> 
N*factorial(N-1).
```

### 13. Operators
- Arithmetic
  - \+-*/
  - rem
  - div
- Relational
  - \==
  - \/=
  - \<
  - \=<
  - \>
  - \>=
- Logical
  - or
  - and
  - not
  - xor
- Bitwise
  - band
  - bor
  - bxor
  - bnot

### 14. Atoms
- Name with small letter. Literal
- temp.erl:
```erl
-module(temp).
%% comment
-export([convert/2]).
convert(F,fahrenheit)-> (F-32)*5/9;
convert(C,celsius)-> C*9/5 +32.
```
- Build and test
```bash
13> c(temp).
{ok,temp}
14> temp:convert(32,fahrenheit).
0.0
15> temp:convert(temp:convert(100,fahrenheit),celsius).
100.0
```

### 15. Data types
- Boolean: true/false
- Number: 3, 5., $a (ASCII value of the char)
- String: "hello world"
- Atom: `celsius`, `fahrenheit`
- Function: `convert()` shown above
- Tuple: `{First,Second}`
- List: `[A,B,C]`
- Map: `#{a=>2, b=>3}`
- PortId: `pid`
- ProcessId: `<0.130.0>`

### 16. Tuples
- Temperature conversion using tuples
```erl
-module(temp).
%% comment
-export([convert/2,convert/1]).
convert(F,fahrenheit)-> (F-32)*5/9;
convert(C,celsius)-> C*9/5 +32.
convert({fahrenheit,X})->
  Y = (X-32)*5/9,
  {celsius,Y};
convert({celsius,X})-> 
  Y = X*9/5+32,
  {fahrenheit,Y}.
```
- When the definition of function is not done, use comma (,)
- For function arguments, only Capital letter (to distinguish from atoms)
```bash
4> c(temp).                       
{ok,temp}
5> temp:convert({fahrenheit,100}).
{celsius,37.77777777777778}
```

### 17. Lists
```erl
8> [ A, B | R] = [1,2,3,4,5].
[1,2,3,4,5]
9> A.
1
10> B.
2
11> R.
[3,4,5]
12> length([1,2,3,4]).
4
```

### 18. Maps
- `maps:get(key,mapData,-1)`: return value from mapData when key is given
```erl
-module(hw18).
%% comment <--- this seems necessary
-export([getAge/1]).
getAge(Name)->
  AgeMap = #{"Alice" => 23, "Bob"=>33, "Cynthia"=>19},
  maps:get(Name, AgeMap, -1).
```
- Testing:
```bash
22> c(hw18).
{ok,hw18}
23> hw18:getAge("Alice").
23
24> hw18:getAge("Cynthia").
19
25> hw18:getAge("April").  
-1
```

## Section 5: Control Structures

19. If Else
- Basic structure:
```erl
if
condition ->
   statement#1;
true ->
   statement #2
end.
```
- Sample code
```erl
-module(hw19).
%% comment
-export([jump/1]).
jump(Input)->
  if
     Input rem 2 == 0 -> pass1;
     Input rem 3 == 0 -> pass2;
     Input rem 5 == 0 -> pass3;
     true -> non_pass  %%  this corresponds to ELSE
  end.
```
  - `non-pass` will not work as this employs `-`, which is arithmetic operawtion
- Practice
```bash
20> c(hw19).
{ok,hw19}
21> hw19:jump(2).
pass1
22> hw19:jump(3).
pass2
23> hw19:jump(6).
pass1
24> hw19:jump(7).
non_pass
```

20. Case
```erl
-module(hw20).
%%
-export([numbering/1]).
numbering(N)->
  case N of
    1 -> jan;
    2 -> feb;
    3 -> mar;
    4 -> apr;
    (_) -> null  %% default case
  end.
```
- Practice:
```bash
35> c(hw20).          
{ok,hw20}
36> hw20:numbering(5).
null
37> hw20:numbering(2).
feb
```

21. Loops
- There is no loop in Erlang
- Let's use list elements to repeat the function call
```erl
-module(hw21).
%%
-export([greet/1]).
greet([])-> true;
greet([First | Rest])-> io:fwrite("Hello " ++ First ++ "\n"),
greet(Rest).
```
- Test
```bash
48> c(hw21).                                
{ok,hw21}
49> hw21:greet(["world", "folks", "there"]).
Hello world
Hello folks
Hello there
true
```

## Section 6: Car Dealership

### 22. Challenge
- Context
  - Car dealership
  - List of cars
  - Map of prices in usd
- Data
  - ["I8", "LH", "F12"]
  - #{"I8"=> 150000, "LH"=> 500000, "F12" => 120000}
- Objective
  - Print prices in other currencies
  - Create listPrices(Currency) function
  - Helper functions
    - round(Number)
    - io.fwrite("Price" ++ interger_to_list(convertedPrice) ++ "\n")

### 23. Solution
```erl
module(hw22).
%% comment
-export([getPrice/1,getPriceInUSD/1]).
getPriceInUSD(P) -> P/1.1.
%% assuming 1.1 for USD to EURO
getPrice(Name) ->
  PriceMap = #{"I8" => 150000, "LH" => 500000, "F12" => 120000},
  E = maps:get(Name, PriceMap, -1),
  S =  getPriceInUSD(E),
  ConvertedPrice = round(S),
  io:fwrite("Price is $" ++ integer_to_list(ConvertedPrice) ++ "\n").
```
- Internal variable name must begin with a Capital letter
- Test:
```bash
65> c(hw22).            
{ok,hw22}
66> hw22:getPrice("I8").
Price is $136364
ok
```

## Section 7: Functions

### 24. Pattern Matching
```bash
2> [First,Rest] = [1,2].
[1,2]
3> First.
1
4> Rest.
2
5> {atom1, X} = {atom1, 1.23}.
{atom1,1.23}
6> X.
1.23
```

### 25. Guards
- Apply constraints/conditions
```erl
-module(hw25).
%%
-export([getType/1]).
getType(N) when N < 13 -> child;
getType(N) when N < 18 -> teen;
getType(N) when N > 17 -> adult.
```
- Test:
```bash
1> c(hw25).
{ok,hw25}
2> hw25:getType(7).
child
3> hw25:getType(14).
teen
...
7> hw25:getType(18).
adult
```

### 26. Built in Functions
```bash
8> round(5.6).
6
9> trunc(5.6).
5
10> length([1,2,3,4]).
4
11> float(5).
5.0
12> is_atom(hello).
true
13> is_atom('hello').
true
14> is_atom("hello").
false
15> is_tuple({abc, 123}).
true
16> atom_to_list(hello).
"hello"
```

### 27. Higher Order Functions
- Lambda function in other language
```erl
-module(hw27).
%%
-export([double/0]).
double() ->
  F = fun(X) -> 2*X end,
  map(F,[1,2,3,4]).

map(F,[])-> [];
map(F,[First | Rest]) -> [F(First) | map(F,Rest)].
```
- Test:
```bash
19> c(hw27).      
hw27.erl:8: Warning: variable 'F' is unused
{ok,hw27}
20> hw27:double().
[2,4,6,8]
```

## Section 8: Concurrent Processing

### 28. Processes
- Threads can share data but Processes don't
```erl
-module(hw28).
%%
-export([run/0,say/2]).
say(What,0) -> done;
say(What, Times) -> io:fwrite(What ++ "\n"), say(What, Times -1).
run()-> spawn(hw28, say, ["Hi",3]), %% runs parallel
        spawn(hw28, say, ["Bye",3]). %%  runs parallel
```
- Test:
```bash
5> c(hw28).   
hw28.erl:4: Warning: variable 'What' is unused
{ok,hw28}
6> hw28:run().
Hi
Bye
<0.102.0>
Hi
Bye
Hi
Bye
```
  - say() with Hi/Bye runs in parallel, not waiting for other side

### 29. Message Passing
- Erlang uses the exclamation mark (!) as the operator for sending a message. 
```erl
%send message Message to the process with pid Pid
Pid ! Message
```
- Sample message send/receive
```erl
-module(hw29).
%%
-export([alice/0,bob/2,run/0]).
alice() ->
  receive {message,PId} -> io:fwrite("Alice got a message\n"), 
                           PId ! message,
                           alice();
                           finished-> io:fwrite("Alice is finished\n")
  end.
bob(0,PId) -> PId ! finished, io:fwrite("Bob is finished\n");
bob(N,PId) -> PId ! {message,self()},
              receive message -> 
                 io:fwrite("Bob got a message\n")
              end, 
              bob(N-1,PId).
run() ->
  PId = spawn(hw29,alice,[]),
  spawn(hw29,bob,[3,PId]).
```
- Test:
```bash
10> c(hw29).
{ok,hw29}
11> hw29:run().
Alice got a message
<0.124.0>
Bob got a message
Alice got a message
Bob got a message
Alice got a message
Bob got a message
Bob is finished
Alice is finished
```

### 30. Registered Process Names
- register(): arguments of atom, pid
- Instead of using pid, we may use atom to address the target process
```erl
-module(hw30).
%%
-export([alice/0,bob/1,run/0]).
alice() ->
  receive message -> io:fwrite("Alice got a message\n"),      
                           bob ! message,
                           alice();
                           finished-> io:fwrite("Alice is finished\n")
  end.
bob(0) -> alice ! finished, io:fwrite("Bob is finished\n");
bob(N) -> alice ! message,
              receive message ->
                 io:fwrite("Bob got a message\n")
              end,
              bob(N-1).
run() ->
  register(alice, spawn(hw30,alice,[])),
  register(bob, spawn(hw30,bob,[3])).
```
- Demo:
```bash
13> hw30:run().
Alice got a message
true
Bob got a message
Alice got a message
Bob got a message
Alice got a message
Bob got a message
Bob is finished
Alice is finished
```
  - Same results of Ch29

### 31. Distributed Programming
```erl
-module(hw31).
%%
-export([alice/0,bob/2,run/0,startAlice/0,startBob/1]).
alice() ->
  receive {message,BobNode} -> io:fwrite("Alice got a message\n"),
                           BobNode ! message,
                           alice();
                           finished-> io:fwrite("Alice is finished\n")
  end.
bob(0,AliceNode) -> {alice,AliceNode} ! finished, io:fwrite("Bob is finished\n");
bob(N,AliceNode) -> {alice,AliceNode} ! {message,self()},
              receive message ->
                 io:fwrite("Bob got a message\n")
              end,
              bob(N-1,AliceNode).
run() ->
  register(alice, spawn(hw31,alice,[])),
  register(bob, spawn(hw31,bob,[3])).
startAlice() ->
  register(alice, spawn(hw31, alice,[])).
startBob(AliceNode) ->
  spawn(hw31,bob,[3,AliceNode]).
```
- Demo is shown in next chapter. Note that we need 2 different computers to test distributed computing

### 32. Running the code on a Mac

### 33. Running the code on a Windows PC
- On a single Ubuntu
  - First terminal
```bash  
$ erl -sname hakune
(hakune@hakune)6> node().           
hakune@hakune
(hakune@hakune)7> c(hw31).          
{ok,hw31}
(hakune@hakune)8> hw31:startAlice().
true
Alice got a message
Alice got a message
Alice got a message
Alice is finished 
(hakune@hakune)9> 
```
  - Second terminal
```bash  
$ erl -sname miku
(miku@hakune)5> node().                            
miku@hakune
(miku@hakune)6> c(hw31).
{ok,hw31}
(miku@hakune)7> hw31:startBob(hakune@hakune).
<0.109.0>
Bob got a message
Bob got a message
Bob got a message
Bob is finished 
```

## Section 9: Conclusion

### 34. Conclusion

### 35. Further resources

### 36. Thank you
