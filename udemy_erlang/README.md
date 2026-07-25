# Erlang Masterclass: The Fundamentals
- Instructor: Daniel Hjerpe

## Section 1: Practical info

### 1. Introduction

### 2. Course materials

### 3. Setting up the environment on a Mac

### 4. Setting up the environment on a PC

### 5. Setting up the environment on Linux
- https://www.erlang.org/
- sudo apt install erlang
- erlang plugin for VScode

## Section 2: Getting started

### 6. Introduction to the Erlang shell
```bash
$ erl
Erlang/OTP 25 [erts-13.2.2.5] [source] [64-bit] [smp:8:8] [ds:8:8:10] [async-threads:1] [jit:ns]

Eshell V13.2.2.5  (abort with ^G)
1> 1+7 %% does nothing
1> .   %% now executes
8
2> q() %% when exiting the console
2> .
ok
```

### 7. The slightly awkward syntax of Erlang
- Most of syntax from Prolog

### 8. Different types of data
- Data types
  - Number
    - Integer
    - Float
  - Atom: start with lower characters
  - Tuples: collection of a fixed terms
  - Maps: contains variable number of key-value pairs
  - Lists: collection of a variable number of terms
  - fun: function data type
  - References: unique tag
    - Enables us to identify processes
  - Ports: communicate with the external world
  - Records: tuple. Access field by name

### 9. Numbers and operators
- Integers
  - Arbitrary precision arithmetic
  - No limit
  - Up to the physical memory
- Floats
  - Approximation of real numbers
```erlang
4> 123456789123456789+1.
123456789123456790
5> 1+0.00000000001.
1.00000000001
6> 1+0.0000000000000001. %% truncated
1.0
7> 1 == 1.
true
8> 5 > 7.
false
9> 2/1 =:= 1
9> .
false
13> (1 /= 2) and (5 > 7).
false
14> (1 /= 2) or (5 > 7). 
true
15> (1 /= 2) or not (5 > 7).
true
16> (1 /= 2) xor (5 > 7).   
true
```

### 10. Putting it all together
- Tuples: 
  - `{"banana", "apple"}`
  - `{fruit, "apple", 123}`
    - Here fruit is a tag
- List
  - `[1,2,3]`
  - [ Head | Tail]
    - `["apple" | ["banana", "pear"]]`
```erlang
24> hd(["apple" | ["banana","pear"]])
24> .
"apple"  %% head
25> tl(["apple" | ["banana","pear"]]).
["banana","pear"] %% tail
...
23> ["a","b"] ++ ["c","d"].
["a","b","c","d"]
```

### 11. Splitting the atom
- Atom is not garbage-collected
- Similar to enum in C/C++
```erlang
26> apple.
apple
27> true.
true
28> false.
false
29> 'banana and fruit'
29> .
'banana and fruit'
```

### 12. Oh variables, where art thou?
- Variables: start with a capital letter
  - Immutable: after creation, we cannot modify
```erlang  
30> MyFruit = {banana, apple}.
{banana,apple}
31> MyFruit = {banana, apple, pear}.
** exception error: no match of right hand side value 
                    {banana,apple,pear}
32> myBag = {fruit, "Apple", 123}.
** exception error: no match of right hand side value 
                    {fruit,"Apple",123}
33> MyBag = {fruit, "Apple", 123}.
{fruit,"Apple",123}
34> {Type, Name, Price} = MyBag.
{fruit,"Apple",123}
35> Type.
fruit
36> Name.
"Apple"
37> Price.
123
32> myBag = {fruit, "Apple", 123}. %% error as the first char is not Capital
** exception error: no match of right hand side value 
                    {fruit,"Apple",123}
33> MyBag = {fruit, "Apple", 123}.
{fruit,"Apple",123}
34> {Type, Name, Price} = MyBag.
{fruit,"Apple",123}
35> Type.
fruit
36> Name.
"Apple"
37> Price.
123
38> b().  %% showing current variables
MyBag = {fruit,"Apple",123}
MyFruit = {banana,apple}
Name = "Apple"
Price = 123
Type = fruit
ok
39> f(Price). %% freeing variables
ok
40> b(). %% now Price is gone
MyBag = {fruit,"Apple",123}
MyFruit = {banana,apple}
Name = "Apple"
Type = fruit
ok
```

### 13. Exercises

## Section 3: Sequential Erlang

### 14. Greetings from the world of Erlang!
- hello.erl:
```erlang
-module(hello).
-author("abc def").
greetings() -> io:format("Greetings from the world of Erlang! ~n").
```
- Demo:
```bash
$ erl
Erlang/OTP 25 [erts-13.2.2.5] [source] [64-bit] [smp:8:8] [ds:8:8:10] [async-threads:1] [jit:ns]

Eshell V13.2.2.5  (abort with ^G)
1> c(hello).
hello.erl:3:1: Warning: function greetings/0 is unused
%    3| greetings() -> io:format("Greetings from the world of Erlang! ~n").
%     | ^

{ok,hello}
2> c(hello).

hello.erl:3:1: Warning: function greetings/0 is unused
%    3| greetings() -> io:format("Greetings from the world of Erlang! ~n").
%     | ^

{ok,hello}
2> hello:greetings().
** exception error: undefined function hello:greetings/0 # not defined
3> c(hello,[export_all]).
{ok,hello}
4> hello:greetings().    
Greetings from the world of Erlang! 
ok # now runs OK
```
- Let's add export_all to the source code:
```erlang
-module(hello).
-author("abc def").
-compile(export_all).
greetings() -> io:format("Greetings from the world of Erlang! ~n").
hull_speed(Lwl) -> 
  Vhull = 1.34*math:sqrt(Lwl),
  knots_to_kmph(Vhull).
knots_to_kmph(Knots) -> 
  Knots *1.852001.
```
- Demo again:
```bash
11> c(hello).            
hello.erl:3:2: Warning: export_all flag enabled - all functions will be exported
%    3| -compile(export_all).
%     |  ^

{ok,hello}
12> hello:hull_speed(40).
15.695530922277458
```  
- Instead of export_all, we can select which functions will be exported:
```erlang
-module(hello).
-author("abc def").
%%-compile(export_all).
-export([greetings/0, hull_speed/1]).
greetings() -> io:format("Greetings from the world of Erlang! ~n").
hull_speed(Lwl) -> 
  Vhull = 1.34*math:sqrt(Lwl),
  knots_to_kmph(Vhull).
knots_to_kmph(Knots) -> 
  Knots *1.852001.
```
- Calling hello:knots_to_kmph() will error as it is not exported

### 15. Go with the flow
- cf.erl:
```erlang
-module(cf).
-author("John Scott").
-export([greetings/1]).
%% Pattern matching
greetings([]) -> "Hello stranger";
greetings(Name) -> "Hello " ++ Name.
```
- Demo
```bash
17> c(cf).
{ok,cf}
18> cf:greetings("").
"Hello stranger"
19> cf:greetings("Amy").
"Hello Amy"
20> cf:greetings(Amy).  
* 1:14: variable 'Amy' is unbound
```
- Shadowing:
  - For the following code, 3rd definition is never executed as function call with argument will activate 2nd definition, shadowing 3rd definition
```erlang
greetings([]) -> "Hello stranger";
greetings(Name) -> "Hello " ++ Name;
greetings({Firstname,Surname}) -> %% SHADOWED!
"hello " ++ Firstname ++ " " ++ Surname.
```
  - Therefore, we change the order of definitions as:
```erlang
greetings([]) -> "Hello stranger";
greetings({Firstname,Surname}) -> %% SHADOWED!
"hello " ++ Firstname ++ " + " ++ Surname;
greetings(Name) -> "Hello " ++ Name.
```
- cf.erl:
```erlang
-module(cf).
-author("John Scott").
-export([greetings/1]).
-export([beverage/1]).
-export([beverage3/2]).
%% Pattern matching
greetings([]) -> "Hello stranger";
greetings({Firstname,Surname}) -> %% SHADOWED!
"hello " ++ Firstname ++ " + " ++ Surname;
greetings(Name) -> "Hello " ++ Name.
%%
beverage(Type) ->
  case Type of 
    coffee -> "Good coffee";
    tea -> "Boild water";
    _ -> unknown
  end.
beverage3(Type,Temp) ->
  case {Type, Temp>70} of
    {coffee, true} -> "Hot coffee";
    {coffee,_} -> "Not Hot";
    {tea, _} -> "boild water";
    _ -> unknown
  end.
```
- Demo:
```bash
31> c(cf).                     
{ok,cf}
32> cf:beverage(coffee).
"Good coffee"
33> cf:beverage(tea).   
"Boild water"
34> cf:beverage(coke).
unknown
52> cf:beverage3(coffee,70).
"Not Hot"
53> cf:beverage3(coffee,73).
"Hot coffee"
```

### 16. Recursion
- rec.erl:
```erlang
-module(rec).
-author("Amy Brown").
-export([factorial/1]).
-export([list_len/1]).
%% Factorial
factorial(0) -> 1;
factorial(N) when N > 0, is_integer(N) -> 
  N*factorial(N-1).
list_len([]) -> 0;
list_len([_Head|Tail]) ->
  1+ list_len(Tail).
```
- Demo:
```bash
55> c(rec).
{ok,rec}
56> rec:factorial(3).
6
57> rec:factorial(-1).
** exception error: no function clause matching 
                    rec:factorial(-1) (rec.erl, line 5)
58> rec:factorial(1.234).
** exception error: no function clause matching 
                    rec:factorial(1.234) (rec.erl, line 5)
61> rec:list_len(["a", "b", "c", "D"]).
4
```

### 17. Tail recursion
```erlang
-module(rec).
-author("Amy Brown").
-export([factorial/1, list_len/1, list_len2/1, list_len2/2, reverse/1, reverse/2]).
%% Factorial
factorial(0) -> 1;
factorial(N) when N > 0, is_integer(N) -> 
  N*factorial(N-1).
list_len([]) -> 0;
list_len([_Head|Tail]) ->
  1+ list_len(Tail).
list_len2(L) -> list_len2(L,0).
list_len2([],Acc) -> Acc;
list_len2([_|Tl],Acc) -> 
  list_len2(Tl,1+Acc).
reverse(L) -> reverse(L,[]).
reverse([],Acc) -> Acc;
reverse([Hd|Tl], Acc) -> reverse(Tl,[Hd|Acc]).
```
- Demo:
```bash
7> c(rec).
{ok,rec}
8> rec:list_len2(["Jussi","Peter","Christ"],0).
3
9> rec:reverse([1,2,3,4,5,6]).                 
[6,5,4,3,2,1]
10> 
```
- Why we use accumulator?
  - Erlang will optimize and can save stack memory
  - Tail Call Optimization (TCO)
  - https://medium.com/@themissouri.md/recursion-tail-recursion-in-erlang-59caf740b345

### 18. Tail or body recursion?
- How to avoid stack overflow?
  - Use accumulator
```erlang
-module(rec).
-author("Amy Brown").
-export([doubles/1, doubles2/1, doubles2/2]).
%%
doubles([]) -> [];
doubles([Hd|Tl]) ->
  [Hd*2|doubles(Tl)].
%%
doubles2(L) -> doubles2(L,[]).
doubles2([],Acc) -> Acc;
doubles2([Hd|Tl],Acc) -> 
  doubles2(Tl,[Hd*2|Acc]).
```  
- Demo:
```bash
17> c(rec).
{ok,rec}
18> rec:doubles([1,2,3,4]).
[2,4,6,8]
19> rec:doubles2([1,2,3,4]).
[8,6,4,2]
```

### 19. Keep calm and let it crash!
- Compile time error
  - Syntax error
  - Head mismatch
  - Variables are not used
- Logical error
  - Can be resolved through tests/unit-tests
- Run-time error
  - Bad match, where pattern match fails
  - Bad argument of built-in functions
  - Case-clause fails
- Generated error
  - Exceptions at runtime
  - Exit/throw

### 20. Exercises

## Section 4: Becoming a functional hipster

### 21. Fun fun functions!
- Anonymous function
  - Lambda
  - Can be defined as an expression "on the fly"
```bash
20> Even = fun(Num) -> Num rem 2 ==0 end.
#Fun<erl_eval.42.3316493>
21> Even(1).
false
22> Even(2).
true
```
- fff.erl:
```erlang
-module(fff).
-author("ABC DEF").
-export([doubles/1]).
doubles([]) -> [];
doubles([Hd|Tl]) -> 
  Double = fun(X) -> X*2 end,
  [Double(Hd)|doubles(Tl)].
```
- Demo:
```bash
23> c(fff).
{ok,fff}
24> fff:doubles([1,2,3,4,5]).
[2,4,6,8,10]
```

### 22. Map
```bash
27> Double = fun(X) -> X*2 end.
#Fun<erl_eval.42.3316493>
28> lists:map(Double,[1,2,3]).
[2,4,6]
```
- hof.erl:
```erlang
-module(hof).
-author("ABC DEF").
-export([opera/0]).
opera() ->
  OperaSingers = [
    {tenor, "Jssi"},
    {baritone, "Peter"},
    {soprano, "Elin"},
    {mezzo, "Malena"}],
  VoiceMap = fun({Voice,Name}) ->
    NewVoice = case Voice of 
      soprano -> "High voice";
      tenor -> "High voice";
      baritone -> "Middle voice";
      mezzo -> "Low voice"
    end,
    {NewVoice,Name} end,
  _OperaForDummies = lists:map(VoiceMap,OperaSingers).
```
- Demo:
```bash
35> c(hof).
{ok,hof}
36> hof:opera().              
[{"High voice","Jssi"},
 {"Middle voice","Peter"},
 {"High voice","Elin"},
 {"Low voice","Malena"}]
```

### 23. Filter
- Using a predicate function, input data are filtered
  - `lists:filter(predicate_ftn, input)`
- hof2.erl:
```erlang
-module(hof2).
-author("ABC DEF").
-export([double_r_diner/0]).
double_r_diner() ->
  DoubleROrders = [
    {coffee, "Dale"},
    {coffee, "John"},
    {pie, "Harry"},
    {pancakes, "Nadine"}],
    FilterItems = fun(Orders, ItemPredicate) ->
      lists:filter(fun({Item,_Customer}) ->
        Item == ItemPredicate end,
        Orders) end,
    CoffeeOrders = FilterItems(DoubleROrders,coffee),
    PieOrders = FilterItems(DoubleROrders, pie),
    io:format("Coffee orders: ~p~n Pie orders: ~p~n",
      [CoffeeOrders,PieOrders]).
```
- Demo:
```bash
41> c(hof2).             
{ok,hof2}
42> hof2:double_r_diner().
Coffee orders: [{coffee,"Dale"},{coffee,"John"}]
 Pie orders: [{pie,"Harry"}]
ok
```

### 24. Fold
- lists::foldl/3: a higher order function used to reduce or collapse a list into a single value by trasverse it from left to right
- hof3.erl:
```erlang
-module(hof3).
-author("dummy John").
-export([fruit_market/0]).
fruit_market() ->
  Fruits = [
    {banana,0.95,2},
    {apple,1.20, 3},
    {grapes,1, 2.25}],
  lists:foldl(fun({_Item,Price,Quantity},Sum) ->
    (Price*Quantity) + Sum end, 0, Fruits).
```
- Demo:
```bash
44> c(hof3).
{ok,hof3}
45> hof3:fruit_market().
7.75
```

### 25. Just add another layer of abstraction

### 26. List comprehension
- hof4.erl
```erlang
-module(hof4).
-author("ABC DEF").
-export([main/0]).
main() ->
  Evens = [X || X <- [1,2,3,4], X rem 2 == 0],
  io:format("Even nubmers: ~p~n", [Evens]),
  OperaSingers = [
    {tenor, "Jssi"},
    {baritone, "Peter"},
    {soprano, "Elin"},
    {mezzo, "Malena"}],
  FormattedOperaSingers = 
    [Name ++ ": " ++ atom_to_list(Voice) || {Voice,Name} <- OperaSingers],
  io:format("Opera singers: ~p~n",[FormattedOperaSingers]).
```
- Demo:
```bash
60> c(hof4).     
{ok,hof4}
61> hof4:main().
Even nubmers: [2,4]
Opera singers: ["Jssi: tenor","Peter: baritone","Elin: soprano",
                "Malena: mezzo"]
ok
```

### 27. A short note on side effects

### 28. Exercises

## Section 5: Bonus: More ways to work with data

### 29. Records
- Records
  - Access field by name
  - Add fields over time
  - Allows us to use default values of fields
- rr(): Read records
- twin_records.erl:
```erlang
-module(twin_records).
-author("ABD DEF").
-export([is_suspect/1, make_suspect/1, clear_suspect/1]).
-record(citizen,
  {name,
    date_of_birth,
    address,
    suspect=false}).
%is_suspect({_Name,_DoB,Suspect,_Street,_PostalCode,_City}) ->
%  Suspect.
%is_suspect(C) ->
%  C#citizen.suspect.
is_suspect(#citizen{suspect=Suspect} = _C) ->
  Suspect.
make_suspect(C) ->
  C#citizen{suspect=true}.
clear_suspect(C) ->
  C#citizen{suspect=false}.
```
- Demo:
```bash
62> c(twin_records).
twin_records.erl:4:2: Warning: record citizen is unused
%    4| -record(citizen,
%     |  ^

{ok,twin_records}
63> #citizen(name="John", date_of_birth="Apri 19,
 1999").
* 1:9: syntax error before: '('
63> #citizen{name="John", date_of_birth="Apri 19,
 1999"}.
* 1:1: record citizen undefined
64> rr(twin_records).
[citizen]
65> Audrey = #citizen{date_of_birth="August 24, 1
997", name="Audrey Horne"}.
#citizen{name = "Audrey Horne",
         date_of_birth = "August 24, 1997",
         address = undefined,suspect = false}
66> record_info(fields, citizen).
[name,date_of_birth,address,suspect]
67> record_info(size, citizen).
5
68> c(twin_records).                             
twin_records.erl:15:1: Warning: function make_suspect/1 is unused
%   15| make_suspect(C) ->
%     | ^

twin_records.erl:17:1: Warning: function clear_suspect/1 is unused
%   17| clear_suspect(C) ->
%     | ^

{ok,twin_records}
69> c(twin_records).                     
{ok,twin_records}
70> PrimeSuspect = twin_records:make_suspect(Audr
ey).
#citizen{name = "Audrey Horne",
         date_of_birth = "August 24, 1997",
         address = undefined,suspect = true}
71> twin_records:is_suspect(PrimeSuspect).
true
```

### 30. Macros
- Make code more readable
- Using `?XXX`

### 31. Macros and debug flags
```erlang
-ifdef(my_debug_flag).
  -define(DEBUG(Statement),io:format("*DEBUG* ~p~n", [Statement])).
-else.
  -define(DEBUG(Statement), ok).
-endif.
```

### 32. Maps
```erlang
85> Tea = #{price=>3, ingredients=>["boiled water", "
dried leaves"]}.
#{ingredients => ["boiled water","dried leaves"],
  price => 3}
86> FlammKuchen = #{price=>7, ingredients=>["spinnage
", "garlic"]}.
#{ingredients => ["spinnage","garlic"],price => 7}
87> Menu = #{flammkuchen=>FlammKuchen, tea => Tea}.
#{flammkuchen =>
      #{ingredients => ["spinnage","garlic"],
        price => 7},
  tea =>
      #{ingredients =>
            ["boiled water","dried leaves"],
        price => 3}}
88> erlang:system_info(atom_limit).
1048576
```

## Section 6: Thank you !

### 33. Outro

************************************************

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
