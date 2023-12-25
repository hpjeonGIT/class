## Title: The Complete Haskell Course: From Zero to Expert!
- Instructor: Lucas Bazilio

## Section 1: Course Introduction

1. Introduction to Haskell
- No assignment
- No loops
- No side effects
- No explicit memory management
- Lazy evaluation
- Functions are first-order objects
- Static type system
- Automatic type inference

2. Basic Fundamentals
- Install Haskell
  - curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
- Running interactive haskell: ghci
  - To exit: ctrl+d
```hs
ghci> 3+2*2
7
ghci> even 4
True
ghci> even(4)
True
ghci> div 11 3
3
ghci> :type 'D'
'D' :: Char
ghci> :type not
not :: Bool -> Bool
ghci> :type length
length :: Foldable t => t a -> Int
```

3. Tools
- Glasgow Haskell compiler (GHC)
  - Comiler: ghc
  - Interpreter: ghci

4. Haskell Intepreter
- File extension is \*.hs
- test.hs
```hs
factorial:: Integer -> Integer
factorial 0 = 1 -- Comment
factorial n = n* factorial(n-1)
```
- From ghci
```hs
ghci> :load section1/test.hs
[1 of 2] Compiling Main             ( section1/test.hs, interpreted )
Ok, one module loaded.
ghci> factorial 5
120
ghci> map factorial[0..5]
[1,1,2,6,24,120]
ghci> :type factorial
factorial :: Integer -> Integer
```
- Now modify test.hs as:
```hs
-factorial:: Integer -> Integer
--factorial 0 = 1 -- Comment
--factorial n = n* factorial(n-1)
double x = 2*x
```
- Reload the module as:
```hs
ghci> :reload
[1 of 2] Compiling Main             ( section1/test.hs, interpreted ) [Source file changed]
Ok, one module loaded.
ghci> double 3
6
ghci> double(4)
8
ghci> double(-3)
-6
ghci> double -3 -- Use () for negative number !!!
<interactive>:26:1: error:
```
  - As factorial is commented out, it doesn't work anymore

## Section 2: Basic Type

5. Basic Types - Part 1
- Booleans: Bool
```hs
rue
ghci> True && False
False
ghci> 0 && False  -- 0 is not Bool and this operation is not allowed
<interactive>:30:1: error:
ghci> True || True
True
ghci> not (not True)
True
ghci> not not True -- This is understood as (not not) True
<interactive>:35:1: error:
```
- Integers: Int, Integer
  - Int: 64bit integer
  - Integer: arbitrarily long integers
  - Negative number needs (): (-5)
```hs
ghci> 2^6
64
ghci> div 11 2
5
ghci> mod 11 2
1
ghci> rem 13 3
1
ghci> mod (-11) 2 -- mod yield positive number always
1
ghci> rem (-13) 3
-1
ghci> 4 /= 3
True
ghci> div 15 3
5
ghci> 15 `div` 3  -- to be more readable
5
```
- Reals: Float, Double
  - Float: 32bit floating point
  - Double: 64bit floating point
  - Power operation: ** (in integer, it is ^)
  - Converting Integer to Real: `fromIntegral`
  - From Real to Integer: `round`,`floor`,`ceiling`
```hs
ghci> round 3.6
4
ghci> round (-3.6)
-4
ghci> map round [1.1, -4.1, 5.5]
[1,-4,6]
```
  - `1.` is NOT allowed. `1.0` is allowed
- Characters: Char

6. Basic Types - Part 2
```hs
ghci> import Data.Char
ghci> ord('A')
65
ghci> chr(65)
'A'
```
- "A" is NOT allowed. Only 'A'

## Section 3: Functions

7. Introduction to Functions
- Functions in Haskell are pures: they only return results calculated relative to their parameters
- Functions do not have side effects
  - They do not modify the parameters
  - They do not modify the memory
  - They do not modify the input/output
- A function always returns the same result applied to the same parameters
- Definition of Functions
  - Function identifiers start with a lower case
  - First, its type declaration (header) is given
  - Then its definition is given, using formal parameters
- simple.hs
```hs
double :: Int -> Int
double x = 2*x

perimeter:: Int -> Int -> Int
perimeter width height = double (width + height)

xOr :: Bool -> Bool -> Bool
xOr a b = (a || b) && not (a && b)

factorial :: Integer -> Integer
factorial n = if n==0 then 1 else n* factorial(n-1)
```

8. Definitions with Patterns
- Evaluation of the patterns is from top to bottom
- Patterns can become more elegant than the if-then-else and they have more applications
- `_` for anonymous variable
```hs
nand :: Bool -> Bool -> Bool
nand True True = False
nand _ _  = True  -- True nand True is defined above. All the other cases are True
```
9. Definition with Guards
```hs
valAbs:: Int -> Int
valAbs n
  | n >= 0    = n
  | otherwise = -n
```
- Equality goes after every guard

10. Local Definitions
- Local names in an expression can be used in the `let-in`
```hs
fastExp:: Integer-> Intger -> Integer
fastExp _ 0 = 1
fastExp x n =
    let y        = fastExp  x n_halved
        n_halved = div n 2
    in
      if even n
      then y*y
      else y*y*x
```
- Or `where` allows names to be defined in more than on expression
```hs
fastExp :: Integer -> Integer -> Integer
fastExp _ 0 = 1
fastExp x n 
  | even n    = y*y
  | otherwise = y*y*x
  where
    y = fastExp x n_halved
    n_halved = div n 2
```    

11. Currying
- All functions have a single parameter
- Functions of more than one parameter actually return a new function
- No need to pass all parameters (partial application)
  - Ex: `prod 3 5` is same as `(prod 3) 5`

## Section 4: Solved Problems - Functions

12. Absolute Value
```hs
absValue :: Int -> Int

absValue x
    | x >= 0 = x
    | otherwise = -x
```
    
13. Power
```hs
power :: Int -> Int -> Int

power x 0 = 1   -- Base Case
power x p
    | even p = n*n
    | otherwise = n*n*x
    where
        n = power x (p `div` 2)
```        

14. Prime Number
```hs
isPrime :: Int -> Bool
isPrime 0 = False -- Base Case
isPrime 1 = False -- Base Case

isPrime x = not (hasDivisor(x-1))
    where
        hasDivisor :: Int -> Bool
        hasDivisor 1 = False
        hasDivisor n = mod x n == 0 || hasDivisor(n-1)
```

15. Fibonacci
```hs
fib :: Int -> Int
fib 0 = 0   -- Base Case
fib 1 = 1   -- Base Case
fib n = fib (n-1) + fib (n-2)
```

## Section 5: Tuples

16. Introduction to Tuples
- A tuple is a structured type that allows you to store different type values on a single value of type
- The number of fields is fixed
- The fields are of heterogeneous type
```hs
timeDecomposition::Int -> (Int, Int, Int)
timeDecomposition seconds = (h,m,s)
where
   h = div seconds 3600
   m = div (mod seconds 3600) 60
   s = mod seconds 60
```

17. Access to Tuples
- For 2 element tuples, `fst`(first) and `snd`(second)
```hs
ghci> fst (1,2)
1
ghci> snd(1,2)
2
ghci> fst(1,2,3) -- fst/snd for 2 element tuple only
<interactive>:13:4: error:
```
- No access function for general tuples

18. Decomposition of Tuples into Patterns
```hs
distance:: (Float,Float) -> (Float,Float) -> Float
distance (x1,y1) (x2,y2) = sqrt((x1-x2)^2 + (y1-y2)^2)
```

19. Empty Tuple (Unit)
- A tuple without any data
- Similar to void in C

## Sectin 6: Lists

20. Introduction to Lists
- A list is a structured type that contains a sequence of elements, all of the same type
```hs
ghci> []
[]
ghci> [1,2,3]
[1,2,3]
ghci> [(1, "abc"), (2,"def")]
[(1,"abc"),(2,"def")]
ghci> [1 .. 10]
[1,2,3,4,5,6,7,8,9,10]
ghci> [1, 3 .. 10]
[1,3,5,7,9] -- Note that 10 is NOT included !!!
```

21. Construction and Implementation
- List constructor
```hs
ghci> [1,2,3]
[1,2,3]
ghci> 1 : 2 : 3 : []
[1,2,3]
ghci> 1 : (2 : (3 : []))
[1,2,3]
```
- Lists in Haskell are simply linked lists
  - Constructors [] and : work in constant time
```hs
ghci> l1 = 3:2:1:[]
ghci> l1
[3,2,1]
ghci> l2 = 4:l1
ghci> l2
[4,3,2,1]
```

22. Lists and Patterns
- Pattern discrimination
  - Decomposing lists:
```hs
mysum [] = 0
mysum (x:xs) = x + mysum xs
```
  - Using `sum` instead of `mysum` may raise error due to built-in function names

23. Syntax in Patterns
```hs
suml list = 
  case list of
    []   -> 0
    x:xs -> x + suml xs

divImod n m
  | n < m   = (0,n)
  | otherwise = (q+1,r)
  where (q,r) = divImod (n-m) m

firstandsecond list = 
  let first:second:rest = list
  in (first,second)
```

24. Texts
- Texts in Haskell are lists of characters
  - String is equivalent to [Char]
- Double quotes are required
```hs
ghci> n1 = ['j','i','m']
ghci> n1
"jim"
ghci> n2 = "jimal"
ghci> n2
"jimal"
ghci> n1 == n2
False
ghci> n1 < n2  -- in terms of length
True
ghci> "aaa" < "aaaa" -- aaaa is larger than aaa
True
ghci> "aaz" > "aaaa" -- z is larger than a
True
```

25. Common Functions
- `head` returns the first element of the list
- `last` returns the last element of the list
- `init` returns the list without the last element
- `tail` returns the list without the first element
- `null` checks if the list is empty or not
- `elem` checks if the element exists in the list or not
- `!!` returns the element of the given index (starts from 0)
- `++` concatenates two lists
- `take` return first N elements
- `drop` return the left-over when first N elements are removed
- `zip` return the list of tuples using argument lists
- `repeat` produces an infinite list containing x
- `concat` returns a list using the lists of lists
```hs
ghci> head(n1)
'j'
ghci> last(n1)
'm'
ghci> init [1 .. 10]
[1,2,3,4,5,6,7,8,9]
ghci> tail [1 .. 10]
[2,3,4,5,6,7,8,9,10]
ghci> reverse [1 .. 4]
[4,3,2,1]
ghci> null []
True
ghci> null [ 1, 2,3]
False
ghci> elem 6 [1..10]
True
ghci> 6 `elem` [1..10]
True
ghci> [1..10] !! 3
4
ghci> [0, 4, 7, 9 ] !! 0
0
ghci> [0, 4, 7, 9 ] !! 2
7
ghci> "Hell World" !! 5
'W'
ghci> "Hello "++ "World!"
"Hello World!"
ghci> maximum "Hello"
'o'
ghci> minimum "World!"
'!'
ghci> maximum [1..10]
10
ghci> sum [1,2,3,4]
10
ghci> product [1,2,3,4]
24
ghci> take 3 "Hello world"
"Hel"
ghci> drop 3 "Hello world"
"lo world"
ghci> zip "Hello" "World!"
[('H','W'),('e','o'),('l','r'),('l','l'),('o','d')] -- Note that "!" is gone
ghci> concat [[1..10] , [1,2,3] , [11..14]]
[1,2,3,4,5,6,7,8,9,10,1,2,3,11,12,13,14]
```

## Section 7: Solved Problems - Lists

## Section 8: Higher Order Functions
