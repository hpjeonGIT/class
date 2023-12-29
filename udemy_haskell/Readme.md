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

26. Problem 1 - Last Elment of a List
```hs
myLast:: [a] -> a
myLast [] = error "Empty list"
myLast [x] = x
myLast (_:xs) = myLast xs
```

27. Problem 2 - Penultime Object
- Return the 2nd last element in the list
```hs
my2Last:: [a] -> a
my2Last [] = error "Empty list"
my2Last [x] = error "Single element list"
my2Last [x,_] = x
my2Last (_:(x:xs)) = my2Last (x:xs)
```

28. Problem 3 - Duplicate Elements
- 2x duplicate the elements in a list
```hs
mydupe:: [a] -> [a]
mydupe [] = []
mydupe (x:xs) = x:x:mydupe xs
```

29. Problem 4 - Average
- Return float type average number of a list of integers
```hs
myav:: [Int] -> Float
myav [] = error "Empty list"
myav x = sumx/lenx
   where
    sumx = fromIntegral(sum(x))      
    lenx = fromIntegral(length(x))
```

30. Problem 5 - Insertion in Position
```hs
insertIn:: a -> [a] -> Int -> [a]
insertIn x,(xs), n = (take (n-1) xs) ++ [x] ++ (drop (n-1) xs)
```
- Answer:
```hs
insertIn :: a -> [a] -> Int -> [a]
insertIn x ys 1 = x:ys
insertIn x (y:ys) n = y:insertIn x ys (n-1)
```

## Section 8: Higher Order Functions

31. Higher Order Functions

32. Anonymous Functions
- Or lambda function
- A Function without a name
```hs
ghci> (\x -> x + 5) 3
8
ghci> map even [1, 2, 3]
[False,True,False]
ghci> map (\x -> 2*x) [1,2,3]
[2,4,6]
ghci> (\x y -> x + y) 2 3  -- multiple parameter functions
5
ghci> (\x -> \y-> x + y) 2 3
5
ghci> (\x -> (\y -> x+y)) 2 3
5
```
- Useful when the function is short/used only once

33. Sections
- Allows partial infix operators to be applied
  - Regarding the operation, left or right operands can be given
```hs
ghci> map (/2) [1,2,3]
[0.5,1.0,1.5]
ghci> map (2/) [1,2,3]
[2.0,1.0,0.6666666666666666]
```

## Section 9: Solved Problems - Higher Order Functions

34. Problem 1 - Equality Problem
- Compare two lists if their elements are same or not
```hs
eql:: [Int] -> [Int] -> Bool
eql [] [] = True
eql (x:xs) (y:ys) =  (x == y) && eql xs ys
```
- Answer:
```hs
eql :: [Int] -> [Int] -> Bool
eql x y
  | length x /= length y    = False
  | otherwise               = and $ zipWith (==) x y
```
- Using `zipWith`  
```hs
ghci> zipWith (+) [1,2,3] [3,2,1] 
[4,4,4]
ghci> zipWith (-) [0,0,0] [3,2,1]
[-3,-2,-1]
ghci> zipWith (==) [1,2,3] [1,2,4]
[True,True,False]
```

35. Problem 2 - Product of Elements
```hs
myprod:: [Int] -> Int
myprod [] = error "Empty array"
myprod [x] = x
myprod (x:xs) = x * myprod xs
```
- Answer
```hs
prod :: [Int] -> Int
prod                        = foldl (*) 1
```
- foldl (fold from left)
```hs
ghci> foldl (/) 64 [4, 2, 4] -- 64 / 4/ 2/ 4 = 2.0
2.0
```

36. Problem 3 - Even Result
- List product using even elements only
```hs
prodEvens:: [Int] -> Int
prodEvens [] = error "Empty list"
prodEvens [x] 
  | even x = x
  | otherwise = 1
prodEvens (x:xs)
  | even x = x*prodEvens xs
  | otherwise = prodEvens xs
```
- Answer:
```hs
prod :: [Int] -> Int
prod = foldl (*) 1
prodEvens :: [Int] -> Int
prodEvens = prod . filter even
```
- filter
```hs
ghci> filter even [1,2,3,4]
[2,4]
ghci> filter (> 3) [1,2,3,4]
[4]
```
- Note that ` f . g == f(g(x))`

37. Problem 4 - Infinite Powers of Two
- Find elements which are power of 2 in the list, which is infinite 
```hs
div2:: Int -> Bool
div2 1 = True
div2 x = if (rem x 2) ==0 then (div2 (div x 2)) else False
pOf2:: [Int] -> [Int]
pOf2 = filter div2 -- this is for when a finite list is given
```
- Answer
```hs
powersOf2 :: [Int]
powersOf2 = iterate (*2) 1
```

38. Problem 5 - Scalar Product
- scalarProduct [3.0,4.0] [5.0,3.0] -> 27.0
```hs
sProd:: [Float] -> [Float] -> Float
sProd [x] [y] = x*y
sProd (x:xs) (y:ys) = x*y + sProd xs ys
``` 
- Answer:
```hs
scalarProduct :: [Float] -> [Float] -> Float
scalarProduct x y           = sum $ zipWith (*) x y
```
- `sum zipWith(*) x y` crashes as sum -> zipWith cannot work. ZipWith must be done before sum. Using `$`, `zipWith()` can be done earlier than `sum` works

39. Problem 5 - Scalar Product - Extension
40. Problem 6 - Flattening of Lists
- foldr (fold from right)
```hs
ghci> foldr (+) 10 [1,2] -- 10 + 2 + 1
13
ghci> [1,2] ++ [3,4] -- ++ is concat operator
[1,2,3,4]
```
- Answer:
```hs
flatten :: [[Int]] -> [Int]
flatten = foldr (++) [] -- (implicit input)
-- flatten x = foldr (++) [] x (explicit input)
```
- ~~In this example, if foldl is used, it will change the order~~ foldl yields the same results

41. Problem 7 - Length
- mylength "John" => 4
```hs
mylength::String -> Int
mylength x = sum ( map (const 1) x )
```

42. Problem 8 - Reverse
- myRev [1,2,3] => [3,2,1]
```hs
myRev::[Int] -> [Int]
myRev [x] = [x]
myRev (x:xs) = (myRev xs) ++ [x]
```
- Answer:
```hs
myReverse :: [Int] -> [Int]
myReverse = foldl (flip(:)) []
```
- flip function: changes the order of arguments
```hs
ghci> flip (/) 1 2
2.0
ghci> flip (>) 3 5
True
```

43. Problem 9 - Occurrences
- In [[1,2,3],[1,1], [1],[3,4]], how many 1 exists? 
  - Answer: [1,2,1,0]
```hs
countIn :: [[Int]] -> Int -> [Int]

countIn l x = map count l
    where
        count :: [Int] -> Int
        count = length . (filter (==x))
```
- In Haskell, f.g == f(g(x))

44. Problem 10 - First Word
- dropWhile() and takeWhile()
```hs
ghci> dropWhile (<3) [1,2,3,4,2]
[3,4,2]
ghci> takeWhile (<3) [1,2,3,4,2]
[1,2]
```
- Answer:
```hs
firstWord :: String -> String
firstWord = takeWhile(/= ' ') . dropWhile (== ' ')
-- firstWord s = takeWhile(/= ' ') (dropWhile (== ' ') s)
```

45. Problem 11 - Conditional Count
- countIf even [1,2,3,4] -> 2
- How to read predicate (conditional statement) in function definition?
  - In this problem, the predicate reads Int then produces Bool
  - (>3): larger than 3 then True
  - even: even number then True
```hs
countIf:: (Int->Bool) -> [Int] -> Int
countIf p x = length ( filter p x )
```

46. Problem 12 - Combination of Applications
- Answer
```hs
combined::[Int] -> [Int->Int] -> [[ Int]]
combined l fs = [ map f l | f <- fs]
```
- Results
```hs
ghci> combined [1,2,3] [(*2),(^2),(+3)]
[[2,4,6],[1,4,9],[4,5,6]]
```

47. Problem 13 - Consecutive Functions
- Answer:
```hs
consecutive :: [Int] -> [Int -> Int] -> [[Int]]
consecutive l fs = [[f x | f <- fs] | x <- l]
```
- Results
```hs
ghci> consecutive [1,2] [(*2),(^2),(+3)]
[[2,1,4],[4,4,5]]
```

48. Problem 14 - Filter Fold
```hs
filterFoldl :: (Int -> Bool) -> (Int -> Int -> Int) -> Int -> [Int] -> Int
filterFoldl c f y xs = foldl f y (filter c xs)
```

## Section 10: Solved Problems - Infinite Lists

49. Problem 1 - Infinite Ones

50. Problem 2 - Natural Numbers

51. Problem 3 - Infinite Integers

52. Problem 4 - Triangular Numbers - Part 1

53. Problem 4 - Triangular Numbers - Part 2

54. Problem 5 - Factorial Dimension

55. Problem 6 - Fibonacci Sequence

56. Problem 7 - Prime Numbers

57. Problem 8 - Hamming Numbers
```hs
hammings :: [Integer]

hammings = 1 : (merge (map (*2) hammings) $ merge (map (*3) hammings) (map (*5) hammings))
    where
        merge :: [Integer] -> [Integer] -> [Integer]
        merge (x:xs) (y:ys)
            | x < y = x : merge xs (y:ys)
            | x == y = x : merge xs ys
            | otherwise = y : merge (x:xs) ys
```           
           
## Section 11: Binary Trees

58. Fundamentals of Binary Trees

## Section 12: Solved Problems - Binary Trees

59. Problem 1 - Tree Size
- Definition of the tree
  - `data Tree a = Node a (Tree a) (Tree a) | Empty deriving (Show)`
  - A tree with of a type a (given when defined, Int or Real or String) is, either an empty tree, either a node with an element (of type a) and two other trees of the same type. The `deriving (Show)` statement enables visualization of trees
```hs
data Tree a = Node a (Tree a) (Tree a) | Empty 
    deriving (Show)
size :: Tree a -> Int
size Empty = 0  -- Base Case: Empty Tree
size (Node _  lc rc) = 1 + size lc + size rc
```

60. Problem 2 - Height
```hs
data Tree a = Node a (Tree a) (Tree a) | Empty 
    deriving (Show)
height :: Tree a -> Int
height Empty = 0
height (Node _ lc rc)  = 1 + max (lc) (height rc)
```

61. Problem 3 - Equivalent Trees
```hs
data Tree a = Node a (Tree a) (Tree a) | Empty 
    deriving (Show)
equal :: Eq a => Tree a -> Tree a -> Bool 
equal Empty Empty = True
equal _ Empty = False
equal Empty _ = False
equal (Node x la ra) (Node y lb rb)
    | x /= y = False
    | otherwise = equal la lb && equal ra rb
```

62. Problem 4 - Isomorphism
```hs
data Tree a = Node a (Tree a) (Tree a) | Empty 
    deriving (Show)      
isomorphic :: Eq a => Tree a -> Tree a -> Bool
isomorphic Empty Empty = True
isomorphic Empty _ = False
isomorphic _ Empty = False
isomorphic (Node x l_a r_a) (Node y l_b r_b)
    | x /= y = False
    | otherwise = (isomorphic l_a l_b && isomorphic r_a r_b) ||
                  (isomorphic l_a r_b && isomorphic r_a l_b)
```

63. Problem 5 - Preorder Traversal
```hs
data Tree a = Node a (Tree a) (Tree a) | Empty deriving (Show)
preOrder :: Tree a -> [a]
preOrder Empty = []
preOrder (Node x lt rt) = [x] ++ preOrder (lt) ++ preOrder (rt)
-- preOrder
-- 1.ROOT
-- 2.LEFT
-- 3.RIGHT
```

64. Problem 6 - Postorder Traversal
```hs
data Tree a = Node a (Tree a) (Tree a) | Empty deriving (Show)
postOrder :: Tree a -> [a]
postOrder Empty = []
postOrder (Node x left right) = (postOrder left) ++
                                (postOrder right) ++
                                [x]
-- 1.LEFT
-- 2.RIGHT
-- 3.ROOT
```

65. Problem 7 - Inorder Traversal
```hs
data Tree a = Node a (Tree a) (Tree a) | Empty deriving (Show)
inOrder :: Tree a -> [a]
inOrder Empty = []
inOrder (Node x l_c r_c) = inOrder (l_c) ++ [x] ++ inOrder (r_c)                    
--Inorder : LEFT, ROOT, RIGHT                    
```                

66. Problem 8 - Breadth First Search
```hs
data Tree a = Node a (Tree a) (Tree a) | Empty deriving (Show)
breadthFirst :: Tree a -> [a]
breadthFirst t = bfs [t]
bfs :: [Tree a] -> [a]
bfs [] = []
bfs (Empty:xs) = bfs xs
bfs ((Node x lc rc):xs) = x : (bfs $ xs ++ [lc,rc])
```

## Section 13: Multiway Trees

67. Fundamentals of Multiway Trees
- Haskell Multiway Tree definition:
```hs 
data Tree a = Node a  [Tree a]
       deriving (Eq, Show)
```       

## Section 14: Solved Problems - Multiway Trees

68. Problem 1 - Number of Nodes
```hs
data Tree a = Node a [Tree a]
        deriving (Eq, Show)
numnodes :: Tree a -> Int
numnodes (Node _ children) = 1 + sum (map numnodes children)
```

69. Problem 2 - Construction on Information
- Using '^' as backtrack
```
        a
       /|\
      f c b
      |   /\
      g  d  e
```
- The above tree is represented as 'afg^^c^bd^e^^^'
```hs
data Tree a = Node a [Tree a]
        deriving (Eq, Show)        
stringToTree :: String -> Tree Char
stringToTree (x:xs) = Node x (fst (stringToTrees xs))
    where stringToTrees (x:xs)
            | x == '^' = ([],xs)
            | otherwise = ([Node x trees0] ++ trees1, rest1)
                where  (trees0,rest0) = stringToTrees xs
                       (trees1,rest1) = stringToTrees rest0
```

70. Problem 3 - Path Length
```hs
data Tree a = Node a [Tree a]
        deriving (Eq, Show)
pathLength :: Tree a -> Int
pathLength = pathLengthAux 0
  where pathLengthAux d (Node _ ts) = d + sum (map (pathLengthAux (d+1)) ts)
```

71. Problem 4 - Bottom-Up
```hs
data Tree a = Node a [Tree a]
        deriving (Eq, Show)        
bottomUp :: Tree a -> [a]
bottomUp (Node x ts) = concatMap bottomUp ts ++ [x]
```

## Section 15: Graphs

72. Fundamentals of Graphs
```
   g        b--c
   |         \/
   h        f
           /    d
          k
```          
- Edge Notation
  - Edge [(g,h),(k,f),(f,b),(f,c),(b,c)]
```hs
data Graph a = Edge [(a,a)]
       deriving (Show, Eq)
```
  - Isolated nodes cannot be represented
- Graph Notation
  - Combination of nodes and edges
  - Graph([b,c,d,f,g,h,k], [(b,c),(b,f),(c,f),(f,k),(g,h)])
```hs
data Graph a = Graph [a] [(a,a)]  
        deriving (Show, Eq)
```        
- Adjacency List Notation
  - Adj[('b',"cf"),('c',"bf"),('d',""),('f',"bck"),('g',"h"),('h',"g"),('k',"f")]
```hs
data Graph a = Adj [(a, [a])]
           deriving (Show, Eq)
```

## Section 16: Solved Problems - Graphs

73. Problem 1 - Acyclic Paths
```hs
acyclicPaths :: Eq a => a -> a -> [(a,a)] -> [[a]]
acyclicPaths source sink edges
    | source == sink = [[sink]]
    | otherwise = [
        source:path | edge <- edges, (fst edge) == source,
            path <- (acyclicPaths (snd edge) sink
            [e | e <- edges, e /= edge]) ]
```

74. Problem 2 - Depth First Search
```hs
type Node = Int
type Edge = (Node,Node)
type Graph = ([Node],[Edge])

depthFirst :: Graph -> Node -> [Node]

depthFirst (v,e) n
    | [x|x<-v,x==n] == [] = []
    | otherwise = recdepth (v,e) [n]

recdepth :: Graph -> [Node] -> [Node]
recdepth ([],_) _ = []
recdepth (_,_) [] = []
recdepth (v,e) (top:stack)
        | [x|x<-v,x==top] == [] = recdepth (newv,e) stack
        | otherwise = top : recdepth (newv,e) (adjacent ++ stack)
                where
                        adjacent = [x| (x,y)<-e, y == top] ++
                                   [x| (y,x)<-e, y == top]
                        newv = [x|x<-v, x /= top]
```

75. Problem 3 - Connected Components
```hs
import Data.List
type Node = Int
type Edge = (Node,Node)
type Graph = ([Node],[Edge])
depthfirst :: Graph -> Node -> [Node]
depthfirst (v,e) n
    | [x|x<-v, x==n] == [] = []
    | otherwise = dfrecursive (v,e) [n]

dfrecursive :: Graph -> [Node] -> [Node]
dfrecursive ([],_) _ = [] -- Base Case 1: V is empty
dfrecursive (_,_) [] = [] -- Base Case 2: Solution empty
dfrecursive (v,e) (top:stack)
    | [x|x<-v,x==top] == [] = dfrecursive (newv,e) stack
    | otherwise = top : dfrecursive (newv,e) (adjacent ++ stack)
    where
        newv = [x|x<-v,x/=top]
        adjacent = [x | (x,y)<-e,y==top] ++ [x | (y,x)<-e,y==top]

connectedcomponents :: Graph -> [[Node]]

connectedcomponents ([],_) = []
connectedcomponents (top:v,e)
    | remaining == [] = [connected]
    | otherwise = connected : connectedcomponents (remaining,e)
    where
        connected = depthfirst (top:v,e) top
        remaining = (top:v) \\ connected
```

## Section 17: Advanced Types


76. Maybe Data Type
- `data Maybe a = Just a | Nothing`
- Of type a with the constructor Just or absence (empty constructor Nothing)
- Applications
  - Indicates possible null value
  - Indicates absence of a result
  - Report an error
```hs
f::Int -> Maybe Int
f 0 = Nothing
f x = Just x

g::Int -> Maybe Int
g 100 = Nothing
g x = Just x

k:: String -> Maybe String
k "Field" = Nothing
k "Sea" = Nothing
k s = Just s
```

77. Either Data Type
- The polymorphic type `Either a b` is predefined as:
```hs
data Either a b = Left a | Right b
```
- There are two possibilities for a value of type a (using Left constructor) or type b (using Right constructor)
- Usually right side is the correct result type
  - Left side or a could be String and can be used for error diagnosis
- test.hs
```hs
secDiv:: Float -> Float -> Either String Float
secDiv _ 0 = Left "Division by zero"
secDiv x y = Right ( x/y )
```
  - Not only two cases, many more cases can be defined, producing different messages or different float number operations
- Test run
```hs
ghci> :load "section17/test.hs"
ghci> secDiv 3 0
Left "Division by zero"
ghci> secDiv 3 1
Right 3.0
```

78. Algebraic Types
- Algebraic types can be deconstructed using patterns
```hs
data Shape
   = Rectangle Float Float  -- height/width
   | Square Float           -- size
   | Circle Float           -- radius
   | Point

area :: Shape -> Float
area (Rectangle width height) = width * height
area (Square size) = area (Rectangle size size)
area (Circle radius) = pi * radius * radius
area Point = 0
```
- Test run
```hs
ghci> area (Rectangle 4 3)
12.0
ghci> area (Square 2.0)
4.0
ghci> area (Circle 1)
3.1415927
```

## Section 18: Functors

79. Introduction to Funtors
```hs
ghci> (+3) [1,2,3]
<interactive>:33:1: error:
ghci> fmap (+3) [1,2,3]
[4,5,6]
ghci> +3 Nothing
<interactive>:35:1: error: parse error on input ‘+’
ghci> fmap(+3) Nothing
Nothing
ghci> (fmap (*2) (+3)) 2
10
```

80. Implementation of fmap
- `fmap` applies a function to the elements of a generic container `f a` returning a container of the same time
- `fmap` is a function of the instances of the class Functor:
```hs
ghci> :t fmap
fmap :: Functor f => (a -> b) -> f a -> f b
ghci> :info fmap
type Functor :: (* -> *) -> Constraint
class Functor f where
  fmap :: (a -> b) -> f a -> f b
```

81. Maybe Instance as Functor
```hs
ghci> :info Maybe
type Maybe :: * -> *
data Maybe a = Nothing | Just a
  	-- Defined in ‘GHC.Maybe’
instance Semigroup a => Monoid (Maybe a) -- Defined in ‘GHC.Base’
instance Semigroup a => Semigroup (Maybe a)
  -- Defined in ‘GHC.Base’
instance Foldable Maybe -- Defined in ‘Data.Foldable’
instance Traversable Maybe -- Defined in ‘Data.Traversable’
instance Eq a => Eq (Maybe a) -- Defined in ‘GHC.Maybe’
instance Ord a => Ord (Maybe a) -- Defined in ‘GHC.Maybe’
instance Show a => Show (Maybe a) -- Defined in ‘GHC.Show’
instance Read a => Read (Maybe a) -- Defined in ‘GHC.Read’
instance Applicative Maybe -- Defined in ‘GHC.Base’
instance Functor Maybe -- Defined in ‘GHC.Base’
instance MonadFail Maybe -- Defined in ‘Control.Monad.Fail’
instance Monad Maybe -- Defined in ‘GHC.Base’
```

82. Either Instance as Functor
```hs
ghci> :info Either
type Either :: * -> * -> *
data Either a b = Left a | Right b
  	-- Defined in ‘Data.Either’
instance Traversable (Either a) -- Defined in ‘Data.Traversable’
instance Semigroup (Either a b) -- Defined in ‘Data.Either’
instance Applicative (Either e) -- Defined in ‘Data.Either’
instance Foldable (Either a) -- Defined in ‘Data.Foldable’
instance Functor (Either a) -- Defined in ‘Data.Either’
instance Monad (Either e) -- Defined in ‘Data.Either’
instance (Read a, Read b) => Read (Either a b)
  -- Defined in ‘Data.Either’
instance (Eq a, Eq b) => Eq (Either a b)
  -- Defined in ‘Data.Either’
instance (Ord a, Ord b) => Ord (Either a b)
  -- Defined in ‘Data.Either’
instance (Show a, Show b) => Show (Either a b)
  -- Defined in ‘Data.Either’
```

83. Lists and Functions - Instances of the Functor class

84. Binary Trees as Functors

85. Laws of Functors
- Functor instances must have these properties
  - Identity: `fmap id == id`
  - Composition: `fmap(g1 . g2) == fmap g1 . fmap g2`

## Section 19: Applicatives

86. Introduction to Applicatives
- When a function is inside of a container
- `<*>`
```hs
ghci> [(*2),(+3)] <*> [2]
[4,5]
ghci> Right (+3) <*> Right 2
Right 5
ghci> Right (+3) <*> Left "Err"
Left "Err"
ghci> Left "err" <*> Right 2
Left "err"
```

87. Implementation of the Applicative Operator
- The operator `<*>` is an operation of the class Applicative (which must be also functor)
- `<*>` applies a function inside a container to values inside a container. Containers are generic and of the same type
- `pure` constructs a container with a value
```hs
ghci> :info <*>
type Applicative :: (* -> *) -> Constraint
class Functor f => Applicative f where
  ...
  (<*>) :: f (a -> b) -> f a -> f b
  ...
  	-- Defined in ‘GHC.Base’
infixl 4 <*>
ghci> :info pure
type Applicative :: (* -> *) -> Constraint
class Functor f => Applicative f where
  pure :: a -> f a
  ...
  	-- Defined in ‘GHC.Base’
```

88. Instantiations of Applicative
- Laws of Applicatives
  - Identity: `pure id <*>v == v`
  - Homomorphism: `pure f <*> pure x == pure (f x)`
  - Exchange: `u <*> pure y === pure ($ y) <*> u`
  - Composition: `u <*> (v <*> w) == pure (.) <*> u <*> v <*> w`
  - Relation with the functor: `fmap g x == pure g <*> x`

## Section 20: Monads

89. Introduction to Monads
- A monad is a structure that combines program fragments (functions) and wraps their return values in a type with additional computation
  - Ref: https://en.wikipedia.org/wiki/Monad_(functional_programming)
- Operator `>>=` is an operation of the class Monad
  - A function that unpack, applies the given function then leaves encapsulated
- Type `Maybe` is an instance of Monad

90. Bind Operator (>>=)
```hs
ghci> :info >>=
type Monad :: (* -> *) -> Constraint
class Applicative m => Monad m where
  (>>=) :: m a -> (a -> m b) -> m b
  ...
  	-- Defined in ‘GHC.Base’
infixl 1 >>=
```

91. Monads Operations
```hs
ghci> :info Monad
type Monad :: (* -> *) -> Constraint
class Applicative m => Monad m where
  (>>=) :: m a -> (a -> m b) -> m b
  (>>) :: m a -> m b -> m b
  return :: a -> m a
```
- Monads have three operations
  - `return` wrap
  - `>>=` unwrap, apply and wrap
  - `>>` is purely esthetic

92. Monads Instances - Either, Maybe, and Lists
```hs
instance Monad Maybe where
  return        = Just
  Nothing >>= f = Nothing
  Just x  >>= f = f x
instance Monad (Either a) where
  return        = Right
  Left x  >>= f = Left x
  Right x >>= f = f x
instance Monad [] where
  return x = [x]
  xs >>= f = contcatMap f xs
```  

93. Monad Laws
- Identity on the left
  - `return x >>= f == fx`
- Identity on the right
  - `m >>= return == m`
- Associativity
  - `(m >>= f) >>= g == m >>= (\x -> f x >>= g)`

94. Do Notation
- Syntactic sugar to facilitate the use of Monads
- With `do`, functional code looks like imperative code with assignment

95. Do Notation Example

96. State Monad
- State type: Represents a stateful computation
- State constructor: Wraps a function that takes an initial state and produces a result and a new state
- runState function: Executes the stateful computation
```hs
import Control.Monad.State

-- Define a stateful computation to increment the counter
incrementCounter :: State Int ()
incrementCounter = do
    counter <- get
    put (counter + 1)

-- Define a stateful computation to decrement the counter
decrementCounter :: State Int ()
decrementCounter = do
    counter <- get
    put (counter - 1)

-- Main function to demonstrate stateful computations
main :: IO ()
main = do
    -- Initialize the state with an initial value of 0
    let initialState = 0

    -- Run the stateful computations using execState
    let finalState = execState (do { incrementCounter;
                        incrementCounter; decrementCounter; incrementCounter}) initialState

    -- Print the final state
    putStrLn $ "Final State: " ++ show finalState
```

## Section 21: Input and Output

97. Introduction to Input and Output
- Input/Output in Haskell is based on Monad
  - The main program is `main::IO()`
  - The IO type constructor is used to handle input/output
  - IO is an instance of Monad
  - It is usually used with do notation
- Basic operations
  - getChar :: IO Char
  - getLine :: IO String
  - getContents :: IO String
  - putChar :: Char -> IO()
  - putStr :: String -> IO()
  - putStrLn :: String -> IO()
  - print :: Show a => a -> IO()
  - () is a zero field tuple and () is the only value of type()
- Sample read/write
```hs
main = do
  putStrLn "What is your name?"
  name <- getLine
  putStrLn $ "Hello " ++  name ++ "!"
```  

98. Working with Input and Output
```hs
main :: IO ()
-- Write a text backwards
main = do
    line <- getLine
    if line /= "*" then do
        putStrLn $ reverse line
        main
    else
        return ()
```

## Section 22: Solved Problems - Input and Output

99. Problem 1 - Groups
```hs
main :: IO ()
main = do
    name <- getLine
    putStrLn $ convert name        
convert :: String -> String
convert s =
    if myElem 'a' s || myElem 'A' s then
        "You belong to Group A!"
    else
        "You belong to Group B!" 
myElem :: Eq a => a -> [a] -> Bool   
myElem _ [] = False
myElem x (y:ys) = if x == y then True else myElem x ys
```

100. Problem 2 - Cities
```hs
main :: IO()

main = do
    line <- getLine
    if line /= "*" then do
        putStrLn $ city_sol line
        main
    else
        return ()
        
city_sol :: String -> String
city_sol line = name ++ ": " ++ sol
    where
        (name,km) = parse line
        sol = interpret km
        
parse :: String -> (String,Float)
parse line = (n,km)
    where
        [n,_km] = words line
        km = read _km
        
interpret :: Float -> String
interpret x
    | x < 18 = "Napa"
    | x < 25 = "Davenport"
    | x < 30 = "Naperville"
    | x <= 45 = "Phoenix"
    | otherwise = "Carbondale"
```

101. Problem 3 - Sum of Elements
```hs
import System.IO

prompt t = do putStr t
              hFlush stdout

sumOfNumbers =
 do putStrLn "Compute the sum of some numbers."
    prompt "How many numbers?"
    n <- readLn
    let ask n = do prompt ("Enter a number:")
                   readLn
    list_numbers <- mapM ask [1..n]
    putStr "The sum of the numbers is "
    print (sum list_numbers)
```

102. Problem 4 - Conditional Sort
```hs
import System.IO
import Data.List
prompt t = do putStr t
              hFlush stdout
sortNumbers = do unsorted_solution <- readUntilZero
                 print (sort unsorted_solution)
readUntilZero =
    do prompt "Enter a number (0 to end)"
       n <- readLn
       if n == 0 then return [] else do sub_sol <- readUntilZero
                                        return (n:sub_sol)
```

103. Problem 5 - Computer Game
```hs
import System.IO
prompt t = do putStr t
              hFlush stdout
game :: IO()
game =
    do putStrLn "Think of a number between 1 and 100!"
       play 1 100
    
play :: Int -> Int -> IO()
play lo hi | lo>hi = putStrLn "This is not possible!"
play lo hi =
  do let mid = (lo+hi) `div` 2
     ans <- validM (`elem` ["lower","higher","yes"]) $
                   do prompt ("Is it "++show mid++"? ")
                      getLine
     case ans of
       "yes" -> putStrLn "Great, I won!"
       "lower" -> play lo (mid-1)
       "higher" -> play (mid+1) hi
validM :: (a->Bool) -> IO a -> IO a
validM valid ask =
  do ans <- ask
     if valid ans then return ans
                  else do putStrLn "Answer properly!"
                          validM valid ask
```

## Section 23: Final Exams

104. Exam 1 - Problem 1

105. Exam 1 - Problem 2

106. Exam 2 - Problem 1

107. Exam 2 - Problem 2

108. Exam 2 - Problem 3

109. Exam 3 - Problem 1

110. Exam 3 - Problem 2
