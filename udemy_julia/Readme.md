## Title: Programming with Julia
- Instructor: Dr. İlker Arslan

## Section 1: Introduction

1. Introduction

2. History of Julia
- Speed of C and the dynamicsm of Ruby

3. Why Julia?
- On top of LLVM
- https://juliahub.com/ui/packages

4. Codes and resources
- https://github.com/ilkerarslan/JuliaCourseCodes.git

## Section 2: Starting with Julia

5. Installing Julia in Windows

6. Installing Julia in Linux
- https://julialang.org/downloads/

7. Installing Julia with juliaup

8. Julia REPL
- Julia CLI
```julia
julia> println("hello world\n")
hello world
julia> 4+3;

julia> 4+3
7
```
- `;` starts shell
- Back-space to stop shell
- For a saved file, use extension of `.jl`

9. Julia Editor and IDEs
- vscode, jupyter, pluto.jl

## Section 3: Variables, Data Type, and Operations

10. Introduction
- Dynamically typed language

11. Variables
```julia
julia> x = 11; y=12;z=31
31

julia> x
11

julia> y
12

julia> z
31

julia> y = x^3 + 3*x
1364

julia> x= "Hello world"
"Hello world"

julia> typeof(x)
String
```
- \alpha + TAB for alpha character
- \beta + TAB for beta character
- \_0 TAB for subscript 0
- \^1 TAB for superscript 1
- \pi TAB for pi number
- \euler TAB for *e*
```julia
ulia> π
π = 3.1415926535897...

julia> \euler
ERROR: syntax: "\" is not a unary operator
Stacktrace:
 [1] top-level scope
   @ none:1

julia> ℯ
ℯ = 2.7182818284590...
```
- A single line comment: `#`
- Multi line comments: `#= .... =#`
```julia
julia> a,b,c = 1,2,3
(1, 2, 3)

julia> a,b = b,a
(2, 1)

julia> println(a, ",", b)
2,1
```
- DataType declaration
  - Not working in CLI or global variables
  - For function arguments
  - (expression)::DataType
    - Ex: x::Int64, y::String
- For constant variables, use `const` and name as full Capitals
  - Re-assigning with different datatype will generate error
  - Re-assigning with same datatype will generate warning

12. Type Hierarchy in Julia
![datatype](./ch12_datatype.png)
- Red label: Abstract types. Cannot be instantiated but can constraint argument type
- Blue label: Concrete types
```julia
julia> subtypes(Any)
587-element Vector{Any}:
 AbstractArray
 AbstractChannel
 AbstractChar
 AbstractDict
 AbstractDisplay
 AbstractMatch
 AbstractPattern
 AbstractSet
 AbstractString
 Any
 ⋮
 Timer
 Tuple
 Type
 TypeVar
 UndefInitializer
 Val
 VecElement
 VersionNumber
 WeakRef

julia> supertypes(Any)
(Any,)

julia> supertypes(Number)
(Number, Any)

julia> subtypes(Number)
2-element Vector{Any}:
 Complex
 Real
```
- `<:`: is subtype?
```julia
julia> Int64 <: Number
true

julia> Number <: Int64
false

julia> Float64 <: Real
true
```

13. Numerical Data Types: Integers and Floating-Point Numbers
```julia
ulia> typemax(Int8)
127

julia> typemax(Int)
9223372036854775807

julia> typemax(Int16)
32767

julia> typemax(Int32)
2147483647

julia> typemax(Int64)
9223372036854775807

julia> typemin(Int32)
-2147483648

julia> Sys.WORD_SIZE
64

julia> 10^50
-5376172055173529600 #<----- overflow !!!

julia> big(10)^50
100000000000000000000000000000000000000000000000000

julia> typeof(1)
Int64

julia> typeof(1.0)
Float64

julia> typeof(1.e-2)
Float64

julia> typeof(1.f-2)
Float32

julia> x = 1_000_000
1000000

julia> y = Inf
Inf

julia> sizeof(Inf)
8

julia> typeof(Inf)
Float64

ulia> NaN
NaN

julia> typeof(NaN)
Float64

julia> 0/0
NaN

julia> isinf(0/0)
false

julia> isinf(1/0)
true

julia> eps()
2.220446049250313e-16 #<-------- machine epsilon

julia> eps(Float64)
2.220446049250313e-16

julia> eps(Float32)
1.1920929f-7

julia> eps(Float16)
Float16(0.000977)

julia> typeof(true)
Bool

julia> true == 1
true

julia> false == 0
true

julia> Bool(1)
true
```

14. Numerical Data Types: Complex and Rational Numbers
```julia
julia> x = 3+4im
3 + 4im

julia> real(x)
3

julia> imag(x)
4

julia> conj(x)
3 - 4im

julia> typeof(x)
Complex{Int64}

julia> sqrt(-1)
ERROR: DomainError with -1.0: #<----- Throws

julia> √Complex(-1) #<--- \sqrt + TAB 
0.0 + 1.0im

julia> √(-1+0im)
0.0 + 1.0im

julia> sqrt(-1+0im)
0.0 + 1.0im

julia> x = 5/2
2.5

julia> x = 5//2
5//2

julia> typeof(x)
Rational{Int64}

julia> isa(1, Int)
true

```

15. Character and String Types

16. Primitive and Composite Types

17. Parametric Types

18. Basic Operations

19. Exercises: Variables, Data Types & Operations

20. Solutions to Exercises: Variables, Data Types & Operations

## Section 4: Data Structures

## Section 5: Conditionals and Loops

## Section 6: Functions

## Section 7: Methods

## Section 8: Modules and Packages

## Section 9: Metaprogramming

## Section 10: Streams and Networking

## Section 11: Parallel Programming
