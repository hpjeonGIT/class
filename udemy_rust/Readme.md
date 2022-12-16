## Summary
- Title: Rust Programming For Beginners
- Instructor: Jayson Lennon
- src: https://github.com/jayson-lennon/rust-programming-for-beginners

## Section 1: Getting Started

1. Download Data files

2. Introduction

3. Installation
- https://www.rust-lang.org/tools/install
- curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  - Will install at ~/.cargo
- source ~/.carbo/env
- Rust is ready now

4. Fundamentals: Data types
- Boolean: true, false
- Integer
- Double/Float
- Character : 'A', '7', '$'
- String: "hello world"

5. Fundamentals: Variables
```rust
let two = 2;
let hello = "hello";
let mut ny_name = "Bill";
let j = 'j';
let quit_program = false;
```

6. Fundamentals: Functions
```rs
fn add(a: i32, b:i32) -> i32 { a+b}
```

7. Fundamentals: println macro
```rs
let life = 42;
println!("hello");
println!("{:?}",life);
println!("{:?}" "{:?}",life, life);
```
- `:?` means debug mode
- Macros use an exlcamation point `!` to call
- Generates additional Rust code

8. Fundamentals: Control Flow
```rs
if ... {
} else if  ... {
} else {  
}
```

9. Fundamentals: Repetition
- Loop
```rs
let mut a = 0;
loop {
  if a==5 {
    break;
  }
  println!("{:?}",a);
  a = a + 1;
}
```
- While loop
```rs
let mut a = 0;
while a != 5 {
  println!("{:?}",a);
  a = a + 1;
}
```

10. Comments
```rs
//  Single line comment
fn main() {
  println!("hello world");
}
```

11. Coding Exercise: Functions
- a1.rs
```rs
fn first_name() {
  println!("Hello");
}
fn second_name() {
  println!("world");
}
fn main() {
  first_name();
  second_name();
}
```
- Command to run: `cargo new a1; cd a1`
  - Edit `src/main.rs`
  - `cargo run`
  - Or
    - Use `rustc src/main.rs -o a1; ./a1`

12. Demo: Numeric Types and Arithmetic

13. Coding Exercise: Basic Math

## Section 2: Making Decisions

14. Coding Exercise: Logic with If & Else (part 1)

15. Coding Exercise: Logic with If & Else (part 2)

16. Fundamentals: Match
- Similar to if..else
- Exhaustive
```rs
fn main() {
  let some_int = 3;
  match some_int {
    1 => println!("it is 1"),
    2 => println!("it is 2"),
    _ => println!("it is unknown"),
  }
}
```
- `_` to matchy anything else

17. Demo: Match

18. Coding Exercise: Basic Match (Part 1)

19. Coding Exercise: Basic Match (Part 2)

## Section 3: Repetition

20. Demo: Loop
- `loop {...}`: infinite loop
  - May use `break` to exit the loop
```rs
fn main() {
  let mut i=3;
  loop {
    println!("{:?}",i);
    i = i - 1;
    if i == 0 {
      break;
    }
  }
  println!("Done");
}
```

21. Coding Exercise: Loop

22. Demo: While Loop
- `while condition {...}`
```rs
fn main() {
  let mut i=1;
  while  i<=3 {
    println!("{:?}",i);
    i = i + 1;
  }
}
```

23. Coding Exercise: While Loop

## Section 4: Working with Data

24. Working with Data: Enums
- Enumeration
  - Data that can be one of multiple different possibilities
    - Each possibility is called a variant
  - Provides information about your program to the compiler
    - More robust programs with match keyword

25. Demo: Enums
```rs
enum Direction {
  Left,
  Right,
}
fn main() {
  let go = Direction::Left;
  match go {
   Direction::Left  => println!("go left"),
   Direction::Right => println!("go right"),
   _                => println!("get lost"),
  }
}
```

26. Coding Exercise: Enums
```rs
enum Direction {
  Left,
  Right,
}
fn print_Direction(go: Direction) {
  match go {
   Direction::Left  => println!("go left"),
   Direction::Right => println!("go right"),
   _                => println!("get lost"),
  }
}
fn main() {
   print_Direction(Direction::Right);
}
```

27. Working With Data: Structs
- Structure
  - A type that contains multiple pieces of data
  - All or nothing - cannot have partial data
  - Each piece of data is called a field
```rs
struct ShippingBox {
  depth: i32,
  width: i32,
  height: i32,
}
```

28. Demo: Structs
```rs
struct GroceryItem {
  stock: i32,
  price: f64,
}
fn main() {
  let cereal = GroceryItem{ 
    stock: 10,
    price: 2.99,
  };
  println!("stock: {:?}", cereal.stock);
}
```

29. Coding Exercise: Structs

30. Working With Data: Tuples
- Tuples
  - A type of record
  - Stores data anonymously
  - No need to name fields
```rs
enum Access {
  Full,
}
fn three_tuples() -> (i32,i32,i32) {
  (1,2,3) 
}
fn main() {
  let n3 = three_tuples();
  let (x,y,z) = three_tuples();
  println!("{:?}, {:?}",x,n3.0);
  println!("{:?}, {:?}",y,n3.1);
  println!("{:?}, {:?}",z,n3.2);
  let (name,access) = ("Jake", Access::Full);
}
```
  
31. Demo: Tuples
```rs
let coord = (2,3);
println!(coord.0, coord.1);
let (x,y) = (2,3);
println!(x,y);
```
- More than 2 items, struct might be better

32. Coding Exercise: Tuples

## Section 5: Intermediate Concepts

33. Fundamentals: Expressions
- Rust is an expression-based language
- Expression values caolesce to a single point

34. Demo: Expressions

35. Coding Exercise: Expressions
```rs
let value = 100;
let is_gt_100 = value > 100 ; // will be false
```

36. Fundamentals: Intermediate Memory
- Memory uses addresses and offsets
  - Offsets can be used to index into some data

37. Ownership
- Programs must track memory - to avoid leak
- Rust utilizes an ownership model to manage memory
  - The owner of memory is responsible for cleaning up the memory
- Memory can either be moved or borrowed
```rs
enum Light {
   Bright,
   Dull,
}
fn display_light(light: Light) {
   match light {
    Light::Bright => println!("bright"),
    Light::Dull   => println!("dull"),
   }
}
fn display_num(a: i32) {
   println!("{:?}",a);
}
fn main() {
   let x = 321;
   display_num(x);
   display_num(x); // allowed as implements copy trait
   let dull = Light::Dull;
   display_light(dull);
   //display_light(dull); dull data is moved already in the above. Not allowed as Copy trait is not implemented. 
}
```
- `&` or referencing: borrows data, not moving
```rs
enum Light {
   Bright,
   Dull,
}
fn display_light(light: &Light) {
   match light {
    Light::Bright => println!("bright"),
    Light::Dull   => println!("dull"),
   }
}
fn display_num(a: i32) {
   println!("{:?}",a);
}
fn main() {
   let x = 321;
   display_num(x);
   display_num(x);
   let dull = Light::Dull;
   display_light(&dull);
   display_light(&dull); // this is allowed
}
```

38. Demo: Ownership
```rs
struct Book {
  pages: i32,
  rating: i32,
}
fn display_page_count(book: Book) {
  println!("pages = {:?}", book.pages);
}
fn display_rating(book: Book) {
  println!("rating = {:?}", book.rating);
}
fn main() {
  let book = Book {
      pages: 5,
      rating: 9,
  };
  display_page_count(book); // data is moved
  // display_rating(book); // not allowed as the data is gone
}
```

39. Coding Exercise: Ownership

40. Demo: Implementing Functionality
- `impl`: implementing functionality
```rs
struct Temperature {
   degrees_f: f64,
}
impl Temperature {
   fn show_temp_using_arg(temp: &Temperature) {
      println!("{:?} degrees F", temp.degrees_f);
   }
   fn show_temp_using_self(&self) {
     println!("{:?} degrees F", self.degrees_f);
  }
}
fn main() {
   let hot = Temperature { degrees_f: 99.9};
   Temperature::show_temp_using_arg(&hot);
   hot.show_temp_using_self();
}
```

41. Coding Exercise: Implementing Functionality

## Section 6: Data Collections

42. Data Structures: Vectors
- Multiple pieces of data
- Must be the same type
- Can add, remove, and traverse the entries
```rs
let my_numbers = vec![1,2,3]; // vec macro
for num in my_numbers {
  println!("{:?}",num);
}
let mut my_nb = Vec::new();
my_nb.push(1);
my_nb.push(2);
my_nb.push(3);
my_nb.pop();
my_nb.len();
let two = my_nb[1];
```

43. Demo: Vectors & For loops

44. Coding Exercise: Vectors & For loops

45. Strings
```rs
let str1 = "owned string".to_owned();
let str2 = String::from("another string");
```
- Strings are automatically borrowed
- To createan owned copy of a string slice, use .to_owned() or String::from()
- Use an owned String when storing in a struct

46. Demo: Strings
```rs
struct LineItem {
  name: String,
  count: i32,
}
fn main() {
  let receipt = vec![
      LineItem{
        name: "cereal".to_owned(),
        // name: "cereal", will generate error
        count: 1,
      },
      LineItem{
        name: String::from("fruit"),
        count: 3,
      },
  ];
  for item in receipt {
     println!("name: {:?}, count: {:?}", item.name, item.count);
  }
}
```

47. Coding Exercise: Strings
```rs
struct Person {
  name: String,
  fav_color: String,
  age: i32,
}
fn print(data: &str) {
  println!("{:?}", data);
}
fn main() {
  let people = vec![
    Person {
      name: String::from("George"),
      fav_color: String::from("green"),
      age: 7,
    },
    Person {
      name: String::from("Anna"),
      fav_color: String::from("purple"),
      age: 9,
    },
    Person {
      name: String::from("Katie"),
      fav_color: String::from("blue"),
      age: 14,
    },
  ];
  for person in people {
    if person.age <= 10 {
      print(&person.name);
      print(&person.fav_color);
    }
  }
}
```

## Section 7: Advanced Concepts

48. Demo: Deriving Functionality
- `derive`: adds additional functionality to enum and struct
```rs
#[derive(Debug, Clone, Copy)]
enum Position {
  Manager,
  Supervisor,
  Worker,
}
#[derive(Debug)]
struct Employee {
  position: Position,
  work_hours: i64,
}
fn main() {
  let me = Employee {
    position: Position::Worker,
    work_hours: 40,
  };
  println!("{:?}", me); // without derive, this fails at compilation
}
```

49. Type Annotations
- Required for function signatures
- Types are usually inferred but explicity type annotations can be specified
- Ex)
  - Function signature: `fn print_some(msg: &str, count: i32) {...}`
  - `let a = 132;` or `let a: i32 = 132;`
```rs
let numbers: Vec<i32> = vec![1,2,3];
let letters: Vec<char> = vec!['a','b'];
let clicks: Vec<Mouse> = vec![
  Mouse::LeftClick,
  Mouse::RightClick,
  ];
```

50. Enums Revisited
- Each variant of enum can optionally contain additional data
```rs
enum Mouse {
  LeftClick,
  RightClick,
  MiddleClick,
  Scroll(i32),
  Move(i32,i32), // x,y
}
enum PromoDiscount {
  NewUser,
  Holiday(String),
}
enum Discount {
  Percent(f64),
  Flat(i32)
  Promo(PromoDiscount),
  Custom(String),
}
```

51. Demo: Advanced match
```rs
enum Discount {
  Percent(i32),
  Flat(i32),
}
struct Ticket{
  event: String,
  price: i32,
  location: String,
}
fn main() {
  let n=4;
  match n {
    3=> println!("three"),
    other=> println!("number: {:?}",other), // instead of '_', can name it other cases. 'etc' works as well
  }
  let flat = Discount::Flat(2);
  match flat {
    Discount::Flat(2) => println!("flat 2"),
    Discount::Flat(some_amount) => println!("flat discout of {:?}", some_amount),
    _ => (), // does nothing
  }
  let concert = Ticket {
    event: "concert".to_owned(),
    price: 50,
    location: "park".to_owned(),
  };
  match concert {
    Ticket {price:50, event, location} => println!("event@50 =  {:?}",event),
    Ticket {price, ..} => println!("price= {:?}",price),
    // two dot .. means other cases
  }
}
```

52. Coding Exercise: Advanced match

53. Working with data: option
- A type that may be one of two things
  - Some data of a specified type
  - Nothing
- Used when:
  - Unable to find something
  - Ran out of items in a list
  - Form field not filled out
```rs
enum Option<T> {
  Some(T),
  None
}
```
- Instead of `Option::Some()`, we can use `Some()` only


54. Demo: option
```rs
struct Survey {
  q1: Option<i32>,
  q2: Option<bool>,
  q3: Option<String>,
}
fn main() {
  let response = Survey {
    q1: Some(12),
    q2: Some(true),
    q3: Some("A".to_owned()),
  };
  match response.q1 {
    Some(ans) =>println!("q1:{:?}",ans),
    None => println!(""),
  }
}
```

55. Coding Exercise: option

56. Demo: Generating Documentation
- Using `///` as documentation comment
- Use `rustdoc`

57. Demo: Accessing Standard Library Documentation

58. Coding Exercise: Accessing Standard Library Documentation
- `rustup doc`: open rust documentation

59. Working With Data: Result
- A data type that contains one of two types of data:
  - Successful data
  - Error
```rs
enum Result<T,E> {
  Ok(T),
  Err(E)
}
```

60. Demo: Result
```rs
#[derive(Debug)]
enum MenuChoice{
  MainMenu,
  Start,
  Quit,
}
fn get_choice(input: &str) -> Result<MenuChoice, String> {
  match input {
     "mainmenu" => Ok(MenuChoice::MainMenu),
     "start"    => Ok(MenuChoice::Start),
     "quit"     => Ok(MenuChoice::Quit),
     _ => Err("Menu choice not found".to_owned()),
  }
}
fn main() {
  let choice = get_choice("mainmenu");
  println!("choice = {:?}", choice);
}
```

61. Coding Exercise: Result
- () type: unit
```rs
struct Customer {
  age: i32
}
fn try_purchase(customer: &Customer) -> Result <(),String> {
  if customer.age < 21 {
    Err("customer must be at least 21 years old".to_owned())
  } else {
    Ok(())
  }
}
fn main() {
  let ashley = Customer {age: 20};
  let purchased = try_purchase(&ashley);
  println!("{:?}", purchased);
}
```

62. Coding Exercise: Result & Question Operator
- `?` operator: 
  - When the function returns `Result`
  - When Err is returned from the given funtion, run return immediately without doing further work
```rs
fn try_access(x: i32) -> Result<(), String> {
   if x >= 0 {
     return Ok(());
   } else {
     return Err("Negative number found".to_owned());
   }
}
fn print_access(x: i32) -> Result<(),String> {
   let tr = try_access(x)?;
   println!("status = {:?}",tr); // this is not executed if x is negative and Err(...) is returned
   return Ok(());
}
fn main(){
  let x = -3;
  let y = print_access(x);
  println!("final result = {:?}", y);
}
```

63. Data Structures: Hashmaps
- Collection that stores data as key-value pairs
- Very fast to retrieve data using the key
- Random order

64. Demo: Hashmap Basics
```rs
use std::collections::HashMap;
#[derive(Debug)]
struct Contents {
  content: String,
}
fn main() {
  let mut lockers = HashMap::new();
  lockers.insert(1, Contents {content: "shirts".to_owned(),});
  lockers.insert(2, Contents {content: "pants".to_owned(),});
  for (locker_number, content) in lockers.iter() {
    println!("number: {:?}, content: {:?}", locker_number, content);
  }
}
```

65. Coding Exercise: Hashmap Basics
```rs
use std::collections::HashMap;
fn main() {
  let mut stock = HashMap::new();
  stock.insert("Chair",5);
  stock.insert("Bed",0);
  stock.insert("Table",2);
  let mut total_stock = 0;
  for (item, qty) in stock.iter() {
    total_stock = total_stock + qty;
    // let stock_count = if qty == 0 { // this yields : no implementation for `&{integer} == {integer}`
    let stock_count = if qty == &0 { // note that we use reference of 0
      "out of stock".to_owned()
    } else {
      format!("{:?}",qty)
    };
    println!("item={:?}, stock={:?}", item, stock_count);
  }
  println!("total stock = {:?}", total_stock);
}
```
- .iter() yields borrowed memory

## Section 8: Real World

66. Demo: User input
- `io::Result<>` has its own error type. No need for error type
```rs
use std::io;
fn get_input() -> io::Result<String> {
  let mut buffer = String::new();
  io::stdin().read_line(&mut buffer)?; // borrows buffer muttably
  Ok(buffer.trim().to_owned())
}
fn main() {
  let mut all_input = vec![];
  let mut times_input = 0;
  while times_input < 2 {
    match get_input() {
      Ok(words) => {
        all_input.push(words);
        times_input += 1;
      }
      Err(e) => println!("error: {:?}", e),
    }
  }
  for input in all_input {
    println!("Original: {:?}, captialized: {:?}",
              input,
              input.to_uppercase()
    );
  }
} 
```

67. Coding Exercise: User input

68. Project 1: Menu-driven billing application
- 
69. Demo: Basic Closures
- Closure: anonymous function
```rs
fn add_fn(a:i32, b:i32) -> i32 { a + b }
fn main() {
  let sum_from_fn = add_fn(1,1);
  // closure
  //
  let add = | a: i32, b: i32| -> i32 { a +b };
  // Simpler form
  // let add = |a,b| a + b;
  let sum = add(1,1);
}
```

70. Demo: Map combinator

71. Coding Exercise: Map Combinator
```rs
#[derive(Debug)]
struct User {
    user_id: i32,
    name: String,
}
/// Locates a user id based on the name.
fn find_user(name: &str) -> Option<i32> {
    let name = name.to_lowercase();
    match name.as_str() {
        "sam" => Some(1),
        "matt" => Some(5),
        "katie" => Some(9),
        _ => None,
    }
}
fn main() {
    let user_name = "amy";
    let user = find_user(user_name).map(|user_id| User {
        user_id,
        name: user_name.to_owned(),
    });
    match user {
        Some(user) => println!("{:?}", user),
        None => println!("user not found"),
    }
}
```

72. Demo: Modules
```rs
mod greet {
  fn hello() { println!("hello"); } 
  fn goodbye() { println!("goodbye"); }
}
fn main() {
  use greet::hello;
  hello();
  greet::goodbye();
}
```

73. Demo: Testing
```rs
fn all_caps(word: &str) -> String {
   word.to_uppercase() 
}
fn main() {}
#[cfg(test)]
mod test {
  use crate::*; // scope up to the main src
  #[test]
  fn check_all_caps() {
    let result = all_caps("hello");
    let expected = String::from("HELLO");
    assert_eq!(result, expected, "string must be uppercase");
  }
}
```
- Command:
```bash
$ rustc --test ./a73.rs
$ ./a73 
running 1 test
test test::check_all_caps ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

74. Coding Exercise: Testing

## Section 9: Refining Your Code

75. Demo: Option Combinators
```rs
fn main() {
  let a: Option<i32> = Some(1);
  dbg!(a);
  let a_is_some = a.is_some(); // having Some() or not
  dbg!(a_is_some);
  let a_is_none = a.is_none(); // having None or not
  dbg!(a_is_none);
  let a_mapped = a.map(|num| num + 1); // map works only when Some() exists
  dbg!(a_mapped);
  let a_filtered = a.filter(|num| num == &1); 
  dbg!(a_filtered);
  let a_or_else = a.or_else(|| Some(4));
  dbg!(a_or_else);
  let unwrapped = a.unwrap_or_else(|| 0);
  dbg!(unwrapped);
}
```

76. Coding Exercise: Option Combinators

77. Demo: Iterators
```rs
fn main() {
   let numbers = vec! [1,2,3,4,5];
   /*
   let mut plus_one = vec![];
   for num in numbers {
      plus_one.push(num+1);
   }
   */
   let plus_one: Vec<_> = numbers.iter().map(|num| num+1).collect();
  let even_num: Vec<_> = numbers.iter().filter(|num| *num%2 == 0).collect(); // note that dereferencing *num is used
}
```

78. Coding exercise: Iterators

79. Demo: Ranges
```rs
fn main() {
  let range1 = 1..=3; // [1,3]
  let range2 = 1..4;  // [1,4)
  for num in range2 {println!("{:?}",num)};
}
```

80. Fundamentals: Traits
- A way to specify that some functionality exists
- Standardize functionality across multiple different types

81. Demo: Traits
```rs
trait Noise {
  fn make_noise(&self);
}
struct Person;
struct Dog;
impl Noise for Person { // implements Noise trait for Person struct
  fn make_noise(&self) {
    println!("hello");
  }
}
impl Noise for Dog {
  fn make_noise(&self) {
    println!("woof");
  }
}
fn hello(noisy: impl Noise) { // argument is any object implement Noise trait
  noisy.make_noise();
}
fn main() {
  hello(Person {});
  hello(Dog {});
  let mydog = Dog { } ; // No field data for now
  hello(mydog);
}
```

82. Coding Exercise: Traits
```rs
trait Perimeter {
  fn find_perimeter(&self) -> i32; // return type must be shown
}
struct Square {
  side: i32,
}
struct Triangle {
  a: i32,
  b: i32, 
  c: i32,
}
impl Perimeter for Square {
  fn find_perimeter(&self) -> i32 { self.side*4 }
}
impl Perimeter for Triangle {
  fn find_perimeter(&self) -> i32 {self.a*self.b*self.c}
}
fn main() {
  let mySQ = Square { side: 3};
  let myTR = Triangle {a:1,b:2,c:3};
  println!("{:?}",mySQ.find_perimeter());
  println!("{:?}",myTR.find_perimeter());
}
```

83. Demo: if..let..else
- Alternative of match
- When only one case is considered
```rs
fn main() {
   let potential_user = Some("Jerry");
   // match case
   match potential_user {
       Some(user) => println!("user = {:?}", user),
       None       => println!("no user"),
   }
   // if let else (same as above)
   if let Some(user) = potential_user {
      println!("user = {:?}", user);
   } else {
      println!("no user");
  }
}
```

84. Demo : While...let
```rs
fn main() {
  let mut data = Some(3);
  while let Some(i) = data {
    println!("loop");
    data = None;
  }
}
```

85. Demo: External Crates
- External libraries (like npm)
- https://doc.rust-lang.org/cargo/guide/dependencies.html

86. Activity: External Crates
- cargo new a86
- cd a86
- Edit Cargo.toml
```yaml
[package]
name = "a86"
version = "0.1.0"
edition = "2021"
# See more keys and their definitions at https://doc.rust-lang.or
g/cargo/reference/manifest.html
[dependencies]
chrono = "0.4.15"
```
- src/main.rs
```rs
use chrono::prelude::*;
fn main() {
    let local: DateTime<Local> = Local::now();
    println!("{}",local.format("%Y-%m-%d %H:%M:%S").to_string());
}
```
- Command: cargo run

87. Demo: Default Trait 
- Enumeration with default value
```rs
struct Package {
  weight: f64,
}
impl Package {
  fn new(weight: f64) -> Self {
    Self {weight }
  }
}
impl Default for Package {
  fn default() -> Self {
    Self { weight: 3.0}
  }
}
fn main() {
  let p = Package::default();
  println!("{:?}",p.weight);
}
```

88. Demo: const
```rs
const MAX_SPEED: i32 = 9000;
fn clamp_speed ( speed: i32) -> i32 {
   if speed > MAX_SPEED {
      MAX_SPEED
   } else {
      speed
   }
}
fn main() {}
```

89. Demo: Modules as separate files
- cargo new a89
- 
- Cargo.toml
```yaml
[package]
name = "a89"
version = "0.1.0"
edition = "2021"
# See more keys and their definitions at https://doc.rust-lang.or
g/cargo/reference/manifest.html
[lib]
name = "demo"
path = "src/lib.rs"
[dependencies]
```
- src/main.rs
```rs
//use demo::*;
use demo::print_from_lib;
fn main() {
  print_from_lib();
}
```
- src/lib.rs
```rs
pub fn print_from_lib() {
  println!("hello from lib");
}
```

## Section 10: Final Project

90. Demo: Custom Error Types
- Using an external crate: thiserror
- Cargo.toml
```yaml
[package]
name = "a90"
version = "0.1.0"
edition = "2021"
# See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html
[dependencies]
thiserror = "1.0"
```

91. Coding Exercise: Custom Error Types
```rs
use thiserror::Error;
#[derive(Debug, Error)]
enum ProgramError {
    #[error("menu error")]
    Menu(#[from] MenuError),
    #[error("math error")]
    Math(#[from] MathError),
}
#[derive(Debug, Error)]
enum MenuError {
    #[error("menu item not found")]
    NotFound,
}
#[derive(Debug, Error)]
enum MathError {
    #[error("divide by zero error")]
    DivideByZero,
}
fn pick_menu(choice: &str) -> Result<i32, MenuError> {
    match choice {
        "1" => Ok(1),
        "2" => Ok(2),
        "3" => Ok(3),
        _ => Err(MenuError::NotFound),
    }
}
fn divide(a: i32, b: i32) -> Result<i32, MathError> {
    if b != 0 {
        Ok(a / b)
    } else {
        Err(MathError::DivideByZero)
    }
}
fn run(step: i32) -> Result<(), ProgramError> {
    if step == 1 {
        pick_menu("4")?;
    } else if step == 2 {
        divide(1, 0)?;
    }
    Ok(())
}
fn main() {
    println!("{:?}", run(1));
    println!("{:?}", run(2));
}
```

92. Demo: New type pattern
```rs
#[derive(Debug,Copy,Clone)]
struct NeverZero(i32);
impl NeverZero {
  pub fn new(i: i32) -> Result<Self, String> {
    if i==0 {
       Err ("cannot be zero".to_owned())
    } else {
       Ok(Self(i))
    }
  }
}
fn divide(a:i32, b: NeverZero) -> i32{
   let b = b.0; // tuple data
   println!("b={:?}",b);
   a/b
}
fn main() {
  match NeverZero::new(0) {
   Ok(nz) => println!("{:?}", divide(10,nz)),
   Err(e) => println!("{:?}", e),
  }
}
```

93. Coding Exercise: New type pattern
- Derived class and the dedicated method
- ShirtColor/ShoesColor are inherited from Color
- print_shirt_color() runs only on ShirtColor, not on ShoesColor
```rs
#[derive(Debug)]
enum Color {
    Black,
    Blue,
    Brown,
    Custom(String),
    Gray,
    Green,
    Purple,
    Red,
    White,
    Yellow,
}
#[derive(Debug)]
struct ShirtColor(Color);
impl ShirtColor {
    fn new(color: Color) -> Self {
        Self(color)
    }
}
#[derive(Debug)]
struct ShoesColor(Color);
impl ShoesColor {
    fn new(color: Color) -> Self {
        Self(color)
    }
}
#[derive(Debug)]
struct PantsColor(Color);
impl PantsColor {
    fn new(color: Color) -> Self {
        Self(color)
    }
}
fn print_shirt_color(color: ShirtColor) {
    println!("shirt color = {:?}", color);
}
fn print_shoes_color(color: ShoesColor) {
    println!("shoes color = {:?}", color);
}
fn print_pants_color(color: PantsColor) {
    println!("pants color = {:?}", color);
}
fn main() {
    let shirt_color = ShirtColor::new(Color::Gray);
    let pants_color = PantsColor::new(Color::Blue);
    let shoes_color = ShoesColor::new(Color::White);
    print_shirt_color(shirt_color);
    print_pants_color(pants_color);
    print_shoes_color(shoes_color);
}
```

94. Project2: Personal Contact Manager

95. Fundamentals: Advanced Memory
- Stack
  - Data placed sequentially
  - Limited space
  - Very fast to work with
- Heap
  - Data placed algorithmically
  - Slower than stack
  - Uses pointers
```rs
enum Answer {
  Yes,
  No,
}
fn main() {
  let yes = Answer::Yes;
  let yes_heap: Box<Answer> = Box::new(yes);
  let yes_stack = *yes_heap;
}
```
- struct std::boxed::Box
  - https://doc.rust-lang.org/std/boxed/struct.Box.html

96. Demo: Passing Closures to Functions
- Using `Fn(...)->...`
```rs
fn math(a: i32, b:i32, op: Box<dyn Fn(i32,i32)->i32>) -> i32 {
  op(a,b)
}
fn main() {
  let add = |a,b| a+b;
  let add = Box::new(add);
  println!("{}", math(2,2,add));
  let mul = Box::new(|a,b| a*b);
  println!("{}", math(2,2,mul));
}
```

97. Demo: Lifetimes
- Lifetime annotation: `'a`, `'b`, ...
  - https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html
