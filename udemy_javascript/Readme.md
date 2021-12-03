# Summary
- Title: JavaScript - The Complete Guide 2022 (Beginner + Advanced)
- Instructor: Maximilian Schwarzmüller

6. Dynamic and weakly typed language
- Dynamic interpreted programming language
    - Not precompiled but compiled on the fly
    - code can change at run time, including type of a variable
- Weakly typed programming language
    - Data types are assumed automatically

7. Javascrit runs on a host environment
- Browser-side
    - works with http requests
    - Doesn't access local filesystem & local OS
- Server-side
    - Standalone tool (Node.js)

19. Variables and constants
- Use constants as often as possible

20. Variable naming
- Use camelCase
- snake_case (user_name) is not recommended

23. Data type
- string + number will be string: str(number) is done automatically

25. String
```
let Description = `hello world with ${currentResult}`;
```
- This will print hello word with the value of currentResult in the code
```
let Description = 'hello world with ${currentResult}';
```
- Single or double quotation cannot translate `${}`

Assignment 1.
- Create 2 variables (a, b). One that holds a fictional user input (a=1.234) and the other without pre-assigned value
- b = a + 18
- b = b*a
- b = b/a
- alert(a)
- alert(b)

30. Code order
- The location of a function definition can be located after calling the function

35. A function as a function argument
- For a funciton `add()`, use as `add` inside of a function parentheses
```
function add() {
    currentResult = currentResult + userInput.value;
}
addBtn.addEventListener('click', add);
outputResult(currentResult,`hello with ${currentResult}`);
```
- This will not show results in the web-browser as outputResult() function is executed when the code is executed
```
function add() {
    currentResult = currentResult + userInput.value;
    outputResult(currentResult,`hello with ${currentResult}`);
}
addBtn.addEventListener('click', add);
```
- Now the result is shown when the add button is clicked

Assignment 2.
- Create 2 new functions. 1) no parameter simply alert() 2) passes a name and alert()
- Call both functions from the code
- Use task3Element and use event listener
- 3) a new function with 3 parameters, returns a concatenated string
- Call the function from the code and alert() the results
```
function assign2_1(){
    alert('assignment 2-1');
}
function assign2_2(inputs){
    alert('print: ' + inputs);
}
function assign2_3(param1, param2, param3) {
    alert('concatenated string:' + param1 + param2 + param3);
}
assign2_1();
assign2_2('hello world');
task3Element.addEventListener('click', assign2_1);
assign2_3('hello world', " in the ", " weekend");
```

36. Converting data type
- parseInt() or parseFloat()


39. functions
```
function createAndWriteOutput(operator, resultBefore, calcNumber) {
    const calcDescription = `${resultBefore} ${operator} ${calcNumber}`
    outputResult(currentResult, calcDescription);
}
function multiply() {
    const enteredNumber = getUserNumberInput();
    const initialResult = currentResult;
    currentResult = currentResult * enteredNumber;
    createAndWriteOutput('*', initialResult, enteredNumber)
}
```
- Passing `*` or `+` into function arguments

40. Comments
- Use `//` or a block of `/* ... */`

42. Data types
- Numbers: 2, -1.234
- Strings: 'HI', "Hi", `Hi`
- Booleans: true/false
- Objects: JSON
- Arrays: [1, 2, 3]

43. Arrays
```
let logEntries = [];
...
    logEntries.push(enteredNumber);
    console.log(logEntries);
```
- `console.log()` can be viewed from `Inspect element->Console`
- Each element in arrays can be accessed like `logEntries[0]`
    - If the element value doesn't exist, `undefined` is printed

48. Undefined, null, and NaN
- Undefined: default value of uninitialized variables
    - No manual assignment as Undefined
    - It works but bad-practice
- null: shouldn't be default value
    - Can be used for reset or clear of a variable
- NaN: still stays at memory and can be checked
    - `typeof NaN` yields `number`

50. defer and sync
- `Inspect element->Performance->Record` then load a page. Then click stop record
![Snapshot of performance profiling](./snapshot_performance.png)
```
    <script src="assets/scripts/app.js" defer></script>
    <script src="assets/scripts/hw.js" defer></script>
```
- Using defer, loading those js files (locally or remotely) is done in parallel with loading the html file
```
    <script src="assets/scripts/app.js" async></script>
    <script src="assets/scripts/hw.js" async></script>
```
- Using async will load/execute those files independently, and may cause race conditions

62. MDN
- https://developer.mozilla.org/en-US/docs/Web/JavaScript

65. Debugging javascript
- Read error messages
- Use console.log()
- Use Chrome debugging tools
- Use IDE debugging tools

68. Debugging with Chrome
- Inspect element -> Sources, find source file and click line numbers to add break points
![Snapshot of debugging in chrome](./debugging_chrome.png)
- Check Call Stack and local variables

75. Comparisons
- `==` checks value only while `===` checks data type. `===` is more favored for strict check
```
> 2 == 2
> true
> 2 === 2
> true
> 2 == "2" // NOTE !!!
> true
> 2 === "2" // NOTE !!!
> ​false
> 2 != "2"
> false
> 2 !== "2"
> true￼
```
￼
​78. Comparing Objects and Arrays
- Equality of objects or array may not work
```
> xyz = {name: 'Max'}
> abc = {name: 'Max'}
> abc === xyz // NOTE !!!
> false
> abc.name === xyz.name
> true
> arr = [1,2,3]
> brr = [1,2,3]
> arr == brr // NOTE !!!
> false
> crr = arr 
> arr === crr // But if a variable is copied from existing array/object, equality works
> true
```
￼
81. Truthy vs Falsy
- Truthy: when a value can be converted to true
- Falsy: when a value is a candidate of false
    - '', null, undefined, NaN, 0, 0.0, 0.
- ​Non-empty string is converted to true : `if (txt) {}`
    - '' is false
- 0 is converted to false
    - 0.0 or 0. as well
- Empty array ([]) or object ({}) is true
- null, undefined, NaN are false

90. Validating user input
- Use prompt() to read a user input
![Prompt in Chrome](./prompt_chrome.png)
```
const enteredValue = prompt('Maximum life for you and the monster','100');
chosenMaxLife = parseInt(enteredValue);
if (isNaN(chosenMaxLife) || chosenMaxLife <= 0) {
    chosenMaxLife = 100;
}
```

93. Ternary operator
- `const userName = isLogin ? 'Max':null;`
- May need parenthesis when another conditional check is made
```
> 'Max' === isLogin ? 'Max':null
> null
> 'Max' === (isLogin ? 'Max':null);
> true
```
- The lecture shows that ternary operator works OK with === but Opera runs === prior to ternary operator

95. Boolean tricks
- Double bang: `!!`
    - negate 2x: `!!1`=> true, `!!''`=> false
    - Can convert truthy/falsy value to true/false
        - `boolVal = !!mytxt` => true when non-empty string. When empty, false
- Assigning a default value using OR
    - ` const name = usrInput || 'DefaultName'`
    - If usrInput is an empty string or undefined, it is falsy and `DefaultName` will be assigned
- OR operation: 
    - Ref: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Logical_OR
    - `expr1 || expr2`
    - If expr1 can be converted to true, returns expr1; else, returns expr2. 
```
> isLogin = true
> isLogin || 'Max' // NOTE - true is resulted
> true
> 'Max' || isLogin // NOTE- 'Max' is resulted
> 'Max'
> wasLogin = false
> wasLogin || 'Max'
> 'Max'
> 'Max' || wasLogin
> 'Max'
```
- AND operation:
    - `expr1 && expr2`
    - If expr1 can be converted to true, returns expr2; else, returns expr1. 
```
> isLogin && 'Max'  // NOTE - 'Max' is resulted
> 'Max'
> 'Max' && isLogin  // NOTE - true is resulted
> true
> wasLogin && 'Max'
> false
> 'Max' && wasLogin
> false
> '' && 'Max' // NOTE - '' is resulted. Not true or false.
> ''
> 'Max' && '' // NOTE - '' is resulted. Not true or false.
''
```

97. switch-case
- It uses === for comparison (type of data is checked)
- break is necessary - unless, below cases will be executed

98. For loops
- for loop: `for (let i=0;i<3;i++) {}`
- for-of loop: `for(const el of array) {}`
    - Make sure to use `of`, not `in`
    - `in` will return the index, not value
```
> for (const el of ['a','b','c']) {console.log(el)}
a
b
c
> for (const el in ['a','b','c']) {console.log(el)} // Note that it returns index, not value of 'a','b','c'
0
1
2
```
- for-in loop: `for(const key in obj) {}`
- while loop: `while(isTrue) {}`

104. Continue
- break: exits the loop
- continue: returns to the next iterator

105. Labeled statement with break
- Using the label of the loop, break can control which loop it may exit
```
> for (i of ['a','b','c']) {
    console.log('Outer',i);
    for (j in [1, 2, 3]) {
        if (j==2) {
            break;
        }
        console.log('Inner',j);
    }
}
Outer a
Inner 0
Inner 1
Outer b
Inner 0
Inner 1
Outer c
Inner 0
Inner 1
> OuterLoop: for (i of ['a','b','c']) {
    console.log('Outer',i);
    InnerLoop: for (j in [1, 2, 3]) {
        if (j==2) {
            break OuterLoop;
        }
        console.log('Inner',j);
    }
}
Outer a
Inner 0
Inner 1
```

107. Throwing an error message
```
> throw { message: 'help!'}
Uncaught 
{message: 'help!'}
```

108. try-catch
- For the errors beyond the control by the developer
    - user input typo, network outage, ...
- try {}: the code which may throw an error
- catch {}: error handling and fallback logic
- finally: optional but can be used to cleanup work in both cases of success or fail
```
> function myftn() { throw { message: 'crashed!'}}
> try {
    myftn();
} catch (error) {
    console.log(error);
    throw error;
} finally {
    console.log('testing try-catch-finally');
}
> {message: 'crashed!'}
> testing try-catch-finally
```

112. ES5 vs ES6+
- ES: ECMA Script
- ES5: only var. No support of let and const

113. var vs let vs const
- var: creates a variable over function & global scope
    - Don't use var in ES6+ as a good practice
- let: creates a variable over a block scope
- const: creates a constant over a block scope

114. Hoisting
- Similar to functions, the location of var variables may not matter as JS will read the entire script and loads var variables into memory

115. Strict mode and writing a good code
- `'use strict';` or `"use strict";`
    - Applies strict rules of JS
    - Only in the single JS file
- Do not initialize a new variable without let. It may confuse other folks to find the location of the initial declaration of the variable

117. Inside the Javascript engine
- Long term memory like function defintion on heap
- Function calls, short-lived data, and communication on stack
    - Found from Debugging -> Call Stack

119. Primitive vs Reference values
- Primitive
    - strings, numbers, booleans, null, undefined, symbol
    - Copies by values for copy operation
    - Stored on stack
    - Array as well? The lecture shows an array is reference but Opera shows it behaves as primitive
- Reference
    - all other objects which are expensive to create
    - Stored on heap
    - Copies the address of the variable in the memory for copy operation
    - In order to have copying value, use `...`
    - This is why `===` of two same objects doesn't work as it compares the address
        - `const array` or `const object` implies the constant address
        - push() still works as the address is constant
        - new assignment will fail as the address will change
```
> let a1 = { age: 30};
> let b1 = a1;
> let c1 = {...a1} // now copy values, not address
> a1.height = 5.09
5.09
> b1
{age: 30, height: 5.09} // copy by reference. Updated automatically
> c1
{age: 30} // copy by value. No change
> const aobj = {age:30};
> aobj.height = 6.01  // push or adding new key works OK for const obj
> aobj = {age:30, height:6.01} // new assignment fails as it changes the address
VM1899:1 Uncaught TypeError: Assignment to constant variable.
    at <anonymous>:1:6
```

120. Garbage collection
- Management of heap memory
- Checks periodically for unused memory (no reference)
