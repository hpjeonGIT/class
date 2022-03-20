## Go: The Complete Developer's Guide (Golang)
- Instructor: Stephen Grider

7. Hello world
```
package main
import "fmt"
func main(){
	fmt.Println("Hi there!")
}
```

8. Go command
- go build: build only
- go run: build and run
- go fmt: 
- go install/get: download packages
- go test: like ctest

9. Go packages
- `package` is used to show which project each source code belongs
    - For multiple files which have `package main`: 
        - `go run main.go state.go`
- Types of package
    - Executable
    - Reusable: library or helper code
- `package main` is an assigned name for an executable
- `package myworld` might be used for reusable or libraries

15. Variable declarations
- Dynamic types language: JS, Ruby, Python
- Static types language: C++, java, Go
    - `var card string = "Ace of Spades"`
    - variable type might be skipped as infering is done
        - `var card string = "Ace of Spades"`
	    - `card := "Ace of Spaces"`
        - `:=` for a new variable only
- Basic Go types
    - bool: true/false
    - string
    - int
    - float64
- Variable initialization on globe is allowed
```
var card string
func main() {...}
```
    - Global variable assignment is not allowed
```
var card string = "hello" # => this is not allowed
func main() {...}
```

16. Function and return type
- `func newCard() string { return "hello" }`

17. Slices and for loops
- array: fixed length list
    - `cards := [10] int`
- slice: array that can grow or shrink
    - `cards := [] string { "hello", "word"}`
    - `cards = append(cards, "wonderful")`
- Sample for loop
```
package main
import "fmt"
func main() {
	cards := [] string {"hello", "word"}
	cards = append(cards, "awesome")
	for i, val := range cards {
		fmt.Println(i,val)
	}
}
```
- if `i` is not used within loop, build will produce an error message


18. OO vs Go approach
- Go is NOT object-oriented

19. Custom type
- `type deck [] string`

20. Receiver functions
- Methods of a new type
```
func (d_this deck) print() {
    for i, val := range d_this {
        fmt.Println(i,val)
    }
}
```
- `d_this` is arbitrary and corresponds to `this` or `self` in C/Python

21. Creating a new deck
- Use `_` in the loop when index or variable is not used
```
package main
type deck [] string
func newDeck() deck {
	cards := deck{}
	cardSuits := deck{"Spaces", "Diamonds"}
	cardValues := deck{"Ace", "Two"}
	for _,s:=range cardSuits{
		for _,v:=range cardValues{
			cards = append(cards, v+" of " + s)
		}
	}
	return cards
}
func main() {
	cards := newDeck()
	cards.print()
}
```

22. Slice Range Syntx
- cards[0:2] = cards[:2] = cards[0], cards[1]
    - card[2] is NOT included
- cards[:] = all of cards
- cards[2:] = cards[2], cards[3], ...
- cards[len(cards)-1]): the last element

23. Multiple Return Value
- Function argument: `func deal(d deck, handSize int)`
    - argument name + argument type within (...)
```
func deal(d deck, handSize int) (deck,deck) {
    return d[:handSize], d[handSize:]
}
...
tmp1, tmp2 := deal(cards,3)
```
- As tmp1, tmp2 are defined (:=), not need declaration beforehand

24. Byte slices
- In order to use WriteFile(), string needs to be converted into bytes

25. Deck to string
- Type conversion
    - `[]byte("Hello World")`

26. Joining a slice of Strings
- Loading multiple package : needs new line
```
import ("fmt"  
       "strings")
```
- Or `import ("fmt" ;  "strings")`
- Aggregating strings: `strings.Join(d, ",")`

27. Saving data into a text file
- `ioutil.WriteFile("myfile.txt", []byte(sumstring), 0666)`
- file mode: https://schadokar.dev/to-the-point/how-to-read-and-write-a-file-in-golang/
```
0000     no permissions
0700     read, write, & execute only for owner
0770     read, write, & execute for owner and group
0777     read, write, & execute for owner, group and others
0111     execute
0222     write
0333     write & execute
0444     read
0555     read & execute
0666     read & write
0740     owner can read, write, & execute; group can only read; others have no permissions 
```

28. Reading from a text file
```
bytes, err := ioutil.ReadFile(filename)
if err != nil { 
    //throw an error 
    os.Exit(1)
} else {
    ...
}
```

30. Shuffling a deck
- `rand.Intn(N)`: yields 0....N-1

31. Random number generation
- Random seed using the current time
```
src := rand.NewSource(time.Now().UnixNano())
r := rand.New(src)
...
fmt.Println(r.Intn(5))
```

32. Testing with go
- Make a file `*_test.go`
```
import ("testing")
func Test1(t *testing.T) {
    ...
    // At failed condition
    t.Errorf("as expected")
    ...
}
```
- `go test`
- If a following message is found:
```
go: cannot find main module, but found .git/config in /home/hpjeon/hw/class
	to create a module there, run:
	cd ../../.. && go mod init
```
    - Run: `go env -w GO111MODULE=auto`

39. Declaring structs
```
package main
import "fmt"
type person struct {
	firstName string
	lastName string
}
func main(){
	//alex := person{"Alex", "Anderson"} // this works too
	alex := person{firstName:"Alex", lastName: "Anderson"}
	fmt.Println(alex)
}
```

40. Updating struct values
- Zero value of each type
    - string: ""
    - int: 0
    - float: 0
    - bool: false
```
package main
import "fmt"
type person struct {
	firstName string
	lastName string
}
func main(){
	var alex person
	alex.firstName = "Alex"
	alex.lastName = "Anderson"
	fmt.Println(alex)
	fmt.Printf("%+v\n",alex)
}
```

41. Embedding struct
- When struct is assigned, add **comma** in the last element as well
```
package main
import "fmt"
type contactInfo struct {
	email string
	zipCode int
}
type person struct {
	firstName string
	lastName string
	contact contactInfo
}
func main(){
	jim := person {firstName: "Jim", lastName: "Party",
					contact: contactInfo {email: "jim@g.com", 
										zipCode: 12345},}
	fmt.Println(jim)
	fmt.Printf("%+v\n",jim)
}
```

42. Struct with receiver function
- `type person struct {	firstName string; 	lastName string; contactInfo }` is as same as `type person struct {	firstName string; 	lastName string; contactInfo contactInfo} }`
    - When nesting a struct, the struct name can become the variable name

43. Pass by value
```
func (p person) updateName(newName string) {
	p.firstName = newName 
}
...
jim.updateName("jimmy")
```
- This does not update jim's firstname as "jimmy" - pass by value

44. Structs with pointers
```
	jimP := &jim
	jimP.updateName("jimmy")
	jim.print()
}
func (p *person) updateName(newName string) {
	(*p).firstName = newName
}
```
- Now the firstName is updated
- `(p *person)`: this `*` is the description
- `(*p).firstName`: this `*` is the operation

46. Pointer shortcut
```
	jim.updateName("jimmy")
	jim.print()
}
func (p *person) updateName(newName string) {
	(*p).firstName = newName
}
```
- Still works OK, replacing with "jimmy"

47. Gotchas with Pointers
- Slice is pass by reference as default

48. Reference vs value types
- Value types: needs pointers to change them in a function
    - int, float, string, bool, structs
    - Again, **struct** is value type!!!
- Reference types
    - slices, maps, channels, pointers, functions

49. What is a map?
- Dict() in Python
- But keys must be the same types
- values must be the same types
```
package main
import "fmt"
func main() {
	colors := map[string]string {
		"red":"#ff0000","green":"#5bf745", 
	}
	fmt.Println(colors)
}
```

50. Manipulating map
- `var colors map[string]string`: making a nil map. Elements cannot be added
- `colors := make(map[string]string)`: making an empty map.
```
package main
import "fmt"
func main() {
	colors := make(map[string]string)
	colors["white"] = "#fffff"
	fmt.Println(colors)
	delete(colors,"white")
	fmt.Println(colors)
}	
```

53. Purposes of interfaces
- Similar logic/behavior but differen inputs
    - Not exactly same

54. Without interface
```
package main
import "fmt"
type englishBot struct{}
type spanishBot struct{}

func main() {
	eb := englishBot{}
	//sb := spanishBot{}
	printGreeting(eb)
	//printGreeting(sb)
}

func printGreeting(eb englishBot) {
	fmt.Println(eb.getGreeting())
}
/*
func printGreeting(sb englishBot) {
	fmt.Println(sb.getGreeting())
}
*/
func(eb englishBot) getGreeting() string{
	return "Hello"
}
func(sb spanishBot) getGreeting() string{
	return "Hola"
}
```
- Cannot execute same functions with different argument
    - Golang doesn't support overloading
    - Producing different results cannot be handled by overloading

55. Interfaces in practice
```
package main
import "fmt"
type bot interface{
	getGreeting() string
}
type englishBot struct{}
type spanishBot struct{}
type intBot int
func main() {
	eb := englishBot{}
	sb := spanishBot{}
	var ab intBot
	printGreeting(ab)
	printGreeting(eb)
	printGreeting(sb)
}
func printGreeting(b bot) {
	fmt.Println(b.getGreeting())
}
func(i intBot) getGreeting() string{
	return "integer"
}
func(eb englishBot) getGreeting() string{
	return "Hello"
}
func(sb spanishBot) getGreeting() string{
	return "Hola"
}
```
- Command:
```
$ go run bot_interface.go 
integer
Hello
Hola
```
- Note that method getGreeting() may have different data types, struct, int, ...

56. Rules of interfaces
```
type bot interface{
    getGreeting(string, int) (string, error)    
}
```
- Can have multiple arguments/mutiple return values

57. Extra interface notes

58. HTTP package
```
package main
import (
	"net/http"
	"os"
	"fmt"
)
func main() {
	resp, err := http.Get("http://google.com")
	if err != nil {
		fmt.Println("Error:",err)
		os.Exit(1)
	}
	fmt.Println(resp)
}
```
- Command: 
```
$ go run main.go
&{200 OK 200 HTTP/1.1 1 1 map[Cache-Control:[private, 
...
```
- Body is delivered as io.ReadCloser, which is interface type. See below

59. Reading the Docs
- Response -> Body -> io.ReadCloser -> Reader/Closer -> io.Reader/Closer interface

60. More interface syntax

61. Interface review
- Interface condition is implicit
- Different data type must have the same name of interface function, with same type inputs/returns

62. Reader interface
- Difference source of input: http, text on disk, image on disk, text from command line, data from sensor, ...
- Different return type: string, jpeg, byte, float, ...
- May need different functions to print
- Can be generalized using **interface**
    - all sources -> Reader -> [] byte

64. Working with the Read function
```
package main
import (
	"net/http"
	"os"
	"fmt"
)
func main() {
	resp, err := http.Get("http://google.com")
	if err != nil {
		fmt.Println("Error:",err)
		os.Exit(1)
	}
	bs := make([]byte, 99999) // makes a big slice
	resp.Body.Read(bs)
	fmt.Println(string(bs))
}
```

65. The Writer interface
- `io.Copy(os.Stdout, resp.body)` will dump resp.body into stdout
- [] byte -> Writer -> outgoing http request/text file on disk/image file on disk/terminal/...

66. io.Copy function
- 

Assignment 2
- Define an interface getArea() for square and triangle struct
```
package main
import "fmt"
type shape interface{
	getArea() float64
}
type triangle struct {
	height float64
	base   float64
}
type square struct {
	sideLength float64
}
func main() {
	mytriangle := triangle{10,1}	
	mysquare := square{5}
	printArea(mytriangle)
	printArea(mysquare)
}
func printArea(s shape) {
	fmt.Println(s.getArea())
}
func (t triangle) getArea() (float64) {
	return t.height*t.base*0.5
}
func (s square) getArea() (float64) {
	return s.sideLength*s.sideLength
}
```

Assignment 3
- Read a text file and dump the content to the screen
```
package main
import (
	"fmt"
	"os"
	"io"
)
func main() {
	args := os.Args
	fname := args[1]
	content, err := os.Open(fname)
	if err != nil {
		fmt.Println("File not read:", fname)
	}
	io.Copy(os.Stdout, content)
}
```

69. Website status checker
- Channels/Go routine: for concurrent/parallel operation
- we write a code checking multiple web-sites
```
package main
import (
	"fmt"
	"net/http"
)
func main() {
	links := [] string{
		"http://google.com",
		"http://facebook.com",
		"http://stackoverflow.com",
		"http://amazon.com",
	}
	for _,link := range links {
		checkLink(link)
	}
}
func checkLink(link string) {
	_, err := http.Get(link)
	if err != nil {
		fmt.Println(link, "cannot connect!")
		return
	}
	fmt.Println(link, "is up!")
}
```

71. Serial link checking
- There is a delay b/w links above

72. Go routines
- `go` + func() inside of for-loop

73. Theory of Go routines
- One CPU core <-> Go Scheduler <-> each Go routine by default
- Using Multiple CPUs will be parallelism
- **Concurrency is NOT parallelism**

74. Channels
- When the main routine generates child routines, it doens't couple with child routines
- Channels communicate b/w main and child routines
- main routine -> channel <- child routines
- `channel <- v` : send the value of a variable v into channel
- `v <- channel` : send value from channel to a variable v
- `fmt.Println(<-c)` : prints the channel value

76. Blocking channels
- If the number of threads is not matched in channel handling, it may miss or hang

78. Repeating routines
- Infinite loop: `for { ...}`

80. Sleeping a routine
- `time.Sleep(5 * time.Second)`: works in the corresponding go routine
    - Instead of the main routine, add this to the each checkLink() function

81. Function literals
- Lambda in C++/Python or anonymous function in JS
