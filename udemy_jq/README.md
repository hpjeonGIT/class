## Beginners Course: Jq Command Tutorials to Parse JSON Data
- Instructor: VR Technologies

## Section 1: Introduction

1. Introduction to Jq command
- Jq is a command line tool to parse JSON data on unix/Linux CLI
Section 2: Environment setup

## Section 2: Environment Setup

2. Installing Jq Command on Unix/Linux OS

3. Installing Any Linux Distribution and CentOS on Windows with WSL
Section3: Jq Command Syntax, Filters & Options

## Section 3: Jq Command syntax, filters, and options

4. Jq Command syntax
- jq [options] 'LogicToParse' [input] > output
- jq --help

5. Identity & Field Filters
- Identity filter: validate the json data
    - input and output are identical
    - Also applies reformat
    - jq . sample.json
    - jq '.' sample.json
    - jq "." sample.json
- sample.json:
```json
{
  "gender":"male",
  "name": {
    "title":"Mr",     "first": "Arlo",
    "last": "King"
   }
}
```
- Field fielter: handles key/property values
    - Chaining of keys: jq '.k1.k2.k3' fileName
    - Filters are also called as operators
```bash
$ jq '.gender' ./sample.json
"male"
$ jq '.name' ./sample.json
{
  "title": "Mr",
  "first": "Arlo",
  "last": "King"
}
$ jq '.name.first' ./sample.json
"Arlo"
```

6. Jq Command with Raw String Options |  How to read a non JSON data with Jq?
- `-R`: To read raw string
= `-r`: Removes quotations from output
```bash
$ more test.txt
docker
postman
k8s
$ jq -R '.' ./test.txt
"docker"
"postman"
"k8s"
$ jq '.name.first' ./sample.json
"Arlo"
$ jq -r  '.name.first' ./sample.json
Arlo
```

7. Jq Command with Field Filters and Comma
- For multiple keywords
```
$ jq -r '.gender, .name.first' ./sample.json
male
Arlo
```

8. How to Sort Keys in alphabetical order of an Object using Jq command?
- Using `-S`
```bash
]$ jq -S '.' ./sample.json
{
  "gender": "male",
  "name": {
    "first": "Arlo",
    "last": "King",
    "title": "Mr"
  }
```

9. Combining  Multiple JSON Files using -s option of Jq Command
- jq -s file1 file2 ...
- Merges the results
```bash
$ cat a.json
[
1,
3
]
 
$ cat b.json
[
2,
4
]$ jq '.' a.json b.json
[
  1,
  3
]
[
  2,
  4
]
$ jq -s '.' a.json b.json
[
  [
    1,
    3
  ],
  [
    2,
    4
  ]
]
```

## Section 4: Different ways to Pass JSON Data as an Input to jq command

10. How to pass JSON File, String, Linux Command Output as an input Jq command?
- `jq '.' <<< "$(command_here)"` # note triple <<<
- `command_here | jq '.'`
```
$ jq '.' <<< "$(echo '[1,2,3]')"
[
  1,
  2,
  3
]
$ echo '[1,2,3]' | jq '.'
[
  1,
  2,
  3
]
$ jq '.' <<< '{"first": "hello"}'
{
  "first": "hello"
}
```

11. Parsing API Response with Jq command
- For any REST API: `curl https::/xxx.xxx/api/`
- Ex: `jq '.info' <<< "$(curl -s https://randomuser.me/api/)"
 
12. Parsing Cloud CLI command output and kubectl command output with jq
- AWS: `aws iam list-users --output json`
- Kubectl: `kubectl config view --output json`

## Section 5: Working with Arrays

13. Iterator Filter With Index of Array Items/Values/Elements
- Array must be surrounded by bracket []
- "key":"filter" cannot be the element of an array
-  There is an index number or position for each value in arguments
```bash
$ cat test.json
[
"docker",
"postman",
"k8s"
]
$ jq '.[]' test.json
"docker"
"postman"
"k8s"
$ jq '.[0]' test.json
"docker"
$ jq '.[1]' test.json
"postman"
$ jq '.[2]' test.json
"k8s"
$ jq '.[-1]' test.json
"k8s"
$ jq '.[-2]' test.json
"postman"
$ cat tools.json
{
"techonogies": "devops",
"tools": ["jenkins", "bamboo",
  "ansible", "docker"
]
}
$ jq '.tools[1]' ./tools.json
"bamboo"
$ jq '.tools[1:4]' ./tools.json
[
  "bamboo",
  "ansible",
  "docker"
]
```
 
14. Iterator or Array Filter to take and print one by one value from a JSON Array

## Section 6: Constructing null, number, string, Boolean, arrays, and objects using jq

15. Constructing basic JSON Data's using Jq Command without any input JSON Data
- `-n` : no input. Null
```bash
]$ jq -n '.'
null
$ jq -n "HELLO world"
jq: error: syntax error, unexpected IDENT, expecting $end (Unix shell quoting issues?) at <top-level>, line 1:
HELLO world
jq: 1 compile error
$ jq -n '"HELLO world"' # string must be surrounded with "" and inside of quotation
"HELLO world"
$ jq -n 'true'
true
$ jq -n 'false'
false
$ jq -n '5'
5
```

16. Constructing JSON Array with Jq Command without any Input jSON Data
```bash
$ jq -n '[]'
[]
$ jq -n '[1,2,3]'
[
  1,
  2,
  3
]
$ jq -n '{"k1":"v1", "k2":"v2"}'
{
  "k1": "v1",
  "k2": "v2"
}
```

17. Constructing JSON Object with Jq command with and without any input JSON Data
```bash
$ cat tools.json
{
"techonogies": "devops",
"tools": ["jenkins", "bamboo",
  "ansible", "docker"
]
}
$ jq '{"mytools": .tools[2] }' tools.json
{
  "mytools": "ansible"
}
```

18. Jq Command with --tab and -c options
```bash
$ jq '.' tools.json
{
  "techonogies": "devops",
  "tools": [
    "jenkins",
    "bamboo",
    "ansible",
    "docker"
  ]
}
$ jq --tab '.' tools.json
{
        "techonogies": "devops",
        "tools": [
                "jenkins",
                "bamboo",
                "ansible",
                "docker"
        ]
}
$ jq -c '.' tools.json
{"techonogies":"devops","tools":["jenkins","bamboo","ansible","docker"]}
```

19. Adding and modifying a Key Value for a given JSON object
- `jq '.key=value' [input]`
- `jq '.=.+{"key":value}' [input]`
- `jq '.+={"key":value}' [input]`
```bash
$ jq '.year=2023' tools.json
{
  "techonogies": "devops",
  "tools": [
    "jenkins",
    "bamboo",
    "ansible",
    "docker"
  ],
  "year": 2023
}
$ jq '.=.+{"year":2023}' tools.json
{
  "techonogies": "devops",
  "tools": [
    "jenkins",
    "bamboo",
    "ansible",
    "docker"
  ],
  "year": 2023
}
$ jq '.+={"year":2023}' tools.json
{
  "techonogies": "devops",
  "tools": [
    "jenkins",
    "bamboo",
    "ansible",
    "docker"
  ],
  "year": 2023
}
$ jq '.tools+=["git"]' tools.json
{
  "techonogies": "devops",
  "tools": [
    "jenkins",
    "bamboo",
    "ansible",
    "docker",
    "git"
  ]
}
```
 
20. Adding and modifying a Value for a given JSON Array
- `|length` to find the size of array/dictionary
- `jq '.+=[value]' [input]`
```bash
$ jq '.tools[0]="hello"'  tools.json
{
  "techonogies": "devops",
  "tools": [
    "hello",  #<----- replacing 0th array element
    "bamboo",
    "ansible",
    "docker"
  ]
}
$ jq '.tools|length'  tools.json
4
$ jq '.tools[.tools|length]="k8"'  tools.json
{
  "techonogies": "devops",
  "tools": [
    "jenkins",
    "bamboo",
    "ansible",
    "docker",
    "k8"  #<--- appending one more element in the end
  ]
}
```
 
21. Recreating an Object: Creating a new object where key-vlaue pairs are from input
```bash
$ jq '{"mytool": .tools[2], "mytech": .techonogies}' tools.json
{
  "mytool": "ansible",
  "mytech": "devops"
}
```
 
## Section 7: Jq Command with Exit Status & Creating Shell Variables from jq Result

22. How to Find the exit status of a Jq command?
- The default exit status of jq is zero, regardles of the operation results
    - Use `-e` to get exit status
```bash
$ date
Fri Jan 12 15:18:04 EST 2024
$ echo $?
0   # success
$ dateXX
bash: dateXX: command not found...
$ echo $?
127  # failed
$ jq -e  '.tools' tools.json
[
  "jenkins",
  "bamboo",
  "ansible",
  "docker"
]
[bxj670@BRCLWC5T32G3 ~]$ echo $?
0
[bxj670@BRCLWC5T32G3 ~]$ jq -e  '.tools2' tools.json
null
[bxj670@BRCLWC5T32G3 ~]$ echo $?
1
```

23. How to store Jq Command Output into a Shell Variable?

## Section 8: jq Pipes

24. How to combine multiple jq commands and their logics with one jq command?
```bash
$ jq . sample.json 
{
  "techonogies": "devops",
  "tools": [
    "jenkins",
    "bamboo",
    "ansible",
    "docker"
  ],
  "year": 2023
}
$ jq . sample.json | jq '.tools' # value for tools
[
  "jenkins",
  "bamboo",
  "ansible",
  "docker"
]
$ jq . sample.json | jq '.tools' | jq 'length' # length of values of tools
4
$ jq '.tools | length' sample.json  # pipe in jq options
4
```

25. Create an Array with jq command and items are from input file
- How to extract username from {Users [{username:...} {username:...} {user:...} ]} ?
- `jq '.Users[].UserName' userarray.json`
- To have an array form, `jq '[.Users[].UserName]' userarray.json`

## Section 9: Introduction to jq Functions

26. Introduction to jq Functions
- keys
```bash
$ jq '.Users[0]' userarray.json |jq 'keys'
[
  "Arn",
  "CreateDate",
  "Path",
  "UserId",
  "UserName"
]
$ jq '.Users[0]|keys' userarray.json 
[
  "Arn",
  "CreateDate",
  "Path",
  "UserId",
  "UserName"
]
```
- length
- min, max, add
- reverse, sort
- unique
- del
- env
- join, split
- has
- map, reduce
- select
- match
- tonumber, tostring, uppercase, lowercase, ...
- Do not add '.' to the function name

27. min, max, add, sort, reverse and unique functions of jq command
```bash
$ jq -n '[2,4,1,3,7]' |jq 'min'
1
$ jq -n '[2,4,1,3,7] | max'
7
$ jq -n '[2,4,1,3,7] | add'
17
$ jq 'sort' <<< '[2,4,1,3,7]'
[
  1,
  2,
  3,
  4,
  7
]
$ jq 'reverse' <<< '[2,4,1,3,7]'
[
  7,
  3,
  1,
  4,
  2
]
$ jq 'unique' <<< '[2,4,1,3,7]' # sorted as well by default
[
  1,
  2,
  3,
  4,
  7
]
```

28. min_by, max_by, sort_by, group_by, unique_by and reverse functions for Array
```bash
$ jq . arrayobject.json 
[
  {
    "id": 1,
    "name": "jenkins",
    "usage": "integration"
  },
  {
    "id": 2,
    "name": "stackstom",
    "usage": "integration and event-driven automation"
  },
  {
    "id": 3,
    "name": "bamboo",
    "usage": "integration"
  },
  {
    "id": 4,
    "name": "abc",
    "usage": "integration and event-driven automation"
  },
  {
    "id": 5,
    "name": "docker",
    "usage": "container"
  }
]
$ jq 'max_by(.id)' arrayobject.json 
{
  "id": 5,
  "name": "docker",
  "usage": "container"
}
$ jq 'sort_by(.usage)' arrayobject.json 
[
  {
    "id": 5,
    "name": "docker",
    "usage": "container"
  },
  {
    "id": 1,
    "name": "jenkins",
    "usage": "integration"
  },
  {
    "id": 3,
    "name": "bamboo",
    "usage": "integration"
  },
  {
    "id": 2,
    "name": "stackstom",
    "usage": "integration and event-driven automation"
  },
  {
    "id": 4,
    "name": "abc",
    "usage": "integration and event-driven automation"
  }
]
$ jq 'group_by(.usage)' arrayobject.json 
[
  [
    {
      "id": 5,
      "name": "docker",
      "usage": "container"
    }
  ],
  [
    {
      "id": 1,
      "name": "jenkins",
      "usage": "integration"
    },
    {
      "id": 3,
      "name": "bamboo",
      "usage": "integration"
    }
  ],
  [
    {
      "id": 2,
      "name": "stackstom",
      "usage": "integration and event-driven automation"
    },
    {
      "id": 4,
      "name": "abc",
      "usage": "integration and event-driven automation"
    }
  ]
]
```

## Section 10: Working with OS Variables

29. Accessing OS Level Variables from jq command with env function
- `jq [options] env [input]`
```bash
$ jq -n 'env'
{
  "SHELL": "/bin/bash",
  "SESSION_MANAGER": "local/hakune:@/tmp/.ICE-unix/9480,unix/hakune:/tmp/.ICE-unix/9480",
  "QT_ACCESSIBILITY": "1",
  "COLORTERM": "truecolor",
...
}
$ jq -n 'env.USER'
"hpjeon"
$ jq -n 'env.TERM'
"xterm-256color"
```
30. Shell Script to display output as json data using jq command

31. --arg option to work with environment or custom variables
- `--arg argument1 argument2` will deliver the variable to jq
```bash
$ jq -n --arg myTMP "/tmp/TMP" '$myTMP'
"/tmp/TMP"
$ jq -n --arg today "$(date)" '$today'
"Fri 12 Jan 2024 06:52:50 PM EST"
```

## Section 11: Join & Split and range Functions

32. join and split functions | How to get /etc/shells as an array through jq command?
- jq [options] 'join("separator)' [input]
- jq [options] 'split("separator)' [input]
```bash
$ jq 'join(" ")' <<< '["hello", "world"]'
"hello world"
$ jq '.tools' ./sample.json | jq 'join(":")'
"jenkins:bamboo:ansible:docker"
$ jq 'split(" ")' <<< '"hello world"'
[
  "hello",
  "world"
]
```

33. range Function
```bash
$ jq -n 'range(4)'
0
1
2
3
$ jq -n 'range(4;7)'
4
5
6
$ jq -n 'range(1;10;3)'
1
4
7
$ jq -n '{"myValues": [range(0;4;2)] }'
{
  "myValues": [
    0,
    2
  ]
}
```

## Section 12: Operations on Strings

34. Concatenation of String with + operator and join function
- Using + or join to merge strings
- simple.json
```json
{
  "gender": "male",
  "name": {
    "title": "Mr",
    "first": "Arlo",
    "last": "King"
  }
}
```
- Demo:
```bash
$ jq '.name.last + " " +  .name.first' simple.json 
"King Arlo"
$ jq '[.name.last, .name.first] | join(" ")' simple.json 
"King Arlo"
```

35. String's Case Conversion with jq functions
- ascii_upcase
- ascii_downcase
```bash
$ jq '[.name.last, .name.first] | join(" ") | ascii_upcase' simple.json 
"KING ARLO"
$ jq '[.name.last, .name.first] | join(" ") | ascii_downcase' simple.json 
"king arlo"
```

## Section 13: Type Conversion

36. Finding Data Type and Converting Number to String and String to Number
```bash
$ jq -n '4 | type'
"number"
hpjeon@hakune:~/hw/class/udemy_jq$ jq -n '"hello" | type'
"string"
hpjeon@hakune:~/hw/class/udemy_jq$ jq -n 'true | type'
"boolean"
hpjeon@hakune:~/hw/class/udemy_jq$ jq -n '[true, false] | type'
"array"
hpjeon@hakune:~/hw/class/udemy_jq$ jq -n '{"key":"val"} | type'
"object"
$ jq -n '"hello" + 4'
jq: error (at <unknown>): string ("hello") and number (4) cannot be added
$ jq -n '"hello" +  (4 | tostring)'
"hello4"
$ export a=4
$ export b=7
$ jq -n '(env.a | tonumber) + (env.b | tonumber)' 
11
```

37. Arithmetic Operations with jq command

38. How to use variables to get required field or index value?
- When key name is given from system variable, how we can use it to couple with jq?
```bash
$ jq '.gender' simple.json 
"male"
$ export myKey=gender
$ jq '.[env.myKey]' simple.json 
"male"
```

## Section 14: has, select, map & reduce functions

39. has Function
- Check if the object has the required key or not
```bash
$ jq 'has("gender")' simple.json 
true
$ jq '.name|has("title")' simple.json  # for the nested object
true
$ jq .[] arrayobject.json |jq 'has("id")' # for the array items
true
true
true
true
true
```

40. map Function
- How to extract values from array object?
  - map function will MAP them into an array
```bash
$ jq .[] arrayobject.json |jq '.name'
"jenkins"
"stackstom"
"bamboo"
"abc"
"docker"
$ jq 'map(.name)' arrayobject.json 
[
  "jenkins",
  "stackstom",
  "bamboo",
  "abc",
  "docker"
]
$ jq  '.tools' sample.json  | jq 'map(ascii_upcase)'
[
  "JENKINS",
  "BAMBOO",
  "ANSIBLE",
  "DOCKER"
]
$ jq .name simple.json | jq 'map(ascii_upcase)'
[
  "MR",
  "ARLO",
  "KING"
]
$ jq .name simple.json | jq 'map_values(ascii_upcase)' # value only. 
{
  "title": "MR",
  "first": "ARLO",
  "last": "KING"
}
```

41. select Function
- When condition inside of select() is true, it shows results
```bash
$ jq .[] arrayobject.json  | jq 'select(.id >3)'
{
  "id": 4,
  "name": "abc",
  "usage": "integration and event-driven automation"
}
{
  "id": 5,
  "name": "docker",
  "usage": "container"
}
$ jq .[] arrayobject.json  | jq 'select(.usage=="integration")'
{
  "id": 1,
  "name": "jenkins",
  "usage": "integration"
}
{
  "id": 3,
  "name": "bamboo",
  "usage": "integration"
}
```

## Section 15: Formatting JSON Data

42. Converting JSON data into csv and tsv formats
- Can export to CSV only when value of array is array
- jq -r '.[] | @csv' [input]
- jq -r '.[] | @tsv' [input]
- test.json:
```json
[
 [ 1, "hello", "world"],
 [ 2, "Fine", "thanks"]
]
```
- Demo:
```bash
$ jq -r '.[] | @csv' test.json 
1,"hello","world"
2,"Fine","thanks"
$ jq -r '["ID","prefix", "postfix"], .[] | @csv' test.json  # Adding header line
"ID","prefix","postfix"
1,"hello","world"
2,"Fine","thanks"
```

43. Encoding and Decoding a String
```bash
$ jq -n 'env.TERM'
"xterm-256color"
$ jq -n 'env.TERM | @base64' # encoding as a binary
"eHRlcm0tMjU2Y29sb3I="
```

## Section 16: Practice

44. Shell Script to get k8s namespaces details
