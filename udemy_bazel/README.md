## Modern C++ development with bazel, googletest and googlemock
- Instructor: Balaji Ambresh Rajkumar

## Section 1: Introduction

### 1. Introduction
- 3 Tools for C++ code development
    1. Bazel: scalable build tool
    2. Googletest: Testing framework for unit-testing
    3. Googlemock: Makes unit testing easier by stubbing function calls

### 2. Setup
- Bazel install
  1. Download bazel-*-installer-linux-x86_64.sh from https://github.com/bazelbuild/bazel/releases
  2. bash ./bazel-8.5.0-installer-linux-x86_64.sh --prefix=/XXXX/sw_local/bazel/8.5.0
  
## Section 2: Bazel

### 3. Introduction
- Multi platform build management system capable of building C++, Java, iOS and android code
- Written by Skylark language (subset of Python)
- Scales to 100k+ source files
- Can be extended to support any framework/language
- Bazel rules
  - cc_library: Creates static/dynamic libraries
  - cc_test: Compiles and runs our test code and dumps test result outpu to a test.log
  - cc_binary: Creates an executable

### 4. hello bazel - compiling a hello world program using bazel
- Source/folder structure:
```bash
modern_cpp/00-hello-world/
├── src
│   ├── BUILD
│   └── HelloWorld.cc
└── WORKSPACE
```
- src/BUILD file:
```bash
cc_binary(
    name = "HelloWorld",
    srcs = ["HelloWorld.cc"]
)
```
- Build process:
```bash
$ bazel build //src:HelloWorld
INFO: Analyzed target //src:HelloWorld (69 packages loaded, 480 targets configured).
INFO: Found 1 target...
Target //src:HelloWorld up-to-date:
  bazel-bin/src/HelloWorld
INFO: Elapsed time: 0.780s, Critical Path: 0.20s
INFO: 6 processes: 4 internal, 2 processwrapper-sandbox.
INFO: Build completed successfully, 6 total actions
```
  - The name after `//src:` is found from BUILD file contents
  - This process produces many folders like bazel-bin, bazel-out, bazel-testlogs, which are link of files generated at $HOME/.cache/bazel folder
    - To clean them, `bazel clean`

### 5. compiling files including headers
- At 01-build-with-headers:
```bash
../01-build-with-headers/
├── src
│   ├── BUILD
│   ├── Calculator.cc
│   ├── Calculator.h
│   └── Main.cc
└── WORKSPACE
```
- src/BUILD:
```bash
cc_library(
    name = "Calculator",
    srcs = ["Calculator.cc"],
    hdrs = ["Calculator.h"]
)

cc_binary(
    name = "Main",
    srcs = ["Main.cc"],
    deps = [
        ":Calculator"
    ]
)
```
- Build command: `bazel build //src:Main`
- When Main.cc is modified, it will build without rebuilding Calculator then the entire process is very fast

### 6. creating and using a library
```bash
../02-simple-lib/
├── lib
│   └── calc
│       ├── BUILD
│       ├── include
│       │   └── calc
│       │       └── Calculator.h
│       └── src
│           ├── Adder.cc
│           ├── Adder.h
│           └── Calculator.cc
├── main
│   ├── BUILD
│   ├── Main.cc
│   └── searchbar
│       ├── include
│       │   └── searchbar
│       │       └── SearchBar.h
│       └── src
│           └── SearchBar.cc
└── WORKSPACE

11 directories, 10 files
```
- lib/calc/BUILD: 
```bash
cc_library(
    name = "Calculator",
    srcs = [
        "src/Calculator.cc",
        "src/Adder.h",
        "src/Adder.cc"
    ],
    hdrs = [
        "include/calc/Calculator.h"
    ],
    includes = [
        "include",
        "src"
    ],
    visibility = ["//main:__pkg__"]
)
```
- main/BUILD:
```bash
cc_library(
    name = "SearchBar",
    srcs = ["searchbar/src/SearchBar.cc"],
    hdrs = [
        "searchbar/include/searchbar/SearchBar.h"
    ],
    includes = [
        "searchbar/include"
    ],
    deps = [
        "//lib/calc:Calculator"
    ]
)

cc_binary(
    name = "Main",
    srcs = ["Main.cc"],
    deps = [
        ":SearchBar"
    ],
)
```
- Build command: ` bazel build //main:Main`
  - Note that it is NOT `//src`

### 7. generating product flavors
- Why? Beep sounds in the audio?
- At 06-more-bazel-concepts$ tree
```bash
├── e01
│   ├── BUILD
│   ├── include
│   │   └── e01
│   │       └── Greeter.h
│   └── src
│       ├── GreeterDebug.cc
│       ├── GreeterDefault.cc
│       ├── GreeterLocal.cc
│       ├── GreeterProduction.cc
│       └── Main.cc
└── WORKSPACE
```
- e01/BUILD:
```bash
cc_library(
    name = "GreeterDebug",
    hdrs = glob(["include/**/*.h"]),
    srcs = ["src/GreeterDebug.cc"],
    includes = ["include"]
)

cc_library(
    name = "GreeterProduction",
    hdrs = glob(["include/**/*.h"]),
    srcs = ["src/GreeterProduction.cc"],
    includes = ["include"]
)

cc_library(
    name = "GreeterLocal",
    hdrs = glob(["include/**/*.h"]),
    srcs = ["src/GreeterLocal.cc"],
    includes = ["include"]
)

cc_library(
    name = "GreeterDefault",
    hdrs = glob(["include/**/*.h"]),
    srcs = ["src/GreeterDefault.cc"],
    includes = ["include"]
)

cc_binary(
    name = "Main",
    srcs = ["src/Main.cc"],
    deps = 
    select({
        ":debug": [":GreeterDebug"],
        ":production": [":GreeterProduction"],
        ":localDevelopment": [":GreeterLocal"],
        "//conditions:default": [":GreeterDefault"]
    })
)

config_setting(
    name = "debug",
    values = {
        "compilation_mode": "dbg"
    }
)

config_setting(
    name = "production",
    define_values = {
        "type": "production"
    }
)

config_setting(
    name = "localDevelopment",
    values = {
        "compilation_mode": "dbg"
    },
    define_values = {
        "type": "local"
    }
)
```
- `bazel build //e01:Main`
- `bazel run //e01:Main`  yields `Greeter default`
- `bazel run //e01:Main -c dbg`  yields `Greeter debug`
  - `-c` for compilation mode: opt, dbg
- `bazel run //e01:Main --define type=local -c dbg` yields `Greeter local`
  - `--define` to feed KEY-VALUE pair
- `bazel run e01:Main  -c dbg --define type=production` yields Build failed as thow two options are exclusive

## Section 3: Google test

### 8. Introduction
- At 03-gtest-intro$ tree
``` bash
├── lib
│   └── calc
│       ├── BUILD
│       ├── src
│       │   ├── Calculator.cc
│       │   └── Calculator.h
│       └── test
│           └── CalculatorTest.cc
├── main
│   ├── BUILD
│   ├── src
│   │   ├── Main.cc
│   │   └── searchbar
│   │       ├── SearchBar.cc
│   │       └── SearchBar.h
│   └── test
│       └── SearchBarTest.cc
└── WORKSPACE
```
- lib/calc/BUILD:
```bash 
cc_library(
    name = "Calculator",
    srcs = glob(["src/*.cc"]),
    hdrs = glob(["src/*.h"]),
    includes = ["src"],
    visibility = ["//visibility:public"]
)

cc_test (
    name = "CalculatorTest",
    srcs = [
        "test/CalculatorTest.cc"
    ],
    timeout="short",
    deps = [
        "@gtest//:gtest",
        "@gtest//:gtest_main",
        ":Calculator"
    ]
)
```
- main/BUILD:
```bash
cc_library(
    name = "SearchBar",
    srcs = [
        "src/searchbar/SearchBar.cc"
    ],
    hdrs = [
        "src/searchbar/SearchBar.h"
    ],
    includes = ["src"],
    deps = [
        "//lib/calc:Calculator"
    ]
)

cc_test (
    name = "SearchBarTest",
    srcs = [
        "test/SearchBarTest.cc"
    ],
    timeout="short",
    deps = [
        "@gtest//:gtest",
        "@gtest//:gtest_main",
        ":SearchBar"
    ]
)

cc_binary(
    name = "Main",
    srcs = ["src/Main.cc"],
    deps = [
        ":SearchBar"
    ],
)
```
- WORKSPACE:
```bash
 more WORKSPACE 
# See : https://docs.bazel.build/versions/master/external.html
# See : https://docs.bazel.build/versions/master/repo/git.html

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "gtest",
    remote = "https://github.com/google/googletest",
    commit = "703bd9caab50b139428cea1aaff9974ebee5742e"
)
```
  - This will install exterinal libraries
  - Workspace is deprecated since BAZEL8+

### 9. Demo part 1
- WORKSPACE is not supported anymore
- Make MODULE.bazel in the root location:
```bash
# MODULE.bazel

module(name = "pr03again")
bazel_dep(name="googletest", version="1.17.0")
```
  - In the BUILD files, rename gtest into googletest
- Running command
  - bazel build //... # builds everything
  - bazel clean # cleans the root folder
  - To build by components
    - bazel build //lib/calc:Calculator
    - bazel build //lib/calc:CalculatorTest
    - bazel build //main:SearchBar
    - bazel build //main:SearchBarTest
  - bazel test //... # Test all
  - Test by Component: bazel test //main:SearchBarTest
  - bazel test //main:SearchBarTest --test_output=all
```bash
 ==================== Test output for //main:SearchBarTest:
Running main() from gmock_main.cc
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from SearchBar
[ RUN      ] SearchBar.AdditionTest
[       OK ] SearchBar.AdditionTest (0 ms)
[ RUN      ] SearchBar.InvalidOp
[       OK ] SearchBar.InvalidOp (176 ms)
[----------] 2 tests from SearchBar (176 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (176 ms total)
[  PASSED  ] 2 tests.
================================================================================
```

### 10. Demo part 2

### 11. Deepdive into gtest part 1
- At 04-gtest-deepdive with modifications above-mentioned:
```bash
├── e01
│   ├── BUILD
│   ├── src
│   │   ├── SimpleMath.cc
│   │   └── SimpleMath.h
│   └── test
│       └── SimpleMathTest.cc
├── e02
│   ├── BUILD
│   ├── src
│   │   ├── Stack.cc
│   │   └── Stack.h
│   └── test
│       ├── InheritedStackTest.cc
│       └── StackTest.cc
├── e03
│   ├── BUILD
│   ├── src
│   │   ├── Utils.cc
│   │   └── Utils.h
│   └── test
│       ├── TestMain.cc
│       ├── UtilsTest1.cc
│       ├── UtilsTest2.cc
│       ├── UtilsTest3.cc
│       └── UtilsTest4.cc
└── MODULE.bazel
```
- bazel build //...
- bazel test //...

### 12. Deepdive into gtest part 2

### 13. Deepdive into gtest part 3

## Section 4: Google mock

### 14. Introduction

### 15. Demo

### 16. Deepdive

