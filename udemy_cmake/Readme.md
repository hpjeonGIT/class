## Master CMake for Cross-Platform C++ Project Building
- Instructor: Milan Yadav

## Section 1: Introduction

## Section 2: CMake installation and Building the First Target

6. Build File Generation Process
- A CMakeLists.txt file
- Folder structures

7. Generating the first executable using CMake
- add_executable(<ExecName> <SRCfiles>)
- A single line CMakeLists.txt
```cmake
add_executable (calculator addition.cpp  division.cpp  main.cpp  print_result.cpp)
```
  - mkdir build; cd build; cmake ../; make works OK
- An improved version:
```cmake
cmake_minimum_required(VERSION 3.0.0)
project(Calc_project VERSION 1.0.0)
add_executable (calculator addition.cpp  division.cpp  main.cpp  print_result.cpp)
```
  - We need to implement file hierarchies

8. Generating the First library
- add_library(<LibraryName> <SRCFILES>)
```cmake
cmake_minimum_required(VERSION 3.0.0)
project(Calc_project VERSION 1.0.0)
add_library( my_math addition.cpp division.cpp)
add_library( my_print print_result.cpp)
add_executable(calculator main.cpp)
target_link_libraries(calculator my_math my_print)
```

9. Target's Properties and Dependencies
- Properties
  - INTERFACE_LINK_DIRECTORIES
  - INCLUDE_DIRECTORIES
  - VERSION
  - SOURCE
  - Commands:
    - set_target_properties()
    - set_property()
    - get_property()
    - get_target_property()
- Dependencies
  - The order of building objects
- Propagating Target Properties (Scope specifier)
  - PUBLIC
  - INTERFACE
  - PRIVATE

10. FAQ on Targets

## Section 3: Managing Project Files and Folders using Subdirectories

11. Sub-directories
```bash
├── addition.h
├── build
├── CMakeLists.txt
├── division.h
├── main.cpp
├── my_math_dir
│   ├── addition.cpp
│   ├── CMakeLists.txt
│   └── division.cpp
├── my_print_dir
│   ├── CMakeLists.txt
│   └── print_result.cpp
└── print_result.h
```
- CMakeLists.txt 
```cmake
cmake_minimum_required(VERSION 3.0.0)
project(Calc_project VERSION 1.0.0)
add_subdirectory(my_math_dir)
add_subdirectory(my_print_dir)
add_executable(calculator main.cpp)
target_link_libraries(calculator my_math my_print)
```
- my_math_dir/CMakeLists.txt 
```cmake
add_library(my_math addition.cpp division.cpp)
```
- my_print_dir/CMakeLists.txt 
```cmake
add_library(my_print print_result.cpp)
```

12. Managing header files
```bash
├── CMakeLists.txt
├── inc
│   └── print_result.h
└── src
    └── print_result.cpp
```
- Need to update print_result.cpp, addition.cpp, division.cpp to have header files respectively

13. CMake way of including the header files
```bash
├── build
├── CMakeLists.txt
├── main.cpp
├── my_math_dir
│   ├── CMakeLists.txt
│   ├── inc
│   │   ├── addition.h
│   │   └── division.h
│   └── src
│       ├── addition.cpp
│       └── division.cpp
└── my_print_dir
    ├── CMakeLists.txt
    ├── inc
    │   └── print_result.h
    └── src
        └── print_result.cpp
```
- my_math_dir/CMakeLists.txt 
```cmake
add_library(my_math src/addition.cpp src/division.cpp)
target_include_directories(my_math PUBLIC inc)
```
- my_print_dir/CMakeLists.txt 
```cmake
add_library(my_print src/print_result.cpp)
target_include_directories(my_print PUBLIC inc)
```
- PUBLIC keyword in my_math_dir/my_print_dir/CMakeLists.txt let main.cpp use header files definition from my_math_dir and my_print_dir
  - When defined PRIVATE or INTERFACE, build will fail

14. Target Properties and Propagation Scopes

|Question | Answer | Answer |Answer|
|---------|--------|--------|------|
| Does my_math need the folder? | Yes | No  | Yes |
|Does the other targets, depending on my_math, need the inc folder? | Yes |Yes |No |
| | PUBLIC | INTERFACE | PRIVATE|

15. Propagation of Target Properties

## Section 4: Variables, Lists, and Strings

16. Normal Variables
- MESSAGE(<mode>, "hello word")
- Strings & Lists
  - SET(Name "Bob Smith") => String 'Name' = Bob Smith
  - SET(Name Bob Smith) => List 'Name' = Bob; Smith
```bash
$ cat CMakeLists.txt 
SET(Name "Bob Smith")
SET(List Bob Smith)
message("hello world from ${Name}  and ${List} " )
hpjeon@hakune:~/hw/class/udemy_cmake/chap16$ 
$ cmake -p CMakeLists.txt 
hello world from Bob Smith  and Bob;Smith 
-- Configuring done
-- Generating done
```

17. Quoted and Unquoated Arguments

| Set Commands | Value of VAR | message(${VAR}) | message("${VAR}") |
|------------|--|--|--|
|set(VAR aa bb cc) | aa;bb;cc | aabbcc | aa;bb;cc |
|set(VAR aa;bb;cc) | aa;bb;cc | aabbcc | aa;bb;cc |
|set(VAR "aa" "bb" "cc") | aa;bb;cc | aabbcc | aa;bb;cc |
|set(VAR "aa bb cc") | aa bb cc | aa bb cc | aa bb cc |
|set(VAR "aa;bb;cc") | aa;bb;cc | aabbcc | aa;bb;cc |

18. Manipulating Variables
```bash
cat CMakeLists.txt 
SET(Name 3.14)
SET(3.14 Jonny)
message(Name ${Name} ${${Name}})
message("Name ${Name} ${${Name}}")
$ cmake -p CMakeLists.txt 
Name3.14Jonny
Name 3.14 Jonny
-- Configuring done
-- Generating done
```

19. Lists and Strings
- list(<subcommand> <nameoflist> ... <returnvariable>)
  - APPEND
  - INSERT
  - FILTER
  - GET
  - JOIN
```bash
$ cat CMakeLists.txt 
cmake_minimum_required(VERSION 3.0.0)
SET(VAR a b c;d "e;f" 2.7 "hello world")
list(APPEND VAR 1.6 XX)
list(REMOVE_AT VAR 2 -3)
message(${VAR})
message("${VAR}")
$ cmake -p CMakeLists.txt 
abdef2.71.6XX
a;b;d;e;f;2.7;1.6;XX
```
- string()
  - FIND
  - REPLACE
  - PREPEND
  - APPEND
  - TOLOWER
  - TOUPPER
  - COMPARE
- file()
  - READ
  - WRITE
  - RENAME
  - REMOVE
  - COPY
  - DOWNLOAD
  - LOCK

## Section 5: Control Flow Commands, Functions, Macros, Scopes and Listfiles

20. If-Else Command
- Flow control
  - if-else
  - loop
    - while
    - foreach
- Function command
- Scopes
- Macro command
- Modules
- Constants
  - TRUE: 1, ON, YES, TRUE, Y, a nonzero number
  - FALSE: 0, OFF, NO, FALSE, N, IGNORE, NOTFOUND, empty string, string ending with -NOTFOUND
- IF() conditions
  - Unary tests
    - DEFINED
      - `IF(DEFINED VAR1) ... ENDIF()`
    - COMMAND
      - `IF(COMMAND target_link_libraries) ... ENDIF()`
      - Check if the command is valid 
    - EXISTS
      - `IF(EXISTS /etc/log) ... ENDIF()`
    - Binary tests
      - STRLESS
      - STRGREATER
      - STREQUAL
    - Boolean operators
      - OR
      - AND

21. Looping Commands
- WHILE
  - `WHILE() ... ENDWHILE()`
- FOREACH
  - `FOREACH() ... ENDFOREACH()`
```bash
$ cat CMakeLists.txt 
cmake_minimum_required(VERSION 3.0.0)
FOREACH (NAME abc;def; xyz) 
 MESSAGE(${NAME})
ENDFOREACH()
foreach(x RANGE 101 104)
  message("x=${x}")
endforeach()
$ cmake -p CMakeLists.txt 
abc
def
xyz
x=101
x=102
x=103
x=104
```

22. Functions
- `function(<functionName> <functionArgs>) <commands> endfunction()`
- Can de-reference recursively:
```bash
$ cat CMakeLists.txt 
cmake_minimum_required(VERSION 3.0.0)
set(Name Charlie)
set(Age 45)
function(print_detail var)
  message("My ${var} is ${${var}}")
endfunction()
print_detail(Name)
print_detail(Age)
$ cmake -p CMakeLists.txt 
My Name is Charlie
My Age is 45
```
- When the same function name is repeated, the last one is used
  - The previous function might be called as `_functionName()`

23. Optional Arguments of Functions

| Special variables | Description|
|--|--|
| ARGC  | Total count of arguments                 |
| ARGV  | List of all arguments including optional |
| ARGN  | List of optional arguments               |
| ARGV1 | Second Argument                          |
| ARGV2 | Third Argument                           |

```bash
$ cat CMakeLists.txt 
cmake_minimum_required(VERSION 3.0.0)
function(print_detail var)
  message("testing ${ARGC} ${ARGV} ${ARGN} ${ARGV0} ${ARGV1}")
endfunction()
print_detail("Name")
print_detail("Name" "AGE")
$ cmake -p CMakeLists.txt 
testing 1 Name  Name 
testing 2 Name;AGE AGE Name AGE
```
- Note that function arguments may not need comma

24. Scopes
- The scope of a function is isolated from PARENT
- To access parent scope, use `PARENT_SCOPE`
```cmake
set(Name pName)
function(myFunction)
  set(Name NewNAME PARENT_SCOPE)
  ...
end function()
```

25. Macros
- `macro(<functionName> <functionArgs>) ... endmacro()`
- Difference to functions
  - No local scope. Same as parent scope
- Function/macro/cmake functions are case insensitive

26. Listfiles and Modules

## Section 6: Cache Variables

27. Setting a Cache Variable
- Cache variables
  - Set by CMake, depending on the development environment
  - Set by commands inside CMakeLists.txt
- `set(A "123" CACHE STRING "some description")`
  - Will be shown at CMakeCache.txt
```cmake
//Some description
A:STRING=123
```
  - Can be dereferenced as `$CACHE{A}`
    - `${A}` works OK too
    - if `${A}` doesn't exist, `$CACHE{A}` will search CMakeCache.txt
- Environment variables
  - Global scope
  - Not stored in CMakeCache.txt
  - `$ENV{variable}`

28. Modification of Cache Variables
- Modifying Cache Variables
  - Edit existing CMakeCache.txt file
  - Use FORCE keyword or -D flag

29. Cache Variables: CMAKE_VERSION, CMAKE_PROJECT_NAME, CMAKE_GENERATOR
- CMAKE_VERSION, CMAKE_MAJOR_VERSION, CMAKE_MINOR_VERSION, CMAKE_PATCH_VERSION
- CMAKE_GENERATOR: may use ninja or other build tools

## Section 7: Installing and Exporting Package

30. Requirements for installing/exporting package

31. Installation Process of Packages
- `install(FILES <fileName> DESTINATION <dir>)`
- `install(TARGETS <tgtName> DESTINATION <dir>)`

32. Exporting a Package
- `find_package()` command
  - Add targets to export group
  - Install the export group
  - Modify the target_include_directories() commands
  
33. Using a 3rd party Package in our Project
- `find_package(my_math)`
  - Searches Module then Config
  - Module mode: `Findmy_math.cmake`
  - Config mode: `my_math-config.cmake`
    - Lapack package installation has lib/cmake/lapack-x.x.x/lapack-config.cmake
  - Uses `CMAKE_MODULE_PATH`
- `find_package(my_math MODULE)` : module mode only
- `find_package(my_math CONFIG)` : config mode only

## Section 8: Tips/FAQs

35. Commenting in CMake
- Multi-line comment: `#[[` and `#]]`

## Section 9: Linking External Libraries

39. Problems with Linking External Libraries
- Using External library
  - Uses CMake based build generation process
    - *config.cmake present
    - No *config.cmake present => Use pkg-config
  - Uses NON-CMake based build generation process
    - CMake or Library provides Find* or *config modules
    - Uses pkg-config file
    - No pkg-config file present => write own find_package()

40. Installation of OpenCV

41. Using OpenCV in a Project

42. Using Pkg-Config to link GTK3 library
- Using pkg-config when a SW is installed using non-cmake build process
- To find package:
```cmake
find_package(PkgConfig REQUIRED)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/opt/sw/myapp")
set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/opt/sw/myapp")
```

43. find_library() and find_path() commands
- Example
  - Project: MyProject
    - Executable: MyApp
      - Dependency: External library abc
  - abc.h at /home/mky/Downloads/abc/include/abc.h
  - libabc.so at /home/mky/Downloads/abc/lib/libabc.so
- `find_library(<VAR> <lib-name> <path1><path2>...)`
  - May use HINTS
  - Will not find the sub-directory of path1, path2, ...
    - To search the sub folder, add `PATH_SUFFIXES`
  - May find multiple selections using NAMES
    - libabc.so or libabc-1.14.so or libabc-1.15.so
  - Ex) `find_library(abc_LIBRARY NAMES abc abc-1.14 abc-1.15 HINTS /home/mky/Downloads/abc /opt/abcp PATH_SUFFIXES lib lib/abc-1.14)`
- find_path() is similar to find_library()
  - May be used to find header files
- find_library() vs find_path()
  - find_library()
    - Default search path is /usr/lib or /usr/libx86_64-linux-gnu
    - Returns the directory path + file name such as /home/mky/Downloads/abc/libabc.so
  - find_path()
    - Default search path is /usr/include or /usr/include/x86_64-linux-gnu
    - Ruturns the directory path only such as /home/mky/Downloads/abc/include

44. Writing a Find* module
- In the main CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.0.0)
...
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)
find_package(GTK3 REQUIRED)
```
- At cmake/modules, Add FindGTK3.cmake as:
```cmake
find_library( GTK3_LIBRARY
		NAMES gtk-3)
find_path(GTK3_INCLUDE_DIR
		NAMES gtk/gtk.h
		PATH_SUFFIXES gtk-3.0)
find_path(GLIB_INCLUDE_DIR
		NAMES glib.h
		PATH_SUFFIXES glib-2.0)
find_path(GLIBCONFIG_INCLUDE_DIR
		NAMES glibconfig.h
		HINTS /usr/lib/x86_64-linux-gnu
		PATH_SUFFIXES glib-2.0/include) 
find_path(PANGO_INCLUDE_DIR
		NAMES pango/pango.h
		PATH_SUFFIXES pango-1.0)
find_path(CAIRO_INCLUDE_DIR
		NAMES cairo.h
		PATH_SUFFIXES cairo)
find_path(GDK_PIXBUF_INCLUDE_DIR
		NAMES gdk-pixbuf/gdk-pixbuf.h
		PATH_SUFFIXES gdk-pixbuf-2.0)
find_path(ATK_INCLUDE_DIR
		NAMES atk/atk.h
		PATH_SUFFIXES atk-1.0)
find_library(GIO_LIBRARY
		NAMES gio-2.0)
find_library(GOBJECT_LIBRARY
		NAMES gobject-2.0)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GTK3 DEFAULT_MSG
	GTK3_LIBRARY 
	GIO_LIBRARY 
	GOBJECT_LIBRARY
	GTK3_INCLUDE_DIR 
	GLIB_INCLUDE_DIR 
	GLIBCONFIG_INCLUDE_DIR
	PANGO_INCLUDE_DIR 
	CAIRO_INCLUDE_DIR 
	GDK_PIXBUF_INCLUDE_DIR 
	ATK_INCLUDE_DIR)
if(GTK3_FOUND)
    set(GTK3_INCLUDE_DIRS  
		${GTK3_INCLUDE_DIR}  
		${GLIB_INCLUDE_DIR} 
		${GLIBCONFIG_INCLUDE_DIR}
		${PANGO_INCLUDE_DIR}  
		${CAIRO_INCLUDE_DIR} 
		${GDK_PIXBUF_INCLUDE_DIR} 
		${ATK_INCLUDE_DIR})
    set(GTK3_LIBRARIES  
		${GTK3_LIBRARY} 
		${GIO_LIBRARY} 
		${GOBJECT_LIBRARY})
endif()
```

## Section 10: Bonus



## Extra work

### 1. Nvidia hpc sdk
- CMakeLists.txt:
```bash
cmake_minimum_required(VERSION 3.10)
project(MyMPIProject VERSION 1.0.0)
# Find MPI with C++ components
find_package(MPI)
# Add an executable target
add_executable(my_mpi_app main.cpp)
# (Optional) Display information about the found MPI setup
message(STATUS "MPI CXX Compiler: ${MPI_CXX_COMPILER}")
message(STATUS "MPI CXX Libraries: ${MPI_CXX_LIBRARIES}")
message(STATUS "MPI CXX Include Path: ${MPI_CXX_INCLUDE_PATH}")
message(STATUS "MPI CXX Compile Flags: ${MPI_CXX_COMPILE_FLAGS}")
message(STATUS "MPI CXX Link Flags: ${MPI_CXX_LINK_FLAGS}")
```
- Screen shot:
```bash
$ cmake ..
-- The C compiler identification is NVHPC 25.3.0
-- The CXX compiler identification is NVHPC 25.3.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /home/hpjeon/sw_local/hpc_sdk/Linux_x86_64/25.3/compilers/bin/nvc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /home/hpjeon/sw_local/hpc_sdk/Linux_x86_64/25.3/compilers/bin/nvc++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found MPI_C: /home/hpjeon/sw_local/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/lib/libmpi.so (found version "3.1") 
-- Found MPI_CXX: /home/hpjeon/sw_local/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/lib/libmpi.so (found version "3.1") 
-- Found MPI: TRUE (found version "3.1")  
-- MPI CXX Compiler: /home/hpjeon/sw_local/hpc_sdk/Linux_x86_64/25.3/comm_libs/mpi/bin/mpicxx
-- MPI CXX Libraries: /home/hpjeon/sw_local/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/lib/libmpi.so
-- MPI CXX Include Path: /home/hpjeon/sw_local/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/include;/home/hpjeon/sw_local/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/include/openmpi;/home/hpjeon/sw_local/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/include/openmpi/opal/mca/hwloc/hwloc201/hwloc/include;/home/hpjeon/sw_local/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/include/openmpi/opal/mca/event/libevent2022/libevent;/home/hpjeon/sw_local/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/include/openmpi/opal/mca/event/libevent2022/libevent/include
-- MPI CXX Compile Flags: -pthread
-- MPI CXX Link Flags: -Wl,-rpath -Wl,/home/hpjeon/sw_local/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/lib -Wl,--enable-new-dtags -pthread
-- Configuring done (1.5s)
-- Generating done (0.0s)
```
