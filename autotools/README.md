## Using Autotools
- References
  - https://earthly.dev/blog/autoconf
  - https://www.gnu.org/software/autoconf/manual/autoconf-2.66/html_node/C_002b_002b-Compiler.html

### Component of Autotools
- autoconf
  - Written in M4sh, using m4 macros
  - Write configure.ac, defining release name, version, compiler, .... then autoconf converts this to configure script
  - Basically collects system information to populate Makefile.in
- automake
  - Makefile.am script creates Makefile.in
  - Primaries: such as `_PROGRAMS` suffix. Gives knolwedges of your program to automake
- aclocal
  - Generates m4 macros
  - aclocal runs before running autoconf

### Installing autotools
- RHEL: sudo yum install autoconf automake
- Ubuntu: sudo apt install autoconf autotools-dev

### A sample demo:
- main.cxx:
```cpp
#include <iostream>
int main(int argc, char* argv[])
{
  std::cout << "Hello world\n";
  return 0;
}
```
- configure.ac:
```bash
AC_INIT([helloworld], [3.14], [abc@iam.com]) # name of application, version, maintainer email
AM_INIT_AUTOMAKE  # we use automake
AC_PROG_CXX  # we use C++ compiler. For C, AC_PROG_CC
AC_CONFIG_FILES([Makefile]) # Will search Makefile.in
AC_OUTPUT
```
- Makefile.am
```bash
AUTOMAKE_OPTIONS = foreign # source is at root
bin_PROGRAMS = myExe # my application name
myExe_SOURCES = main.cxx # source files
```
  - In this example , `_PROGRAMS` and `_SOURCES` are primaries
- Demo:
```bash
$ ls
configure.ac  main.cxx  Makefile.am
$ aclocal
$ ls
aclocal.m4  autom4te.cache  configure.ac  main.cxx  Makefile.am
$ autoconf
$ ls
aclocal.m4  autom4te.cache  configure  configure.ac  main.cxx  Makefile.am
$ automake --add-missing
configure.ac:2: installing './install-sh'
configure.ac:2: installing './missing'
Makefile.am: installing './depcomp'
$ ls
aclocal.m4      configure     depcomp     main.cxx     Makefile.in
autom4te.cache  configure.ac  install-sh  Makefile.am  missing
$ ./configure --prefix=/home/hpjeon/sw_local/myExe
checking for a BSD-compatible install... /usr/bin/install -c
checking whether build environment is sane... yes
checking for a race-free mkdir -p... /usr/bin/mkdir -p
checking for gawk... no
checking for mawk... mawk
checking whether make sets $(MAKE)... yes
checking whether make supports nested variables... yes
checking for g++... g++
checking whether the C++ compiler works... yes
checking for C++ compiler default output file name... a.out
checking for suffix of executables... 
checking whether we are cross compiling... no
checking for suffix of object files... o
checking whether the compiler supports GNU C++... yes
checking whether g++ accepts -g... yes
checking for g++ option to enable C++11 features... none needed
checking whether make supports the include directive... yes (GNU style)
checking dependency style of g++... gcc3
checking that generated files are newer than configure... done
configure: creating ./config.status
config.status: creating Makefile
config.status: executing depfiles commands
$ make
g++ -DPACKAGE_NAME=\"helloworld\" -DPACKAGE_TARNAME=\"helloworld\" -DPACKAGE_VERSION=\"3.14\" -DPACKAGE_STRING=\"helloworld\ 3.14\" -DPACKAGE_BUGREPORT=\"abc@iam.com\" -DPACKAGE_URL=\"\" -DPACKAGE=\"helloworld\" -DVERSION=\"3.14\" -I.     -g -O2 -MT main.o -MD -MP -MF .deps/main.Tpo -c -o main.o main.cxx
mv -f .deps/main.Tpo .deps/main.Po
g++  -g -O2   -o myExe main.o  
$ make install
make[1]: Entering directory '/home/hpjeon/hw/class/earthly_autotools/ex'
 /usr/bin/mkdir -p '/home/hpjeon/sw_local/myExe/bin'
  /usr/bin/install -c myExe '/home/hpjeon/sw_local/myExe/bin'
make[1]: Nothing to be done for 'install-data-am'.
make[1]: Leaving directory '/home/hpjeon/hw/class/earthly_autotools/ex'
$ /home/hpjeon/sw_local/myExe/bin/myExe 
Hello world
$ make dist # this will generate helloworld-3.14.tar.gz for distribution
```
- The content of helloworld-3.14.tar.gz:
```bash
$ ls
aclocal.m4  configure.ac  install-sh  Makefile.am  missing
configure   depcomp       main.cxx    Makefile.in
```
