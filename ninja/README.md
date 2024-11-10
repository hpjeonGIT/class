## Ninja build system

### Make vs ninja
- Sample make file for MD code
```
#Makfile to compile MD example
.SUFFIXES: .o .cxx
CXX = g++
FLAGS = -g -ansi
LIB = -lm
FOMP = -fopenmp
INCLUDE =
OBJ = main.o force.o verlet.o print.o
TRG = md_run
${TRG}: ${OBJ}
        ${CXX} ${FLAGS} ${FOMP} -o ${TRG} ${OBJ} ${LIB}
main.o:main.cxx
        ${CXX} ${FLAGS} ${FOMP} -c $<
force.o:force.cxx
        ${CXX} ${FLAGS} ${FOMP} -c $<
.cxx.o:
        ${CXX} ${FLAGS} -c $<
clean:
        rm ${OBJ} ${TRG} *~
```
- Ninja version. Save as `build.ninja`
```
pool link_pool
  depth = 1
rule cxx
  deps = gcc
  depfile = $out.d
  command = g++ -g $cxxflags -c $in -o $out
rule link 
  pool = link_pool
  command = g++ $ldflags $in -o $out
outdir = build_ninja
ldflags = -lm -fopenmp
cxxflags = -std=c++17 -ansi -fopenmp
build $outdir/main.o: cxx main.cxx
build $outdir/force.o: cxx force.cxx
build $outdir/verlet.o: cxx verlet.cxx
build $outdir/print.o: cxx print.cxx
build $outdir/md_run: link $outdir/main.o $outdir/force.o $outdir/verlet.o $outdir/print.o
```
- Run `ninja`

## Why no pattern matching in ninja?
- For speed-up
  - Ref: https://www.youtube.com/watch?v=AkGt0fsQ17o
