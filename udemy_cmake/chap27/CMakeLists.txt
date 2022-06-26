cmake_minimum_required(VERSION 3.0.0)
SET(A "123" CACHE STRING "Some description")
message(${A} $CACHE{A})
