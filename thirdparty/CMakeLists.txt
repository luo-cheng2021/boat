cmake_minimum_required(VERSION 3.2)
#set(EXEC ON CACHE BOOL "" FORCE)
set(ASMJIT_STATIC ON CACHE BOOL "" FORCE)

add_subdirectory(asmjit EXCLUDE_FROM_ALL)
