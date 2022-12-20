cmake_minimum_required(VERSION 3.2)

project(root)

OPTION(BUILD_TESTS "Build with tests" ON)

MESSAGE(INFO "--------------------------------")
MESSAGE(STATUS "Build with tests: ${BUILD_TESTS}")
MESSAGE(INFO "--------------------------------")

set(CMAKE_CXX_STANDARD 17)
if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long")
endif()

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(BOAT_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(BOAT_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(BOAT_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

set(BOAT_HEADERS_DIR ${PROJECT_SOURCE_DIR}/src/)

include_directories(${BOAT_INSTALL_INCLUDE_DIR})
include_directories(${BOAT_HEADERS_DIR})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(thirdparty)
add_subdirectory(src)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
