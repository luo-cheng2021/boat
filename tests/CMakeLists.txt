cmake_minimum_required(VERSION 3.2)
project(boat_tests)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(lib/googletest)
add_subdirectory(lib/gflags)

include_directories(${BOAT_HEADERS_DIR})
include_directories(lib/googletest/googletest/include)

file(GLOB_RECURSE TEST_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

set(SOURCE_FILES lib/main.cpp ${TEST_SOURCE_FILES})

add_executable(boat_tests ${SOURCE_FILES})
target_link_libraries(boat_tests boat gtest)
install(TARGETS boat_tests DESTINATION bin)

add_executable(bench bench.cpp)
target_link_libraries(bench boat gflags::gflags)
