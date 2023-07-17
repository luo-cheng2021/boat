cmake_minimum_required(VERSION 3.2)
project(boat)

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(boat STATIC ${SOURCE_FILES})
add_compile_definitions(DNNL_CPU_THREADING_RUNTIME=DNNL_RUNTIME_TBB)
#add_compile_definitions(DNNL_CPU_THREADING_RUNTIME=DNNL_RUNTIME_OMP)
target_compile_definitions(boat PRIVATE BOAT_EXPORT)
target_include_directories(boat INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(boat asmjit::asmjit tbb)
#target_link_libraries(boat asmjit::asmjit gomp)
install(TARGETS boat DESTINATION ${BOAT_INSTALL_BIN_DIR})
#install(FILES gemm.h DESTINATION ${BOAT_INSTALL_INCLUDE_DIR})
