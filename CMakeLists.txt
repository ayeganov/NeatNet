cmake_minimum_required(VERSION 2.6)
project(NeatNet)

add_definitions(-std=c++11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(NeatNet SHARED src/genome.cpp)
file(GLOB TEST_SOURCES "test/*.cpp" "src/*.cpp")
add_executable(test_neat_net ${TEST_SOURCES})
