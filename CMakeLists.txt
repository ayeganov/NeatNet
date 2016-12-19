cmake_minimum_required(VERSION 2.8)
project(NeatNet)

add_definitions(-std=c++14)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(include)
set(test_genome_sources "./src/genome.cpp" "./src/genes.cpp" "./src/innovation.cpp" "./src/params.cpp" "./test/test_genome.cpp")
add_executable(test_genome ${test_genome_sources})

set(test_innovation_sources "./src/innovation.cpp" "./src/genes.cpp" "./src/params.cpp" "./test/test_innovation.cpp")
add_executable(test_innovation ${test_innovation_sources})

set(test_xor_sources "./src/genalg.cpp" "./test/test_xor.cpp" "./src/genes.cpp" "./src/genome.cpp" "./src/innovation.cpp" "./src/phenotype.cpp" "./src/species.cpp" "./src/params.cpp" "./src/netvisualize.cpp")
add_executable(test_xor ${test_xor_sources})
target_link_libraries(test_xor ${OpenCV_LIBS})

set(test_params_sources "./src/params.cpp" "./test/test_params.cpp")
set(test_params_json "./test/test_params.json")
file(COPY ${test_params_json} DESTINATION ${CMAKE_BINARY_DIR})
add_executable(test_params ${test_params_sources})

set(test_visualize_sources "./test/test_visualize.cpp" "./src/netvisualize.cpp" "./src/phenotype.cpp" "./src/params.cpp" "./src/genome.cpp" "./src/innovation.cpp" "./src/genes.cpp")
set(test_images "./test/test_image.png")
file(COPY ${test_images} DESTINATION ${CMAKE_BINARY_DIR})
add_executable(test_visualize ${test_visualize_sources})
target_link_libraries(test_visualize ${OpenCV_LIBS})

#add_library(NeatNet SHARED src/genome.cpp)
#add_executable(test_neat_net ${TEST_SOURCES})
