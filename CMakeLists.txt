cmake_minimum_required(VERSION 2.8)
project(NeatNet)

add_definitions(-std=c++14)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Unit tests
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


set(test_serialize_sources "./test/test_serialize.cpp" "./src/phenotype.cpp" "./src/params.cpp" "./src/genome.cpp" "./src/innovation.cpp" "./src/genes.cpp" "./src/serialize.cpp")
add_executable(test_serialize ${test_serialize_sources})


# Build the library
set(SRC_FILES src/genalg.cpp src/genome.cpp src/netvisualize.cpp src/phenotype.cpp src/species.cpp src/genes.cpp src/innovation.cpp src/params.cpp src/serialize.cpp)

set(INCLUDE_FILES include/genalg.h include/genome.h include/json.hpp include/params.h include/serialize.h include/utils.h include/genes.h include/innovation.h include/netvisualize.h include/phenotype.h include/species.h)

set(VERSION 1.0.0)
add_library(NeatNet_${VERSION} SHARED ${SRC_FILES} ${INCLUDE_FILES})
