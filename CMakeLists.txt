cmake_minimum_required(VERSION 2.8)
project(NeatNet)

add_definitions(-std=c++14)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Build the library
set(SRC_FILES src/genalg.cpp src/genome.cpp src/netvisualize.cpp src/phenotype.cpp src/species.cpp src/genes.cpp src/innovation.cpp src/params.cpp src/serialize.cpp)

set(INCLUDE_FILES include/genalg.h include/genome.h include/json.hpp include/params.h include/serialize.h include/utils.h include/genes.h include/innovation.h include/netvisualize.h include/phenotype.h include/species.h)

enable_testing()
add_subdirectory(test)

set(VERSION 1.0.0)
add_library(NeatNet_${VERSION} SHARED ${SRC_FILES} ${INCLUDE_FILES})

install(FILES ${INCLUDE_FILES} DESTINATION include/neatnet)
install(TARGETS NeatNet_${VERSION} LIBRARY DESTINATION lib NAMELINK_ONLY)
