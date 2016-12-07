cmake_minimum_required(VERSION 2.8)
project(NeatNet)

add_definitions(-std=c++11)

include_directories(include)
set(test_genome_sources "./src/genome.cpp" "./src/genes.cpp" "./src/innovation.cpp" "./test/test_genome.cpp")
add_executable(test_genome ${test_genome_sources})

set(test_innovation_sources "./src/innovation.cpp" "./src/genes.cpp" "./test/test_innovation.cpp")
add_executable(test_innovation ${test_innovation_sources})

set(test_genalg_sources "./src/genalg.cpp" "./test/test_genalg.cpp" "./src/genes.cpp" "./src/genome.cpp" "./src/innovation.cpp" "./src/phenotype.cpp" "./src/species.cpp")
add_executable(test_genalg ${test_genalg_sources})

#add_library(NeatNet SHARED src/genome.cpp)
#add_executable(test_neat_net ${TEST_SOURCES})
