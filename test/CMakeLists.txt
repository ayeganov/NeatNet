# Unit tests
set(test_genome_sources "../src/genome.cpp" "../src/genes.cpp" "../src/innovation.cpp" "../src/params.cpp" "test_genome.cpp")
add_executable(test_genome ${test_genome_sources})
target_include_directories(test_genome PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

set(test_innovation_sources "../src/innovation.cpp" "../src/genes.cpp" "../src/params.cpp" "test_innovation.cpp")
add_executable(test_innovation ${test_innovation_sources})
target_include_directories(test_innovation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

set(test_xor_sources "../src/genalg.cpp" "test_xor.cpp" "../src/genes.cpp" "../src/genome.cpp" "../src/innovation.cpp" "../src/phenotype.cpp" "../src/species.cpp" "../src/params.cpp")
set(xor_params_json "./xor_params.json")
file(COPY ${xor_params_json} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_executable(test_xor ${test_xor_sources})
target_include_directories(test_xor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

set(test_params_sources "../src/params.cpp" "test_params.cpp")
set(test_params_json "./test_params.json")
file(COPY ${test_params_json} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_executable(test_params ${test_params_sources})
target_include_directories(test_params PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

if(NOT DEFINED NO_OPENCV)
  set(test_visualize_sources "test_visualize.cpp" "../src/netvisualize.cpp" "../src/phenotype.cpp" "../src/params.cpp" "../src/genome.cpp" "../src/innovation.cpp" "../src/genes.cpp")
  set(test_images "./test_image.png")
  file(COPY ${test_images} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  add_executable(test_visualize ${test_visualize_sources})
  target_link_libraries(test_visualize ${OpenCV_LIBS})
  target_include_directories(test_visualize PUBLIC ${OpenCV_INCLUDE_DIRS})
  target_include_directories(test_visualize PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
  enable_asan(test_visualize)
endif()


set(test_serialize_sources "test_serialize.cpp" "../src/phenotype.cpp" "../src/params.cpp" "../src/genome.cpp" "../src/innovation.cpp" "../src/genes.cpp" "../src/serialize.cpp")
add_executable(test_serialize ${test_serialize_sources})
target_include_directories(test_serialize PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)


add_test(test_genome test_genome)
add_test(test_innovation test_innovation)
add_test(test_xor test_xor)
add_test(test_serialize test_serialize)
add_test(test_params test_params)
