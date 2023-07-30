cmake_minimum_required(VERSION 3.5)
project(NeatNet)

add_definitions(-std=c++20)

function(enable_asan TARGET)
  target_compile_options(${TARGET} PRIVATE
    -fsanitize=address
    -fno-omit-frame-pointer
    -O0
    -g
  )
  target_link_options(${TARGET} PRIVATE
    -fsanitize=address
  )
endfunction()


# Build the library
if(DEFINED NO_OPENCV)
  message("Not building visualization of the library")
  set(SRC_VISUAL "")
  set(INCLUDE_VISUAL "")
else()
  message("Building with OpenCV and visualization of network functionality")
  find_package(OpenCV REQUIRED)
  set(SRC_VISUAL src/netvisualize.cpp)
  set(INCLUDE_VISUAL include/netvisualize.h)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SRC_FILES src/genalg.cpp src/genome.cpp ${SRC_VISUAL} src/phenotype.cpp src/species.cpp src/genes.cpp src/innovation.cpp src/params.cpp src/serialize.cpp)

set(INCLUDE_FILES include/genalg.h include/genome.h include/json.hpp include/params.h include/serialize.h include/utils.h include/genes.h include/innovation.h ${INCLUDE_VISUAL} include/phenotype.h include/species.h)

enable_testing()
add_subdirectory(test)

set(VERSION 1.0.0)
add_library(NeatNet_${VERSION} SHARED ${SRC_FILES} ${INCLUDE_FILES})
target_include_directories(NeatNet_${VERSION} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(NeatNet_${VERSION} PROPERTIES MACOSX_RPATH ON)
target_link_libraries(NeatNet_${VERSION} ${OpenCV_LIBS})
target_include_directories(NeatNet_${VERSION} PUBLIC ${OpenCV_INCLUDE_DIRS})

install(FILES ${INCLUDE_FILES} DESTINATION include/neatnet)
install(TARGETS NeatNet_${VERSION} LIBRARY DESTINATION lib)
