cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(example_app)

find_package(Torch REQUIRED)
find_package(PkgConfig REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(neuron_op SHARED IMPORTED)
set_target_properties(neuron_op PROPERTIES IMPORTED_LOCATION "${CMAKE_PREFIX_PATH}/lib/libneuron_op.so")

add_library(tokenizers SHARED IMPORTED)
set_target_properties(tokenizers PROPERTIES IMPORTED_LOCATION "${CMAKE_PREFIX_PATH}/../tokenizers_binding/lib/libtokenizers.so")

add_executable(example_app utils.cpp example_app.cpp)

target_link_libraries(example_app "${TORCH_LIBRARIES}")
target_link_libraries(example_app tokenizers)
target_link_libraries(example_app neuron_op)

set_property(TARGET example_app PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
