cmake_minimum_required(VERSION 3.10)

project(cuBlackhole CXX CUDA)

find_package(JPEG)

add_executable(blackhole main.cpp tracer.cu)

if (JPEG_FOUND)
    target_include_directories(blackhole PRIVATE ${JPEG_INCLUDE_DIR})
    target_link_libraries(blackhole ${JPEG_LIBRARIES})
endif ()

target_compile_options(blackhole PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_60,code=sm_60>)