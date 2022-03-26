cmake_minimum_required(VERSION 3.19)
project(cg_2_2 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cg_2_2 main.cu)

set_target_properties(
        cg_2_2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)