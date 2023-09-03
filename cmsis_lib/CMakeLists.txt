cmake_minimum_required(VERSION 3.10)

project(cached_dilated_convolutions)

add_compile_definitions(ARM_MATH_CM7)
add_executable(cdcc cdcc_main.cpp)

include_directories(../libDaisy/Drivers/CMSIS/Include/)
include_directories(../libDaisy/Drivers/CMSIS/DSP/Include/)