cmake_minimum_required(VERSION 3.15)
project(pygloo)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# 查找 pybind11 包
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(gloo REQUIRED)

# 包含头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(src)
list(APPEND PYGLOO_SRCS
    main.cc
)

# 定义目标模块
# message(STATUS "pygloo srcs: ${PYGLOO_SRCS}")
pybind11_add_module(pygloo ${PYGLOO_SRCS})

# 可选：添加 include 路径
target_include_directories(pygloo PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(pygloo PRIVATE gloo)
