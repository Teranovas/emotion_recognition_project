cmake_minimum_required(VERSION 3.12)
project(emotion_cpp)

set(CMAKE_CXX_STANDARD 17)

# pybind11 설정
find_package(pybind11 REQUIRED)

# 모듈 빌드
pybind11_add_module(cpp_module bindings.cpp emotion_utils.cpp)

target_link_libraries(cpp_module PRIVATE pybind11::module)
