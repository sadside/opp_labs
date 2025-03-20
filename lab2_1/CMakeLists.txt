cmake_minimum_required(VERSION 3.17)
project(MyOpenMPProject)

set(CMAKE_CXX_STANDARD 17)

# Если вы на macOS, указываем компиляторы из brew LLVM, поддерживающие OpenMP.
if(APPLE)
    # Для Apple Silicon (aarch64):
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
endif()

find_package(OpenMP REQUIRED)

add_executable(my_openmp_app main.cpp)

target_link_libraries(my_openmp_app PRIVATE OpenMP::OpenMP_CXX)
