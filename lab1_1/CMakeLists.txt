cmake_minimum_required(VERSION 3.17)
project(MyMPIProject)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)

add_executable(my_mpi_app main.cpp)

target_link_libraries(my_mpi_app PRIVATE MPI::MPI_CXX)
