cmake_minimum_required(VERSION 3.18)
project(astar_cpp LANGUAGES CXX)

# Enforce modern Python discovery policy
cmake_policy(SET CMP0148 NEW)

# Locate Python interpreter and development libraries
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Instruct pybind11 to use the modern FindPython modules
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)

# Build the pybind11 extension module
pybind11_add_module(_astar_cpp astar_cpp.cpp)

# Emit the shared library into the source directory alongside __init__.py
set_target_properties(_astar_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(_astar_cpp PRIVATE -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

# Link OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_astar_cpp PUBLIC OpenMP::OpenMP_CXX)
endif()
