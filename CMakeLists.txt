cmake_minimum_required(VERSION 3.20.0)

project(mlir-einsum LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
include(AddLLVM)
include(TableGen)

message(STATUS "MLIR_DIR = ${MLIR_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR}/externals/llvm-project)

add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(tests)

option(EINSUM_ENABLE_PYTHON_BINDINGS "Enable building Python bindings" ON)

if(EINSUM_ENABLE_PYTHON_BINDINGS)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )

  if(PYBIND11_CMAKE_DIR)
    message(STATUS "Found pybind11 CMake dir: ${PYBIND11_CMAKE_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
  endif()

  find_package(pybind11 2.6 CONFIG)

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddMLIRPython)

  add_subdirectory(python)

  message(STATUS "Einsum: Python bindings enabled.")
endif()
