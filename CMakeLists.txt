cmake_minimum_required(VERSION 3.20.0)

project(mlir-einsum LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
include(AddLLVM)
include(TableGen)
include(HandleLLVMOptions)

message(STATUS "MLIR_DIR = ${MLIR_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR}/externals/llvm-project)

add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(tests)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
  # Note: for EXTERNAL_PROJECT_BUILD this must be set from the command line.
  if(NOT MLIR_PYTHON_PACKAGE_PREFIX)
    set(MLIR_PYTHON_PACKAGE_PREFIX "einsum_mlir" CACHE STRING "" FORCE)
  endif()
  if(NOT MLIR_BINDINGS_PYTHON_INSTALL_PREFIX)
    set(MLIR_BINDINGS_PYTHON_INSTALL_PREFIX "python_packages/${MLIR_PYTHON_PACKAGE_PREFIX}" CACHE STRING "" FORCE)
  endif()
  add_subdirectory(python)
endif()

