# python/CMakeLists.txt
include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

declare_mlir_python_sources(EinsumPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT EinsumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/einsum_mlir"
  TD_FILE dialects/EinsumOps.td
  SOURCES
    __init__.py
    dialects/__init__.py
    dialects/einsum.py
    #einsum_mlir/dialects/_ods_common.py
  DIALECT_NAME einsum
  DEPENDS MLIREinsumOpsIncGen
)

declare_mlir_python_extension(EinsumPythonSources.Pybind11Extension
  MODULE_NAME _einsum
  ADD_TO_PARENT EinsumPythonSources
  SOURCES
    EinsumModule.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIREinsumCAPI
  PYTHON_BINDINGS_LIBRARY
    pybind11
)

add_mlir_python_common_capi_library(EinsumPythonCAPI
  INSTALL_COMPONENT EinsumPythonModules
  INSTALL_DESTINATION python_packages/einsum/einsum_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_packages/einsum/einsum_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    EinsumPythonSources
)

add_mlir_python_modules(EinsumPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages/einsum/einsum_mlir"
  INSTALL_PREFIX "python_packages/einsum/einsum_mlir"
  DECLARED_SOURCES
    EinsumPythonSources
  COMMON_CAPI_LINK_LIBS
    EinsumPythonCAPI
)

