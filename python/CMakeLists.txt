# python/CMakeLists.txt

declare_mlir_python_sources(EinsumPythonBindings
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT MLIRPythonSources
  SOURCES
    einsum_mlir/__init__.py
    einsum_mlir/dialects/__init__.py
    einsum_mlir/dialects/einsum.py
    einsum_mlir/dialects/_ods_common.py
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT EinsumPythonBindings
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE "einsum_mlir/dialects/EinsumOps.td"
  DIALECT_NAME einsum
  DEPENDS MLIREinsumOpsIncGen
)

declare_mlir_python_extension(EinsumExtension
  MODULE_NAME _einsum
  ADD_TO_PARENT EinsumPythonBindings
  SOURCES
    EinsumModule.cpp
  EMBED_CAPI_LINK_LIBS
    MLIREinsumCAPI
    MLIRPythonCAPI
    MLIREinsum
  PRIVATE_LINK_LIBS
    LLVMSupport
)

add_mlir_python_modules(EinsumPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages"
  INSTALL_PREFIX "python_packages"
  DECLARED_SOURCES
    EinsumPythonBindings
  COMMON_CAPI_LINK_LIBS
    MLIRPythonSources
)

