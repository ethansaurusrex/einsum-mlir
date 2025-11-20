# python/CMakeLists.txt

# python/CMakeLists.txt

# Set the source directory for your TableGen files
set(EINSUM_TD_DIR "${CMAKE_SOURCE_DIR}/lib/Dialect/Einsum")

message(STATUS "Current CMAKE_CURRENT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Current EINSUM_TD_DIR ${EINSUM_TD_DIR}")

################################################################################
# 1. Pure Python Sources (The Root Target)
################################################################################
declare_mlir_python_sources(EinsumPythonBindings
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    einsum_mlir/__init__.py
    einsum_mlir/dialects/__init__.py
    einsum_mlir/dialects/einsum.py
)

################################################################################
# 2. ODS Wrapper Generation (Adds to the Root Target)
################################################################################
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT EinsumPythonBindings
  ROOT_DIR "${EINSUM_TD_DIR}"
  TD_FILE EinsumOps.td
  SOURCES
    EinsumOps.td
  DIALECT_NAME einsum
  DEPENDS MLIREinsumOpsIncGen
)

################################################################################
# 3. C++ Extension
################################################################################
declare_mlir_python_extension(EinsumPythonExtension.Extension
  MODULE_NAME _einsum
  ADD_TO_PARENT EinsumPythonBindings
  SOURCES
    EinsumModule.cpp
  EMBED_CAPI_LINK_LIBS
    MLIREinsumCAPI
    MLIRPythonCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# 4. Build the Package
################################################################################

add_mlir_python_modules(EinsumPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages/einsum_mlir"
  INSTALL_PREFIX "python_packages/einsum_mlir"
  DECLARED_SOURCES
    EinsumPythonBindings
  COMMON_CAPI_LINK_LIBS
    MLIRPythonSources
)

message(STATUS "Building python_packages in: ${CMAKE_BINARY_DIR}/python_packages")
