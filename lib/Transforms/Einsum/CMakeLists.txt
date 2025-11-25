# lib/Transforms/Einsum/CMakeLists.txt
# ===============================================
# Einsum Transforms
# ===============================================

set(LLVM_TARGET_DEFINITIONS Passes.td)
set(PASS_C_API_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR})
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Einsum)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Einsum)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Einsum)
add_public_tablegen_target(MLIREinsumPassesIncGen)

add_dependencies(mlir-headers MLIREinsumPassesIncGen)

add_mlir_doc(Passes EinsumPasses ./ -gen-dialect-doc)

# ---- Build Library ----
add_subdirectory(PTX)

add_mlir_library(MLIREinsumPasses
  EinsumHLToLL.cpp  
  Pipelines.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/lib/Transform/Einsum
  ${PROJECT_SOURCE_DIR}/lib/Transform/Einsum/PTX

  DEPENDS
  MLIREinsum
  MLIREinsumPtxPassesIncGen
  
  LINK_LIBS PUBLIC
  EinsumPTX
)


