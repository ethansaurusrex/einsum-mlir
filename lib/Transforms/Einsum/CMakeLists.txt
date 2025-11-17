# lib/Transforms/Einsum/CMakeLists.txt
# ===============================================
# Einsum Transforms
# ===============================================

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Einsum)

add_public_tablegen_target(MLIREinsumPasses)

add_mlir_doc(Passes EinsumPasses ./ -gen-dialect-doc)

# ---- Build Library ----
add_mlir_library(EinsumPasses
  EinsumHLToLL.cpp  
  
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/lib/Transform/Einsum

  DEPENDS
  MLIREinsum
  MLIREinsumPasses
  
  LINK_LIBS PUBLIC
)
