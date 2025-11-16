# lib/Dialect/CMakeLists.txt
# ===============================================
# Einsum Dialect TableGen + Library Build
# ===============================================

set(LLVM_TARGET_DEFINITIONS EinsumDialect.td)

mlir_tablegen(EinsumDialect.h.inc -gen-dialect-decls -dialect=einsum)
mlir_tablegen(EinsumDialect.cpp.inc -gen-dialect-defs -dialect=einsum)

# Generate op declarations and definitions (empty until ops are added)
#mlir_tablegen(EinsumOps.h.inc -gen-op-decls)
#mlir_tablegen(EinsumOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(MLIREinsumIncGen)

add_dependencies(mlir-headers MLIREinsumIncGen)

add_mlir_doc(EinsumDialect EinsumDialect Einsum/ -gen-dialect-doc)

# ---- Build Dialect Library ----
add_mlir_dialect_library(MLIREinsum
  EinsumDialect.cpp    # You must create this file
  #EinsumOps.cpp        # You must create this file

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/lib/Dialect/Einsum

  LINK_LIBS PUBLIC
  MLIRIR
  )
