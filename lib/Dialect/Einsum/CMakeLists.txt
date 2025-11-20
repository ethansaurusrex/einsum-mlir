# lib/Dialect/Einsum/CMakeLists.txt
# ===============================================
# Einsum Dialect TableGen + Library Build
# ===============================================

set(LLVM_TARGET_DEFINITIONS EinsumDialect.td)
mlir_tablegen(EinsumDialect.h.inc -gen-dialect-decls -dialect=einsum)
mlir_tablegen(EinsumDialect.cpp.inc -gen-dialect-defs -dialect=einsum)

set(LLVM_TARGET_DEFINITIONS EinsumTypes.td)
mlir_tablegen(EinsumTypes.h.inc -gen-typedef-decls -typedefs-dialect=einsum)
mlir_tablegen(EinsumTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=einsum)

set(LLVM_TARGET_DEFINITIONS EinsumOps.td)
mlir_tablegen(EinsumOps.h.inc -gen-op-decls)
mlir_tablegen(EinsumOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(MLIREinsumOpsIncGen)

add_dependencies(mlir-headers MLIREinsumOpsIncGen)

add_mlir_doc(EinsumDialect EinsumDialect Einsum/ -gen-dialect-doc)

# ---- Build Dialect Library ----
add_mlir_dialect_library(MLIREinsum
  EinsumDialect.cpp    
  EinsumOps.cpp        

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/lib/Dialect/Einsum

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  )
