# lib/Conversion/EinsumToLinalg/CMakeLists.txt
# ===============================================
# EinsumToLinalg Conversion
# ===============================================

set(LLVM_TARGET_DEFINITIONS EinsumToLinalg.td)
mlir_tablegen(EinsumToLinalg.h.inc -gen-pass-decls -name EinsumToLinalg)
mlir_tablegen(EinsumToLinalg.capi.h.inc -gen-pass-capi-header --prefix EinsumToLinalg)
mlir_tablegen(EinsumToLinalg.capi.cpp.inc -gen-pass-capi-impl --prefix EinsumToLinalg)
add_dependencies(mlir-headers MLIREinsumOpsIncGen)
add_public_tablegen_target(EinsumToLinalgPassIncGen)
add_mlir_doc(EinsumToLinalg EinsumToLinalg EinsumToLinalg/ -gen-pass-doc)

add_mlir_library(MLIREinsumToLinalg
  EinsumToLinalg.cpp

  ${PROJECT_SOURCE_DIR}/lib/Conversion/EinsumToLinalg/
  ADDITIONAL_HEADER_DIRS

  DEPENDS
  EinsumToLinalgPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIREinsum
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
  )
