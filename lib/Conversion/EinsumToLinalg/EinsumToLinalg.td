#ifndef LIB_CONVERSION_EINSUMTOLINALG_EINSUMTOLINALG_TD_
#define LIB_CONVERSION_EINSUMTOLINALG_EINSUMTOLINALG_TD_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// EinsumToLinalg pass declaration
//===----------------------------------------------------------------------===//

def EinsumToLinalg : Pass<"einsum-ll-to-linalg"> {
  let summary = "Lower `einsum.ll` to linalg.generic";

  let description = [{
    This pass lowers the `einsum.ll` dialect to `linalg.generic`.
    It also converts `NamedAxesTensorType` to `RankedTensorType`.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::einsum::EinsumDialect",
  ];
}

#endif  // LIB_CONVERSION_EINSUMTOLINALG_EINSUMTOLINALG_TD_
