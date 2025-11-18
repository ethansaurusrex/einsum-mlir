#ifndef LIB_CONVERSION_EINSUMTOLINALG_EINSUMTOLINALG_H_
#define LIB_CONVERSION_EINSUMTOLINALG_EINSUMTOLINALG_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

// Extra includes needed for dependent dialects
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"     // from @llvm-project

namespace mlir::einsum {

#define GEN_PASS_DECL
#include "lib/Conversion/EinsumToLinalg/EinsumToLinalg.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Conversion/EinsumToLinalg/EinsumToLinalg.h.inc"

}  // namespace mlir::einsum

#endif  // LIB_CONVERSION_EINSUMTOLINALG_EINSUMTOLINALG_H_
