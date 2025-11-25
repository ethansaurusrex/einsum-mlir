
#ifndef LIB_LINEAR_OFFSET_H_
#define LIB_LINEAR_OFFSET_H_

// Add the required forward declarations as the generated code doesn't contain the include statements
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace einsum {

// #define switches used to add the required components from the generated code
#define GEN_PASS_DECL_LINEAROFFSET
#include "../include/Passes.h.inc"

} // namespace einsum
} // namespace mlir

#endif // LIB_LINEAR_OFFSET_H_
