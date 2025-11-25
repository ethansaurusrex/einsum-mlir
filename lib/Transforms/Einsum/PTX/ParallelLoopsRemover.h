
#ifndef LIB_TRANSFORMS_EINSUM_PTX_PARALLEL_LOOPS_REMOVER_H_
#define LIB_TRANSFORMS_EINSUM_PTX_PARALLEL_LOOPS_REMOVER_H_

// Add the required forward declarations as the generated code doesn't contain the include statements
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include <vector>

namespace mlir {
namespace einsum {

// #define switches used to add the required components from the generated code
#define GEN_PASS_DECL_PARALLELLOOPSREMOVER
#include "lib/Transforms/Einsum/PTX/Passes.h.inc"

} // namespace einsum
} // namespace mlir

#endif // LIB_TRANSFORMS_EINSUM_PTX_PARALLEL_LOOPS_REMOVER_H_
