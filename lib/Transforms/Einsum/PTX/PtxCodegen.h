
#ifndef LIB_PTX_CODEGEN_H_
#define LIB_PTX_CODEGEN_H_

// Add the required forward declarations as the generated code doesn't contain the include statements
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include <sstream>

namespace mlir {
namespace einsum {

// #define switches used to add the required components from the generated code
#define GEN_PASS_DECL_PTXCODEGEN
#include "../include/Passes.h.inc"

} // namespace einsum
} // namespace mlir

#endif // LIB_PTX_CODEGEN_H_
