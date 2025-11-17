#ifndef LIB_TRANSFORMS_EINSUM_EINSUMHLTOLL_H_
#define LIB_TRANSFORMS_EINSUM_EINSUMHLTOLL_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::einsum {

#define GEN_PASS_DECL_EINSUMHLTOLL
#include "lib/Transforms/Einsum/Passes.h.inc"

} // namespace mlir::einsum

#endif // LIB_TRANSFORMS_EINSUM_EINSUMHLTOLL_H_
