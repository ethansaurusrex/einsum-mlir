#ifndef LIB_TRANSFORMS_EINSUM_PASSES_H_
#define LIB_TRANSFORMS_EINSUM_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir::einsum {

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/Einsum/Passes.h.inc"

} // namespace mlir::einsum

#endif // LIB_TRANSFORMS_EINSUMPASSES_H_
