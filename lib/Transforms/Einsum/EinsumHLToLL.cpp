#include "lib/Transforms/Einsum/EinsumHLToLL.h"
#include "lib/Dialect/Einsum/EinsumOps.h"

namespace mlir::einsum {

#define GEN_PASS_DEF_EINSUMHLTOLL
#include "lib/Transforms/Einsum/Passes.h.inc"
  
} //namespace mlir::einsum
