#include "lib/Dialect/Einsum/EinsumDialect.h"

#include "lib/Dialect/Einsum/EinsumTypes.h"
#include "lib/Dialect/Einsum/EinsumOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"


#include "lib/Dialect/Einsum/EinsumDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Einsum/EinsumTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/Einsum/EinsumOps.cpp.inc"

namespace mlir {
  namespace einsum {
    void EinsumDialect::initialize() {
    addTypes<
      #define GET_TYPEDEF_LIST
      #include "lib/Dialect/Einsum/EinsumTypes.cpp.inc"
      >();
    addOperations<
      #define GET_OP_LIST
      #include "lib/Dialect/Einsum/EinsumOps.cpp.inc"
      >();
    };
  } // einsum
} // namespace mlir
