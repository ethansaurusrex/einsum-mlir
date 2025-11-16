#include "lib/Dialect/Einsum/EinsumDialect.h"

#include "lib/Dialect/Einsum/EinsumTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "lib/Dialect/Einsum/EinsumDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Einsum/EinsumTypes.cpp.inc"

namespace mlir {
  namespace einsum {
    void EinsumDialect::initialize() {
    addTypes<
      #define GET_TYPEDEF_LIST
      #include "lib/Dialect/Einsum/EinsumTypes.cpp.inc"
      >();
    };
  } // einsum
} // namespace mlir
