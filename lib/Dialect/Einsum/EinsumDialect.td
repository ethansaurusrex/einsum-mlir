#ifndef LIB_DIALECT_EINSUM_EINSUMDIALECT_TD_
#define LIB_DIALECT_EINSUM_EINSUMDIALECT_TD_

// lib/Dialect/Einsum/EinsumOps.td

include "mlir/IR/DialectBase.td"

def EinsumDialect : Dialect {
  let name = "einsum";

  let summary = "Dialect for defining einsum operations";

  let description = [{
      The dialect allows users to define einsum computations with tensors
      }];

  let cppNamespace = "::mlir::einsum";

  let useDefaultTypePrinterParser = 1;  
}

#endif // LIB_DIALECT_EINSUM_EINSUMDIALECT_TD_