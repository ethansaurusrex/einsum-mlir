#ifndef LIB_DIALECT_EINSUM_EINSUMOPS_H_
#define LIB_DIALECT_EINSUM_EINSUMOPS_H_

#include "lib/Dialect/Einsum/EinsumDialect.h"
#include "lib/Dialect/Einsum/EinsumTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"


#define GET_OP_CLASSES
#include "lib/Dialect/Einsum/EinsumOps.h.inc"

#endif // LIB_DIALECT_EINSUM_EINSUMOPS_H_

