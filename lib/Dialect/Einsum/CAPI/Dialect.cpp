#include "lib/Dialect/Einsum/CAPI/Dialect.h"
#include "mlir/CAPI/Registration.h"
#include "lib/Dialect/Einsum/EinsumDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Einsum, einsum,
                                       mlir::einsum::EinsumDialect)
