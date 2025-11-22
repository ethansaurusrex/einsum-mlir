#ifndef LIB_CAPI_DIALECT_H
#define LIB_CAPI_DIALECT_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Einsum, einsum);

// === Type C-API ===

MLIR_CAPI_EXPORTED bool mlirTypeIsAEinsumNamedAxesTensorType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirEinsumNamedAxesTensorTypeGet(
    MlirContext ctx, MlirType tensorType, MlirAttribute axisNames);  

#ifdef __cplusplus
}
#endif

#endif // LIB_CAPI_DIALECT_H
