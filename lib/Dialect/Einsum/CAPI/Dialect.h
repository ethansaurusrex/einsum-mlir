#ifndef LIB_DIALECT_EINSUM_CAPI_DIALECT_H
#define LIB_DIALECT_EINSUM_CAPI_DIALECT_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Einsum, einsum);

#ifdef __cplusplus
}
#endif

#endif // LIB_DIALECT_EINSUM_CAPI_DIALECT_H
