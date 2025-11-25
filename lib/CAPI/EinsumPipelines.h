#ifndef LIB_CAPI_EINSUMPIPELINES_H
#define LIB_CAPI_EINSUMPIPELINES_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void mlirRegisterEinsumPipelines(void);  

#ifdef __cplusplus
}
#endif

#endif // LIB_CAPI_EINSUMDIALECT_H
