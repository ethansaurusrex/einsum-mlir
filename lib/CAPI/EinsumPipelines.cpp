#include "mlir/CAPI/Pass.h"
#include "lib/Transforms/Einsum/Pipelines.h"
#include "mlir/Pass/Pass.h"

using namespace mlir::einsum;

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void mlirRegisterEinsumPipelines(void) {
  registerEinsumPipelines();
}  

#ifdef __cplusplus
}
#endif
