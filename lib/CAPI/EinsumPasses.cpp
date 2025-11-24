#include "mlir/CAPI/Pass.h"
#include "lib/Transforms/Einsum/Passes.h"
#include "mlir/Pass/Pass.h"

#include "lib/Transforms/Einsum/Passes.capi.h.inc"

using namespace mlir::einsum;

#ifdef __cplusplus
extern "C" {
#endif

#include "lib/Transforms/Einsum/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
