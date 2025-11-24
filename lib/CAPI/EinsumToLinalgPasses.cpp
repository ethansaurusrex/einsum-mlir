#include "mlir/CAPI/Pass.h"
#include "lib/Conversion/EinsumToLinalg/EinsumToLinalg.h"
#include "mlir/Pass/Pass.h"

#include "lib/Conversion/EinsumToLinalg/EinsumToLinalg.capi.h.inc"

using namespace mlir::einsum;

#ifdef __cplusplus
extern "C" {
#endif

#include "lib/Conversion/EinsumToLinalg/EinsumToLinalg.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
