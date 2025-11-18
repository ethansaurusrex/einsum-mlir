#include "lib/Conversion/EinsumToLinalg/EinsumToLinalg.h"

#include "lib/Dialect/Einsum/EinsumOps.h"
#include "lib/Dialect/Einsum/EinsumTypes.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::einsum {

#define GEN_PASS_DEF_EINSUMTOLINALG
#include "lib/Conversion/EinsumToLinalg/EinsumToLinalg.h.inc"

struct EinsumToLinalg : impl::EinsumToLinalgBase<EinsumToLinalg> {
  using EinsumToLinalgBase::EinsumToLinalgBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    // TODO: implement lowering from einsum.ll -> linalg.generic
    // Steps will include:
    // 1. Define a ConversionTarget marking einsum.ll ops as illegal.
    // 2. Define a TypeConverter converting NamedAxesTensorType -> RankedTensorType.
    // 3. Add OpConversionPattern(s) for einsum.ll ops.
    // 4. Apply the partial conversion on the module.
  }
};

}  // namespace mlir::einsum
