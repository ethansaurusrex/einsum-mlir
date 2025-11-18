#include "lib/Conversion/EinsumToLinalg/EinsumToLinalg.h"

#include "lib/Dialect/Einsum/EinsumOps.h"
#include "lib/Dialect/Einsum/EinsumTypes.h"

#include "lib/Dialect/Einsum/EinsumOps.h"
#include "lib/Dialect/Einsum/EinsumTypes.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::einsum {

#define GEN_PASS_DEF_EINSUMTOLINALG
#include "lib/Conversion/EinsumToLinalg/EinsumToLinalg.h.inc"

class EinsumToLinalgTypeConverter : public TypeConverter {
public:
  explicit EinsumToLinalgTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });

    // convert NamedAxesTensorType -> RankedTensorType
    addConversion([](NamedAxesTensorType natType) -> Type {
      return natType.getTensorType();
    });
  }
};
  
struct ConvertEinsumLL : public OpConversionPattern<EinsumLL> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      EinsumLL op, EinsumLL::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // At this point, adaptor.getInputs() returns operands with converted types
    // (i.e., NamedAxesTensorType has already been lowered to RankedTensorType).
    
    // Example: retrieve converted operand types
    SmallVector<RankedTensorType, 4> inputTensors;
    for (Value v : adaptor.getInputs()) {
      auto rankedTy = dyn_cast<RankedTensorType>(v.getType());
      if (!rankedTy)
        return op.emitOpError("expected RankedTensorType after type conversion");
      inputTensors.push_back(rankedTy);
    }
    
    // TODO: create the equivalent linalg.generic op here using rewriter.
    // Use rewriter.create<linalg::GenericOp>(...) and map the operands/results.

    // Replace original op
    //rewriter.replaceOp(op, /* new linalg results */);

    return success();
  }
};

void populateEinsumLLToLinalgPatterns(RewritePatternSet &patterns,
                                      TypeConverter &typeConverter) {
  patterns.add<ConvertEinsumLL>(typeConverter, patterns.getContext());
}
  
struct EinsumToLinalg : impl::EinsumToLinalgBase<EinsumToLinalg> {
  using EinsumToLinalgBase::EinsumToLinalgBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addIllegalDialect<EinsumDialect>(); // dont want anymore of einsum

    RewritePatternSet patterns(context);
    EinsumToLinalgTypeConverter typeConverter(context);
    populateEinsumLLToLinalgPatterns(patterns, typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
    // TODO: implement lowering from einsum.ll -> linalg.generic
    // Steps will include:
    // 1. Define a ConversionTarget marking einsum.ll ops as illegal.
    // 2. Define a TypeConverter converting NamedAxesTensorType -> RankedTensorType.
    // 3. Add OpConversionPattern(s) for einsum.ll ops.
    // 4. Apply the partial conversion on the module.
  }
};

}  // namespace mlir::einsum
