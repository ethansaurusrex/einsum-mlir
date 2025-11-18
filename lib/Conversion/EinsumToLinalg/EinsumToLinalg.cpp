#include "lib/Conversion/EinsumToLinalg/EinsumToLinalg.h"

#include "lib/Dialect/Einsum/EinsumOps.h"
#include "lib/Dialect/Einsum/EinsumTypes.h"

#include "lib/Dialect/Einsum/EinsumOps.h"
#include "lib/Dialect/Einsum/EinsumTypes.h"
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project

#include "mlir/Support/DebugStringHelper.h" // for llvm::dbgs()

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

static Value createInitTensor(ImplicitLocOpBuilder &b, RankedTensorType outType) {
  // build the dynamic dims for tensor.empty
  SmallVector<Value, 4> dims;
  for (int64_t i = 0; i < outType.getRank(); ++i) {
    if (outType.isDynamicDim(i)) { // should all be static
      llvm_unreachable("dynamic dims not yet supported");
    }
  }

  // ceate empty output tensor
  Value empty =
    tensor::EmptyOp::create(b, outType.getShape(), outType.getElementType());

  // get constant zero & zero fill
  Value zero = arith::ConstantOp::create(b, b.getZeroAttr(outType.getElementType()));
  return linalg::FillOp::create(b, zero, empty).getResult(0);
}

static Value multiplyAll(ImplicitLocOpBuilder &b, ValueRange vals) {
  assert(!vals.empty() && "multiplyAll requires at least one value");
  
  Value acc = vals.front();
  for (Value v : vals.drop_front())
    acc = arith::MulFOp::create(b, acc, v);
  
  return acc;
}  
  
struct ConvertEinsumLL : public OpConversionPattern<EinsumLL> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      EinsumLL op, EinsumLL::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // get input RankTensorType
    SmallVector<RankedTensorType, 4> inputTensors;
    for (Value v : adaptor.getInputs()) {
      auto rankedTy = dyn_cast<RankedTensorType>(v.getType());
      if (!rankedTy)
        return op.emitOpError("expected RankedTensorType after type conversion");
      inputTensors.push_back(rankedTy);
    }

    // get output RankTensorType
    auto *tc = this->getTypeConverter();
    Type convertedOut = tc->convertType(op.getResult().getType());
    auto outputType = dyn_cast<RankedTensorType>(convertedOut);
    if (!outputType)
      return op.emitOpError("expected output to convert to RankedTensorType");

    // get the loop  order attribute for affine 
    ArrayAttr loopOrderAttr = op.getLoopOrder();
    SmallVector<char> loopOrder;
    loopOrder.reserve(loopOrderAttr.size());
    for (Attribute a : loopOrderAttr) {
      auto s = cast<StringAttr>(a).str();
      loopOrder.push_back(s.front());
    }
    unsigned numLoops = loopOrder.size();    


    ArrayAttr indexingMapsAttr = op.getIndexingMaps();
    SmallVector<AffineMap> indexingMaps;
    for (Attribute attr : indexingMapsAttr) {
      indexingMaps.push_back(cast<AffineMapAttr>(attr).getValue());
    }

    // Convert iterator types from ArrayAttr of StringAttr to ArrayRef<utils::IteratorType>
    ArrayAttr iteratorTypesAttr = op.getIteratorTypes();
    SmallVector<utils::IteratorType> iteratorTypes;
    for (Attribute attr : iteratorTypesAttr) {
      StringRef str = cast<StringAttr>(attr).getValue();
      if (str == "parallel") {
        iteratorTypes.push_back(utils::IteratorType::parallel);
      } else if (str == "reduction") {
        iteratorTypes.push_back(utils::IteratorType::reduction);
      } else {
        return op.emitOpError("unknown iterator type: ") << str;
      }
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value init = createInitTensor(b, outputType);

    auto generic = linalg::GenericOp::create
      (b,
       TypeRange{outputType},
       adaptor.getInputs(),
       ValueRange{init},
       indexingMaps,
       iteratorTypes,
       [&](OpBuilder &b, Location loc, ValueRange args) {
         // args = (%a, %b, ..., %out)
         // multiply/reduce here…
         ImplicitLocOpBuilder bb(loc, b);
         Value acc = args.back();
         Value mul = multiplyAll(bb, args.drop_back());
         Value sum = arith::AddFOp::create(bb, acc, mul);
         linalg::YieldOp::create(bb, sum);
       }
      );

    if(!generic)
      return op.emitOpError("Failed to create generic op");

    rewriter.replaceOp(op, generic.getResults());

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

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      llvm::errs() << "=== Partial conversion failed! Dumping module ===\n";
      module->dump();      
      signalPassFailure();
    }
  }
};

}  // namespace mlir::einsum
