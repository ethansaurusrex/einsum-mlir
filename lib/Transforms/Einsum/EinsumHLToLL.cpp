// lib/Transforms/Einsum/EinsumHLToLL.cpp
#include "lib/Transforms/Einsum/EinsumHLToLL.h"
#include "lib/Dialect/Einsum/EinsumOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir::einsum {

#define GEN_PASS_DEF_EINSUMHLTOLL
#include "lib/Transforms/Einsum/Passes.h.inc"

namespace {

  static LogicalResult parseEinsumEquation(StringRef equation,
                                           SmallVectorImpl<StringRef> &inputSubs,
                                           StringRef &outputSub) {
    if (equation.empty())
      return failure();
    
    StringRef lhs, rhs;
    std::tie(lhs, rhs) = equation.split("->");
    if (lhs.empty())
      return failure();
    
    lhs.split(inputSubs, ',');
    outputSub = rhs;
    return success();
  }

  static std::optional<RankedTensorType> getRankedTensorTypeFromNamed(Value v) {
    if(auto natType = dyn_cast<NamedAxesTensorType>(v.getType()) )
      return dyn_cast<RankedTensorType>(natType.getTensorType());
    if(auto rType = dyn_cast<RankedTensorType>(v.getType()) )
      return rType;
    return std::nullopt;
  }

  void computeParallelAndReductionIndices(ArrayRef<StringRef> inputSubs,
                                        StringRef outputSub,
                                        SmallVectorImpl<char> &parallel,
                                        SmallVectorImpl<char> &reduction) {
    llvm::SmallDenseSet<char, 8> inputIndices;
    for (auto &sub : inputSubs) {
      for (char c : sub) {
        inputIndices.insert(c);
      }
    }
    
    llvm::SmallDenseSet<char, 8> outputIndices;
    
    for (char c : outputSub) {
      outputIndices.insert(c);
    }
    
    parallel.clear();
    reduction.clear();
    
    // Parallel indices: preserve output order
    for (char c : outputSub) {
      parallel.push_back(c);
    }
    
    // Reduction indices: input-only, preserve first occurrence
    for (auto &sub : inputSubs) {
      for (char c : sub) {
        if (!outputIndices.contains(c) && !is_contained(reduction, c)) {
          reduction.push_back(c);
        }
      }
    }
  }
                             

  static SmallVector<AffineMap, 8>
  generateAffineMapsFromEquation(OpBuilder &builder,
                                 const SmallVector<char, 8> &loopOrder,
                                 ArrayRef<StringRef> inputSubs,
                                 StringRef outputSub,
                                 ArrayRef<RankedTensorType> inputTensors,
                                 RankedTensorType outputType) {
    MLIRContext *ctx = builder.getContext();

    unsigned totalDims = loopOrder.size();

    // map subscript char to pos in loopOrder
    llvm::SmallDenseMap<char, unsigned, 8> posMap;
    for(unsigned i = 0; i < loopOrder.size(); ++i) {
      posMap[loopOrder[i]] = i;
    }

    SmallVector<AffineMap, 8> maps;
    maps.reserve(inputTensors.size() + 1);

    // handle inputs
    for(unsigned i = 0; i < inputSubs.size(); ++i) {
      StringRef subs = inputSubs[i];
      llvm::SmallVector<AffineExpr, 8> exprs;
      exprs.reserve(subs.size());

      for(char c : subs) {
        auto it = posMap.find(c);
        assert(it != posMap.end() && "input axis must be present in loop order");
        exprs.push_back(mlir::getAffineDimExpr(it->second, ctx));
      }

      maps.push_back(AffineMap::get(totalDims, /*symbolCount=*/0, exprs, ctx));
    }

    // handle output map
    SmallVector<AffineExpr, 8> outExprs;
    outExprs.reserve(outputSub.size());
    for(char c : outputSub) {
      auto it = posMap.find(c);
      assert(it != posMap.end() && "output axis must be present in loop order");
      outExprs.push_back(mlir::getAffineDimExpr(it->second, ctx));
    }
    maps.push_back(AffineMap::get(totalDims,  /*symbolCount=*/0, outExprs, ctx));

    return maps;
  }

  struct EinsumHLToLLPattern : public OpRewritePattern<einsum::EinsumHL> {
    using OpRewritePattern::OpRewritePattern;
    
    LogicalResult matchAndRewrite(einsum::EinsumHL op,
                                  PatternRewriter &rewriter) const override {
      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      // parse HL op equation
      StringRef equation = op.getEquation();
      SmallVector<StringRef, 4> inputSubs;
      StringRef outputSubs;
      
      if(failed(parseEinsumEquation(equation, inputSubs, outputSubs)))
         return op.emitError("invalid equation attribute, expected 'lhs->rhs' form");
      
      // get the ins RankedTensorTypes
      SmallVector<Value, 4> castedInputs;
      SmallVector<RankedTensorType, 4> inputTensors;
      inputTensors.reserve(op.getInputs().size());
      
      for(Value v : op.getInputs()) {
        auto maybeRT = getRankedTensorTypeFromNamed(v);
        if(!maybeRT)
          return op.emitOpError("input is not a RankedTensorType or NamedAxesTensorType");
        inputTensors.push_back(*maybeRT);
      }

      // quick check that num input subs == num inputs
      // note: above should be checked by EinsumHL::verify()
      if (inputSubs.size() != inputTensors.size()) {
        return op.emitOpError()
	  << "number of input subs (" << inputSubs.size()
	  << ") does not match number of operands (" << inputTensors.size()
	  << ")";
      }
      
      // get the output RankedTensorType
      Value outputOperand = op.getOutput();
      auto maybeRT = getRankedTensorTypeFromNamed(outputOperand);
      if(!maybeRT)
        return op.emitOpError("output is not a RankedTensorType or NamedAxesTensorType");
      \
      RankedTensorType outputRT = *maybeRT;

      // get paralle and reduction indices
      SmallVector<char> parallel;
      SmallVector<char> reduction;
      computeParallelAndReductionIndices(inputSubs, outputSubs, parallel, reduction);      

      // build unified loop ordering
      SmallVector<char, 8> loopOrder;
      loopOrder.append(parallel.begin(), parallel.end());
      loopOrder.append(reduction.begin(), reduction.end());

      // probably a better way to do this
      SmallVector<Attribute, 8> iteratorTypes;
      auto pStr = rewriter.getStringAttr("parallel");
      auto rStr = rewriter.getStringAttr("reduction");

      for(char c : loopOrder) {
	iteratorTypes.push_back(
				llvm::is_contained(parallel, c) ? pStr : rStr
				);
      }

      // now we have iterator type array attr
      ArrayAttr iteratorTypesAttr = rewriter.getArrayAttr(iteratorTypes);

      // must biuld the affine maps now
      SmallVector<AffineMap, 8> affineMaps =
        generateAffineMapsFromEquation(rewriter,
                                       loopOrder,
                                       inputSubs,
                                       outputSubs,
                                       inputTensors,
                                       outputRT);
      
      ArrayAttr mapsAttr = rewriter.getAffineMapArrayAttr(affineMaps);

      // build loop order attr

      SmallVector<Attribute> loopOrderAttrs;
      loopOrderAttrs.reserve(loopOrder.size());

      for (char c : loopOrder) {
        loopOrderAttrs.push_back(rewriter.getStringAttr(StringRef(&c, 1)));
      }

      ArrayAttr loopOrderAttr = rewriter.getArrayAttr(loopOrderAttrs);

      auto llOp = EinsumLL::create(b,
				   /*output=*/op.getResult().getType(),
				   /*inputs=*/op.getInputs(),
				   /*out_operand=*/outputOperand,
				   /*equation=*/rewriter.getStringAttr(equation),
				   /*indexing_maps=*/mapsAttr,
				   /*iteartor_types=*/iteratorTypesAttr,
				   /*loop_order=*/loopOrderAttr);
      
      rewriter.replaceOp(op, llOp.getResult());
                                            
      return success();
    }
  };
  
  struct EinsumHLToLL : impl::EinsumHLToLLBase<EinsumHLToLL> {
    using EinsumHLToLLBase::EinsumHLToLLBase;
      
    void runOnOperation() override {
      // get rewrite set:
      mlir::RewritePatternSet patterns(&getContext());
      patterns.add<EinsumHLToLLPattern>(&getContext());
      (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }
};

} // namespace
} // namespace mlir::einsum
