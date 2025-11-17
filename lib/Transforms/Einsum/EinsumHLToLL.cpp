// lib/Transforms/Einsum/EinsumHLToLL.cpp
#include "lib/Transforms/Einsum/EinsumHLToLL.h"
#include "lib/Dialect/Einsum/EinsumOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"

namespace mlir::einsum {

#define GEN_PASS_DEF_EINSUMHLTOLL
#include "lib/Transforms/Einsum/Passes.h.inc"

namespace {

  struct EinsumHLToLLPattern : public OpRewritePattern<einsum::EinsumHL> {
    using OpRewritePattern::OpRewritePattern;
    
    LogicalResult matchAndRewrite(einsum::EinsumHL op,
                                  PatternRewriter &rewriter) const override {
      return success();
    }
  };
  
  struct EinsumHLToLL : impl::EinsumHLToLLBase<EinsumHLToLL> {
    using EinsumHLToLLBase::EinsumHLToLLBase;
      
    void runOnOperation() {
      // get rewrite set:
      mlir::RewritePatternSet patterns(&getContext());
      // patterns.add<EinsumHLToLLPattern>(&getContext());
      (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }
};

} // namespace
  
} // namespace mlir::einsum
