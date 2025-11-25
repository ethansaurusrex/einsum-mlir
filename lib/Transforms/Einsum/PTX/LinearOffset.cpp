#include "lib/Transforms/Einsum/PTX/LinearOffset.h"

namespace mlir {
namespace einsum {

#define GEN_PASS_DEF_LINEAROFFSET
#include "lib/Transforms/Einsum/PTX/Passes.h.inc"

using namespace mlir;
using namespace mlir::memref;

Value calculateLinearOffset(OpBuilder &builder, Location loc,
                        MemRefType memrefType,
                        ValueRange indices) {
    auto shape = memrefType.getShape();
    assert(indices.size() == shape.size() && "Indices must match memref rank");

    Value offset = nullptr;

    for (size_t i = 0; i < indices.size(); ++i) {
        Value term = indices[i];
        
        int64_t stride = 1;
        for (size_t j = i + 1; j < shape.size(); j++) {
            if (shape[j] == ShapedType::kDynamic) {
                // Handle dynamic dimensions - would need runtime dimension queries
                llvm::errs() << "Dynamic dimensions not supported\n";
                stride = ShapedType::kDynamic;
                break;
            }
            stride *= shape[j];
        }
        
        if (stride != 1 && stride != ShapedType::kDynamic) {
            Value strideVal = builder.create<arith::ConstantIndexOp>(loc, stride);
            term = builder.create<arith::MulIOp>(loc, term, strideVal);
        }
        
        if (offset == nullptr) {
            offset = term;
        } else {
            offset = builder.create<arith::AddIOp>(loc, offset, term);
        }

    }

    return offset;
}

mlir::MemRefType createLinearizedMemRefType(mlir::MemRefType old) {
    int64_t totalSize = 1;
    for (auto dim : old.getShape()) {
      if (dim == ShapedType::kDynamic) {
        totalSize = ShapedType::kDynamic;
        break;
      }
      totalSize *= dim;
    }
    
    auto flatMemrefType = MemRefType::get(
        {totalSize},
        old.getElementType(),
        MemRefLayoutAttrInterface{},
        old.getMemorySpace());
    
    return flatMemrefType;
}

struct SimplifyMemRefLoadPattern : public OpRewritePattern<LoadOp> {
  using OpRewritePattern<LoadOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto memrefType = loadOp.getMemRefType();
    
    if (memrefType.getRank() <= 1)
      return failure();
    
    Location loc = loadOp.getLoc();
    Value linearOffset = calculateLinearOffset(rewriter, loc, memrefType, loadOp.getIndices());
    
    int64_t totalSize = 1;
    for (auto dim : memrefType.getShape()) {
      if (dim == ShapedType::kDynamic) {
        totalSize = ShapedType::kDynamic;
        break;
      }
      totalSize *= dim;
    }
    
    auto flatMemrefType = createLinearizedMemRefType(memrefType);
    
    OpFoldResult offset = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(totalSize)};
    SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};
    
    auto flatMemref = rewriter.create<ReinterpretCastOp>(
        loc,
        flatMemrefType,
        loadOp.getMemRef(),
        offset,
        sizes,
        strides);
    
    auto newLoad = rewriter.create<LoadOp>(loc, flatMemref,
                                           ValueRange{linearOffset});
    
    rewriter.replaceOp(loadOp, newLoad);
    return success();
  }
};

struct SimplifyMemRefStorePattern : public OpRewritePattern<StoreOp> {
  using OpRewritePattern<StoreOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto memrefType = storeOp.getMemRefType();
    
    if (memrefType.getRank() <= 1)
      return failure();
    
    Location loc = storeOp.getLoc();
    Value linearOffset = calculateLinearOffset(rewriter, loc, memrefType, storeOp.getIndices());
    
    int64_t totalSize = 1;
    for (auto dim : memrefType.getShape()) {
      if (dim == ShapedType::kDynamic) {
        totalSize = ShapedType::kDynamic;
        break;
      }
      totalSize *= dim;
    }
    
    // auto flatMemrefType = MemRefType::get(
    //     {totalSize},
    //     memrefType.getElementType(),
    //     MemRefLayoutAttrInterface{},
    //     memrefType.getMemorySpace());

    auto flatMemrefType = createLinearizedMemRefType(memrefType);
    
    OpFoldResult offset = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(totalSize)};
    SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};
    
    auto flatMemref = rewriter.create<ReinterpretCastOp>(
        loc,
        flatMemrefType,
        storeOp.getMemRef(),
        offset,
        sizes,
        strides);
    
    auto newStore = rewriter.create<StoreOp>(loc, storeOp.getValue(), flatMemref,
                                           ValueRange{linearOffset});
    
    rewriter.replaceOp(storeOp, newStore);
    return success();
  }
};

struct SimplifyLoadWithReinterpretCast : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;
  
  mlir::LogicalResult matchAndRewrite(
      mlir::memref::LoadOp loadOp,
      mlir::PatternRewriter &rewriter) const override {
    
    mlir::Value memref = loadOp.getMemRef();
    
    auto reinterpretOp = memref.getDefiningOp<mlir::memref::ReinterpretCastOp>();
    if (!reinterpretOp)
      return mlir::failure();
    
    mlir::Value sourceMemref = reinterpretOp.getSource();
    auto sourceType = llvm::cast<mlir::MemRefType>(sourceMemref.getType());
    auto castType = llvm::cast<mlir::MemRefType>(memref.getType());
    
    if (!isReinterpretCastUnnecessary(sourceType, castType))
      return mlir::failure();
    
    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(
        loadOp,
        sourceMemref,
        loadOp.getIndices()
    );
    
    return mlir::success();
  }
  
private:
  bool isReinterpretCastUnnecessary(mlir::MemRefType sourceType,
                                     mlir::MemRefType castType) const {
    if (sourceType.getElementType() != castType.getElementType())
      return false;
    
    if (sourceType.getShape() != castType.getShape())
      return false;
    
    if (sourceType.getMemorySpace() != castType.getMemorySpace())
      return false;
    
    if (!sourceType.getLayout().isIdentity() || 
        !castType.getLayout().isIdentity())
      return false;
    
    return true;
  }
};

struct SimplifyStoreWithReinterpretCast : public mlir::OpRewritePattern<mlir::memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;
  
  mlir::LogicalResult matchAndRewrite(
      mlir::memref::StoreOp storeOp,
      mlir::PatternRewriter &rewriter) const override {
    
    mlir::Value memref = storeOp.getMemRef();
    
    auto reinterpretOp = memref.getDefiningOp<mlir::memref::ReinterpretCastOp>();
    if (!reinterpretOp)
      return mlir::failure();
    
    mlir::Value sourceMemref = reinterpretOp.getSource();
    auto sourceType = llvm::cast<mlir::MemRefType>(sourceMemref.getType());
    auto castType = llvm::cast<mlir::MemRefType>(memref.getType());
    
    if (!isReinterpretCastUnnecessary(sourceType, castType))
      return mlir::failure();
    
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(
        storeOp,
        storeOp.getValue(),
        sourceMemref,
        storeOp.getIndices()
    );
    
    return mlir::success();
  }
  
private:
  bool isReinterpretCastUnnecessary(mlir::MemRefType sourceType,
                                     mlir::MemRefType castType) const {
    return sourceType.getElementType() == castType.getElementType() &&
           sourceType.getShape() == castType.getShape() &&
           sourceType.getMemorySpace() == castType.getMemorySpace() &&
           sourceType.getLayout().isIdentity() &&
           castType.getLayout().isIdentity();
  }
};

struct LinearOffset : impl::LinearOffsetBase<LinearOffset> {
  using LinearOffsetBase::LinearOffsetBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();

    RewritePatternSet patterns(context);
    patterns.add<SimplifyMemRefLoadPattern>(context);
    patterns.add<SimplifyMemRefStorePattern>(context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        funcOp.emitError("Failed to apply memref simplification patterns");
    }

    mlir::OpBuilder builder(funcOp.getContext());

    mlir::FunctionType oldType = funcOp.getFunctionType();
    llvm::SmallVector<mlir::Type> inputTypes(oldType.getInputs().begin(), 
                                            oldType.getInputs().end());
    llvm::SmallVector<mlir::Type> resultTypes(oldType.getResults().begin(),
                                            oldType.getResults().end());
    
    for (size_t i = 0; i < inputTypes.size(); i++) {
        if (isa<mlir::MemRefType>(inputTypes[i])) {
            inputTypes[i] = createLinearizedMemRefType(dyn_cast<mlir::MemRefType>(inputTypes[i]));
        }
    }
    for (size_t i = 0; i < resultTypes.size(); i++) {
        if (isa<mlir::MemRefType>(resultTypes[i])) {
            resultTypes[i] = createLinearizedMemRefType(dyn_cast<mlir::MemRefType>(resultTypes[i]));
        }
    }
    auto newFuncType = builder.getFunctionType(inputTypes, resultTypes);
    funcOp.setType(newFuncType);

    funcOp.walk([&](mlir::Block *block) {
        for (unsigned i = 0; i < block->getNumArguments(); ++i) {
            mlir::BlockArgument arg = block->getArgument(i);
            mlir::Type oldType = arg.getType();
            if (auto memrefType = llvm::dyn_cast<mlir::MemRefType>(oldType)) 
                arg.setType(createLinearizedMemRefType(memrefType));
        }
    });

    RewritePatternSet patterns2(context);
    patterns2.add<SimplifyLoadWithReinterpretCast>(context);
    patterns2.add<SimplifyStoreWithReinterpretCast>(context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns2)))) {
        funcOp.emitError("Failed to apply memref simplification patterns");
    }
  }

};
    
} // namespace einsum
} // namespace mlir
