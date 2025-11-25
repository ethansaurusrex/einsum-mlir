#include "lib/Transforms/Einsum/PTX/ParallelLoopsRemover.h"
#include "lib/Transforms/Einsum/PTX/ParallelLoopsTracker.h"

namespace mlir {
namespace einsum {

#define GEN_PASS_DEF_PARALLELLOOPSREMOVER
#include "lib/Transforms/Einsum/PTX/Passes.h.inc"

struct ParallelLoopsRemover : impl::ParallelLoopsRemoverBase<ParallelLoopsRemover> {
  using ParallelLoopsRemoverBase::ParallelLoopsRemoverBase;

  void yolo() {
    // auto& parallelLoopsTrackerAnalysis = getAnalysis<ParallelLoopsTrackerAnalysis>();
    // auto totalCount = parallelLoopsTrackerAnalysis.getTotalCount();
    // auto functionCounts = parallelLoopsTrackerAnalysis.getFunctionCounts();
    // // Print summary
    // llvm::outs() << "\n┌─────────────────────────────────────────────┐\n";
    // llvm::outs() << "│  SCF Parallel Loop Count Summary           │\n";
    // llvm::outs() << "├─────────────────────────────────────────────┤\n";
    // llvm::outs() << "│  Total parallel loops: " << totalCount << "\n";
    // llvm::outs() << "│  Functions with parallel loops: " 
    //              << functionCounts.size() << "\n";
    // llvm::outs() << "└─────────────────────────────────────────────┘\n\n";
    
    // if (!functionCounts.empty()) {
    //   llvm::outs() << "Per-function breakdown:\n";
    //   for (auto &entry : functionCounts) {
    //     llvm::outs() << "  " << entry.first << ":\n"; 
    //     // entry.second.print(llvm::outs());
    //     printParallelLoopInfo(entry.second, llvm::outs());
    //     llvm::outs() << "\n";
    //   }
    //   llvm::outs() << "\n";
    // }
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // yolo();
    
    SmallVector<scf::ParallelOp, 4> opsToRemove;
    
    func.walk([&](scf::ParallelOp parallelOp) {
      opsToRemove.push_back(parallelOp);
    });
    
    for (scf::ParallelOp parallelOp : opsToRemove) {
      removeParallelLoop(parallelOp);
    }
    
  }

  void removeParallelLoop(scf::ParallelOp parallelOp) {
    OpBuilder builder(parallelOp);
    
    Block &body = parallelOp.getRegion().front();
    
    // If the parallel loop produces results, we need to handle them
    if (parallelOp.getNumResults() > 0) {
      // Replace all uses with undef values or handle reduction
      for (OpResult result : parallelOp.getResults()) {
        // Create an undefined value of the same type
        Value undefValue = builder.create<arith::ConstantOp>(
            parallelOp.getLoc(), 
            result.getType(),
            builder.getZeroAttr(result.getType()));
        result.replaceAllUsesWith(undefValue);
      }
    }
    
    // parallelOp.getNumLoops();
    // parallelOp.getInductionVars();
    // parallelOp.getLowerBound();
    // parallelOp.getUpperBound(); 
    // parallelOp.getStep();

    auto ivs = parallelOp.getInductionVars();
    int count = 0;
    for (auto iv : ivs) {
        llvm::outs() << "IV: " << iv << "\n";
        auto val = builder.create<mlir::ub::PoisonOp>(parallelOp.getLoc(), builder.getIndexType(), nullptr);
        std::string attrVal = std::to_string(count);
        val->setAttr("GPU", builder.getStringAttr(attrVal));
        iv.replaceAllUsesWith(val);

        count++;
        // if (count >= 2) break;
    }

    Operation *terminator = body.getTerminator();
    Block *parentBlock = parallelOp->getBlock();
    
    Block::iterator insertPt = Block::iterator(parallelOp);
    
    for (Operation &op : llvm::make_early_inc_range(body.without_terminator())) {
      op.moveBefore(parentBlock, insertPt);
    }
    
    parallelOp.erase();

    markAnalysesPreserved<ParallelLoopsTrackerAnalysis>();
  }
};
    
} // namespace einsum
} // namespace mlir
