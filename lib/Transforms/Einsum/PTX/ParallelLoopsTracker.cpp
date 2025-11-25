#include "lib/Transforms/Einsum/PTX/ParallelLoopsTracker.h"

namespace mlir {
namespace einsum {

#define GEN_PASS_DEF_PARALLELLOOPSTRACKER
#include "lib/Transforms/Einsum/PTX/Passes.h.inc"

void printParallelLoopInfo(std::tuple<SmallVector<int, 4>, SmallVector<int, 4>, SmallVector<int, 4>> data, raw_ostream &os) {
    auto [lowerBounds, upperBounds, steps] = data;
    os << "  Lower bounds: [";
    for (auto lb : lowerBounds) os << lb << " ";
    os << "]\n";
    os << "  Upper bounds: [";
    for (auto ub : upperBounds) os << ub << " ";
    os << "]\n";
    os << "  Steps: [";
    for (auto step : steps) os << step << " ";
    os << "]\n";
}

ParallelLoopsTrackerAnalysis::ParallelLoopsTrackerAnalysis(ModuleOp module) {
  llvm::errs() << "  [ANALYSIS] Computing ParallelLoops...\n";

  module.walk([&](func::FuncOp funcOp) {
    funcOp.walk([&](scf::ParallelOp parallelOp) {
      totalCount++;

      SmallVector<int, 4> lowerBounds;
      for (auto lb: parallelOp.getLowerBound()) {

      }
      SmallVector<int, 4> upperBounds;
      for (auto ub: parallelOp.getUpperBound()) {

      }
      SmallVector<int, 4> steps;
      for (auto step: parallelOp.getStep()) {

      }
      // ParallelLoopInfo pli(lowerBounds, upperBounds, steps);

      functionCounts[funcOp.getName()] = {lowerBounds, upperBounds, steps};
    });
    
  });
}

struct ParallelLoopsTracker : impl::ParallelLoopsTrackerBase<ParallelLoopsTracker> {
  using ParallelLoopsTrackerBase::ParallelLoopsTrackerBase;

  void runOnOperation() override {
    auto& parallelLoopsTrackerAnalysis = getAnalysis<ParallelLoopsTrackerAnalysis>();
    auto totalCount = parallelLoopsTrackerAnalysis.getTotalCount();
    auto functionCounts = parallelLoopsTrackerAnalysis.getFunctionCounts();

    llvm::outs() << "\n┌─────────────────────────────────────────────┐\n";
    llvm::outs() << "│  SCF Parallel Loop Count Summary           │\n";
    llvm::outs() << "├─────────────────────────────────────────────┤\n";
    llvm::outs() << "│  Total parallel loops: " << totalCount << "\n";
    llvm::outs() << "│  Functions with parallel loops: " 
                 << functionCounts.size() << "\n";
    llvm::outs() << "└─────────────────────────────────────────────┘\n\n";
    
    if (!functionCounts.empty()) {
      llvm::outs() << "Per-function breakdown:\n";
      for (auto &entry : functionCounts) {
        llvm::outs() << "  " << entry.first << ":\n"; 
        printParallelLoopInfo(entry.second, llvm::outs());
        llvm::outs() << "\n";
      }
      llvm::outs() << "\n";
    }
  }
};

} // namespace einsum
} // namespace mlir
