
#ifndef LIB_PARALLEL_LOOPS_TRACKER_H_
#define LIB_PARALLEL_LOOPS_TRACKER_H_

// Add the required forward declarations as the generated code doesn't contain the include statements
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
#include <tuple>

namespace mlir {
namespace einsum {

struct ParallelLoopInfo {
    ParallelLoopInfo(SmallVector<int, 4> lowerBounds, SmallVector<int, 4> upperBounds, SmallVector<int, 4> steps) : lowerBounds(lowerBounds), upperBounds(upperBounds), steps(steps) {}
    
    void print(raw_ostream &os) const {
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
private:
    SmallVector<int, 4> lowerBounds;
    SmallVector<int, 4> upperBounds;
    SmallVector<int, 4> steps;
};

void printParallelLoopInfo(std::tuple<SmallVector<int, 4>, SmallVector<int, 4>, SmallVector<int, 4>> data, raw_ostream &os);

class ParallelLoopsTrackerAnalysis {
public: 
  ParallelLoopsTrackerAnalysis(ModuleOp);
  unsigned getTotalCount() const { return totalCount; }
  // DenseMap<StringRef, ParallelLoopInfo> getFunctionCounts() {return functionCounts; }
  DenseMap<StringRef, std::tuple<SmallVector<int, 4>, SmallVector<int, 4>, SmallVector<int, 4>>> getFunctionCounts() {return functionCounts; }

private: 
  unsigned totalCount = 0;
  // DenseMap<StringRef, ParallelLoopInfo> functionCounts;
  DenseMap<StringRef, std::tuple<SmallVector<int, 4>, SmallVector<int, 4>, SmallVector<int, 4>>> functionCounts;
};

// #define switches used to add the required components from the generated code
#define GEN_PASS_DECL_PARALLELLOOPSTRACKER
#include "../include/Passes.h.inc"

} // namespace einsum
} // namespace mlir

#endif // LIB_PARALLEL_LOOPS_TRACKER_H_
