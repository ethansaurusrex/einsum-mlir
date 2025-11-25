#ifndef TRANSFORMS_EINSUM_PTX_PASSES_TD_
#define TRANSFORMS_EINSUM_PTX_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ParallelLoopsTracker : Pass<"parallel-loops-tracker", "ModuleOp"> {
  let summary = "Convert parallel loops to GPU threads and blocks";
  let dependentDialects = ["mlir::scf::SCFDialect, mlir::arith::ArithDialect"];
}

def ParallelLoopsRemover : Pass<"parallel-loops-remover", "func::FuncOp"> {
  let summary = "Remove parallel loops";
  let dependentDialects = ["mlir::scf::SCFDialect, mlir::arith::ArithDialect", "func::FuncDialect", "mlir::ub::UBDialect"];
}

def LinearOffset : Pass<"linearize-offset", "func::FuncOp"> {
  let summary = "Linearize offsets on memref loads and stores";
  let dependentDialects = ["mlir::arith::ArithDialect", "memref::MemRefDialect", "func::FuncDialect"];
}

def PtxCodegen : Pass<"scf-to-ptx", "ModuleOp"> {
  let summary = "Generate PTX from SCF";
  let dependentDialects = ["mlir::scf::SCFDialect, mlir::arith::ArithDialect", "memref::MemRefDialect", "func::FuncDialect", "mlir::ub::UBDialect"];
}

#endif  // TRANSFORMS_EINSUM_PTX_PASSES_TD_