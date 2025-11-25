#include "lib/Conversion/EinsumToLinalg/EinsumToLinalg.h"
#include "lib/Dialect/Einsum/EinsumDialect.h"
#include "lib/Transforms/Einsum/Passes.h"
#include "lib/Transforms/Einsum/Pipelines.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Conversion/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::einsum::EinsumDialect>();
  mlir::registerAllPasses();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerConvertFuncToLLVMPass();

  mlir::einsum::registerEinsumPasses();
  mlir::einsum::registerEinsumToLinalgPasses();
  mlir::einsum::registerEinsumPipelines();
  
  return mlir::asMainReturnCode(
				mlir::MlirOptMain(argc, argv, "Einsum MLIR optimizer\n", registry));
}
