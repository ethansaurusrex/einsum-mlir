#include "lib/Conversion/EinsumToLinalg/EinsumToLinalg.h"
#include "lib/Dialect/Einsum/EinsumDialect.h"
#include "lib/Transforms/Einsum/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::einsum::EinsumDialect>();
  mlir::registerAllPasses();
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  mlir::einsum::registerEinsumPasses();
  mlir::einsum::registerEinsumToLinalgPasses();
  
  return mlir::asMainReturnCode(
				mlir::MlirOptMain(argc, argv, "Einsum MLIR optimizer\n", registry));
}
