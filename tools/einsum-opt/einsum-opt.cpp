#include "lib/Dialect/Einsum/EinsumDialect.h"
#include "lib/Transforms/Einsum/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::einsum::EinsumDialect>();
  mlir::registerAllDialects(registry);

  mlir::einsum::registerEinsumPasses();
  
  return mlir::asMainReturnCode(
				mlir::MlirOptMain(argc, argv, "Einsum MLIR optimizer\n", registry));
}
