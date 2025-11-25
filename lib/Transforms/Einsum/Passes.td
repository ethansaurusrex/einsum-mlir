//===- Passes.td - Einsum Transform Passes -------------------------===//
//
// Passes that rewrite EinsumHL to EinsumLL.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def EinsumHLToLL : Pass<"einsum-hl-to-ll", "mlir::ModuleOp"> {
  let summary = "Lower einsum.hl ops to einsum.ll ops";
  let description = [{
    Converts symbolic high-level EinsumHL into lower-level EinsumLL,
    constructing indexing-maps and iterator-types.
  }];
  let dependentDialects = ["mlir::einsum::EinsumDialect"];
}
