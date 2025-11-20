#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "lib/Dialect/Einsum/EinsumDialect.h"
#include "lib/Transforms/Einsum/Passes.h"

namespace py = pybind11;
using namespace mlir::einsum;

// dialect registration hook
void populateEinsumDialect(py::module &m) {
  m.def("register_dialect", [](MlirContext context, bool load) {
    MlirDialectHandle handle = mlirGetDialectHandle__einsum__();
    mlir::MLIRContext *cppCtx = unwrap(context);
    cppCtx->getOrLoadDialect<EinsumDialect>();
  });
}

// pass registration hook
void populateEinsumPasses(py::module &m) {
  m.def("register_passes", []() {
    registerEinsumPasses();
  });
}

PYBIND11_MODULE(_einsum, m) {
  m.doc() = "Einsum MLIR Dialect Python Bindings";
  populateEinsumDialect(m);
  populateEinsumPasses(m);
}
