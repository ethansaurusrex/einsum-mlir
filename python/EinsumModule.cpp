
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "lib/CAPI/Dialect.h"
#include "lib/Dialect/Einsum/EinsumDialect.h"
#include "lib/Dialect/Einsum/EinsumTypes.h"
#include "lib/Transforms/Einsum/Passes.h"
#include "lib/Conversion/EinsumToLinalg/EinsumToLinalg.h"

namespace py = pybind11;
using namespace mlir::einsum;
using namespace mlir::python;
using namespace mlir::python::adaptors;

// dialect registration hook
void populateEinsumDialect(py::module &m) {
  
  
  m.def("register_dialect", [](MlirContext context, bool load) {
    MlirDialectHandle handle = mlirGetDialectHandle__einsum__();
    mlirDialectHandleRegisterDialect(handle, context);
    if (load) {
      mlirDialectHandleLoadDialect(handle, context);
    }
  }, py::arg("context"), py::arg("load") = true);
}
/*
    mlir::MLIRContext *cppCtx = unwrap(context);
    cppCtx->getOrLoadDialect<EinsumDialect>();
  });
}
*/

void populateEinsumTypes(py::module &m) {
  // We use mlir_type_subclass, which automatically handles the MlirType
  // lifecycle and registers the CAPI type check function:
  // bool mlirTypeIsAEinsumNamedAxesTensor(MlirType type)
  auto namedAxesTensorType = mlir_type_subclass(
      m, "NamedAxesTensorType", mlirTypeIsAEinsumNamedAxesTensorType);

  // 2. Bind the static 'get' factory method to Python.
  // This maps NamedAxesTensorType.get(...) in Python to the C-API function:
  // MlirType mlirEinsumNamedAxesTensorGet(...)
  namedAxesTensorType.def_classmethod(
      "get",
      [](py::object cls, MlirType tensorType, MlirAttribute axisNames, MlirContext ctx) {
        // Call the C-API function defined in lib/CAPI/Dialects/Einsum.cpp
        return cls(mlirEinsumNamedAxesTensorTypeGet(
            ctx, tensorType, axisNames));
      },
      "Gets an instance of NamedAxesTensor in the given context.",
      py::arg("cls"),
      py::arg("tensor_type"),
      py::arg("axis_names"),
      py::arg("context") = py::none());
}

// pass registration hook
void populateEinsumPasses(py::module &m) {
  m.def("register_passes", []() {
    registerEinsumPasses();
    registerEinsumToLinalgPasses();
  });
}


PYBIND11_MODULE(_einsum, m) {
  m.doc() = "Einsum MLIR Dialect Python Bindings";
  populateEinsumDialect(m);
  populateEinsumTypes(m);
  populateEinsumPasses(m);  
}
