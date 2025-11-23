// python/EinsumModule.cpp

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "lib/CAPI/Dialect.h"
#include "lib/Dialect/Einsum/EinsumDialect.h"
#include "lib/Dialect/Einsum/EinsumTypes.h"
#include "lib/Transforms/Einsum/Passes.h"
#include "lib/Conversion/EinsumToLinalg/EinsumToLinalg.h"

namespace nb = nanobind;
using namespace mlir::einsum;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

// dialect registration hook
static void populateEinsumDialect(nb::module_ &m) {
  
  
  m.def("register_dialect", [](MlirContext context, bool load) {
    MlirDialectHandle handle = mlirGetDialectHandle__einsum__();
    mlirDialectHandleRegisterDialect(handle, context);
    if (load) {
      mlirDialectHandleLoadDialect(handle, context);
    }
  }, nb::arg("context"), nb::arg("load") = true);
}
/*
    mlir::MLIRContext *cppCtx = unwrap(context);
    cppCtx->getOrLoadDialect<EinsumDialect>();
  });
}
*/

static void populateEinsumTypes(nb::module_ &m) {
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
      [](nb::object cls, MlirType tensorType, MlirAttribute axisNames, MlirContext ctx) {
        // Call the C-API function defined in lib/CAPI/Dialects/Einsum.cpp
        return cls(mlirEinsumNamedAxesTensorTypeGet(
            ctx, tensorType, axisNames));
      },
      "Gets an instance of NamedAxesTensor in the given context.",
      nb::arg("cls"),
      nb::arg("tensor_type"),
      nb::arg("axis_names"),
      nb::arg("context") = nb::none());
}

// pass registration hook
static void populateEinsumPasses(nb::module_ &m) {
  m.def("register_passes", []() {
    //registerEinsumPasses();
    //registerEinsumToLinalgPasses();
  });
}


NB_MODULE(_einsumNanobind, m) {
  m.doc() = "Einsum MLIR Dialect Python Bindings";
  populateEinsumDialect(m);
  populateEinsumTypes(m);
  //populateEinsumPasses(m);  
}
