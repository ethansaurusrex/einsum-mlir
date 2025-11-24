// python/EinsumPassesNanobind.cpp

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "lib/CAPI/EinsumDialect.h"
#include "lib/Dialect/Einsum/EinsumDialect.h"
#include "lib/Dialect/Einsum/EinsumTypes.h"
#include "lib/Conversion/EinsumToLinalg/EinsumToLinalg.h"

namespace nb = nanobind;
using namespace mlir::einsum;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

NB_MODULE(_einsumToLinalgPassesNanobind, m) {
  m.doc() = "EinsumToLinalg MLIR Passes Python Bindings";

  mlirRegisterEinsumToLinalgPasses();
}
