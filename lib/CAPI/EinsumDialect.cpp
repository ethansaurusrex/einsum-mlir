#include "lib/CAPI/EinsumDialect.h"
#include "mlir/CAPI/Registration.h"
#include "lib/Dialect/Einsum/EinsumDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "lib/Dialect/Einsum/EinsumTypes.h"

using namespace mlir;
using namespace mlir::einsum;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Einsum, einsum,
                                       mlir::einsum::EinsumDialect)

bool mlirTypeIsAEinsumNamedAxesTensorType(MlirType type)
{
  return isa<mlir::einsum::NamedAxesTensorType>(unwrap(type));  
}

MlirType mlirEinsumNamedAxesTensorTypeGet(
				      MlirContext ctx,
				      MlirType tensorType,
				      MlirAttribute axisNames)
{
  return wrap(mlir::einsum::NamedAxesTensorType::get(
						     unwrap(ctx),
						     cast<RankedTensorType>(unwrap(tensorType)),
						     cast<ArrayAttr>(unwrap(axisNames))));
}
