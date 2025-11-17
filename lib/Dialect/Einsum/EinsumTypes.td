#ifndef LIB_DIALECT_EINSUM_EINSUMTYPES_TD_
#define LIB_DIALECT_EINSUM_EINSUMTYPES_TD_

//===- EinsumTypes.td - Types for Einsum dialect -----------------*- tablegen -*-===//
//
// This file defines types for the Einsum dialect using a ShapedContainerType
//
//===----------------------------------------------------------------------===//

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/AttrTypeBase.td"   // for TypedArrayAttrBase, ArrayAttr
include "mlir/Interfaces/InferTypeOpInterface.td"

include "lib/Dialect/Einsum/EinsumDialect.td"

def NamedAxesTensorType
    : TypeDef<EinsumDialect, "NamedAxesTensorType"> {
  let summary = "Tensor type with optional named axes";
  let description = [{
    A tensor type that stores:
      - an underlying element type (Type)
      - optional axis names (ArrayAttr)
  }];
  // Parameters stored in the TableGen-generated storage
  let parameters = (ins
      "Type":$elementType,
      "ArrayAttr":$axisNames);

  // Assembly format: <axis1,axis2,...> : f32
  let assemblyFormat = "`<` $axisNames `:` $elementType `>`";

  // Mnemonic used for parsing/printing
  let mnemonic = "named_axes_tensor";
}

#endif // LIB_DIALECT_EINSUM_EINSUMTYPES_TD_