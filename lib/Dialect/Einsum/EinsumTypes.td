//===- EinsumTypes.td - Types for Einsum dialect -----------------*- tablegen -*-===//
//
// This file defines types for the Einsum dialect using a ShapedContainerType
//
//===----------------------------------------------------------------------===//

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/AttrTypeBase.td"   // for TypedArrayAttrBase, ArrayAttr

include "lib/Dialect/Einsum/EinsumDialect.td"

class Einsum_Type<string cppName, string typeMnemonic>
    : TypeDef<EinsumDialect, cppName> {
  let mnemonic = typeMnemonic;
}

def NamedAxesTensorType
    : Einsum_Type<"NamedAxesTensorType", "named_axes"> {

  let summary = "Tensor wrapper with optional named axes";
  let description = [{
    A tensor type that stores:
      - an underlying TensorType
      - optional axis names
  }];
}