#ifndef LIB_DIALECT_EINSUM_EINSUMOPS_TD_
#define LIB_DIALECT_EINSUM_EINSUMOPS_TD_

//===- EinsumOps.td - Einsum operations for Einsum dialect --------------===//
//
// High-level symbolic Einsum operation.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "lib/Dialect/Einsum/EinsumDialect.td" 
include "lib/Dialect/Einsum/EinsumTypes.td"

def EinsumOp : Op<EinsumDialect, "op", [NoMemoryEffect]> {
  let summary = "High-level symbolic einsum operation";
  let description = [{
    Accepts N input tensors and an equation string.
    Does not perform any computation; purely symbolic.
  }];

  // Variadic inputs of NamedAxesTensorType
  let arguments = (ins
      Variadic<NamedAxesTensorType>:$inputs,      
      StrAttr:$equation);

  // Single output of NamedAxesTensorType
  let results = (outs NamedAxesTensorType:$output);

  // Attributes: a string specifying the einsum equation
  let assemblyFormat = "`(` $inputs `:` type($inputs) `)` attr-dict `->` type($output)";

}

#endif // LIB_DIALECT_EINSUM_EINSUMOPS_TD_