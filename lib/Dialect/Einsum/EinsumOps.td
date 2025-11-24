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

class EinsumOp<string mnemonic, list<Trait> traits = []>:
      		      Op<EinsumDialect, mnemonic, traits> {
  let cppNamespace = "::mlir::einsum";
}

def EinsumHL : EinsumOp<"hl", [NoMemoryEffect]> {
  let summary = "High-level symbolic einsum operation";
  let description = [{
    Accepts N input tensors and an equation string.
    Does not perform any computation; purely symbolic.
  }];

  // Variadic inputs of NamedAxesTensorType
  let arguments = (ins
      Variadic<NamedAxesTensor>:$inputs,
      NamedAxesTensor:$output,
      StrAttr:$equation);

  // Single output of NamedAxesTensor
  let results = (outs);

  // Attributes: a string specifying the einsum equation
  let assemblyFormat = "`ins``(` $inputs `:` type($inputs) `)` `outs``(` $output `:` type($output) `)` attr-dict";

  // Verifier for checking dim sizes
  let hasVerifier=1;
}


def EinsumLL : EinsumOp<"ll", [NoMemoryEffect]> {
  let summary = "Low-level symbolic einsum operation";
  let description = [{
    Lower level representation of Einsum ops.
    Does not perform any computation; purely symbolic.
  }];

  // Variadic inputs of
  let arguments = (ins
      Variadic<NamedAxesTensor>:$inputs,
      NamedAxesTensor:$output,
      StrAttr:$equation, // original einsum equation string
      AffineMapArrayAttr:$indexing_maps, // affine maps per operand and output
      ArrayAttr:$iterator_types, // parallel/reduction per loop iterator
      ArrayAttr:$loop_order // loop order of the subscripts
      );

  // Single output of NamedAxesTensor
  let results = (outs);

  let assemblyFormat = [{
    `ins``(` $inputs `:` type($inputs) `)` `outs``(` $output `:` type($output) `)`
    attr-dict
  }];

}

#endif // LIB_DIALECT_EINSUM_EINSUMOPS_TD_
