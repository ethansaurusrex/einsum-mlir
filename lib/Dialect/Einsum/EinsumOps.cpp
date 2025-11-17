// lib/Dialect/Einsum/EinsumOps.cpp

#include "lib/Dialect/Einsum/EinsumOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Types.h"

namespace mlir::einsum {

  LogicalResult EinsumHL::verify() {
    /*
    Type firstElemType;
    for (auto input : op.inputs()) {
      auto tensorType = input.getType().cast<NamedAxesTensorType>();
      auto elemType = tensorType.getElementType(); // you need to implement getElementType() in NamedAxesTensorType C++ class
      if (!firstElemType)
	firstElemType = elemType;
      else if (firstElemType != elemType)
	return op.emitOpError("all input tensors must have the same element type");
    }
    
    // 2) Check that dimensions align according to equation
    auto equationAttr = op.equation();
    if (!equationAttr)
      return op.emitOpError("missing equation attribute");
    
    StringRef equation = equationAttr.getValue();
    SmallVector<StringRef, 4> inputEquations;
    StringRef outputEquation;
    
    // Split equation like "ik,kj->ij"
    if (equation.split("->", inputEquations, outputEquation) != 2)
      return op.emitOpError("expected einsum equation of the form 'ik,kj->ij'");
    
    SmallVector<StringRef, 4> inputAxisStrings;
    inputEquations[0].split(',', inputAxisStrings);
    
    if (inputAxisStrings.size() != op.getNumOperands())
      return op.emitOpError("number of inputs does not match equation");
    
    // Map axis letters to dimensions and check alignment
    for (size_t i = 0; i < op.getNumOperands(); ++i) {
      auto tensorType = op.getOperand(i).getType().cast<NamedAxesTensorType>();
      auto axisNames = tensorType.getAxisNames(); // ArrayAttr of StringAttr
      if (axisNames.size() != tensorType.getShape().size())
	return op.emitOpError("number of axes does not match tensor rank");
      
      // You can check matching contracted axes here
      // e.g., for "ik,kj->ij", check that k dimension matches
      // Implementation depends on your axis naming convention
    }
    */
    return success();
  }
 
}
