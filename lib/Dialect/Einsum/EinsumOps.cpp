// lib/Dialect/Einsum/EinsumOps.cpp

#include "lib/Dialect/Einsum/EinsumOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::einsum {

  LogicalResult EinsumHL::verify() {
    Type firstElemType;
    for (auto input : getInputs()) {
      auto natType = dyn_cast<NamedAxesTensorType>(input.getType());
      if(!natType)
        return emitOpError("operand is not a NamedAxesTensorType");
      auto tensorType = dyn_cast<RankedTensorType>(natType.getTensorType());
      auto elemType = tensorType.getElementType();
      if (!firstElemType)
        firstElemType = elemType;
      else if (firstElemType != elemType)
        return emitOpError("all input tensors must have the same element type");
    }

    // 2) check that dimensions align according to equation
    auto equation = getEquation();
    if (equation.empty())
      return emitOpError("missing equation attribute");
    
    StringRef lhsPart, rhsPart;
    std::tie(lhsPart, rhsPart) = equation.split("->");
    
    // now, lhsPart = "ik,kj", rhsPart = "ij"
    // note: rhs can be empty: ij-> (sum), ii-> (trace)
    if (lhsPart.empty())
      return emitOpError("expected einsum equation of the form 'LHS->RHS'");
    
    SmallVector<StringRef, 4> inputAxisStrings;
    lhsPart.split(inputAxisStrings, ',');
    
    if (inputAxisStrings.size() != getInputs().size())
      return emitOpError("number of inputs does not match equation");
    
    // 3) check each NamedAxesTensorType has matching axis count
    for (size_t i = 0; i < getInputs().size(); ++i) {
      auto natType = dyn_cast<NamedAxesTensorType>(getInputs()[i].getType());
      auto tensorType = dyn_cast<RankedTensorType>(natType.getTensorType());
      auto axisNames = natType.getAxisNames();
      auto shape = tensorType.getShape();

      if(shape.size() == 0)
	emitOpError("tensor has unranked shape; cannot verify axes");

      if(axisNames.size() != shape.size())
	emitOpError("number of axes does not match tensor rank");
    }

    
    // 4) check reduction dimensions match (we'll do this again during hlToll)
    llvm::DenseMap<char, SmallVector<std::pair<size_t, size_t>>> axisMap;
    for (size_t i = 0; i < inputAxisStrings.size(); ++i) {
      StringRef axes = inputAxisStrings[i];
      for (size_t pos = 0; pos < axes.size(); ++pos) {
	axisMap[axes[pos]].push_back({i, pos});
      }
    }
    
    // id RHS axes
    llvm::StringSet<> rhsAxes;
    for (char c : rhsPart)
      rhsAxes.insert(std::string(1, c));
    
    // check dimensions of contracted axes (in LHS but not RHS)
    for (auto &entry : axisMap) {
      char axis = entry.first;
      auto &locations = entry.second;
      
      // only check contracted axes
      if (rhsAxes.contains(std::string(1, axis)) || locations.size() < 2)
	continue;

      // reference dimension
      size_t refDim = dyn_cast<RankedTensorType>(dyn_cast<NamedAxesTensorType>(getInputs()[locations[0].first].getType()).getTensorType()).getShape()[locations[0].second];
      
      // compare with all other tensors
      for (size_t i = 1; i < locations.size(); ++i) {
	size_t dim = dyn_cast<RankedTensorType>(dyn_cast<NamedAxesTensorType>(getInputs()[locations[i].first].getType()).getTensorType()).getShape()[locations[i].second];
     
	if (dim != refDim)
	  return emitOpError() << "dimension mismatch for contracted axis '"
			       << axis << "'";
      }
    }
    
    return success();
  }
 
}
