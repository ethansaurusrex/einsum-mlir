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
    auto inputs = getInputs();
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

     // Also check the output tensor type is NamedAxesTensorType
    Value output = getOutOperand();  
    auto outNat = dyn_cast<NamedAxesTensorType>(output.getType());
    if (!outNat)
      return emitOpError("output operand is not a NamedAxesTensorType");
    auto outTensorType = dyn_cast<RankedTensorType>(outNat.getTensorType());
    if (!outTensorType)
      return emitOpError("output is not a ranked NamedAxesTensorType");

    auto resultType = dyn_cast<NamedAxesTensorType>(getOutput().getType());
    if(!resultType) {
      return emitOpError("returned result must be NamedAxesTensorType");
    }
    if(resultType != outNat) {
      return emitOpError("output result type must be equal to output operand type");
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


    // === 6) Verify output axis names and shape match RHS ===
    auto outAxisNames = outNat.getAxisNames();
    auto outShape = outTensorType.getShape();
    
    // (a) RHS rank must match output rank
    if (rhsPart.size() != outAxisNames.size()) {
      return emitOpError() << "output rank (" << outAxisNames.size()
			   << ") does not match RHS rank (" << rhsPart.size() << ")";
    }


    // (b) Axis names: RHS chars correspond to output axis names 1:1
    for (size_t r = 0; r < rhsPart.size(); ++r) {
      char rhsAxis = rhsPart[r];
      
      // axisNames is ArrayRef<Attribute>, so cast
      auto outAxisAttr = llvm::dyn_cast<StringAttr>(outAxisNames[r]);
      if (!outAxisAttr)
	return emitOpError("output axis is not a StringAttr");
      
      StringRef outAxis = outAxisAttr.getValue();
      
      if (outAxis.size() != 1 || outAxis[0] != rhsAxis)
	return emitOpError() << "output axis " << r << " should be '" << rhsAxis
			     << "' but is '" << outAxis << "'";
    }

    // (c) Shape consistency
    for (size_t r = 0; r < rhsPart.size(); ++r) {
      char rhsAxis = rhsPart[r];
      
    // Find any input that contains this RHS axis:
      std::optional<int64_t> chosenDim;
      
      for (size_t i = 0; i < inputAxisStrings.size(); ++i) {
	StringRef axes = inputAxisStrings[i];
	for (size_t pos = 0; pos < axes.size(); ++pos) {

	  if (axes[pos] != rhsAxis)
	    continue;
	  
	  auto nat = dyn_cast<NamedAxesTensorType>(inputs[i].getType());
	  auto tType = dyn_cast<RankedTensorType>(nat.getTensorType());
	  int64_t dim = tType.getShape()[pos];
	
	  if (!chosenDim) {
	    chosenDim = dim;
	  }
	  else if (*chosenDim != dim) {
	    return emitOpError() << "mismatched dimension for output axis '"
				 << rhsAxis << "'";
	  }
	}
      }
      
      if (!chosenDim) {
	return emitOpError() << "RHS axis '" << rhsAxis
			     << "' does not appear in any input operand";
      }

      // Compare against output shape
      if (outShape[r] != *chosenDim) {
	return emitOpError() << "output axis '" << rhsAxis << "' has dimension "
                           << outShape[r] << ", expected " << *chosenDim;
      }
    }
  
    return success();
  }
 
}
