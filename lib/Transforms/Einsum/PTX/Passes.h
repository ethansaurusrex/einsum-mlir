#ifndef LIB_TRANSFORMS_EINSUM_PTX_PASSES_H_
#define LIB_TRANSFORMS_EINSUM_PTX_PASSES_H_

#include "ParallelLoopsTracker.h"
#include "ParallelLoopsRemover.h"
#include "PtxCodegen.h"
#include "LinearOffset.h"

namespace mlir {
namespace einsum {

// pass registrations for all the passes defined in the subdirectory
#define GEN_PASS_REGISTRATION
#include "lib/Transforms/Einsum/PTX/Passes.h.inc"

}  // namespace einsum
}  // namespace mlir

#endif  // TRANSFORMS_EINSUM_PTX_PASSES_H_
