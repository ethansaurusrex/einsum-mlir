# lib/Transforms/Einsum/PTX/CMakeLists.txt
# https://mlir.llvm.org/docs/Tutorials/CreatingADialect/#library-targets
set(LLVM_TARGET_DEFINITIONS Passes.td)
# perform the operation on the input
# args: output file (inside the build directory), specify sub op, flags (-name represents the group name under which the passes are added)
# generates 3 components
#     pass registration hooks
#     method declarations to be added to .h files
#     method implementations to be added to .cpp files
mlir_tablegen(Passes.h.inc -gen-pass-decls -name PtxGen)
# The target is used by other mlir components 
# creates a kind of dependency so that this mlir component is compiled
# otherwise isn't compiled
add_public_tablegen_target(MLIREinsumPtxPassesIncGen)
add_mlir_doc(Passes ArithPasses ./ -gen-pass-doc)


add_mlir_dialect_library(EinsumPTX
    ParallelLoopsTracker.cpp
    ParallelLoopsRemover.cpp
    PtxCodegen.cpp
    LinearOffset.cpp

    ADDITIONAL_HEADER_DIRS
      ${PROJECT_SOURCE_DIR}/lib/Transforms/Einsum/PTX

    # Dialect libraries often depend on the generation of header files from TableGen (specified using the DEPENDS keyword).
    DEPENDS
      MLIREinsumPtxPassesIncGen

    LINK_LIBS PUBLIC
)
