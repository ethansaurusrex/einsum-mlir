# tools/einsum-opt/CMakeLists.txt

# Get the MLIR dialects and conversion libraries registered globally
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# Combine libraries needed for linking
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRRegisterAllExtensions
    MLIRPass
    MLIRFuncToLLVM
    MLIREinsum
    MLIREinsumPasses
    MLIREinsumToLinalg
)

# Create the executable
add_llvm_executable(einsum-opt
    einsum-opt.cpp
)

# Make sure it uses the proper LLVM/MLIR compile flags
llvm_update_compile_flags(einsum-opt)

# Link all the required libraries
target_link_libraries(einsum-opt PRIVATE ${LIBS})

# Verify all linkages
mlir_check_all_link_libraries(einsum-opt)
