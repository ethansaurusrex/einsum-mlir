add_mlir_public_c_api_library(MLIREinsumCAPI
  EinsumDialect.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/lib/CAPI
  ${PROJECT_SOURCE_DIR}/lib/Dialect/Einsum

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIREinsum
)
