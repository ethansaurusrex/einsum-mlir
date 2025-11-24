# tests/CMakeLists.txt'
llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cmake.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cmake.cfg.py
)

set(LLVM_TOOLS_DIR ${CMAKE_BINARY_DIR}/tools)

set (EINSUM_TEST_DEPENDS
    FileCheck count not
    mlir-opt
    mlir-runner
  )

if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND STANDALONE_TEST_DEPENDS EinsumPythonModules)
endif()

add_lit_testsuite(check-einsum "Running Einsum tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${EINSUM_TEST_DEPENDS}
)

add_custom_target(check
  DEPENDS check-einsum
)
