# python/CMakeLists.txt
include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

declare_mlir_python_sources(EinsumPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT EinsumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/einsum_mlir"
  TD_FILE dialects/EinsumOps.td
  SOURCES
    dialects/einsum.py
    #einsum_mlir/dialects/_ods_common.py
  DIALECT_NAME einsum
  DEPENDS MLIREinsumOpsIncGen
)

declare_mlir_python_extension(EinsumPythonSources.Pybind11Extension
  MODULE_NAME _einsumPybind11
  ADD_TO_PARENT EinsumPythonSources
  SOURCES
    EinsumModulePybind11.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS

  MLIRCAPIIR
    MLIRCAPIArith
    MLIRCAPITransforms
    MLIREinsumCAPI
  PYTHON_BINDINGS_LIBRARY
    pybind11
)

declare_mlir_python_extension(EinsumPythonSources.NanobindExtension
  MODULE_NAME _einsumNanobind
  ADD_TO_PARENT EinsumPythonSources
  SOURCES
    EinsumModuleNanobind.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIArith
    MLIRCAPITransforms
    MLIREinsumCAPI
    MLIRCAPIRegisterEverything
  PYTHON_BINDINGS_LIBRARY
    nanobind
)  

add_mlir_python_common_capi_library(EinsumPythonCAPI
  INSTALL_COMPONENT EinsumPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    EinsumPythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
    MLIRCAPIRegisterEverything
    MLIRPythonSources
)

set(EinsumPythonModules_ROOT_PREFIX "${CMAKE_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

set(_core_type_stub_sources
  _mlir/__init__.pyi
  _mlir/ir.pyi
  _mlir/passmanager.pyi
  _mlir/rewrite.pyi
)
get_target_property(_core_extension_srcs MLIRPythonExtension.Core INTERFACE_SOURCES)
mlir_generate_type_stubs(
  MODULE_NAME _mlir
  DEPENDS_TARGETS EinsumPythonModules.extension._mlir.dso
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_mlir_libs"
  OUTPUTS "${_core_type_stub_sources}"
  DEPENDS_TARGET_SRC_DEPS "${_core_extension_srcs}"
  IMPORT_PATHS "${EinsumPythonModules_ROOT_PREFIX}/_mlir_libs"
  VERBOSE
)
set(_mlir_typestub_gen_target "${NB_STUBGEN_CUSTOM_TARGET}")

list(TRANSFORM _core_type_stub_sources PREPEND "_mlir_libs/")
declare_mlir_python_sources(
  EinsumPythonExtension.Core.type_stub_gen
  ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
  ADD_TO_PARENT EinsumPythonSources
  SOURCES "${_core_type_stub_sources}"
)


get_target_property(_einsum_extension_srcs EinsumPythonSources.NanobindExtension INTERFACE_SOURCES)
mlir_generate_type_stubs(
  MODULE_NAME "${MLIR_PYTHON_PACKAGE_PREFIX}._mlir_libs._einsumNanobind"
  DEPENDS_TARGETS
  EinsumPythonModules.extension._mlir.dso
  EinsumPythonModules.extension._einsumNanobind.dso
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_mlir_libs"
  OUTPUTS
  _einsumNanobind/__init__.pyi
  _einsumNanobind/einsum.pyi
  DEPENDS_TARGET_SRC_DEPS "${_einsum_extension_srcs}"
  IMPORT_PATHS "${EinsumPythonModules_ROOT_PREFIX}/.."
)
set(_einsumNanobind_typestub_gen_target "${NB_STUBGEN_CUSTOM_TARGET}")

declare_mlir_python_sources(
  EinsumPythonSources.type_stub_gen
  ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
  ADD_TO_PARENT EinsumPythonSources
  SOURCES
  _mlir_libs/_einsumNanobind/__init__.pyi
  _mlir_libs/_einsumNanobind/einsum.pyi
)

set(_declared_sources
  EinsumPythonSources
  MLIRPythonExtension.RegisterEverything
  MLIRPythonSources 
)


add_mlir_python_modules(EinsumPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages/einsum_mlir"
  INSTALL_PREFIX "python_packages/einsum/einsum_mlir"
  DECLARED_SOURCES
    "${_declared_sources}"
  COMMON_CAPI_LINK_LIBS
    EinsumPythonCAPI
)

