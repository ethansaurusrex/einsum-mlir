#!/bin/bash
set -e

# --- Configurable paths ---
LLVM_SRC="$PWD/externals/llvm-project"
BUILD_DIR="$LLVM_SRC/build"
INSTALL_DIR="$LLVM_SRC/install"

mkdir -p "$BUILD_DIR" "$INSTALL_DIR"

pushd "$BUILD_DIR"

# --- Build type: Release / Debug / RelWithDebInfo ---
BUILD_TYPE=${BUILD_TYPE:-Release}

# --- CMake invocation ---
cmake -G Ninja ../llvm \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON \
    -DLLVM_CCACHE_BUILD=ON \
    -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
    -DLLVM_USE_SPLIT_DWARF=ON

# --- Build LLVM + MLIR and run MLIR tests ---
cmake --build . --target check-mlir

popd
