#!/bin/bash
set -e

# --- Configurable paths ---
LLVM_SRC="$PWD/externals/llvm-project"
BUILD_DIR="$LLVM_SRC/build"
INSTALL_DIR="$LLVM_SRC/install"

mkdir -p "$BUILD_DIR" "$INSTALL_DIR"

# --- Detect Python ---
PYTHON_EXE=$(command -v python3 || command -v python || true)
if [[ -z "$PYTHON_EXE" ]]; then
    echo "Error: No Python interpreter found" >&2
    exit 1
fi
echo "Using Python: $PYTHON_EXE"

# --- Install MLIR Python requirements ---
REQ_FILE="$LLVM_SRC/mlir/python/requirements.txt"
if [[ -f "$REQ_FILE" ]]; then
    echo "Installing MLIR Python requirements..."
    "$PYTHON_EXE" -m pip install -r "$REQ_FILE"
else
    echo "Warning: requirements.txt not found at $REQ_FILE"
fi

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
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(command -v python3)

# --- Build LLVM + MLIR and run MLIR tests ---
cmake --build . --target check-mlir

popd
