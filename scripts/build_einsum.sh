mkdir -p build-ninja
cd build-ninja

LLVM_BUILD_DIR=$(realpath ../externals/llvm-project/build)
echo "pwd: $(pwd)"
echo "LLVM Build Dir: $LLVM_BUILD_DIR"
echo "MLIR DIR: $LLVM_BUILD_DIR/lib/cmake/mlir"
cmake -G Ninja .. \
    -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
    -DBUILD_DEPS=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Debug

ninja
