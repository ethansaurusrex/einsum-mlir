// RUN: einsum-opt %s | FileCheck %s

module {
  // CHECK: einsum.named_axes
  func.func @main(%arg0: !einsum.named_axes) -> !einsum.named_axes {
    return %arg0 : !einsum.named_axes
  }
}