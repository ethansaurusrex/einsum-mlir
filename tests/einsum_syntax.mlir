// RUN: einsum-opt %s | FileCheck %s

module {
  // CHECK: einsum.named_axes_tensor
  func.func @main(%arg0: !einsum.named_axes_tensor<["i"]:tensor<4xf32>>) ->  !einsum.named_axes_tensor<["i"]:tensor<4xf32>> {
    return %arg0 : !einsum.named_axes_tensor<["i"]:tensor<4xf32>>
  }
}