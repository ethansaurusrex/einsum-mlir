// RUN: einsum-opt %s --einsum-to-ptx-pipeline | FileCheck %s

module {
  func.func @main(%arg0: !einsum.named_axes_tensor<["i","k"]:tensor<4x3xf32>>, %arg1: !einsum.named_axes_tensor<["k","j"]:tensor<3x4xf32>>, %arg2: !einsum.named_axes_tensor<["i","j"]:tensor<4x4xf32>>) -> !einsum.named_axes_tensor<["i","j"]:tensor<4x4xf32>> {
    %0 =  einsum.hl ins(%arg0 , %arg1: !einsum.named_axes_tensor<["i","k"]:tensor<4x3xf32>>, !einsum.named_axes_tensor<["k","j"]:tensor<3x4xf32>>) outs(%arg2: !einsum.named_axes_tensor<["i","j"]:tensor<4x4xf32>>)
		{ equation = "ik,kj->ij" }
		-> !einsum.named_axes_tensor<["i","j"]:tensor<4x4xf32>>	
    return %0 : !einsum.named_axes_tensor<["i","j"]:tensor<4x4xf32>>
  }
}

// Check that PTX kernel is emitted
// CHECK: .visible .entry main

// Check for host-side launcher stub
// CHECK: void main_launcher

// Ensure original linalg.generic is gone
// CHECK-NOT: linalg.generic

// Verifify no einsum ops or types remain
// CHECK-NOT: einsum.hl
// CHECK-NOT: einsum.ll
// CHECK-NOT: einsum.named_axes_tensor
