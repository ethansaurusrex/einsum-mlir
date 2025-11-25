// RUN: einsum-opt %s --linalg-to-ptx-pipeline | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func public @main(%arg0: tensor<3x4xf64>, %arg1: tensor<4x3xf64>) -> tensor<3x3xf64> {
    %0 = tensor.empty() : tensor<3x3xf64>
    %cst = arith.constant 0.000000e+00 : f64
    %1 = linalg.fill ins(%cst : f64) outs(%0 : tensor<3x3xf64>) -> tensor<3x3xf64>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x4xf64>, tensor<4x3xf64>) outs(%1 : tensor<3x3xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %3 = arith.mulf %in, %in_0 : f64
      %4 = arith.addf %out, %3 : f64
      linalg.yield %4 : f64
    } -> tensor<3x3xf64>
    return %2 : tensor<3x3xf64>
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
