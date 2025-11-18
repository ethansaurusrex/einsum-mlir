// RUN: einsum-opt %s --einsum-hl-to-ll | einsum-opt --einsum-ll-to-linalg | FileCheck %s

module {
  func.func @main(%arg0: !einsum.named_axes_tensor<["i","k"]:tensor<4x3xf32>>, %arg1: !einsum.named_axes_tensor<["k","j"]:tensor<3x4xf32>>) ->  !einsum.named_axes_tensor<["i","j"]:tensor<4x4xf32>> {
    %0 = einsum.hl(%arg0, %arg1
		: !einsum.named_axes_tensor<["i","k"]:tensor<4x3xf32>>,
       		  !einsum.named_axes_tensor<["k","j"]:tensor<3x4xf32>>)
		{ equation = "ik,kj->ij" }
      		-> !einsum.named_axes_tensor<["i","j"]:tensor<4x4xf32>>
    return %0 : !einsum.named_axes_tensor<["i","j"]:tensor<4x4xf32>>
  }
}

// CHECK: #map = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%arg0: tensor<4x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<4x4xf32>

// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<4x4xf32>) -> tensor<4x4xf32>

// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map1, #map2]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%arg0, %arg1 : tensor<4x3xf32>, tensor<3x4xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<4x4xf32>)

// CHECK: ^bb0(%[[IN0:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[IN0]], %[[IN1]] : f32
// CHECK:   %[[ADD:.*]] = arith.addf %[[OUT]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[ADD]] : f32

// CHECK: return %[[RESULT]] : tensor<4x4xf32>

// Verify no einsum ops remain
// CHECK-NOT: einsum.ll
// CHECK-NOT: einsum.named_axes_tensor
