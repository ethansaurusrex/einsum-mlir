// RUN: einsum-opt %s --einsum-hl-to-ll | einsum-opt --einsum-ll-to-linalg | FileCheck %s

module {
  func.func @sum_all(
      %arg0: !einsum.named_axes_tensor<["i","j"]:tensor<2x3xf32>> // A (2x3)
  ) -> !einsum.named_axes_tensor<[]:tensor<f32>> {               // Output (scalar)
    
    // High-level Einsum op
    %0 = einsum.hl(%arg0 : !einsum.named_axes_tensor<["i","j"]:tensor<2x3xf32>>)
         { equation = "ij->" }
         -> !einsum.named_axes_tensor<[]:tensor<f32>>
    
    return %0 : !einsum.named_axes_tensor<[]:tensor<f32>>
  }
}

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> ()>

// CHECK-LABEL: func.func @sum_all
// CHECK-SAME: (%arg0: tensor<2x3xf32>) -> tensor<f32>

// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<f32>
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[INIT:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<f32>) -> tensor<f32>

// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map1]
// CHECK-SAME: iterator_types = ["reduction", "reduction"]
// CHECK-SAME: ins(%arg0 : tensor<2x3xf32>)
// CHECK-SAME: outs(%[[INIT]] : tensor<f32>)

// CHECK: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:   %[[ADD:.*]] = arith.addf %[[OUT]], %[[IN]] : f32
// CHECK:   linalg.yield %[[ADD]] : f32

// CHECK: return %[[RESULT]] : tensor<f32>

// CHECK-NOT: einsum.hl
// CHECK-NOT: einsum.ll
// CHECK-NOT: einsum.named_axes_tensor
