// RUN: einsum-opt %s --einsum-hl-to-ll | FileCheck %s

module {
  func.func @main(%arg0: !einsum.named_axes_tensor<["i","k"]:tensor<4x3xf32>>,
                  %arg1: !einsum.named_axes_tensor<["k","j"]:tensor<3x4xf32>>) 
                  -> !einsum.named_axes_tensor<["i","j"]:tensor<4x4xf32>> {
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
// CHECK: einsum.ll
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: loop_order = ["i", "j", "k"]  
