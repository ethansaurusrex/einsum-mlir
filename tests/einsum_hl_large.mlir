// RUN: einsum-opt %s --einsum-hl-to-ll | FileCheck %s

// Test complex einsum with multiple contractions: pi,qj,ijkl,rk,sl->pqrs
// This represents: Out[p,q,r,s] = sum_{i,j,k,l} A[p,i] * B[q,j] * C[i,j,k,l] * D[r,k] * E[s,l]

module {
  func.func @large_einsum(
      %arg0: !einsum.named_axes_tensor<["p","i"]:tensor<2x3xf32>>,   // A
      %arg1: !einsum.named_axes_tensor<["q","j"]:tensor<4x5xf32>>,   // B
      %arg2: !einsum.named_axes_tensor<["i","j","k","l"]:tensor<3x5x6x7xf32>>, // C
      %arg3: !einsum.named_axes_tensor<["r","k"]:tensor<8x6xf32>>,   // D
      %arg4: !einsum.named_axes_tensor<["s","l"]:tensor<9x7xf32>>,    // E
      %arg5: !einsum.named_axes_tensor<["p","q","r","s"]:tensor<2x4x8x9xf32>> // outputXS
  ) -> !einsum.named_axes_tensor<["p","q","r","s"]:tensor<2x4x8x9xf32>> {
    
    %0 = einsum.hl ins(%arg0, %arg1, %arg2, %arg3, %arg4
                   : !einsum.named_axes_tensor<["p","i"]:tensor<2x3xf32>>,
                     !einsum.named_axes_tensor<["q","j"]:tensor<4x5xf32>>,
                     !einsum.named_axes_tensor<["i","j","k","l"]:tensor<3x5x6x7xf32>>,
                     !einsum.named_axes_tensor<["r","k"]:tensor<8x6xf32>>,
                     !einsum.named_axes_tensor<["s","l"]:tensor<9x7xf32>>)
		   outs(%arg5: !einsum.named_axes_tensor<["p","q","r","s"]:tensor<2x4x8x9xf32>>)
         { equation = "pi,qj,ijkl,rk,sl->pqrs" }
         -> !einsum.named_axes_tensor<["p","q","r","s"]:tensor<2x4x8x9xf32>>
    
    return %0 : !einsum.named_axes_tensor<["p","q","r","s"]:tensor<2x4x8x9xf32>>
  }
}

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d4)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d5)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d5, d6, d7)>
// CHECK: #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d6)>
// CHECK: #map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d7)>
// CHECK: #map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @large_einsum
// CHECK: einsum.ll
// CHECK-SAME: indexing_maps = [#map, #map1, #map2, #map3, #map4, #map5]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]
// CHECK-SAME: loop_order = ["p", "q", "r", "s", "i", "j", "k", "l"]
