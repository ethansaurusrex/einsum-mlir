// RUN: einsum-opt %s

module {
  func.func @sum_all(%arg0: !einsum.named_axes_tensor<["i","j"]:tensor<2x3xf32>>) 
                     -> !einsum.named_axes_tensor<[]:tensor<f32>> {
    %0 = einsum.hl(%arg0 : !einsum.named_axes_tensor<["i","j"]:tensor<2x3xf32>>)
         { equation = "ij->" }
         -> !einsum.named_axes_tensor<[]:tensor<f32>>
    return %0 : !einsum.named_axes_tensor<[]:tensor<f32>>
  }
}
