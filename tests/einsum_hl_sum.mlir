// RUN: einsum-opt %s

module {
  func.func @sum_all(%arg0: !einsum.named_axes_tensor<["i","j"]:tensor<2x3xf32>>, %arg1: !einsum.named_axes_tensor<[]:tensor<f32>>) -> !einsum.named_axes_tensor<[]:tensor<f32>> {
    %0 = einsum.hl ins(%arg0 : !einsum.named_axes_tensor<["i","j"]:tensor<2x3xf32>>)
                outs(%arg1: !einsum.named_axes_tensor<[]:tensor<f32>>) 
         { equation = "ij->" }
	 -> !einsum.named_axes_tensor<[]:tensor<f32>>
    return %0 : !einsum.named_axes_tensor<[]:tensor<f32>>
  }
}
