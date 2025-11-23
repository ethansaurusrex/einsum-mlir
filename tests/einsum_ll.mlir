// RUN: einsum-opt %s

module {
  func.func @main(%arg0: !einsum.named_axes_tensor<["i","k"]:tensor<4x3xf32>>, %arg1: !einsum.named_axes_tensor<["k","j"]:tensor<3x4xf32>>) ->  !einsum.named_axes_tensor<["i","j"]:tensor<4x4xf32>> {
    %0 = einsum.ll(%arg0, %arg1
		: !einsum.named_axes_tensor<["i","k"]:tensor<4x3xf32>>,
       		  !einsum.named_axes_tensor<["k","j"]:tensor<3x4xf32>>)
		{
		  equation = "ik,kj->ij",
		  indexing_maps = [
			      affine_map<(d0,d1,d2) -> (d0,d2)>,  // A[i,k]
			      affine_map<(d0,d1,d2) -> (d2,d1)>,  // B[k,j]
			      affine_map<(d0,d1,d2) -> (d0,d1)>   // Output[i,j]
    			],
		  iterator_types = ["parallel","parallel","reduction"],
      loop_order = ["i", "j", "k"]
		}
      		-> !einsum.named_axes_tensor<["i","j"]:tensor<4x4xf32>>
    return %0 : !einsum.named_axes_tensor<["i","j"]:tensor<4x4xf32>>
  }
}
