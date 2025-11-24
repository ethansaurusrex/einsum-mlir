# RUN: %python %s | FileCheck %s

import numpy as np
from einsum_mlir.ir import *
from einsum_mlir.ir import Dialect, AffineMap, AffineMapAttr, AffineDimExpr

from einsum_mlir.dialects import func
from einsum_mlir.dialects import einsum

from einsum_mlir.dialects.einsum import EinsumLL
from einsum_mlir.dialects.einsum import NamedAxesTensorType



def mlir_element_type_from_numpy_dtype(dtype):
    from einsum_mlir.ir import F32Type, F64Type, IntegerType

    if dtype == np.float32:
        return F32Type.get()
    if dtype == np.float64:
        return F64Type.get()
    if dtype == np.int32:
        return IntegerType.get_signless(32)
    if dtype == np.int64:
        return IntegerType.get_signless(64)
    if dtype == np.bool_:
        return IntegerType.get_signless(1)

    raise ValueError(f"Unsupported type: {dtype}")

def build_einsum_ll_body(func_op, indexing_maps_attr, iterator_types_attr, loop_order_attr):
    entry_block = func_op.body.blocks[0]
    arg0: Value = entry_block.arguments[0] # Input A
    arg1: Value = entry_block.arguments[1] # Input B

    result_type = func_op.type.results[0]
    
    einsum_op = EinsumLL(
        output=result_type,
        inputs=[arg0, arg1],
        equation="ik,kj->ij",
        indexing_maps=indexing_maps_attr,
        iterator_types=iterator_types_attr,
        loop_order=loop_order_attr,
    )

    func.ReturnOp([einsum_op.output])

def main():
    A = np.random.rand(3,4)
    B = np.random.rand(4,3)
    C = np.zeros((3, 3), dtype=np.float64)
    A_axis_names = ["i","k"]
    B_axis_names = ["k","j"]
    C_axis_names = ["i","j"]
    
    with Context() as ctx:
        with Location.unknown():
            einsum.register_dialect(ctx)
            i, j, k = AffineDimExpr.get(0), AffineDimExpr.get(1), AffineDimExpr.get(2)

            map_A = AffineMap.get(3, 0, [i, k])
            map_B = AffineMap.get(3, 0, [k, j])
            map_C = AffineMap.get(3, 0, [i, j])

            indexing_maps_attr = ArrayAttr.get([
                AffineMapAttr.get(map_A), 
                AffineMapAttr.get(map_B), 
                AffineMapAttr.get(map_C)
            ])            

            iterator_types_attr = ArrayAttr.get([
                StringAttr.get("parallel"), 
                StringAttr.get("parallel"), 
                StringAttr.get("reduction")
            ])

            # --- 3. Define Loop Order ---
            loop_order_attr = ArrayAttr.get([
                StringAttr.get("i"), 
                StringAttr.get("j"), 
                StringAttr.get("k")
            ])

            
    
            A_elem_type = mlir_element_type_from_numpy_dtype(A.dtype)
            A_tensor_type = RankedTensorType.get(A.shape, A_elem_type)        
            A_dense_attr = DenseElementsAttr.get(A, type = A_tensor_type)
            A_string_attrs = [StringAttr.get(name) for name in A_axis_names]
            A_axis_names_attr = ArrayAttr.get(A_string_attrs)
            A_named_tensor_type = NamedAxesTensorType.get(
                A_tensor_type, A_axis_names_attr
            )
            
            B_elem_type = mlir_element_type_from_numpy_dtype(B.dtype)
            B_tensor_type = RankedTensorType.get(B.shape, B_elem_type)        
            B_dense_attr = DenseElementsAttr.get(B, type = B_tensor_type)
            B_string_attrs = [StringAttr.get(name) for name in B_axis_names]
            B_axis_names_attr = ArrayAttr.get(B_string_attrs)
            B_named_tensor_type = NamedAxesTensorType.get(
                B_tensor_type, B_axis_names_attr
            )

            C_elem_type = mlir_element_type_from_numpy_dtype(C.dtype)
            C_tensor_type = RankedTensorType.get(C.shape, C_elem_type)        
            C_dense_attr = DenseElementsAttr.get(C, type = C_tensor_type)
            C_string_attrs = [StringAttr.get(name) for name in C_axis_names]
            C_axis_names_attr = ArrayAttr.get(C_string_attrs)
            C_named_tensor_type = NamedAxesTensorType.get(
            C_tensor_type, C_axis_names_attr
            )        
            
            fnty = FunctionType.get([A_named_tensor_type, B_named_tensor_type], [C_named_tensor_type])
            
            module = Module.create()

            body_builder_lambda = lambda func_op: build_einsum_ll_body(
                func_op, indexing_maps_attr, iterator_types_attr, loop_order_attr
            )
        
            with InsertionPoint.at_block_begin(module.body):
                func_op = func.FuncOp("main",
                                      fnty,
                                      visibility="public",
                                      body_builder=body_builder_lambda)
            
            print(module)
            

if __name__ == "__main__":
    main()

# CHECK: #map = affine_map<(d0, d1, d2) -> (d0, d2)>
# CHECK: #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
# CHECK: #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

# CHECK: module {
# CHECK:   func.func public @main(%[[ARG0:.*]]: !einsum.named_axes_tensor<["i", "k"] : tensor<3x4xf64>>, %[[ARG1:.*]]: !einsum.named_axes_tensor<["k", "j"] : tensor<4x3xf64>>) -> !einsum.named_axes_tensor<["i", "j"] : tensor<3x3xf64>> {
# CHECK:     %[[RES:.*]] = einsum.ll(%[[ARG0]], %[[ARG1]]
# CHECK-SAME:        : !einsum.named_axes_tensor<["i", "k"] : tensor<3x4xf64>>, !einsum.named_axes_tensor<["k", "j"] : tensor<4x3xf64>>) {equation = "ik,kj->ij", indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], loop_order = ["i", "j", "k"]} -> <["i", "j"] : tensor<3x3xf64>>
# CHECK:     return %[[RES]] : !einsum.named_axes_tensor<["i", "j"] : tensor<3x3xf64>>
# CHECK:   }
# CHECK: }    
