import numpy as np
from einsum_mlir.ir import *
from einsum_mlir.dialects.einsum import NamedAxesTensorType, EinsumHL
from einsum_mlir.dialects import func

from einsum_mlir.passmanager import *

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

def parse_einsum_equation(equation: str):
    inputs, outputs = parse_equation(equation)
    inputs_axes = [list(inp) for inp in inputs]   # [["i","k"], ["k","j"]]
    output_axes = list(outputs)                    # ["i","j"]
    return inputs_axes, output_axes

def parse_equation(equation: str):
    lhs, rhs = equation.split("->")
    inputs = lhs.split(",")
    output = rhs
    return inputs, output

def make_named_tensor(arr: np.ndarray, axes: list[str]):
    elem = mlir_element_type_from_numpy_dtype(arr.dtype)
    tensor_type = RankedTensorType.get(arr.shape, elem)
    axis_attrs = [StringAttr.get(a) for a in axes]
    return NamedAxesTensorType.get(tensor_type, ArrayAttr.get(axis_attrs))

def build_einsum_body(func_op, equation):
    block = func_op.body.blocks[0]
    inputs = list(block.arguments)

    einsum_op = EinsumHL(
        output=func_op.type.results[0],
        inputs=inputs,
        out_operand=None,             # no explicit output buffer
        equation=equation,
    )

    func.ReturnOp([einsum_op.output])

import numpy as np

def infer_output_shape(equation: str, *arrays: np.ndarray) -> tuple[int, ...]:
    """
    Given an einsum equation and input arrays, return the shape of the output tensor.
    E.g., "ik,kj->ij" with arrays of shapes (2,3) and (3,4) returns (2,4)
    """
    # Split equation
    lhs, rhs = equation.split("->")
    input_labels = lhs.split(",")
    output_labels = rhs

    if len(input_labels) != len(arrays):
        raise ValueError(f"Expected {len(input_labels)} arrays but got {len(arrays)}")

    # Map each label to its dimension
    label_dim_map = {}

    for labels, array in zip(input_labels, arrays):
        if len(labels) != array.ndim:
            raise ValueError(f"Array with shape {array.shape} does not match labels {labels}")
        for label, dim in zip(labels, array.shape):
            if label in label_dim_map:
                if label_dim_map[label] != dim:
                    raise ValueError(f"Label '{label}' has conflicting dimensions: {label_dim_map[label]} vs {dim}")
            else:
                label_dim_map[label] = dim

    # Construct output shape
    out_shape = tuple(label_dim_map[label] for label in output_labels)
    return out_shape    

def einsum(equation: str, *arrays: np.ndarray):
    input_axes, output_axes = parse_einsum_equation(equation)
    print(parse_einsum_equation)

    with Context() as ctx, Location.unknown():
        from einsum_mlir.dialects import einsum as einsum_dialect
        einsum_dialect.register_dialect(ctx)

        mlir_inputs = []
        for arr, axes in zip(arrays, input_axes):
            mlir_inputs.append(
                make_named_tensor(arr, axes)   # your helper
            )

        output_shape = infer_output_shape(equation, *arrays)

        # 5. Allocate output NumPy array
        output_np = np.zeros(output_shape, arrays[0].dtype)
        mlir_output = make_named_tensor(output_np, output_axes)

        # 6. Build MLIR function
        fnty = FunctionType.get(
            mlir_inputs + [mlir_output],
            [mlir_output]
        )

        module = Module.create()
        with InsertionPoint.at_block_begin(module.body):
            def body(func_op):
                args = func_op.body.blocks[0].arguments
                einsum_op = EinsumHL(
                    output = func_op.type.results[0],
                    inputs = args[:-1],
                    out_operand = args[-1],
                    equation = equation,
                )
                func.ReturnOp([einsum_op.output])

            func.FuncOp("main", fnty, body_builder=body)

        pm = PassManager("any")
        pm.enable_ir_printing(print_after_all=True)
        pm.add('einsum-to-ptx-pipeline')
        pm.run(module.operation)

        return mlir_output  # execute the kernel


if __name__ == '__main__':
    A = np.random.rand(3,3)
    B = np.random.rand(3,3)
    C = einsum("ik,kj->ij", A, B)
