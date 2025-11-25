# Einsum MLIR Dialect:

This repository implements an **Einsum** dialect within MLIR, providing a series of transformations to lower high-level symbolic matrix operations into low-level, executable loop structures suitable for optimization and code generation.

## Building

This project uses **CMake** and **Ninja**.  
You may either use your own LLVM/MLIR build or let the provided script build one.

### Prerequisites

It is strongly recommended by MLIR to use a python virtual environment when building python bindings:

```
python -m venv ~/.venv/einsum_mlir
source ~/.venv/einsum_mlir/bin/activate
```

The bindings will be built using the `scripts/build_llvm.sh` script (see [Build](#2-build))


### 1. Clone the repository

#### **If you do *not* already have `llvm-project`**
```bash
git clone --recurse-submodules https://github.com/ethansaurusrex/einsum-mlir.git
cd einsum-mlir
```

#### **If you already have an LLVM checkout**
```bash
git clone https://github.com/ethansaurusrex/einsum-mlir.git
```

Ensure that:

```
einsum-mlir/externals/llvm-project
```

is either:
- a symlink to your own `llvm-project` checkout, or  
- replaced with your checkout directory.


### 2. Build

#### **Option A: Build LLVM + Einsum dialect together**
```bash
./scripts/build_llvm.sh
./scripts/build_einsum.sh
```

#### **Option B: Use an existing LLVM build**
```bash
./scripts/build_einsum.sh
```

These scripts produce out-of-tree builds under `build/`.


## Overview

### Einsum Notation

The lowering pipeline has two major steps:

```
einsum.hl  →  einsum.ll  →  linalg.generic
```

The goal is to take an equation such as:

```
ik,kj->ij
```

and automatically generate the full loop specification (indexing maps, iterator types, and loop order), eventually producing a canonical loop nest expressed as a `linalg.generic`.


#### 1. High-Level to Low-Level (`einsum.hl` → `einsum.ll`)

This transformation converts the purely symbolic **`einsum.hl`** operation into a more structured **`einsum.ll`** operation.

* **Input (`einsum.hl`):** Uses only the input tensors and the `equation` string (e.g., `"ik,kj->ij"`).
* **Transformation Logic:** The equation string is parsed to determine the roles of each index (subscript), establishing the complete loop space.
* **Output (`einsum.ll`):** Explicitly stores the derived loop structure information, including:
    * **`indexing_maps`**: MLIR Affine Maps defining how each input/output tensor is accessed within the unified loop space.
    * **`iterator_types`**: An array indicating whether each loop index is **`parallel`** (kept in the output) or **`reduction`** (summed over).
    * **`loop_order`**: The explicit ordering of the loops (e.g., parallel indices first, then reduction indices).

#### 2. Low-Level to Linalg (`einsum.ll` → `linalg.generic`)

This transformation eliminates the `einsum.ll` operation entirely by replacing it with a **`linalg.generic`** operation.

* The attributes stored in `einsum.ll` (indexing maps and iterator types) are directly transferred to the `linalg.generic` op.
* This results in a fully specified loop-based kernel. The transformation also inserts the necessary boilerplate:
    * **Buffer Allocation:** A `tensor.empty` op is created for the output tensor.
    * **Initialization:** A `linalg.fill` op initializes the reduction variable (usually to zero).
    * **Region Body:** The required accumulation logic (e.g., $\text{multiply} + \text{add}$) is inserted into the `linalg.generic` op's body region, completing the transformation into a standard, MLIR-friendly operation ready for further optimization (e.g., tiling, fusion, vectorization).


## Pass pipeline

```
einsum-opt input.mlir \
      --einsum-hl-to-ll \
      --einsum-ll-to-linalg
```

## Example

Suppose we have the following einsum 2D matrix-multiply:

```
ik,kj->ij
```

This represents a contraction of the `k`-th dimension. Eventually we will have a python binding to be able to use this, in its stead we will use `numpy` as an example front-end:

```
import numpy as np

# Example matrices
A = np.random.randn(4, 3)   # Shape (4, 3)
B = np.random.randn(3, 4)   # Shape (3, 4)

C = np.einsum("ik,kj->ij", A, B)

print("A:\n", A)
print("B:\n", B)
print("C = A @ B (via einsum):\n", C)
```

Our (future) front end will translate this into an `mlir` module containing the `einsum.hl` op:

```mlir
module {
  func.func @main(%arg0: !einsum.named_axes_tensor<["i","k"]:tensor<4x3xf32>>, %arg1: !einsum.named_axes_tensor<["k","j"]:tensor<3x4xf32>>) ->  !einsum.named_axes_tensor<["i","j"]:tensor<4x4xf32>> {
    %0 = einsum.hl(%arg0, %arg1
		: !einsum.named_axes_tensor<["i","k"]:tensor<4x3xf32>>,
       		  !einsum.named_axes_tensor<["k","j"]:tensor<3x4xf32>>)
    {
      equation = "ik,kj->ij"
    }	-> !einsum.named_axes_tensor<["i","j"]:tensor<4x4xf32>>
    return %0 : !einsum.named_axes_tensor<["i","j"]:tensor<4x4xf32>>
  }
}
```

This will then be lowered using the `--einsum-hl-to-ll` pass to result in the `einsum.ll` op:

```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%arg0: !einsum.named_axes_tensor<["i", "k"] : tensor<4x3xf32>>, %arg1: !einsum.named_axes_tensor<["k", "j"] : tensor<3x4xf32>>) -> !einsum.named_axes_tensor<["i", "j"] : tensor<4x4xf32>> {
    %0 = einsum.ll(%arg0, %arg1 : !einsum.named_axes_tensor<["i", "k"] : tensor<4x3xf32>>, !einsum.named_axes_tensor<["k", "j"] : tensor<3x4xf32>>) {
      equation = "ik,kj->ij",
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      loop_order = ["i", "j", "k"]
    } -> <["i", "j"] : tensor<4x4xf32>>
    return %0 : !einsum.named_axes_tensor<["i", "j"] : tensor<4x4xf32>>
  }
}
```

Finally this will be lowered, and the `einsum.named_axes_tensor` converted to the underlying tensor type:

```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<4x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x3xf32>, tensor<3x4xf32>) outs(%1 : tensor<4x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<4x4xf32>
    return %2 : tensor<4x4xf32>
  }
}
```

## Using MLIR Passes to lower to LLVM

Once we have a `linalg.generic` op we can do a series of lowerings to a number of backends using the builtin passes.

> [!NOTE]  
> Given that we are running a number of builtin passes on our input we will inevitably produce non-optimal code. This is not to say it is not possible to create performant code with builtin passes, just that using these passes in the manner we are doing so will not take advantage of many of the performance modifications you can see in high-end compilers. For example see: [IREE](https://github.com/iree-org/iree)v for a better example of targetting multiple backends in a performant manner.

### Full pass:
```bash
mlir-opt input.mlir \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-math-to-llvm \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts
```

given the following input:
```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<4x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x3xf32>, tensor<3x4xf32>) outs(%arg2 : tensor<4x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
```

### `--one-shot-bufferize="bufferize-function-boundaries"`
This pass converts ops using `tensor` (value) semantics into ops using `memref` (reference) semantics, transforming out immutable tensor usage into explicit reads and writes. The argument `bufferize-function-boundaries` changes the function signature from taking `tensor`s to taking `memref`s forcing the caller to handle memory allocation.
```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%arg0: memref<4x3xf32, strided<[?, ?], offset: ?>>, %arg1: memref<3x4xf32, strided<[?, ?], offset: ?>>, %arg2: memref<4x4xf32, strided<[?, ?], offset: ?>>) -> memref<4x4xf32, strided<[?, ?], offset: ?>> {
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<4x3xf32, strided<[?, ?], offset: ?>>, memref<3x4xf32, strided<[?, ?], offset: ?>>) outs(%arg2 : memref<4x4xf32, strided<[?, ?], offset: ?>>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }
    return %arg2 : memref<4x4xf32, strided<[?, ?], offset: ?>>
  }
}
```

### `--convert-linalg-to-loops`

This pass replaces the declarative `linalg.generic` operation with imperative, nested loops using the `scf` (Structured Control Flow) dialect. It analyzes the `indexing_maps` and `iterator_types` to generate the correct loop nest (in this case, 3 loops for `i`, `j`, `k`).

```mlir
module {
  func.func @main(%arg0: memref<4x3xf32, strided<[?, ?], offset: ?>>, %arg1: memref<3x4xf32, strided<[?, ?], offset: ?>>, %arg2: memref<4x4xf32, strided<[?, ?], offset: ?>>) -> memref<4x4xf32, strided<[?, ?], offset: ?>> {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %arg3 = %c0 to %c4 step %c1 {
      scf.for %arg4 = %c0 to %c4 step %c1 {
        scf.for %arg5 = %c0 to %c3 step %c1 {
          %0 = memref.load %arg0[%arg3, %arg5] : memref<4x3xf32, strided<[?, ?], offset: ?>>
          %1 = memref.load %arg1[%arg5, %arg4] : memref<3x4xf32, strided<[?, ?], offset: ?>>
          %2 = memref.load %arg2[%arg3, %arg4] : memref<4x4xf32, strided<[?, ?], offset: ?>>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %arg2[%arg3, %arg4] : memref<4x4xf32, strided<[?, ?], offset: ?>>
        }
      }
    }
    return %arg2 : memref<4x4xf32, strided<[?, ?], offset: ?>>
  }
}
```

### `--convert-scf-to-cf`

This pass lowers structured loops (`scf.for`) into a standard Control Flow Graph (CFG) as you would see in a traditional control flow-based compiler like LLVM. It removes the concept of a "loop" and replaces it with Basic Blocks (`^bb`), conditional branches (`cf.cond_br`), and unconditional jumps (`cf.br`). 

```mlir
module {
  func.func @main(%arg0: memref<4x3xf32, strided<[?, ?], offset: ?>>, %arg1: memref<3x4xf32, strided<[?, ?], offset: ?>>, %arg2: memref<4x4xf32, strided<[?, ?], offset: ?>>) -> memref<4x4xf32, strided<[?, ?], offset: ?>> {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
    %1 = arith.cmpi slt, %0, %c4 : index
    cf.cond_br %1, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
    %3 = arith.cmpi slt, %2, %c4 : index
    cf.cond_br %3, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%c0 : index)
  ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
    %5 = arith.cmpi slt, %4, %c3 : index
    cf.cond_br %5, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %6 = memref.load %arg0[%0, %4] : memref<4x3xf32, strided<[?, ?], offset: ?>>
    %7 = memref.load %arg1[%4, %2] : memref<3x4xf32, strided<[?, ?], offset: ?>>
    %8 = memref.load %arg2[%0, %2] : memref<4x4xf32, strided<[?, ?], offset: ?>>
    %9 = arith.mulf %6, %7 : f32
    %10 = arith.addf %8, %9 : f32
    memref.store %10, %arg2[%0, %2] : memref<4x4xf32, strided<[?, ?], offset: ?>>
    %11 = arith.addi %4, %c1 : index
    cf.br ^bb5(%11 : index)
  ^bb7:  // pred: ^bb5
    %12 = arith.addi %2, %c1 : index
    cf.br ^bb3(%12 : index)
  ^bb8:  // pred: ^bb3
    %13 = arith.addi %0, %c1 : index
    cf.br ^bb1(%13 : index)
  ^bb9:  // pred: ^bb1
    return %arg2 : memref<4x4xf32, strided<[?, ?], offset: ?>>
  }
}
```

### `--convert-cf-to-llvm`

This pass converts the control flow dialect into llvm ops.

```mlir
module {
  func.func @main(%arg0: memref<4x3xf32, strided<[?, ?], offset: ?>>, %arg1: memref<3x4xf32, strided<[?, ?], offset: ?>>, %arg2: memref<4x4xf32, strided<[?, ?], offset: ?>>) -> memref<4x4xf32, strided<[?, ?], offset: ?>> {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %c0 : index to i64
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    llvm.br ^bb1(%0 : i64)
  ^bb1(%1: i64):  // 2 preds: ^bb0, ^bb8
    %2 = builtin.unrealized_conversion_cast %1 : i64 to index
    %3 = arith.cmpi slt, %2, %c4 : index
    llvm.cond_br %3, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%0 : i64)
  ^bb3(%4: i64):  // 2 preds: ^bb2, ^bb7
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = arith.cmpi slt, %5, %c4 : index
    llvm.cond_br %6, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%0 : i64)
  ^bb5(%7: i64):  // 2 preds: ^bb4, ^bb6
    %8 = builtin.unrealized_conversion_cast %7 : i64 to index
    %9 = arith.cmpi slt, %8, %c3 : index
    llvm.cond_br %9, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %10 = memref.load %arg0[%2, %8] : memref<4x3xf32, strided<[?, ?], offset: ?>>
    %11 = memref.load %arg1[%8, %5] : memref<3x4xf32, strided<[?, ?], offset: ?>>
    %12 = memref.load %arg2[%2, %5] : memref<4x4xf32, strided<[?, ?], offset: ?>>
    %13 = arith.mulf %10, %11 : f32
    %14 = arith.addf %12, %13 : f32
    memref.store %14, %arg2[%2, %5] : memref<4x4xf32, strided<[?, ?], offset: ?>>
    %15 = arith.addi %8, %c1 : index
    %16 = builtin.unrealized_conversion_cast %15 : index to i64
    llvm.br ^bb5(%16 : i64)
  ^bb7:  // pred: ^bb5
    %17 = arith.addi %5, %c1 : index
    %18 = builtin.unrealized_conversion_cast %17 : index to i64
    llvm.br ^bb3(%18 : i64)
  ^bb8:  // pred: ^bb3
    %19 = arith.addi %2, %c1 : index
    %20 = builtin.unrealized_conversion_cast %19 : index to i64
    llvm.br ^bb1(%20 : i64)
  ^bb9:  // pred: ^bb1
    return %arg2 : memref<4x4xf32, strided<[?, ?], offset: ?>>
  }
}
```

### `--convert-arith-to-llvm` and `--convert-math-to-llvm`

These passes convert dialect-specific arithmetic operations (like `arith.addf` or `arith.cmp`) into their direct LLVM dialect equivalents (`llvm.fadd`, `llvm.icmp`). This ensures the instructions map 1-to-1 with LLVM IR instructions. You can see below that `%15 = arith.mulf %12, %13 : f64`  %28 = llvm.fmul %25, %26 : f64` now becomes `%28 = llvm.fmul %25, %26 : f64`

```mlir
module {
  func.func @main(%arg0: memref<4x3xf32, strided<[?, ?], offset: ?>>, %arg1: memref<3x4xf32, strided<[?, ?], offset: ?>>, %arg2: memref<4x4xf32, strided<[?, ?], offset: ?>>) -> memref<4x4xf32, strided<[?, ?], offset: ?>> {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = builtin.unrealized_conversion_cast %1 : index to i64
    %3 = llvm.mlir.constant(4 : index) : i64
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(3 : index) : i64
    llvm.br ^bb1(%2 : i64)
  ^bb1(%6: i64):  // 2 preds: ^bb0, ^bb8
    %7 = builtin.unrealized_conversion_cast %6 : i64 to index
    %8 = llvm.icmp "slt" %6, %3 : i64
    llvm.cond_br %8, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%2 : i64)
  ^bb3(%9: i64):  // 2 preds: ^bb2, ^bb7
    %10 = builtin.unrealized_conversion_cast %9 : i64 to index
    %11 = llvm.icmp "slt" %9, %3 : i64
    llvm.cond_br %11, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%2 : i64)
  ^bb5(%12: i64):  // 2 preds: ^bb4, ^bb6
    %13 = builtin.unrealized_conversion_cast %12 : i64 to index
    %14 = llvm.icmp "slt" %12, %5 : i64
    llvm.cond_br %14, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %15 = memref.load %arg0[%7, %13] : memref<4x3xf32, strided<[?, ?], offset: ?>>
    %16 = memref.load %arg1[%13, %10] : memref<3x4xf32, strided<[?, ?], offset: ?>>
    %17 = memref.load %arg2[%7, %10] : memref<4x4xf32, strided<[?, ?], offset: ?>>
    %18 = llvm.fmul %15, %16 : f32
    %19 = llvm.fadd %17, %18 : f32
    memref.store %19, %arg2[%7, %10] : memref<4x4xf32, strided<[?, ?], offset: ?>>
    %20 = llvm.add %12, %4 : i64
    %21 = builtin.unrealized_conversion_cast %20 : i64 to index
    %22 = builtin.unrealized_conversion_cast %21 : index to i64
    llvm.br ^bb5(%22 : i64)
  ^bb7:  // pred: ^bb5
    %23 = llvm.add %9, %4 : i64
    %24 = builtin.unrealized_conversion_cast %23 : i64 to index
    %25 = builtin.unrealized_conversion_cast %24 : index to i64
    llvm.br ^bb3(%25 : i64)
  ^bb8:  // pred: ^bb3
    %26 = llvm.add %6, %4 : i64
    %27 = builtin.unrealized_conversion_cast %26 : i64 to index
    %28 = builtin.unrealized_conversion_cast %27 : index to i64
    llvm.br ^bb1(%28 : i64)
  ^bb9:  // pred: ^bb1
    return %arg2 : memref<4x4xf32, strided<[?, ?], offset: ?>>
  }
}
```

### `--convert-func-to-llvm`

This pass transforms the function signature and call conventions. It converts `func.func` to `llvm.func` and lowers high-level types in the signature into types compatible with the LLVM ABI (often converting types into pointers or raw structs, in our case `memref` becomes `llvm.ptr` types).

```
module {
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg7, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg8, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg9, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg10, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg12, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg11, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg13, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = builtin.unrealized_conversion_cast %7 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<3x4xf32, strided<[?, ?], offset: ?>>
    %9 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg0, %9[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg1, %10[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg2, %11[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg3, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg5, %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg4, %14[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %arg6, %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = builtin.unrealized_conversion_cast %16 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<4x3xf32, strided<[?, ?], offset: ?>>
    %18 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.insertvalue %arg14, %18[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %arg15, %19[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %arg16, %20[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %arg17, %21[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %arg19, %22[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.insertvalue %arg18, %23[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %arg20, %24[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = builtin.unrealized_conversion_cast %25 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<4x4xf32, strided<[?, ?], offset: ?>>
    %27 = llvm.mlir.constant(0 : index) : i64
    %28 = builtin.unrealized_conversion_cast %27 : i64 to index
    %29 = builtin.unrealized_conversion_cast %28 : index to i64
    %30 = llvm.mlir.constant(4 : index) : i64
    %31 = llvm.mlir.constant(1 : index) : i64
    %32 = llvm.mlir.constant(3 : index) : i64
    llvm.br ^bb1(%29 : i64)
  ^bb1(%33: i64):  // 2 preds: ^bb0, ^bb8
    %34 = builtin.unrealized_conversion_cast %33 : i64 to index
    %35 = llvm.icmp "slt" %33, %30 : i64
    llvm.cond_br %35, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%29 : i64)
  ^bb3(%36: i64):  // 2 preds: ^bb2, ^bb7
    %37 = builtin.unrealized_conversion_cast %36 : i64 to index
    %38 = llvm.icmp "slt" %36, %30 : i64
    llvm.cond_br %38, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%29 : i64)
  ^bb5(%39: i64):  // 2 preds: ^bb4, ^bb6
    %40 = builtin.unrealized_conversion_cast %39 : i64 to index
    %41 = llvm.icmp "slt" %39, %32 : i64
    llvm.cond_br %41, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %42 = memref.load %17[%34, %40] : memref<4x3xf32, strided<[?, ?], offset: ?>>
    %43 = memref.load %8[%40, %37] : memref<3x4xf32, strided<[?, ?], offset: ?>>
    %44 = memref.load %26[%34, %37] : memref<4x4xf32, strided<[?, ?], offset: ?>>
    %45 = llvm.fmul %42, %43 : f32
    %46 = llvm.fadd %44, %45 : f32
    memref.store %46, %26[%34, %37] : memref<4x4xf32, strided<[?, ?], offset: ?>>
    %47 = llvm.add %39, %31 : i64
    %48 = builtin.unrealized_conversion_cast %47 : i64 to index
    %49 = builtin.unrealized_conversion_cast %48 : index to i64
    llvm.br ^bb5(%49 : i64)
  ^bb7:  // pred: ^bb5
    %50 = llvm.add %36, %31 : i64
    %51 = builtin.unrealized_conversion_cast %50 : i64 to index
    %52 = builtin.unrealized_conversion_cast %51 : index to i64
    llvm.br ^bb3(%52 : i64)
  ^bb8:  // pred: ^bb3
    %53 = llvm.add %33, %31 : i64
    %54 = builtin.unrealized_conversion_cast %53 : i64 to index
    %55 = builtin.unrealized_conversion_cast %54 : index to i64
    llvm.br ^bb1(%55 : i64)
  ^bb9:  // pred: ^bb1
    llvm.return %25 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
}
```

### `--finalize-memref-to-llvm`

This pass lowers the `memref.load` and `memref.store` operations (and the `memref` type itself) into LLVM pointer arithmetic. It creates the standard `MemRef Descriptor` struct (containing allocated pointer, aligned pointer, offset, sizes, and strides) and uses `llvm.getelementptr` (aka GEP) to access data.


```mlir
module {
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg7, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg8, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg9, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg10, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg12, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg11, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg13, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = builtin.unrealized_conversion_cast %7 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<3x4xf32, strided<[?, ?], offset: ?>>
    %9 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg0, %9[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg1, %10[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg2, %11[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg3, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg5, %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg4, %14[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %arg6, %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = builtin.unrealized_conversion_cast %16 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<4x3xf32, strided<[?, ?], offset: ?>>
    %18 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.insertvalue %arg14, %18[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %arg15, %19[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %arg16, %20[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %arg17, %21[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %arg19, %22[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.insertvalue %arg18, %23[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %arg20, %24[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = builtin.unrealized_conversion_cast %25 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<4x4xf32, strided<[?, ?], offset: ?>>
    %27 = llvm.mlir.constant(0 : index) : i64
    %28 = builtin.unrealized_conversion_cast %27 : i64 to index
    %29 = builtin.unrealized_conversion_cast %28 : index to i64
    %30 = llvm.mlir.constant(4 : index) : i64
    %31 = llvm.mlir.constant(1 : index) : i64
    %32 = llvm.mlir.constant(3 : index) : i64
    llvm.br ^bb1(%29 : i64)
  ^bb1(%33: i64):  // 2 preds: ^bb0, ^bb8
    %34 = builtin.unrealized_conversion_cast %33 : i64 to index
    %35 = llvm.icmp "slt" %33, %30 : i64
    llvm.cond_br %35, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%29 : i64)
  ^bb3(%36: i64):  // 2 preds: ^bb2, ^bb7
    %37 = builtin.unrealized_conversion_cast %36 : i64 to index
    %38 = llvm.icmp "slt" %36, %30 : i64
    llvm.cond_br %38, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%29 : i64)
  ^bb5(%39: i64):  // 2 preds: ^bb4, ^bb6
    %40 = builtin.unrealized_conversion_cast %39 : i64 to index
    %41 = llvm.icmp "slt" %39, %32 : i64
    llvm.cond_br %41, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %42 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.extractvalue %16[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.getelementptr %42[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %45 = llvm.extractvalue %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.mul %33, %45 overflow<nsw, nuw> : i64
    %47 = llvm.extractvalue %16[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.mul %39, %47 overflow<nsw, nuw> : i64
    %49 = llvm.add %46, %48 overflow<nsw, nuw> : i64
    %50 = llvm.getelementptr inbounds|nuw %44[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %51 = llvm.load %50 : !llvm.ptr -> f32
    %52 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.getelementptr %52[%53] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %55 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.mul %39, %55 overflow<nsw, nuw> : i64
    %57 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.mul %36, %57 overflow<nsw, nuw> : i64
    %59 = llvm.add %56, %58 overflow<nsw, nuw> : i64
    %60 = llvm.getelementptr inbounds|nuw %54[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %61 = llvm.load %60 : !llvm.ptr -> f32
    %62 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.extractvalue %25[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.getelementptr %62[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %65 = llvm.extractvalue %25[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mul %33, %65 overflow<nsw, nuw> : i64
    %67 = llvm.extractvalue %25[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.mul %36, %67 overflow<nsw, nuw> : i64
    %69 = llvm.add %66, %68 overflow<nsw, nuw> : i64
    %70 = llvm.getelementptr inbounds|nuw %64[%69] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %71 = llvm.load %70 : !llvm.ptr -> f32
    %72 = llvm.fmul %51, %61 : f32
    %73 = llvm.fadd %71, %72 : f32
    %74 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.extractvalue %25[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %76 = llvm.getelementptr %74[%75] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %77 = llvm.extractvalue %25[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.mul %33, %77 overflow<nsw, nuw> : i64
    %79 = llvm.extractvalue %25[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.mul %36, %79 overflow<nsw, nuw> : i64
    %81 = llvm.add %78, %80 overflow<nsw, nuw> : i64
    %82 = llvm.getelementptr inbounds|nuw %76[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %73, %82 : f32, !llvm.ptr
    %83 = llvm.add %39, %31 : i64
    %84 = builtin.unrealized_conversion_cast %83 : i64 to index
    %85 = builtin.unrealized_conversion_cast %84 : index to i64
    llvm.br ^bb5(%85 : i64)
  ^bb7:  // pred: ^bb5
    %86 = llvm.add %36, %31 : i64
    %87 = builtin.unrealized_conversion_cast %86 : i64 to index
    %88 = builtin.unrealized_conversion_cast %87 : index to i64
    llvm.br ^bb3(%88 : i64)
  ^bb8:  // pred: ^bb3
    %89 = llvm.add %33, %31 : i64
    %90 = builtin.unrealized_conversion_cast %89 : i64 to index
    %91 = builtin.unrealized_conversion_cast %90 : index to i64
    llvm.br ^bb1(%91 : i64)
  ^bb9:  // pred: ^bb1
    llvm.return %25 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
}
```

### `--reconcile-unrealized-casts`

This pass cleans up the IR by resolving `builtin.unrealized_conversion_cast` operations. During previous passes, if a type couldn't be fully converted immediately (e.g., between an `arith` integer and an `llvm` integer), a temporary cast was inserted. This pass ensures all those casts are resolved, confirming the entire module is valid LLVM dialect.

```mlir
module {
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg7, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg8, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg9, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg10, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg12, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg11, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg13, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg0, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg1, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg2, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg3, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg5, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg4, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg6, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %arg14, %16[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %arg15, %17[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %arg16, %18[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %arg17, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %arg19, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %arg18, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %arg20, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(0 : index) : i64
    %25 = llvm.mlir.constant(4 : index) : i64
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.mlir.constant(3 : index) : i64
    llvm.br ^bb1(%24 : i64)
  ^bb1(%28: i64):  // 2 preds: ^bb0, ^bb8
    %29 = llvm.icmp "slt" %28, %25 : i64
    llvm.cond_br %29, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%24 : i64)
  ^bb3(%30: i64):  // 2 preds: ^bb2, ^bb7
    %31 = llvm.icmp "slt" %30, %25 : i64
    llvm.cond_br %31, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%24 : i64)
  ^bb5(%32: i64):  // 2 preds: ^bb4, ^bb6
    %33 = llvm.icmp "slt" %32, %27 : i64
    llvm.cond_br %33, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %34 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.getelementptr %34[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %37 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.mul %28, %37 overflow<nsw, nuw> : i64
    %39 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.mul %32, %39 overflow<nsw, nuw> : i64
    %41 = llvm.add %38, %40 overflow<nsw, nuw> : i64
    %42 = llvm.getelementptr inbounds|nuw %36[%41] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %43 = llvm.load %42 : !llvm.ptr -> f32
    %44 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %45 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.getelementptr %44[%45] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %47 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.mul %32, %47 overflow<nsw, nuw> : i64
    %49 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.mul %30, %49 overflow<nsw, nuw> : i64
    %51 = llvm.add %48, %50 overflow<nsw, nuw> : i64
    %52 = llvm.getelementptr inbounds|nuw %46[%51] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %53 = llvm.load %52 : !llvm.ptr -> f32
    %54 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.extractvalue %23[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %57 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.mul %28, %57 overflow<nsw, nuw> : i64
    %59 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.mul %30, %59 overflow<nsw, nuw> : i64
    %61 = llvm.add %58, %60 overflow<nsw, nuw> : i64
    %62 = llvm.getelementptr inbounds|nuw %56[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %63 = llvm.load %62 : !llvm.ptr -> f32
    %64 = llvm.fmul %43, %53 : f32
    %65 = llvm.fadd %63, %64 : f32
    %66 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.extractvalue %23[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.getelementptr %66[%67] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %69 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.mul %28, %69 overflow<nsw, nuw> : i64
    %71 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.mul %30, %71 overflow<nsw, nuw> : i64
    %73 = llvm.add %70, %72 overflow<nsw, nuw> : i64
    %74 = llvm.getelementptr inbounds|nuw %68[%73] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %65, %74 : f32, !llvm.ptr
    %75 = llvm.add %32, %26 : i64
    llvm.br ^bb5(%75 : i64)
  ^bb7:  // pred: ^bb5
    %76 = llvm.add %30, %26 : i64
    llvm.br ^bb3(%76 : i64)
  ^bb8:  // pred: ^bb3
    %77 = llvm.add %28, %26 : i64
    llvm.br ^bb1(%77 : i64)
  ^bb9:  // pred: ^bb1
    llvm.return %23 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
}
```


## PTX-Backend

All credit here goes to Govardhan@https://github.com/chioni16 (save for the tests :D )

### Example Pass and Output

## Sample PTX Backend Output

Running:

```bash
./build-ninja/tools/einsum-opt/einsum-opt \
  --linalg-generalize-named-ops \
  --empty-tensor-to-alloc-tensor \
  --one-shot-bufferize="bufferize-function-boundaries=1" \
  --convert-linalg-to-parallel-loops \
  --parallel-loops-tracker \
  --parallel-loops-remover \
  --scf-to-ptx \
  tmp/linalg.mlir
```
with the same input as above:
```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<4x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x3xf32>, tensor<3x4xf32>) outs(%arg2 : tensor<4x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
```


produces a full end-to-end lowering trace followed by generated PTX and a CUDA launcher stub.

### Parallel Loop Analysis

The backend prints diagnostics after each round of analysis:

```
┌─────────────────────────────────────────────┐
│  SCF Parallel Loop Count Summary           │
├─────────────────────────────────────────────┤
│  Total parallel loops: 2
│  Functions with parallel loops: 1
└─────────────────────────────────────────────┘
```

Later, after transformation passes remove or lower them:

```
┌─────────────────────────────────────────────┐
│  SCF Parallel Loop Count Summary           │
├─────────────────────────────────────────────┤
│  Total parallel loops: 0
│  Functions with parallel loops: 0
└─────────────────────────────────────────────┘
```

This demonstrates that the pipeline is locating SCF `parallel` loops, instrumenting them, and then lowering them to GPU thread-index arithmetic or serial loops depending on legality.

### Generated PTX (Excerpt)

The backend emits valid PTX headers, register declarations, thread-index arithmetic, and a kernel entry point:

```ptx
//
.version 7.0
.target sm_75
.address_size 64

.visible .entry main(
    .param .u64 param_0,
    .param .u64 param_1
)
{
    .reg .pred %p<30>;
    .reg .u32 %r<30>;
    .reg .u64 %rd<30>;
    .reg .f32 %f<30>;
    .reg .f64 %fd<30>;

    ld.param.u64 %rd0, [param_0];
    ld.param.u64 %rd1, [param_1];

    mov.u32 %r7, %tid.x;
    mov.u32 %r8, %ctaid.x;
    mov.u32 %r9, %ntid.x;
    mul.lo.u32 %r10, %r8, %r9;
    add.u32 %r11, %r10, %r7;
```

Thread-to-loop-index mapping is handled explicitly because SCF loops are being lowered directly to raw PTX control flow.

### Unsupported Operations

In this example, operations that do not yet have PTX-side lowering hooks are emitted as comments:

```
// Unsupported operation: memref.alloc
```

This is expected: PTX has no heap or stack allocation. Future support will insert host-side allocations or convert such ops into shared/global GPU buffers.

### Incomplete or Invalid Lowering Cases

Some PTX instructions contain placeholders due to missing address computations or unsupported index expressions:

```
div.u32 %r12, %r11, ;
rem.u32 %r13, %r11, ;
st.global.f64 [], %fd6;
```

These appear whenever the lowering lacks enough information to compute:

* loop bounds,
* induction variable expressions,
* memref address calculations,
* or constant divisors.

Fixing this requires improving legality checks and adding dedicated address-computation patterns for memrefs.

### Host-Side CUDA Launcher Stub

For convenience, the backend emits a simple launcher:

```cpp
void main_launcher(float *param_0, float *param_1)
{
    int threads = 0;
    int blocks = 0;
    main_kernel<<<blocks, threads>>>(param_0, param_1);
}
```

`threads` and `blocks` are placeholders; real launch configuration must be supplied by later passes or by user code.

### Final Lowered MLIR Module

After all SCF lowering and GPU conversion, the backend prints the final MLIR:

```mlir
module {
  func.func public @main(%arg0: memref<3x4xf64>, %arg1: memref<4x3xf64>) -> memref<3x3xf64> {
    %alloc = memref.alloc() : memref<3x3xf64>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      %4 = memref.load %arg0[%2, %arg2]
      %5 = memref.load %arg1[%arg2, %3]
      %6 = memref.load %alloc[%2, %3]
      %7 = arith.mulf %4, %5
      %8 = arith.addf %6, %7
      memref.store %8, %alloc[%2, %3]
    }
    return %alloc : memref<3x3xf64>
  }
}
```

