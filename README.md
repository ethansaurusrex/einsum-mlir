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
  func.func public @main(%arg0: tensor<3x4xf64>, %arg1: tensor<4x3xf64>) -> tensor<3x3xf64> {
    %0 = tensor.empty() : tensor<3x3xf64>
    %cst = arith.constant 0.000000e+00 : f64
    %1 = linalg.fill ins(%cst : f64) outs(%0 : tensor<3x3xf64>) -> tensor<3x3xf64>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x4xf64>, tensor<4x3xf64>) outs(%1 : tensor<3x3xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %3 = arith.mulf %in, %in_0 : f64
      %4 = arith.addf %out, %3 : f64
      linalg.yield %4 : f64
    } -> tensor<3x3xf64>
    return %2 : tensor<3x3xf64>
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
  func.func public @main(%arg0: memref<3x4xf64, strided<[?, ?], offset: ?>>, %arg1: memref<4x3xf64, strided<[?, ?], offset: ?>>) -> memref<3x3xf64> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x3xf64>
    %cst = arith.constant 0.000000e+00 : f64
    linalg.fill ins(%cst : f64) outs(%alloc : memref<3x3xf64>)
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<3x4xf64, strided<[?, ?], offset: ?>>, memref<4x3xf64, strided<[?, ?], offset: ?>>) outs(%alloc : memref<3x3xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %0 = arith.mulf %in, %in_0 : f64
      %1 = arith.addf %out, %0 : f64
      linalg.yield %1 : f64
    }
    %cast = memref.cast %alloc : memref<3x3xf64> to memref<3x3xf64, strided<[?, ?], offset: ?>>
    return %alloc : memref<3x3xf64>
  }
}
```

### `--convert-linalg-to-loops`

This pass replaces the declarative `linalg.generic` operation with imperative, nested loops using the `scf` (Structured Control Flow) dialect. It analyzes the `indexing_maps` and `iterator_types` to generate the correct loop nest (in this case, 3 loops for `i`, `j`, `k`).

```mlir
module {
  func.func public @main(%arg0: memref<3x4xf64, strided<[?, ?], offset: ?>>, %arg1: memref<4x3xf64, strided<[?, ?], offset: ?>>) -> memref<3x3xf64> {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x3xf64>
    scf.for %arg2 = %c0 to %c3 step %c1 {
      scf.for %arg3 = %c0 to %c3 step %c1 {
        memref.store %cst, %alloc[%arg2, %arg3] : memref<3x3xf64>
      }
    }
    scf.for %arg2 = %c0 to %c3 step %c1 {
      scf.for %arg3 = %c0 to %c3 step %c1 {
        scf.for %arg4 = %c0 to %c4 step %c1 {
          %0 = memref.load %arg0[%arg2, %arg4] : memref<3x4xf64, strided<[?, ?], offset: ?>>
          %1 = memref.load %arg1[%arg4, %arg3] : memref<4x3xf64, strided<[?, ?], offset: ?>>
          %2 = memref.load %alloc[%arg2, %arg3] : memref<3x3xf64>
          %3 = arith.mulf %0, %1 : f64
          %4 = arith.addf %2, %3 : f64
          memref.store %4, %alloc[%arg2, %arg3] : memref<3x3xf64>
        }
      }
    }
    return %alloc : memref<3x3xf64>
  }
}
```

### `--convert-scf-to-cf`

This pass lowers structured loops (`scf.for`) into a standard Control Flow Graph (CFG) as you would see in a traditional control flow-based compiler like LLVM. It removes the concept of a "loop" and replaces it with Basic Blocks (`^bb`), conditional branches (`cf.cond_br`), and unconditional jumps (`cf.br`). 

```mlir
module {
  func.func public @main(%arg0: memref<3x4xf64, strided<[?, ?], offset: ?>>, %arg1: memref<4x3xf64, strided<[?, ?], offset: ?>>) -> memref<3x3xf64> {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x3xf64>
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
    %1 = arith.cmpi slt, %0, %c3 : index
    cf.cond_br %1, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
    %3 = arith.cmpi slt, %2, %c3 : index
    cf.cond_br %3, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    memref.store %cst, %alloc[%0, %2] : memref<3x3xf64>
    %4 = arith.addi %2, %c1 : index
    cf.br ^bb3(%4 : index)
  ^bb5:  // pred: ^bb3
    %5 = arith.addi %0, %c1 : index
    cf.br ^bb1(%5 : index)
  ^bb6:  // pred: ^bb1
    cf.br ^bb7(%c0 : index)
  ^bb7(%6: index):  // 2 preds: ^bb6, ^bb14
    %7 = arith.cmpi slt, %6, %c3 : index
    cf.cond_br %7, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    cf.br ^bb9(%c0 : index)
  ^bb9(%8: index):  // 2 preds: ^bb8, ^bb13
    %9 = arith.cmpi slt, %8, %c3 : index
    cf.cond_br %9, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    cf.br ^bb11(%c0 : index)
  ^bb11(%10: index):  // 2 preds: ^bb10, ^bb12
    %11 = arith.cmpi slt, %10, %c4 : index
    cf.cond_br %11, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %12 = memref.load %arg0[%6, %10] : memref<3x4xf64, strided<[?, ?], offset: ?>>
    %13 = memref.load %arg1[%10, %8] : memref<4x3xf64, strided<[?, ?], offset: ?>>
    %14 = memref.load %alloc[%6, %8] : memref<3x3xf64>
    %15 = arith.mulf %12, %13 : f64
    %16 = arith.addf %14, %15 : f64
    memref.store %16, %alloc[%6, %8] : memref<3x3xf64>
    %17 = arith.addi %10, %c1 : index
    cf.br ^bb11(%17 : index)
  ^bb13:  // pred: ^bb11
    %18 = arith.addi %8, %c1 : index
    cf.br ^bb9(%18 : index)
  ^bb14:  // pred: ^bb9
    %19 = arith.addi %6, %c1 : index
    cf.br ^bb7(%19 : index)
  ^bb15:  // pred: ^bb7
    return %alloc : memref<3x3xf64>
  }
}```

### `--convert-cf-to-llvm`

This pass converts the control flow dialect into llvm ops.

```mlir
module {
  func.func public @main(%arg0: memref<3x4xf64, strided<[?, ?], offset: ?>>, %arg1: memref<4x3xf64, strided<[?, ?], offset: ?>>) -> memref<3x3xf64> {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %c0 : index to i64
    %cst = arith.constant 0.000000e+00 : f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x3xf64>
    llvm.br ^bb1(%0 : i64)
  ^bb1(%1: i64):  // 2 preds: ^bb0, ^bb5
    %2 = builtin.unrealized_conversion_cast %1 : i64 to index
    %3 = arith.cmpi slt, %2, %c3 : index
    llvm.cond_br %3, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%0 : i64)
  ^bb3(%4: i64):  // 2 preds: ^bb2, ^bb4
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = arith.cmpi slt, %5, %c3 : index
    llvm.cond_br %6, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    memref.store %cst, %alloc[%2, %5] : memref<3x3xf64>
    %7 = arith.addi %5, %c1 : index
    %8 = builtin.unrealized_conversion_cast %7 : index to i64
    llvm.br ^bb3(%8 : i64)
  ^bb5:  // pred: ^bb3
    %9 = arith.addi %2, %c1 : index
    %10 = builtin.unrealized_conversion_cast %9 : index to i64
    llvm.br ^bb1(%10 : i64)
  ^bb6:  // pred: ^bb1
    llvm.br ^bb7(%0 : i64)
  ^bb7(%11: i64):  // 2 preds: ^bb6, ^bb14
    %12 = builtin.unrealized_conversion_cast %11 : i64 to index
    %13 = arith.cmpi slt, %12, %c3 : index
    llvm.cond_br %13, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%0 : i64)
  ^bb9(%14: i64):  // 2 preds: ^bb8, ^bb13
    %15 = builtin.unrealized_conversion_cast %14 : i64 to index
    %16 = arith.cmpi slt, %15, %c3 : index
    llvm.cond_br %16, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%0 : i64)
  ^bb11(%17: i64):  // 2 preds: ^bb10, ^bb12
    %18 = builtin.unrealized_conversion_cast %17 : i64 to index
    %19 = arith.cmpi slt, %18, %c4 : index
    llvm.cond_br %19, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %20 = memref.load %arg0[%12, %18] : memref<3x4xf64, strided<[?, ?], offset: ?>>
    %21 = memref.load %arg1[%18, %15] : memref<4x3xf64, strided<[?, ?], offset: ?>>
    %22 = memref.load %alloc[%12, %15] : memref<3x3xf64>
    %23 = arith.mulf %20, %21 : f64
    %24 = arith.addf %22, %23 : f64
    memref.store %24, %alloc[%12, %15] : memref<3x3xf64>
    %25 = arith.addi %18, %c1 : index
    %26 = builtin.unrealized_conversion_cast %25 : index to i64
    llvm.br ^bb11(%26 : i64)
  ^bb13:  // pred: ^bb11
    %27 = arith.addi %15, %c1 : index
    %28 = builtin.unrealized_conversion_cast %27 : index to i64
    llvm.br ^bb9(%28 : i64)
  ^bb14:  // pred: ^bb9
    %29 = arith.addi %12, %c1 : index
    %30 = builtin.unrealized_conversion_cast %29 : index to i64
    llvm.br ^bb7(%30 : i64)
  ^bb15:  // pred: ^bb7
    return %alloc : memref<3x3xf64>
  }
}
```

### `--convert-arith-to-llvm` and `--convert-math-to-llvm`

These passes convert dialect-specific arithmetic operations (like `arith.addf` or `arith.cmp`) into their direct LLVM dialect equivalents (`llvm.fadd`, `llvm.icmp`). This ensures the instructions map 1-to-1 with LLVM IR instructions. You can see below that `%15 = arith.mulf %12, %13 : f64`  %28 = llvm.fmul %25, %26 : f64` now becomes `%28 = llvm.fmul %25, %26 : f64`

```mlir
module {
  func.func public @main(%arg0: memref<3x4xf64, strided<[?, ?], offset: ?>>, %arg1: memref<4x3xf64, strided<[?, ?], offset: ?>>) -> memref<3x3xf64> {
    %0 = llvm.mlir.constant(4 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(3 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = builtin.unrealized_conversion_cast %3 : i64 to index
    %5 = builtin.unrealized_conversion_cast %4 : index to i64
    %6 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x3xf64>
    llvm.br ^bb1(%5 : i64)
  ^bb1(%7: i64):  // 2 preds: ^bb0, ^bb5
    %8 = builtin.unrealized_conversion_cast %7 : i64 to index
    %9 = llvm.icmp "slt" %7, %2 : i64
    llvm.cond_br %9, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%5 : i64)
  ^bb3(%10: i64):  // 2 preds: ^bb2, ^bb4
    %11 = builtin.unrealized_conversion_cast %10 : i64 to index
    %12 = llvm.icmp "slt" %10, %2 : i64
    llvm.cond_br %12, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    memref.store %6, %alloc[%8, %11] : memref<3x3xf64>
    %13 = llvm.add %10, %1 : i64
    %14 = builtin.unrealized_conversion_cast %13 : i64 to index
    %15 = builtin.unrealized_conversion_cast %14 : index to i64
    llvm.br ^bb3(%15 : i64)
  ^bb5:  // pred: ^bb3
    %16 = llvm.add %7, %1 : i64
    %17 = builtin.unrealized_conversion_cast %16 : i64 to index
    %18 = builtin.unrealized_conversion_cast %17 : index to i64
    llvm.br ^bb1(%18 : i64)
  ^bb6:  // pred: ^bb1
    llvm.br ^bb7(%5 : i64)
  ^bb7(%19: i64):  // 2 preds: ^bb6, ^bb14
    %20 = builtin.unrealized_conversion_cast %19 : i64 to index
    %21 = llvm.icmp "slt" %19, %2 : i64
    llvm.cond_br %21, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%5 : i64)
  ^bb9(%22: i64):  // 2 preds: ^bb8, ^bb13
    %23 = builtin.unrealized_conversion_cast %22 : i64 to index
    %24 = llvm.icmp "slt" %22, %2 : i64
    llvm.cond_br %24, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%5 : i64)
  ^bb11(%25: i64):  // 2 preds: ^bb10, ^bb12
    %26 = builtin.unrealized_conversion_cast %25 : i64 to index
    %27 = llvm.icmp "slt" %25, %0 : i64
    llvm.cond_br %27, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %28 = memref.load %arg0[%20, %26] : memref<3x4xf64, strided<[?, ?], offset: ?>>
    %29 = memref.load %arg1[%26, %23] : memref<4x3xf64, strided<[?, ?], offset: ?>>
    %30 = memref.load %alloc[%20, %23] : memref<3x3xf64>
    %31 = llvm.fmul %28, %29 : f64
    %32 = llvm.fadd %30, %31 : f64
    memref.store %32, %alloc[%20, %23] : memref<3x3xf64>
    %33 = llvm.add %25, %1 : i64
    %34 = builtin.unrealized_conversion_cast %33 : i64 to index
    %35 = builtin.unrealized_conversion_cast %34 : index to i64
    llvm.br ^bb11(%35 : i64)
  ^bb13:  // pred: ^bb11
    %36 = llvm.add %22, %1 : i64
    %37 = builtin.unrealized_conversion_cast %36 : i64 to index
    %38 = builtin.unrealized_conversion_cast %37 : index to i64
    llvm.br ^bb9(%38 : i64)
  ^bb14:  // pred: ^bb9
    %39 = llvm.add %19, %1 : i64
    %40 = builtin.unrealized_conversion_cast %39 : i64 to index
    %41 = builtin.unrealized_conversion_cast %40 : index to i64
    llvm.br ^bb7(%41 : i64)
  ^bb15:  // pred: ^bb7
    return %alloc : memref<3x3xf64>
  }
}
```

### `--convert-func-to-llvm`

This pass transforms the function signature and call conventions. It converts `func.func` to `llvm.func` and lowers high-level types in the signature into types compatible with the LLVM ABI (often converting types into pointers or raw structs, in our case `memref` becomes `llvm.ptr` types).

```
module {
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg7, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg8, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg9, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg10, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg12, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg11, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg13, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = builtin.unrealized_conversion_cast %7 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<4x3xf64, strided<[?, ?], offset: ?>>
    %9 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg0, %9[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg1, %10[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg2, %11[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg3, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg5, %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg4, %14[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %arg6, %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = builtin.unrealized_conversion_cast %16 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<3x4xf64, strided<[?, ?], offset: ?>>
    %18 = llvm.mlir.constant(4 : index) : i64
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.mlir.constant(3 : index) : i64
    %21 = llvm.mlir.constant(0 : index) : i64
    %22 = builtin.unrealized_conversion_cast %21 : i64 to index
    %23 = builtin.unrealized_conversion_cast %22 : index to i64
    %24 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x3xf64>
    %25 = builtin.unrealized_conversion_cast %alloc : memref<3x3xf64> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb1(%23 : i64)
  ^bb1(%26: i64):  // 2 preds: ^bb0, ^bb5
    %27 = builtin.unrealized_conversion_cast %26 : i64 to index
    %28 = llvm.icmp "slt" %26, %20 : i64
    llvm.cond_br %28, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%23 : i64)
  ^bb3(%29: i64):  // 2 preds: ^bb2, ^bb4
    %30 = builtin.unrealized_conversion_cast %29 : i64 to index
    %31 = llvm.icmp "slt" %29, %20 : i64
    llvm.cond_br %31, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    memref.store %24, %alloc[%27, %30] : memref<3x3xf64>
    %32 = llvm.add %29, %19 : i64
    %33 = builtin.unrealized_conversion_cast %32 : i64 to index
    %34 = builtin.unrealized_conversion_cast %33 : index to i64
    llvm.br ^bb3(%34 : i64)
  ^bb5:  // pred: ^bb3
    %35 = llvm.add %26, %19 : i64
    %36 = builtin.unrealized_conversion_cast %35 : i64 to index
    %37 = builtin.unrealized_conversion_cast %36 : index to i64
    llvm.br ^bb1(%37 : i64)
  ^bb6:  // pred: ^bb1
    llvm.br ^bb7(%23 : i64)
  ^bb7(%38: i64):  // 2 preds: ^bb6, ^bb14
    %39 = builtin.unrealized_conversion_cast %38 : i64 to index
    %40 = llvm.icmp "slt" %38, %20 : i64
    llvm.cond_br %40, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%23 : i64)
  ^bb9(%41: i64):  // 2 preds: ^bb8, ^bb13
    %42 = builtin.unrealized_conversion_cast %41 : i64 to index
    %43 = llvm.icmp "slt" %41, %20 : i64
    llvm.cond_br %43, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%23 : i64)
  ^bb11(%44: i64):  // 2 preds: ^bb10, ^bb12
    %45 = builtin.unrealized_conversion_cast %44 : i64 to index
    %46 = llvm.icmp "slt" %44, %18 : i64
    llvm.cond_br %46, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %47 = memref.load %17[%39, %45] : memref<3x4xf64, strided<[?, ?], offset: ?>>
    %48 = memref.load %8[%45, %42] : memref<4x3xf64, strided<[?, ?], offset: ?>>
    %49 = memref.load %alloc[%39, %42] : memref<3x3xf64>
    %50 = llvm.fmul %47, %48 : f64
    %51 = llvm.fadd %49, %50 : f64
    memref.store %51, %alloc[%39, %42] : memref<3x3xf64>
    %52 = llvm.add %44, %19 : i64
    %53 = builtin.unrealized_conversion_cast %52 : i64 to index
    %54 = builtin.unrealized_conversion_cast %53 : index to i64
    llvm.br ^bb11(%54 : i64)
  ^bb13:  // pred: ^bb11
    %55 = llvm.add %41, %19 : i64
    %56 = builtin.unrealized_conversion_cast %55 : i64 to index
    %57 = builtin.unrealized_conversion_cast %56 : index to i64
    llvm.br ^bb9(%57 : i64)
  ^bb14:  // pred: ^bb9
    %58 = llvm.add %38, %19 : i64
    %59 = builtin.unrealized_conversion_cast %58 : i64 to index
    %60 = builtin.unrealized_conversion_cast %59 : index to i64
    llvm.br ^bb7(%60 : i64)
  ^bb15:  // pred: ^bb7
    llvm.return %25 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
}
```

### `--finalize-memref-to-llvm`

This pass lowers the `memref.load` and `memref.store` operations (and the `memref` type itself) into LLVM pointer arithmetic. It creates the standard `MemRef Descriptor` struct (containing allocated pointer, aligned pointer, offset, sizes, and strides) and uses `llvm.getelementptr` (aka GEP) to access data.


```mlir
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg7, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg8, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg9, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg10, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg12, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg11, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg13, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = builtin.unrealized_conversion_cast %7 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<4x3xf64, strided<[?, ?], offset: ?>>
    %9 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg0, %9[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg1, %10[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg2, %11[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg3, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg5, %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg4, %14[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %arg6, %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = builtin.unrealized_conversion_cast %16 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<3x4xf64, strided<[?, ?], offset: ?>>
    %18 = llvm.mlir.constant(4 : index) : i64
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.mlir.constant(3 : index) : i64
    %21 = llvm.mlir.constant(0 : index) : i64
    %22 = builtin.unrealized_conversion_cast %21 : i64 to index
    %23 = builtin.unrealized_conversion_cast %22 : index to i64
    %24 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %25 = llvm.mlir.constant(3 : index) : i64
    %26 = llvm.mlir.constant(3 : index) : i64
    %27 = llvm.mlir.constant(1 : index) : i64
    %28 = llvm.mlir.constant(9 : index) : i64
    %29 = llvm.mlir.zero : !llvm.ptr
    %30 = llvm.getelementptr %29[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %31 = llvm.ptrtoint %30 : !llvm.ptr to i64
    %32 = llvm.mlir.constant(64 : index) : i64
    %33 = llvm.add %31, %32 : i64
    %34 = llvm.call @malloc(%33) : (i64) -> !llvm.ptr
    %35 = llvm.ptrtoint %34 : !llvm.ptr to i64
    %36 = llvm.mlir.constant(1 : index) : i64
    %37 = llvm.sub %32, %36 : i64
    %38 = llvm.add %35, %37 : i64
    %39 = llvm.urem %38, %32 : i64
    %40 = llvm.sub %38, %39 : i64
    %41 = llvm.inttoptr %40 : i64 to !llvm.ptr
    %42 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %43 = llvm.insertvalue %34, %42[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.insertvalue %41, %43[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %45 = llvm.mlir.constant(0 : index) : i64
    %46 = llvm.insertvalue %45, %44[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %47 = llvm.insertvalue %25, %46[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.insertvalue %26, %47[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %49 = llvm.insertvalue %26, %48[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.insertvalue %27, %49[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = builtin.unrealized_conversion_cast %50 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<3x3xf64>
    %52 = builtin.unrealized_conversion_cast %51 : memref<3x3xf64> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb1(%23 : i64)
  ^bb1(%53: i64):  // 2 preds: ^bb0, ^bb5
    %54 = builtin.unrealized_conversion_cast %53 : i64 to index
    %55 = llvm.icmp "slt" %53, %20 : i64
    llvm.cond_br %55, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%23 : i64)
  ^bb3(%56: i64):  // 2 preds: ^bb2, ^bb4
    %57 = builtin.unrealized_conversion_cast %56 : i64 to index
    %58 = llvm.icmp "slt" %56, %20 : i64
    llvm.cond_br %58, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %59 = llvm.extractvalue %50[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.mlir.constant(3 : index) : i64
    %61 = llvm.mul %53, %60 overflow<nsw, nuw> : i64
    %62 = llvm.add %61, %56 overflow<nsw, nuw> : i64
    %63 = llvm.getelementptr inbounds|nuw %59[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %24, %63 : f64, !llvm.ptr
    %64 = llvm.add %56, %19 : i64
    %65 = builtin.unrealized_conversion_cast %64 : i64 to index
    %66 = builtin.unrealized_conversion_cast %65 : index to i64
    llvm.br ^bb3(%66 : i64)
  ^bb5:  // pred: ^bb3
    %67 = llvm.add %53, %19 : i64
    %68 = builtin.unrealized_conversion_cast %67 : i64 to index
    %69 = builtin.unrealized_conversion_cast %68 : index to i64
    llvm.br ^bb1(%69 : i64)
  ^bb6:  // pred: ^bb1
    llvm.br ^bb7(%23 : i64)
  ^bb7(%70: i64):  // 2 preds: ^bb6, ^bb14
    %71 = builtin.unrealized_conversion_cast %70 : i64 to index
    %72 = llvm.icmp "slt" %70, %20 : i64
    llvm.cond_br %72, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%23 : i64)
  ^bb9(%73: i64):  // 2 preds: ^bb8, ^bb13
    %74 = builtin.unrealized_conversion_cast %73 : i64 to index
    %75 = llvm.icmp "slt" %73, %20 : i64
    llvm.cond_br %75, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%23 : i64)
  ^bb11(%76: i64):  // 2 preds: ^bb10, ^bb12
    %77 = builtin.unrealized_conversion_cast %76 : i64 to index
    %78 = llvm.icmp "slt" %76, %18 : i64
    llvm.cond_br %78, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %79 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.extractvalue %16[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %81 = llvm.getelementptr %79[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %82 = llvm.extractvalue %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %83 = llvm.mul %70, %82 overflow<nsw, nuw> : i64
    %84 = llvm.extractvalue %16[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %85 = llvm.mul %76, %84 overflow<nsw, nuw> : i64
    %86 = llvm.add %83, %85 overflow<nsw, nuw> : i64
    %87 = llvm.getelementptr inbounds|nuw %81[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %88 = llvm.load %87 : !llvm.ptr -> f64
    %89 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %90 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.getelementptr %89[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %92 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.mul %76, %92 overflow<nsw, nuw> : i64
    %94 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %95 = llvm.mul %73, %94 overflow<nsw, nuw> : i64
    %96 = llvm.add %93, %95 overflow<nsw, nuw> : i64
    %97 = llvm.getelementptr inbounds|nuw %91[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %98 = llvm.load %97 : !llvm.ptr -> f64
    %99 = llvm.extractvalue %50[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.mlir.constant(3 : index) : i64
    %101 = llvm.mul %70, %100 overflow<nsw, nuw> : i64
    %102 = llvm.add %101, %73 overflow<nsw, nuw> : i64
    %103 = llvm.getelementptr inbounds|nuw %99[%102] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %104 = llvm.load %103 : !llvm.ptr -> f64
    %105 = llvm.fmul %88, %98 : f64
    %106 = llvm.fadd %104, %105 : f64
    %107 = llvm.extractvalue %50[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.mlir.constant(3 : index) : i64
    %109 = llvm.mul %70, %108 overflow<nsw, nuw> : i64
    %110 = llvm.add %109, %73 overflow<nsw, nuw> : i64
    %111 = llvm.getelementptr inbounds|nuw %107[%110] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %106, %111 : f64, !llvm.ptr
    %112 = llvm.add %76, %19 : i64
    %113 = builtin.unrealized_conversion_cast %112 : i64 to index
    %114 = builtin.unrealized_conversion_cast %113 : index to i64
    llvm.br ^bb11(%114 : i64)
  ^bb13:  // pred: ^bb11
    %115 = llvm.add %73, %19 : i64
    %116 = builtin.unrealized_conversion_cast %115 : i64 to index
    %117 = builtin.unrealized_conversion_cast %116 : index to i64
    llvm.br ^bb9(%117 : i64)
  ^bb14:  // pred: ^bb9
    %118 = llvm.add %70, %19 : i64
    %119 = builtin.unrealized_conversion_cast %118 : i64 to index
    %120 = builtin.unrealized_conversion_cast %119 : index to i64
    llvm.br ^bb7(%120 : i64)
  ^bb15:  // pred: ^bb7
    llvm.return %52 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
}
```

### `--reconcile-unrealized-casts`

This pass cleans up the IR by resolving `builtin.unrealized_conversion_cast` operations. During previous passes, if a type couldn't be fully converted immediately (e.g., between an `arith` integer and an `llvm` integer), a temporary cast was inserted. This pass ensures all those casts are resolved, confirming the entire module is valid LLVM dialect.

```mlir
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
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
    %16 = llvm.mlir.constant(4 : index) : i64
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.mlir.constant(3 : index) : i64
    %19 = llvm.mlir.constant(0 : index) : i64
    %20 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %21 = llvm.mlir.constant(3 : index) : i64
    %22 = llvm.mlir.constant(3 : index) : i64
    %23 = llvm.mlir.constant(1 : index) : i64
    %24 = llvm.mlir.constant(9 : index) : i64
    %25 = llvm.mlir.zero : !llvm.ptr
    %26 = llvm.getelementptr %25[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %27 = llvm.ptrtoint %26 : !llvm.ptr to i64
    %28 = llvm.mlir.constant(64 : index) : i64
    %29 = llvm.add %27, %28 : i64
    %30 = llvm.call @malloc(%29) : (i64) -> !llvm.ptr
    %31 = llvm.ptrtoint %30 : !llvm.ptr to i64
    %32 = llvm.mlir.constant(1 : index) : i64
    %33 = llvm.sub %28, %32 : i64
    %34 = llvm.add %31, %33 : i64
    %35 = llvm.urem %34, %28 : i64
    %36 = llvm.sub %34, %35 : i64
    %37 = llvm.inttoptr %36 : i64 to !llvm.ptr
    %38 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %39 = llvm.insertvalue %30, %38[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.insertvalue %37, %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.mlir.constant(0 : index) : i64
    %42 = llvm.insertvalue %41, %40[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.insertvalue %21, %42[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.insertvalue %22, %43[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %45 = llvm.insertvalue %22, %44[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.insertvalue %23, %45[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%19 : i64)
  ^bb1(%47: i64):  // 2 preds: ^bb0, ^bb5
    %48 = llvm.icmp "slt" %47, %18 : i64
    llvm.cond_br %48, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%19 : i64)
  ^bb3(%49: i64):  // 2 preds: ^bb2, ^bb4
    %50 = llvm.icmp "slt" %49, %18 : i64
    llvm.cond_br %50, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %51 = llvm.extractvalue %46[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.mlir.constant(3 : index) : i64
    %53 = llvm.mul %47, %52 overflow<nsw, nuw> : i64
    %54 = llvm.add %53, %49 overflow<nsw, nuw> : i64
    %55 = llvm.getelementptr inbounds|nuw %51[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %20, %55 : f64, !llvm.ptr
    %56 = llvm.add %49, %17 : i64
    llvm.br ^bb3(%56 : i64)
  ^bb5:  // pred: ^bb3
    %57 = llvm.add %47, %17 : i64
    llvm.br ^bb1(%57 : i64)
  ^bb6:  // pred: ^bb1
    llvm.br ^bb7(%19 : i64)
  ^bb7(%58: i64):  // 2 preds: ^bb6, ^bb14
    %59 = llvm.icmp "slt" %58, %18 : i64
    llvm.cond_br %59, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%19 : i64)
  ^bb9(%60: i64):  // 2 preds: ^bb8, ^bb13
    %61 = llvm.icmp "slt" %60, %18 : i64
    llvm.cond_br %61, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%19 : i64)
  ^bb11(%62: i64):  // 2 preds: ^bb10, ^bb12
    %63 = llvm.icmp "slt" %62, %16 : i64
    llvm.cond_br %63, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %64 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.getelementptr %64[%65] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %67 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.mul %58, %67 overflow<nsw, nuw> : i64
    %69 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.mul %62, %69 overflow<nsw, nuw> : i64
    %71 = llvm.add %68, %70 overflow<nsw, nuw> : i64
    %72 = llvm.getelementptr inbounds|nuw %66[%71] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %73 = llvm.load %72 : !llvm.ptr -> f64
    %74 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %76 = llvm.getelementptr %74[%75] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %77 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.mul %62, %77 overflow<nsw, nuw> : i64
    %79 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.mul %60, %79 overflow<nsw, nuw> : i64
    %81 = llvm.add %78, %80 overflow<nsw, nuw> : i64
    %82 = llvm.getelementptr inbounds|nuw %76[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %83 = llvm.load %82 : !llvm.ptr -> f64
    %84 = llvm.extractvalue %46[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %85 = llvm.mlir.constant(3 : index) : i64
    %86 = llvm.mul %58, %85 overflow<nsw, nuw> : i64
    %87 = llvm.add %86, %60 overflow<nsw, nuw> : i64
    %88 = llvm.getelementptr inbounds|nuw %84[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %89 = llvm.load %88 : !llvm.ptr -> f64
    %90 = llvm.fmul %73, %83 : f64
    %91 = llvm.fadd %89, %90 : f64
    %92 = llvm.extractvalue %46[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.mlir.constant(3 : index) : i64
    %94 = llvm.mul %58, %93 overflow<nsw, nuw> : i64
    %95 = llvm.add %94, %60 overflow<nsw, nuw> : i64
    %96 = llvm.getelementptr inbounds|nuw %92[%95] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %91, %96 : f64, !llvm.ptr
    %97 = llvm.add %62, %17 : i64
    llvm.br ^bb11(%97 : i64)
  ^bb13:  // pred: ^bb11
    %98 = llvm.add %60, %17 : i64
    llvm.br ^bb9(%98 : i64)
  ^bb14:  // pred: ^bb9
    %99 = llvm.add %58, %17 : i64
    llvm.br ^bb7(%99 : i64)
  ^bb15:  // pred: ^bb7
    llvm.return %46 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
}
```