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
