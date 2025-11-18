# Einsum MLIR Dialect:

This repository implements an **Einsum** dialect within MLIR, providing a series of transformations to lower high-level symbolic matrix operations into low-level, executable loop structures suitable for optimization and code generation.

## Einsum Notation

The primary goal is to take a high-level mathematical expression (the Einstein summation convention string),
```
ik,kj->ij
```
and gradually transform it into an explicit, multi-dimensional loop nest using the standard `linalg` dialect. This process is broken into two steps:

### 1. High-Level to Low-Level (`einsum.hl` → `einsum.ll`)

This transformation converts the purely symbolic **`einsum.hl`** operation into a more structured **`einsum.ll`** operation.

* **Input (`einsum.hl`):** Uses only the input tensors and the `equation` string (e.g., `"ik,kj->ij"`).
* **Transformation Logic:** The equation string is parsed to determine the roles of each index (subscript), establishing the complete loop space.
* **Output (`einsum.ll`):** Explicitly stores the derived loop structure information, including:
    * **`indexing_maps`**: MLIR Affine Maps defining how each input/output tensor is accessed within the unified loop space.
    * **`iterator_types`**: An array indicating whether each loop index is **`parallel`** (kept in the output) or **`reduction`** (summed over).
    * **`loop_order`**: The explicit ordering of the loops (e.g., parallel indices first, then reduction indices).

### 2. Low-Level to Linalg (`einsum.ll` → `linalg.generic`)

This transformation eliminates the `einsum.ll` operation entirely by replacing it with a **`linalg.generic`** operation.

* The attributes stored in `einsum.ll` (indexing maps and iterator types) are directly transferred to the `linalg.generic` op.
* This results in a fully specified loop-based kernel. The transformation also inserts the necessary boilerplate:
    * **Buffer Allocation:** A `tensor.empty` op is created for the output tensor.
    * **Initialization:** A `linalg.fill` op initializes the reduction variable (usually to zero).
    * **Region Body:** The required accumulation logic (e.g., $\text{multiply} + \text{add}$) is inserted into the `linalg.generic` op's body region, completing the transformation into a standard, MLIR-friendly operation ready for further optimization (e.g., tiling, fusion, vectorization).
