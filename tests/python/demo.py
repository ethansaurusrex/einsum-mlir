import einsum_mlir as eml
import numpy as np

if __name__ == '__main__':
    A = np.random.rand(3,3)
    B = np.random.rand(3,3)
    C = eml.einsum("ik,kj->ij", A, B)
