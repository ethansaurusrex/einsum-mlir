# python/einsum_mlir/dialects/einsum.py

# Import the generated module itself. We will use this to qualify _Dialect.
from . import _einsum_ops_gen as eog

# Import all generated symbols for convenience and re-exporting.
# This brings EinsumHL into the local scope, which is needed for the class definition.
from ._einsum_ops_gen import *

# 2. Extend the ODS classes if needed (Optional)

from mlir.dialects._ods_common import _cext as _ods_cext


# Fix: Use the qualified name (_einsum_ops_gen._Dialect) to explicitly reference
# the dialect class, resolving the NameError during decorator processing.
@_ods_cext.register_operation(eog._Dialect, replace=True)
class EinsumHL(EinsumHL):
    # We can add convenience methods here in the future
    pass
