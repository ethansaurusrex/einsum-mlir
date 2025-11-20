# python/einsum_mlir/dialects/einsum.py

# 1. Import the auto-generated ODS classes
from ._einsum_ops_gen import *

# 2. Extend the ODS classes if needed (Optional)
# The docs show how to use @_cext.register_operation to extend functionality.
# For example, if you wanted a custom builder for EinsumHL:

from mlir.dialects._ods_common import _cext as _ods_cext

@_ods_cext.register_operation(_Dialect, replace=True)
class EinsumHL(EinsumHL):
    # We can add convenience methods here in the future
    pass
