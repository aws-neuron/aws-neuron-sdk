"""Quick Start example for nki.simulate documentation."""

import nki
import nki.language as nl
import nki.isa as nisa
import numpy as np


# NKI_EXAMPLE_SIMULATE_BEGIN
@nki.jit
def add_kernel(a_ptr, b_ptr):
    # Load tiles from HBM into SBUF
    a = nl.load(a_ptr)
    b = nl.load(b_ptr)
    # Element-wise add
    result = nl.add(a, b)
    # Store result back to HBM
    out = nl.ndarray(a_ptr.shape, dtype=a_ptr.dtype, buffer=nl.shared_hbm)
    nl.store(out, value=result)
    return out
# NKI_EXAMPLE_SIMULATE_END


# NKI_EXAMPLE_SIMULATE_RUN_BEGIN
# Create example inputs
a = np.random.rand(128, 512).astype(np.float32)
b = np.random.rand(128, 512).astype(np.float32)

# Run on the CPU simulator
result = nki.simulate(add_kernel)(a, b)

# Verify correctness
np.testing.assert_allclose(result, a + b, rtol=1e-5)
# NKI_EXAMPLE_SIMULATE_RUN_END
