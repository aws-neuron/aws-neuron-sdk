"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np
import neuronxcc.nki as nki

# NKI_EXAMPLE_0_BEGIN
import neuronxcc.nki.language as nl


@nki.jit
def nki_spmd_kernel(a):
  b = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
  i = nl.program_id(0)
  j = nl.program_id(1)
  
  a_tile = nl.load(a[i, j])
  nl.store(b[i, j], a_tile)

  return b
# NKI_EXAMPLE_0_END


nki_spmd_kernel = nki.jit(nki_spmd_kernel, mode='simulation',
                          platform_target='trn2')


class TestNkiIsaExamplesTensorCopy(unittest.TestCase):
  def test_spmd_grid(self):
    np.random.seed(0)
    src = np.random.random_sample([4, 2, 1, 1]).astype(np.float32)
    dst_golden = np.copy(src)

    # NKI_EXAMPLE_0_BEGIN
    ############################################################################
    # Example 1: Let compiler decide how to distribute the instances of spmd kernel
    ############################################################################
    dst = nki_spmd_kernel[4, 2](src)
    # NKI_EXAMPLE_0_END
    self.assertTrue(np.allclose(dst, dst_golden))

    # NKI_EXAMPLE_0_BEGIN
    ############################################################################
    # Example 2: Distribute SPMD kernel instances to physical NeuronCores with
    # explicit annotations. Expected physical NeuronCore assignments:
    #   Physical NC [0]: kernel[0, 0], kernel[0, 1], kernel[1, 0], kernel[1, 1]
    #   Physical NC [1]: kernel[2, 0], kernel[2, 1], kernel[3, 0], kernel[3, 1]
    ############################################################################
    dst = nki_spmd_kernel[nl.spmd_dim(nl.nc(2), 2), 2](src)
    dst = nki_spmd_kernel[nl.nc(2) * 2, 2](src)  # syntactic sugar
    # NKI_EXAMPLE_0_END
    self.assertTrue(np.allclose(dst, dst_golden))

    # NKI_EXAMPLE_0_BEGIN
    ############################################################################
    # Example 3: Distribute SPMD kernel instances to physical NeuronCores with
    # explicit annotations. Expected physical NeuronCore assignments:
    #   Physical NC [0]: kernel[0, 0], kernel[0, 1], kernel[2, 0], kernel[2, 1]
    #   Physical NC [1]: kernel[1, 0], kernel[1, 1], kernel[3, 0], kernel[3, 1]
    ############################################################################
    dst = nki_spmd_kernel[nl.spmd_dim(2, nl.nc(2)), 2](src)
    dst = nki_spmd_kernel[2 * nl.nc(2), 2](src)  # syntactic sugar
    # NKI_EXAMPLE_0_END
    self.assertTrue(np.allclose(dst, dst_golden))
