"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np
import neuronxcc.nki as nki

import neuronxcc.nki.language as nl

# NKI_EXAMPLE_7_BEGIN
import neuronxcc.nki.isa as nisa
...

# NKI_EXAMPLE_7_END

########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################


@nki.jit(mode="simulation")
def nki_dma_copy(a):
  b = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)

  # NKI_EXAMPLE_7_BEGIN
  ############################################################################
  # Example 1: Copy over the tensor to another tensor
  ############################################################################
  nisa.dma_copy(src=a, dst=b)

  # NKI_EXAMPLE_7_END

  return b

      
class TestNkiIsaExamplesTensorCopy(unittest.TestCase):
  def test_tensor_copy(self):
    np.random.seed(0)
    src = np.random.random_sample([256, 1]).astype(np.float32)
    dst_golden = np.copy(src)

    dst = nki_dma_copy(src)
    self.assertTrue(np.allclose(dst, dst_golden))