"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np
import neuronxcc.nki as nki

# NKI_EXAMPLE_7_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
...

# NKI_EXAMPLE_7_END

########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################


@nki.jit(mode="simulation")
def nki_tensor_copy(in_tensor):
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  # NKI_EXAMPLE_7_BEGIN
  ############################################################################
  # Example 1: Copy over the tensor to another tensor using the Vector engine.
  ############################################################################
  x = nl.load(in_tensor)
  x_copy = nisa.tensor_copy(x, engine=nisa.vector_engine)
  nl.store(out_tensor, value=x_copy)
  # NKI_EXAMPLE_7_END

  return out_tensor

      
class TestNkiIsaExamplesTensorCopy(unittest.TestCase):
  def test_tensor_copy(self):
    np.random.seed(0)
    src = np.random.random_sample([8, 8]).astype(np.float32) * 100
    dst_golden = np.copy(src)

    dst = nki_tensor_copy(src)
    self.assertTrue(np.allclose(dst, dst_golden))