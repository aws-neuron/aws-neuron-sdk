"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np
import neuronxcc.nki as nki
# NKI_EXAMPLE_3_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
...
# NKI_EXAMPLE_3_END

########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################


@nki.jit(mode="simulation")
def nki_tensor_tensor(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)

  # NKI_EXAMPLE_3_BEGIN
  ##################################################################
  # Example 1: add two tiles, a and b, of the same
  # shape (128, 512) element-wise and get
  # the addition result in tile c
  ##################################################################
  a: tensor[128, 512] = nl.load(a_tensor)
  b: tensor[128, 512] = nl.load(b_tensor)

  c: tensor[128, 512] = nisa.tensor_tensor(a, b, op=nl.add)

  # NKI_EXAMPLE_3_END
  nl.store(c_tensor, c)
  return c_tensor


class TestNkiIsaExamplesTensorTensor(unittest.TestCase):
  def test_tensor_tensor(self):
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.random.random_sample([128, 512]).astype(np.float32)
    c = nki_tensor_tensor(a, b)
    
    self.assertTrue(np.allclose(c, np.add(a, b)))
