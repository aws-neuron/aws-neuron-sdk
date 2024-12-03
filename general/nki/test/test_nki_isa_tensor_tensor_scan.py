"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np

import neuronxcc.nki as nki
# NKI_EXAMPLE_4_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
# NKI_EXAMPLE_4_END



########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################

@nki.jit(mode="simulation")
def nki_tensor_tensor_scan(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  a = nl.load(a_tensor)
  b = nl.load(b_tensor)

  # NKI_EXAMPLE_4_BEGIN
  ##################################################################
  # Example 1: scan two tiles, a and b, of the same
  # shape (128, 1024) using multiply/add and get
  # the scan result in tile c
  ##################################################################
  c = nl.ndarray(shape=(128, 1024), dtype=nl.float32)

  c[:, 0:512] = nisa.tensor_tensor_scan(a[:, 0:512], b[:, 0:512],
                                        initial=0, op0=np.multiply, op1=np.add)

  c[:, 512:1024] = nisa.tensor_tensor_scan(a[:, 512:1024], b[:, 512:1024],
                                           initial=c[:, 511],
                                           op0=np.multiply, op1=np.add)
  # NKI_EXAMPLE_4_END

  nl.store(c_tensor, c)
  return c_tensor


class TestNkiIsaExamplesTensorTensorScan(unittest.TestCase):
  def test_tensor_tensor_scan(self):
    a = np.random.random_sample([128, 1024]).astype(np.float32)
    b = np.random.random_sample([128, 1024]).astype(np.float32)
    c = nki_tensor_tensor_scan(a, b)

    golden = np.zeros(c.shape)
    golden[:, 0] = a[:, 0] * 0 + b[:, 0]
    for i in range(1, c.shape[1]):
      golden[:, i] = a[:, i] * golden[:, i - 1] + b[:, i]

    self.assertTrue(np.allclose(c, golden))
