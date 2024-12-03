"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np
import neuronxcc.nki as nki
# NKI_EXAMPLE_1_BEGIN
import neuronxcc.nki.language as nl
...
# NKI_EXAMPLE_1



@nki.jit(mode="simulation")
def example_kernel(in_tensor):
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # NKI_EXAMPLE_1_BEGIN
  for i in nl.affine_range(in_tensor.shape[1] // 512):
    tile = nl.load(in_tensor[:, (i * 512):((i + 1) * 512)])
    # Same as above but use ds (dynamic slice) instead of the native
    # slice syntax
    tile = nl.load(in_tensor[:, nl.ds(i * 512, 512)])
    # NKI_EXAMPLE_1_END
    nl.store(out_tensor[:, nl.ds(i * 512, 512)], tile)

  return out_tensor


class TestNkiExampleNlLoad(unittest.TestCase):
  def test_nl_load(self):
    a = np.random.random_sample([128, 4096]).astype(np.float32)

    b = example_kernel(a)
    self.assertTrue(np.allclose(a, b))
