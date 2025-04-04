"""
Copyright (C) 2025, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
# NKI_EXAMPLE_21_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
# NKI_EXAMPLE_21_END
import numpy as np
...


@nki.jit(mode="simulation")
def nki_copy_predicated(predicate, on_true_tensor, on_false_tensor):
  # NKI_EXAMPLE_21_BEGIN
  ##################################################################
  # Example 1: Conditionally copies elements from the `on_true` tile to 
  # SBUF/PSUM destination tile using Vector Engine, where copying occurs 
  # only at positions where the predicate evaluates to True.
  ##################################################################
  # NKI_EXAMPLE_21_END
  ...
  out_tensor: tensor[128, 512] = nl.ndarray([128, 512], dtype=on_true_tensor.dtype,
                                            buffer=nl.shared_hbm)
  # NKI_EXAMPLE_21_BEGIN
  ...
  pre_tile: tensor[128, 512] = nl.load(predicate)
  src_tile: tensor[128, 512] = nl.load(on_true_tensor)

  ix, iy = nl.mgrid[0:128, 0:512]
  dst_tile: tensor[128, 512] = nl.zeros(shape=src_tile.shape, dtype=src_tile.dtype)
  dst_tile[ix, iy] = nl.load(on_false_tensor)

  nisa.tensor_copy_predicated(src=src_tile, dst=dst_tile, predicate=pre_tile)
  # NKI_EXAMPLE_21_END

  nl.store(out_tensor, dst_tile)
  return out_tensor


class TestNkiIsaExamplescopy_predicated(unittest.TestCase):
  def test_copy_predicated(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.random.random_sample([128, 512]).astype(np.float32)

    b = nki_copy_predicated(np.less_equal(a, 0.8), a, b)
    b_golden = np.where(np.less_equal(a, 0.8), a, b)

    self.assertTrue(np.allclose(b, b_golden))
