"""
Copyright (C) 2025, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
# NKI_EXAMPLE_0_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
# NKI_EXAMPLE_0_END
import numpy as np


@nki.jit(mode="simulation", kernel_return=True)
def nki_nc_match_replace8():
  # NKI_EXAMPLE_0_BEGIN
  ##################################################################
  # Example 1: Generate tile a of random floating point values,
  # get the 8 largest values in each row, then replace their first
  # occurrences with -inf:
  ##################################################################
  N = 4
  M = 16
  data_tile = nl.rand((N, M))
  max_vals = nisa.max8(src=data_tile)

  result = nisa.nc_match_replace8(data=data_tile[:, :], vals=max_vals, imm=float('-inf'))
  result_tensor = nl.ndarray([N, M], dtype=nl.float32, buffer=nl.shared_hbm)
  nl.store(result_tensor, value=result)
  # NKI_EXAMPLE_0_END

  return result_tensor


class TestNkiIsaExamplesMatchReplace8(unittest.TestCase):
  def test_nc_match_replace8(self):
    result = nki_nc_match_replace8()

    self.assertEqual(result.shape, (4, 16))
    self.assertEqual(result.dtype, np.float32)

    # Each row should have exactly 8 -inf values
    inf_count = np.sum(np.isinf(result) & (result < 0), axis=1)
    self.assertTrue(np.all(inf_count == 8))

    # Non-inf values should be between 0 and 1 (from rand)
    non_inf_mask = ~(np.isinf(result) & (result < 0))
    self.assertTrue(np.all(result[non_inf_mask] >= 0))
    self.assertTrue(np.all(result[non_inf_mask] <= 1))

TestNkiIsaExamplesMatchReplace8().test_nc_match_replace8()