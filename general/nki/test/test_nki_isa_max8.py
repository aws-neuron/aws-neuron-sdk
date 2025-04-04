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
def nki_max8():
  # NKI_EXAMPLE_0_BEGIN
  ##################################################################
  # Example 1: Generate tile b of 32 * 128 random floating point values
  # and get the 8 largest values in each row:
  ##################################################################
  expr_a = nl.rand((32, 128))
  a = nisa.max8(src=expr_a)

  a_tensor = nl.ndarray([32, 8], dtype=nl.float32, buffer=nl.shared_hbm)
  nl.store(a_tensor, value=a)
  # NKI_EXAMPLE_0_END

  return a_tensor



class TestNkiIsaExamplesMax8(unittest.TestCase):
  def test_max8(self):
    a = nki_max8()

    self.assertEqual(a.shape, (32, 8))
    self.assertTrue(np.all(a >= 0) and np.all(a <= 1))
    row_diffs = np.diff(a, axis=1)  # Get differences between adjacent elements
    self.assertTrue(np.all(row_diffs <= 0), "Values within rows should be descending")

TestNkiIsaExamplesMax8().test_max8()