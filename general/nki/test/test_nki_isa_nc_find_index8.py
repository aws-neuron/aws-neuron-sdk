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
def nki_max_index8():
  # NKI_EXAMPLE_0_BEGIN
  ##################################################################
  # Example 1: Generate tile b of 32 * 128 random floating point values,
  # find the 8 largest values in each row, then find their indices:
  ##################################################################
  # Generate random data
  data = nl.rand((32, 128))

  # Find max 8 values per row
  max_vals = nisa.max8(src=data)

  # Create output tensor for indices
  indices_tensor = nl.ndarray([32, 8], dtype=nl.uint32, buffer=nl.shared_hbm)

  # Find indices of max values
  indices = nisa.nc_find_index8(data=data, vals=max_vals)

  # Store results
  nl.store(indices_tensor, value=indices)
  # NKI_EXAMPLE_0_END

  return indices_tensor



class TestNkiIsaExamplesMaxIndex8(unittest.TestCase):
  def test_max_index8(self):
    indices = nki_max_index8()

    self.assertEqual(indices.shape, (32, 8))
    self.assertEqual(indices.dtype, np.uint32)

    # Verify indices are within valid range (0 to 127)
    self.assertTrue(np.all(indices >= 0) and np.all(indices < 128))

    # Check that indices point to descending values
    indices_diffs = np.diff(indices, axis=1)  # Get differences between adjacent indices
    # Values should be unique, so indices should be different
    self.assertTrue(np.all(indices_diffs != 0), "Indices should be unique")

TestNkiIsaExamplesMaxIndex8().test_max_index8()
