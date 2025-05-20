"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
# NKI_EXAMPLE_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
# NKI_EXAMPLE_END
import numpy as np


@nki.jit(mode="simulation")
def nki_affine_select(a_tensor):
  b_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

  # NKI_EXAMPLE_BEGIN
  ##################################################################
  # Example 1: Take tile a of shape [128, 128] and replace its
  # upper triangle with -9984.0;
  ##################################################################
  ix, iy = nl.mgrid[0:128, 0:128]
  a = nl.load(a_tensor[ix, iy])

  b = nisa.affine_select(pred=(iy <ix), on_true_tile=a[ix, iy], on_false_value=-9984.0)

  nl.store(b_tensor[ix, iy], b)
  # NKI_EXAMPLE_END

  return b_tensor


class TestNkiIsaExamplesAffineSelect(unittest.TestCase):
  def test_affine_select(self):
    a = np.random.random_sample([128, 128]).astype(np.float32) * 100
    b_golden = np.copy(a)

    b = nki_affine_select(a)

    triui = np.triu_indices_from(b_golden) # upper triangle indicies
    b_golden[triui] = -9984.0

    self.assertTrue(np.allclose(b, b_golden))




 