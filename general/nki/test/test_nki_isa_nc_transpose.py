"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np

import neuronxcc.nki as nki
# NKI_EXAMPLE_1_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
...
# NKI_EXAMPLE_1_END

########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################


@nki.jit(mode="simulation")
def nki_nc_transpose(a_tensor, b_tensor):
  at_tensor = nl.ndarray([a_tensor.shape[1], a_tensor.shape[0]], dtype=a_tensor.dtype,
                         buffer=nl.shared_hbm)
  bt_tensor = nl.ndarray([b_tensor.shape[1], b_tensor.shape[0]], dtype=b_tensor.dtype,
                         buffer=nl.shared_hbm)
  ##################################################################
  i_p_a = nl.arange(128)[:, None]
  i_f_a = nl.arange(64)[None, :]
  a = nl.load(a_tensor[i_p_a, i_f_a])
  # NKI_EXAMPLE_1_BEGIN
  ##################################################################
  # Example 1: transpose tile a of shape (128, 64)
  ##################################################################
  i_p_a = nl.arange(128)[:, None]
  i_f_a = nl.arange(64)[None, :]
  aT = nisa.nc_transpose(a[i_p_a, i_f_a])

  # NKI_EXAMPLE_1_END
  i_p_aT = nl.arange(64)[:, None]
  i_f_aT = nl.arange(128)[None, :]
  nl.store(at_tensor[i_p_aT, i_f_aT], aT)

  ##################################################################
  i_p_b = nl.arange(32)[:, None]
  i_f_b = nl.arange(2)[None, :]
  b = nl.load(b_tensor[i_p_b, i_f_b])
  # NKI_EXAMPLE_1_BEGIN
  ##################################################################
  # Example 2: transpose tile b of shape (32, 2) using Vector Engine
  ##################################################################
  i_p_b = nl.arange(32)[:, None]
  i_f_b = nl.arange(2)[None, :]
  bT = nisa.nc_transpose(b[i_p_b, i_f_b], engine=nisa.vector_engine)
  # NKI_EXAMPLE_1_END

  i_p_bT = nl.arange(2)[:, None]
  i_f_bT = nl.arange(32)[None, :]
  nl.store(bt_tensor[i_p_bT, i_f_bT], bT)
  return at_tensor, bt_tensor

class TestNkiIsaExamplesSbTranspose(unittest.TestCase):
  def test_nc_transpose(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 64]).astype(np.float32) * 100
    b = np.random.random_sample([32, 2]).astype(np.float32) * 100

    aT, bT = nki_nc_transpose(a, b)

    aT_golden = np.transpose(a)
    bT_golden = np.transpose(b)

    self.assertTrue(np.allclose(aT, aT_golden))
    self.assertTrue(np.allclose(bT, bT_golden))
