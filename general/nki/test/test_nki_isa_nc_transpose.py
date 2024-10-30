"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
...
nki_jit = nki.trace
simulate_kernel = nki.simulate_kernel

########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################

@nki_jit
def nki_nc_transpose(a_tensor, at_tensor, b_tensor, bt_tensor):
  ##################################################################
  i_p_a = nl.arange(128)[:, None]
  i_f_a = nl.arange(64)[None, :]
  a = nl.load(a_tensor[i_p_a, i_f_a])
  ##################################################################
  # Example 1: transpose tile a of shape (128, 64)
  ##################################################################
  i_p_a = nl.arange(128)[:, None]
  i_f_a = nl.arange(64)[None, :]
  aT = nisa.nc_transpose(a[i_p_a, i_f_a])

  i_p_aT = nl.arange(64)[:, None]
  i_f_aT = nl.arange(128)[None, :]
  nl.store(at_tensor[i_p_aT, i_f_aT], aT)

  ##################################################################
  i_p_b = nl.arange(32)[:, None]
  i_f_b = nl.arange(2)[None, :]
  b = nl.load(b_tensor[i_p_b, i_f_b])
  ##################################################################
  # Example 2: transpose tile b of shape (32, 2) using Vector Engine
  ##################################################################
  i_p_b = nl.arange(32)[:, None]
  i_f_b = nl.arange(2)[None, :]
  bT = nisa.nc_transpose(b[i_p_b, i_f_b], engine=nisa.vector_engine)

  i_p_bT = nl.arange(2)[:, None]
  i_f_bT = nl.arange(32)[None, :]
  nl.store(bt_tensor[i_p_bT, i_f_bT], bT)

class TestNkiIsaExamplesSbTranspose(unittest.TestCase):
  def test_nc_transpose(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 64]).astype(np.float32)
    aT = np.ndarray(shape=[64, 128], dtype=np.float32)
    b = np.random.random_sample([32, 2]).astype(np.float32)
    bT = np.ndarray(shape=[2, 32], dtype=np.float32)

    simulate_kernel(nki_nc_transpose, a, aT, b, bT)

    aT_golden = np.transpose(a)
    bT_golden = np.transpose(b)

    self.assertTrue(np.allclose(aT, aT_golden))
    self.assertTrue(np.allclose(bT, bT_golden))
