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
def nki_nc_matmul(a_tensor, b_tensor, c_tensor, 
          d_tensor, e_tensor, f_tensor):
          # g_tensor, h_tensor, i_tensor):

  ##################################################################
  # Example 1:
  # multiply matrix a of shape (128, 128) and matrix b of shape (128, 512)
  # to get matrix c in PSUM of shape (128, 512)
  ##################################################################
  i_p_a = nl.arange(128)[:, None]
  i_f_a = nl.arange(128)[None, :]
  i_p_b = nl.arange(128)[:, None]
  i_f_b = nl.arange(512)[None, :]
  a = nl.load(a_tensor[i_p_a, i_f_a])
  b = nl.load(b_tensor[i_p_b, i_f_b])

  c_psum = nisa.nc_matmul(a[i_p_a, i_f_a], b[i_p_b, i_f_b])

  nl.store(c_tensor[i_p_a, i_f_b], c_psum)

  ##################################################################
  # Example 2:
  # multiply matrix d of shape (256, 128) and matrix e of shape (256, 512)
  # to get matrix f in PSUM of shape (128, 512) using psum accumulation
  ##################################################################
  f_psum = nl.zeros((128, 512), nl.float32, buffer=nl.psum)

  i_p_d = nl.arange(128)[:, None]
  i_f_d = nl.arange(128)[None, :]
  i_p_e = nl.arange(128)[:, None]
  i_f_e = nl.arange(512)[None, :]

  for i_contract in nl.affine_range(2):
    d = nl.load(d_tensor[i_contract * 128 + i_p_d, i_f_d])
    e = nl.load(e_tensor[i_contract * 128 + i_p_e, i_f_e])
    f_psum += nisa.nc_matmul(d[i_p_d, i_f_d],
                              e[i_p_e, i_f_e])
    
  nl.store(f_tensor[i_p_d, i_f_e], f_psum)

  # FIXME: packing example doesn't work yet:
  # ##################################################################
  # # Example 3:
  # # multiply matrix g of shape (64, 256) and matrix h of shape (64, 512)
  # # to get matrix i of (256, 512) using two independent psum tiles
  # # and Tensor Engine packing mode
  # ##################################################################
  # i_p_g = nl.arange(64)[:, None]
  # i_f_g = nl.arange(128)[None, :]
  # i_p_h = nl.arange(64)[:, None]
  # i_f_h = nl.arange(512)[None, :]

  # # Since the matrix g tile has a (64, 128) padded shape,
  # # we can enable packing mode in hardware
  # # to run the two nc_matmul instructions simultaneously by setting the rowgrp appropriately
  # i_psum_0 = nisa.nc_matmul(g[i_p_g, i_f_g], h[i_p_h, i_f_h], rowgrp=64)
  # i_psum_1 = nisa.nc_matmul(g[i_p_g, 128 + i_f_g], h[i_p_h, i_f_h], rowgrp=64)

class TestNkiIsaExamplesNcMatmul(unittest.TestCase):
  def test_nc_matmul(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 128]).astype(np.float32)
    b = np.random.random_sample([128, 512]).astype(np.float32)
    c = np.ndarray(shape=[128, 512], dtype=np.float32)

    d = np.random.random_sample([256, 128]).astype(np.float32)
    e = np.random.random_sample([256, 512]).astype(np.float32)
    f = np.ndarray(shape=[128, 512], dtype=np.float32)

    # g = np.random.random_sample([64, 256]).astype(np.float32)
    # h = np.random.random_sample([64, 512]).astype(np.float32)
    # i = np.ndarray(shape=[256, 512], dtype=np.float32)

    simulate_kernel(nki_nc_matmul, a, b, c, d, e, f)

    c_golden = np.matmul(np.transpose(a), b)
    f_golden = np.matmul(np.transpose(d), e)
    # i_golden = np.matmul(np.transpose(g), h)

    self.assertTrue(np.allclose(c, c_golden))
    self.assertTrue(np.allclose(f, f_golden))
    # self.assertTrue(np.allclose(i, i_golden))