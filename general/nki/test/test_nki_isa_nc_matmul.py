"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np

import neuronxcc.nki as nki
# NKI_EXAMPLE_0_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
# NKI_EXAMPLE_0_END

########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################


@nki.jit(mode="simulation")
def nki_nc_matmul(a_tensor, b_tensor, d_tensor, e_tensor, g_tensor, h_tensor):
  c_tensor = nl.ndarray([128, 512], dtype=nl.float32, buffer=nl.shared_hbm)
  f_tensor = nl.ndarray([128, 512], dtype=nl.float32, buffer=nl.shared_hbm)
  i_tensor = nl.ndarray([16, 128, 512], dtype=nl.float32, buffer=nl.shared_hbm)

  # NKI_EXAMPLE_0_BEGIN
  ##################################################################
  # Example 1:
  # multiply matrix a of shape (128, 128) and matrix b of shape (128, 512)
  # to get matrix c in PSUM of shape (128, 512)
  ##################################################################
  a_mgrid = nl.mgrid[0:128, 0:128]
  b_mgrid = nl.mgrid[0:128, 0:512]
  c_mgrid = nl.mgrid[0:128, 0:512]

  a = nl.load(a_tensor[a_mgrid.p, a_mgrid.x])
  b = nl.load(b_tensor[b_mgrid.p, b_mgrid.x])

  c_psum = nisa.nc_matmul(a[a_mgrid.p, a_mgrid.x], b[b_mgrid.p, b_mgrid.x])

  nl.store(c_tensor[c_mgrid.p, c_mgrid.x], c_psum)

  ##################################################################
  # Example 2:
  # multiply matrix d of shape (256, 128) and matrix e of shape (256, 512)
  # to get matrix f in PSUM of shape (128, 512) using psum accumulation
  ##################################################################
  d_mgrid = nl.mgrid[0:128, 0:128]
  e_mgrid = nl.mgrid[0:128, 0:512]
  f_mgrid = nl.mgrid[0:128, 0:512]

  f_psum = nl.zeros((128, 512), nl.float32, buffer=nl.psum)

  for i_contract in nl.affine_range(2):
    d = nl.load(d_tensor[i_contract * 128 + d_mgrid.p, d_mgrid.x])
    e = nl.load(e_tensor[i_contract * 128 + e_mgrid.p, e_mgrid.x])
    f_psum += nisa.nc_matmul(d[d_mgrid.p, d_mgrid.x], e[e_mgrid.p, e_mgrid.x])
    
  nl.store(f_tensor[f_mgrid.p, f_mgrid.x], f_psum)

  ##################################################################
  # Example 3:
  # perform batched matrix multiplication on matrix g of shape (16, 32, 128) 
  # and matrix h of shape (16, 32, 512) to get matrix i of (16, 128, 512) 
  # using Tensor Engine packing mode. 
  ##################################################################
  g_mgrid = nl.mgrid[0:32, 0:128]
  h_mgrid = nl.mgrid[0:32, 0:512]
  i_mgrid = nl.mgrid[0:128, 0:512]

  for i in nl.affine_range(16):
    g = nl.load(g_tensor[i, g_mgrid.p, g_mgrid.x])
    h = nl.load(h_tensor[i, h_mgrid.p, h_mgrid.x])
    i_psum = nisa.nc_matmul(g, h, tile_position=((i % 4) * 32, 0), tile_size=(32, 128))
    nl.store(i_tensor[i, i_mgrid.p, i_mgrid.x], i_psum)

  return c_tensor, f_tensor, i_tensor
  # NKI_EXAMPLE_0_END

@nki.jit(mode="simulation", platform_target='trn2')
def nki_nc_matmul_double_row_gen3(a_input, b_input):
  NUM_PARTITIONS_A, TWO_A, FREE_A = a_input.shape
  NUM_PARTITIONS_B, TWO_B, FREE_B = b_input.shape

  c_output = nl.ndarray([FREE_A, FREE_B], dtype=nl.float32, buffer=nl.shared_hbm)

  assert NUM_PARTITIONS_A == NUM_PARTITIONS_B and TWO_A == 2 and TWO_B == 2

  a_tile = nl.ndarray(
    (NUM_PARTITIONS_A, TWO_A, max(FREE_A, 16)), dtype=nl.float8_e5m2, buffer=nl.sbuf
  )
  a_mgrid = nl.mgrid[0:NUM_PARTITIONS_A, 0:TWO_A, 0:FREE_A]
  a_tile[a_mgrid.p, a_mgrid.x, a_mgrid.y] = nl.load(a_input.view(nl.float8_e5m2))
  b_tile = nl.load(b_input.view(nl.float8_e5m2))
  c_tile = nisa.nc_matmul(
    a_tile[a_mgrid.p, a_mgrid.x, a_mgrid.y], b_tile, perf_mode="double_row_gen3"
  )
  nl.store(c_output, value=c_tile)
  return c_output


class TestNkiIsaExamplesNcMatmul(unittest.TestCase):
  def test_nc_matmul(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 128]).astype(np.float32)
    b = np.random.random_sample([128, 512]).astype(np.float32)

    d = np.random.random_sample([256, 128]).astype(np.float32)
    e = np.random.random_sample([256, 512]).astype(np.float32)

    g = np.random.random_sample([16, 32, 128]).astype(np.float32)
    h = np.random.random_sample([16, 32, 512]).astype(np.float32)
    i = np.ndarray(shape=[16, 128, 512], dtype=np.float32)

    c, f, i = nki_nc_matmul(a, b, d, e, g, h)

    c_golden = np.matmul(np.transpose(a), b)
    f_golden = np.matmul(np.transpose(d), e)
    i_golden = np.matmul(g.transpose(0, 2, 1), h)

    self.assertTrue(np.allclose(c, c_golden))
    self.assertTrue(np.allclose(f, f_golden))
    self.assertTrue(np.allclose(i, i_golden))

  def test_double_row_gen3(self):
    np.random.seed(0)
    a = np.ones((128, 2, 1), dtype=nl.float8_e5m2)
    b = np.ones((128, 2, 512), dtype=nl.float8_e5m2)

    c = nki_nc_matmul_double_row_gen3(a, b)

    c_golden = np.einsum("kli,klj->ij",
                         a.astype(np.float32),
                         b.astype(np.float32))

    self.assertTrue(np.allclose(c, c_golden))
