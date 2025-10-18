"""
Copyright (C) 2025, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
# NKI_EXAMPLE_0_BEGIN # NKI_EXAMPLE_1_BEGIN # NKI_EXAMPLE_2_BEGIN # NKI_EXAMPLE_3_BEGIN # NKI_EXAMPLE_4_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt

# NKI_EXAMPLE_0_END # NKI_EXAMPLE_1_END # NKI_EXAMPLE_2_END # NKI_EXAMPLE_3_END # NKI_EXAMPLE_4_END
import numpy as np


@nki.jit(mode="simulation")
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


@nki.jit(mode="simulation")
def nki_nc_match_replace_indices8(in_tensor: nt.tensor, imm: np.float32):
  # NKI_EXAMPLE_1_BEGIN
  ##################################################################
  # Example 2: Read the 8 largest values in each row of the tensor,
  # replace the first occurrence with imm, write indices, and return
  # the replaced output.
  ##################################################################
  n, m = in_tensor.shape
  # NKI_EXAMPLE_1_END
  out_tensor = nl.ndarray([n, m], dtype=in_tensor.dtype, buffer=nl.hbm)
  idx_tensor = nl.ndarray([n, 8], dtype=nl.uint32, buffer=nl.hbm)
  # NKI_EXAMPLE_1_BEGIN
  dst_idx = nl.ndarray((n, 8), dtype=idx_tensor.dtype)

  ix, iy = nl.mgrid[0:n, 0:8]

  inp_tile: nt.tensor[n, m] = nl.load(in_tensor)
  max_vals: nt.tensor[n, 8] = nisa.max8(src=inp_tile)

  out_tile = nisa.nc_match_replace8(
    dst_idx=dst_idx[ix, iy], data=inp_tile[:, :], vals=max_vals, imm=imm
  )
  # NKI_EXAMPLE_1_END

  nl.store(out_tensor, value=out_tile)
  nl.store(idx_tensor[ix, iy], value=dst_idx[ix, iy])
  return out_tensor, idx_tensor


@nki.jit(mode="simulation")
def nki_nc_match_replace_indices8_mask(in_tensor: nt.tensor, imm: np.float32):
  # NKI_EXAMPLE_2_BEGIN
  ##################################################################
  # Example 3: Read the 8 largest values in each row of the tensor,
  # after applying the specified mask, replace the first occurrence
  # with imm, write indices, and return the replaced output.
  ##################################################################
  n, m = in_tensor.shape
  # NKI_EXAMPLE_2_END
  out_tensor = nl.ndarray([n, m], dtype=in_tensor.dtype, buffer=nl.hbm)
  idx_tensor = nl.ndarray([n, 8], dtype=nl.uint32, buffer=nl.hbm)
  # NKI_EXAMPLE_2_BEGIN
  idx_tile = nisa.memset(shape=(n, 8), value=0, dtype=nl.uint32)

  ix, iy = nl.mgrid[0:n, 0:m]
  inp_tile: nt.tensor[n, m] = nl.load(in_tensor)
  max_vals: nt.tensor[n, 8] = nisa.max8(src=inp_tile[ix, iy], mask=(ix < n //2 and iy < m//2))

  out_tile = nisa.nc_match_replace8(
    dst_idx=idx_tile[:, :],
    data=inp_tile[ix, iy],
    vals=max_vals,
    imm=imm,
    mask=(ix < n // 2 and iy < m // 2),  # mask applies to `data`
  )
  # NKI_EXAMPLE_2_END

  nl.store(out_tensor, value=out_tile)
  nl.store(idx_tensor, value=idx_tile)
  return out_tensor, idx_tensor


@nki.jit(mode="simulation")
def nki_nc_match_replace_indices8_3d(data_tensor: nt.tensor):
  # NKI_EXAMPLE_3_BEGIN
  ##################################################################
  # Example 4: Read the 8 largest values in each row of the tensor,
  # replace the first occurrence with 0.0, write indices, and return 
  # the replaced output.
  ##################################################################
  n, b, m = data_tensor.shape
  # NKI_EXAMPLE_3_END
  out_tensor = nl.ndarray([n, b, m], dtype=data_tensor.dtype, buffer=nl.hbm)
  # NKI_EXAMPLE_3_BEGIN
  n, b, m = data_tensor.shape

  out_tensor = nl.ndarray([n, b, m], dtype=data_tensor.dtype, buffer=nl.hbm)
  idx_tensor = nl.ndarray([n, 8], dtype=nl.uint32, buffer=nl.hbm)

  imm = 0.0
  idx_tile = nisa.memset(shape=(n, 8), value=0, dtype=nl.uint32)
  out_tile = nisa.memset(shape=(n, b, m), value=0, dtype=data_tensor.dtype)

  iq, ir, iw = nl.mgrid[0:n, 0:b, 0:m]
  ip, io = nl.mgrid[0:n, 0:8]

  inp_tile = nl.load(data_tensor[iq, ir, iw])
  max_vals: nt.tensor[n, 8] = nisa.max8(src=inp_tile)

  out_tile[iq, ir, iw] = nisa.nc_match_replace8(
    dst_idx=idx_tile[ip, io],
    data=inp_tile[iq, ir, iw],
    vals=max_vals[ip, io],
    imm=imm,
  )

  # NKI_EXAMPLE_3_END
  nl.store(out_tensor, value=out_tile)
  nl.store(idx_tensor, value=idx_tile)
  return out_tensor, idx_tensor


@nki.jit(mode="simulation")
def nki_nc_match_replace_indices8_3d_inplace(data_tensor: nt.tensor):
  # NKI_EXAMPLE_4_BEGIN
  ##################################################################
  # Example 5: Read the 8 largest values in each row of the tensor,
  # replace the first occurrence with 0.0 in-place and write indices.
  ##################################################################
  n, b, m = data_tensor.shape
  # NKI_EXAMPLE_4_END
  out_tensor = nl.ndarray([n, b, m], dtype=data_tensor.dtype, buffer=nl.hbm)
  # NKI_EXAMPLE_4_BEGIN
  n, b, m = data_tensor.shape

  out_tensor = nl.ndarray([n, b, m], dtype=data_tensor.dtype, buffer=nl.hbm)
  idx_tensor = nl.ndarray([n, 8], dtype=nl.uint32, buffer=nl.hbm)

  imm = 0.0
  idx_tile = nisa.memset(shape=(n, 8), value=0, dtype=nl.uint32)

  iq, ir, iw = nl.mgrid[0:n, 0:b, 0:m]
  ip, io = nl.mgrid[0:n, 0:8]

  inp_tile = nl.load(data_tensor[iq, ir, iw])
  max_vals: nt.tensor[n, 8] = nisa.max8(src=inp_tile)

  inp_tile[iq, ir, iw] = nisa.nc_match_replace8(
    dst_idx=idx_tile[ip, io],
    data=inp_tile[iq, ir, iw],
    vals=max_vals[ip, io],
    imm=imm,
  )

  # NKI_EXAMPLE_4_END
  nl.store(out_tensor, value=inp_tile)
  nl.store(idx_tensor, value=idx_tile)
  return out_tensor, idx_tensor


def match_and_get_index(data, vals):
  row = data.copy()
  vlength = vals.shape[-1]

  result = np.zeros(shape=vals.shape, dtype=np.int32)
  idx = 0
  for j in range(vlength):
    matches = np.where(row == vals[j])[0]
    if matches:
      idx = matches[0]
      row[idx] = np.float32("-inf")
      result[j] = idx
  return result


def get_replaced_output_and_max_indices(a, imm=0):
  axis = -1
  a_reshaped = a.reshape(a.shape[0], -1)
  a_sorted = np.sort(a_reshaped, axis=axis)
  a_sorted_last_8 = a_sorted[:, -8:]
  max_vals = np.flip(a_sorted_last_8, axis=-1)

  c = a_reshaped.copy()
  concat_out_golden_max_vals = np.concatenate([c, max_vals], axis=axis)
  c_idx = np.apply_along_axis(
    # get index for first occurence of max_vals along the specified axis
    lambda x: match_and_get_index(x[:-8], x[-8:]),
    axis=axis,
    arr=concat_out_golden_max_vals,
  ).astype(np.uint32)
  np.put_along_axis(c, indices=c_idx, values=imm, axis=axis)
  c = np.reshape(c, a.shape)
  return c, c_idx


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

  def test_nc_match_replace_indices8(self):
    imm = np.float32('-inf')
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)

    b, b_idx = nki_nc_match_replace_indices8(a, imm=imm)
    c, c_idx = get_replaced_output_and_max_indices(a, imm)

    self.assertTrue(np.allclose(b, c))
    self.assertTrue(np.allclose(b_idx, c_idx))

  def test_nc_match_replace_indices8_mask(self):
    imm = np.float32('-inf')
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b, b_idx = nki_nc_match_replace_indices8_mask(a, imm=imm)
    c, c_idx = get_replaced_output_and_max_indices(a[:64, :256], imm) 

    self.assertTrue(np.allclose(b[:64, :256], c))
    self.assertTrue(np.allclose(b_idx[:64, :256], c_idx))

  def test_nc_match_replace_indices8_3d(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 4, 4]).astype(np.float32)

    b, b_idx = nki_nc_match_replace_indices8_3d(a)
    c, c_idx = get_replaced_output_and_max_indices(a, imm=0)

    self.assertTrue(np.allclose(b, c))
    self.assertTrue(np.allclose(b_idx, c_idx))

  def test_nc_match_replace_indices8_3d_inplace(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 4, 4]).astype(np.float32)

    b, b_idx = nki_nc_match_replace_indices8_3d_inplace(a)
    c, c_idx = get_replaced_output_and_max_indices(a, imm=0)

    self.assertTrue(np.allclose(b, c))
    self.assertTrue(np.allclose(b_idx, c_idx))