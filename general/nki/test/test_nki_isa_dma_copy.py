"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np
import neuronxcc.nki as nki

# NKI_EXAMPLE_1_BEGIN # NKI_EXAMPLE_2_BEGIN # NKI_EXAMPLE_3_BEGIN # NKI_EXAMPLE_4_BEGIN # NKI_EXAMPLE_5_BEGIN # NKI_EXAMPLE_6_BEGIN # NKI_EXAMPLE_7_END
# NKI_EXAMPLE_0_BEGIN
import neuronxcc.nki.isa as nisa
# NKI_EXAMPLE_0_END
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
# NKI_EXAMPLE_1_END # NKI_EXAMPLE_2_END # NKI_EXAMPLE_3_END # NKI_EXAMPLE_4_END # NKI_EXAMPLE_5_END # NKI_EXAMPLE_6_END # NKI_EXAMPLE_7_END

########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################


@nki.jit(mode="simulation")
def nki_dma_copy(a):
  b = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)

  # NKI_EXAMPLE_0_BEGIN
  ############################################################################
  # Example 1: Copy over the tensor to another tensor
  ############################################################################
  nisa.dma_copy(dst=b, src=a)

  # NKI_EXAMPLE_0_END

  return b


@nki.jit(mode="simulation")
def nki_indirect_load_oob_err(in_tensor):
  # NKI_EXAMPLE_1_BEGIN
  ############################################################################
  # Example 2: Load elements from HBM with indirect addressing. If addressing 
  # results out-of-bound access, the operation will fail.
  ############################################################################
  # NKI_EXAMPLE_1_END
  ...
  out_tensor: tensor[64, 512] = nl.ndarray([64, 512], dtype=in_tensor.dtype,
                                            buffer=nl.shared_hbm)
  # NKI_EXAMPLE_1_BEGIN
  ...
  n, m = in_tensor.shape
  ix, iy = nl.mgrid[0:n//2, 0:m]

  expr_arange = 2*nl.arange(n//2)[:, None]
  idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

  out_tile: tensor[64, 512] = nisa.memset(shape=(n//2, m), value=-1, dtype=in_tensor.dtype)
  nisa.dma_copy(src=in_tensor[idx_tile, iy], dst=out_tile[ix, iy], oob_mode=nisa.oob_mode.error)
  # NKI_EXAMPLE_1_END

  nl.store(out_tensor, value=out_tile)
  return out_tensor


@nki.jit(mode="simulation")
def nki_indirect_load_oob_error_negative(in_tensor):
  # NKI_EXAMPLE_2_BEGIN
  ############################################################################
  # Example 3: Load elements from HBM with indirect addressing. If addressing 
  # results in out-of-bounds access, the operation will fail.
  ############################################################################
  # NKI_EXAMPLE_2_END
  ...
  out_tensor: tensor[64, 512] = nl.ndarray([64, 512], dtype=in_tensor.dtype,
                                            buffer=nl.shared_hbm)
  # NKI_EXAMPLE_2_BEGIN
  ...
  n, m = in_tensor.shape
  ix, iy = nl.mgrid[0:n//2, 0:m]

  # indices are out of range on purpose to demonstrate the error
  expr_arange = 3*nl.arange(n//2)[:, None] 
  idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

  out_tile: tensor[64, 512] = nisa.memset(shape=(n//2, m), value=-1, dtype=in_tensor.dtype)
  nisa.dma_copy(src=in_tensor[idx_tile, iy], dst=out_tile[ix, iy], oob_mode=nisa.oob_mode.error)

  # NKI_EXAMPLE_2_END

  nl.store(out_tensor, value=out_tile)
  return out_tensor

  
@nki.jit(mode="simulation")
def nki_indirect_load_oob_skip(in_tensor):
  # NKI_EXAMPLE_3_BEGIN
  ############################################################################
  # Example 4: Load elements from HBM with indirect addressing. If addressing 
  # results in out-of-bounds access, the operation will skip indices.
  ############################################################################
  # NKI_EXAMPLE_3_END
  ...
  out_tensor: tensor[64, 512] = nl.ndarray([64, 512], dtype=in_tensor.dtype,
                                            buffer=nl.shared_hbm)
  # NKI_EXAMPLE_3_BEGIN
  ...
  n, m = in_tensor.shape
  ix, iy = nl.mgrid[0:n//2, 0:m]

  # indices are out of range on purpose
  expr_arange = 3*nl.arange(n//2)[:, None] 
  idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

  out_tile: tensor[64, 512] = nisa.memset(shape=(n//2, m), value=-1, dtype=in_tensor.dtype)
  nisa.dma_copy(src=in_tensor[idx_tile, iy], dst=out_tile[ix, iy], oob_mode=nisa.oob_mode.skip)

  # NKI_EXAMPLE_3_END

  nl.store(out_tensor, value=out_tile)
  return out_tensor


@nki.jit(mode="simulation")
def nki_indirect_store_rmw(in_tensor):
  # NKI_EXAMPLE_4_BEGIN
  ############################################################################
  # Example 5: Store elements to HBM with indirect addressing and with 
  # read-modifed-write operation.
  ############################################################################
  # NKI_EXAMPLE_4_END
  ...
  out_tensor: tensor[128, 512] = nl.ndarray([128, 512], dtype=in_tensor.dtype,
                                            buffer=nl.shared_hbm)
  # NKI_EXAMPLE_4_BEGIN
  ...
  n, m = in_tensor.shape
  ix, iy = nl.mgrid[0:n, 0:m]

  expr_arange = 2*nl.arange(n)[:, None]
  inp_tile: tensor[64, 512] = nl.load(in_tensor[ix, iy])
  idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

  out_tile: tensor[128, 512] = nisa.memset(shape=(2*n, m), value=1, dtype=in_tensor.dtype)
  nl.store(out_tensor, value=out_tile)
  nisa.dma_copy(dst=out_tensor[idx_tile, iy], src=inp_tile, dst_rmw_op=np.add)
  # NKI_EXAMPLE_4_END

  return out_tensor


@nki.jit(mode="simulation")
def nki_indirect_store_oob_err(in_tensor):
  # NKI_EXAMPLE_5_BEGIN
  ############################################################################
  # Example 6: Store elements to HBM with indirect addressing. If indirect 
  # addressing results out-of-bound access, the operation will fail.
  ############################################################################
  # NKI_EXAMPLE_5_END
  ...
  out_tensor: tensor[128, 512] = nl.ndarray([128, 512], dtype=in_tensor.dtype,
                                            buffer=nl.shared_hbm)
  # NKI_EXAMPLE_5_BEGIN
  ...
  n, m = in_tensor.shape
  ix, iy = nl.mgrid[0:n, 0:m]

  expr_arange = 2*nl.arange(n)[:, None]
  inp_tile: tensor[64, 512] = nl.load(in_tensor[ix, iy])
  idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

  out_tile: tensor[128, 512] = nisa.memset(shape=(2*n, m), value=-1, dtype=in_tensor.dtype)
  nl.store(out_tensor, value=out_tile)
  nisa.dma_copy(dst=out_tensor[idx_tile, iy], src=inp_tile, oob_mode=nisa.oob_mode.error)
  # NKI_EXAMPLE_5_END

  return out_tensor


@nki.jit(mode="simulation")
def nki_indirect_store_oob_err_negative(in_tensor):
  # NKI_EXAMPLE_6_BEGIN
  ############################################################################
  # Example 7: Store elements to HBM with indirect addressing. If indirect 
  # addressing results out-of-bounds access, the operation will skip indices.
  ############################################################################
  # NKI_EXAMPLE_6_END
  ...
  out_tensor: tensor[128, 512] = nl.ndarray([128, 512], dtype=in_tensor.dtype,
                                            buffer=nl.shared_hbm)
  # NKI_EXAMPLE_6_BEGIN
  ...
  n, m = in_tensor.shape
  ix, iy = nl.mgrid[0:n, 0:m]

  # indices are out of range on purpose to demonstrate the error
  expr_arange = 3*nl.arange(n)[:, None] 
  inp_tile: tensor[64, 512] = nl.load(in_tensor[ix, iy])
  idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

  out_tile: tensor[128, 512] = nisa.memset(shape=(2*n, m), value=-1, dtype=in_tensor.dtype)
  nl.store(out_tensor, value=out_tile)
  nisa.dma_copy(dst=out_tensor[idx_tile, iy], src=inp_tile, oob_mode=nisa.oob_mode.error)

  # NKI_EXAMPLE_6_END

  return out_tensor

  
@nki.jit(mode="simulation")
def nki_indirect_store_oob_skip(in_tensor):
  # NKI_EXAMPLE_7_BEGIN
  ############################################################################
  # Example 8: Store elements to HBM with indirect addressing. If indirect 
  # addressing results out-of-bounds access, the operation will skip indices.
  ############################################################################
  # NKI_EXAMPLE_7_END
  ...
  out_tensor: tensor[128, 512] = nl.ndarray([128, 512], dtype=in_tensor.dtype,
                                            buffer=nl.shared_hbm)
  # NKI_EXAMPLE_7_BEGIN
  ...
  n, m = in_tensor.shape
  ix, iy = nl.mgrid[0:n, 0:m]

  # indices are out of range on purpose
  expr_arange = 3*nl.arange(n)[:, None] 
  inp_tile: tensor[64, 512] = nl.load(in_tensor[ix, iy])
  idx_tile: tensor[64, 1] = nisa.iota(expr_arange, dtype=np.int32)

  out_tile: tensor[128, 512] = nisa.memset(shape=(2*n, m), value=-1, dtype=in_tensor.dtype)
  nl.store(out_tensor, value=out_tile)
  nisa.dma_copy(dst=out_tensor[idx_tile, iy], src=inp_tile, oob_mode=nisa.oob_mode.skip)

  # NKI_EXAMPLE_7_END

  return out_tensor

@nki.jit(mode='simulation')
def nki_dma_copy_swdge(in_tensor):
  # NKI_EXAMPLE_8_BEGIN
  ############################################################################
  # Example 9: Copy data with SWDGE. Must follow DGE access pattern requirements
  # to use DGE.
  ############################################################################
  # NKI_EXAMPLE_8_END
  ...
  out_tensor: tensor[64, 512] = nl.ndarray([64, 512], dtype=in_tensor.dtype,
                                            buffer=nl.shared_hbm)
  # NKI_EXAMPLE_8_BEGIN
  ...
  nisa.dma_copy(dst=out_tensor, src=in_tensor, dge_mode=nisa.dge_mode.swdge)

  # NKI_EXAMPLE_8_END

  return out_tensor

@nki.jit(mode='simulation', platform_target='trn2')
def nki_dma_copy_hwdge(in_tensor):
  # NKI_EXAMPLE_9_BEGIN
  ############################################################################
  # Example 10: Copy data with HWDGE. Must follow DGE access pattern requirements,
  # and further have (1) accessed partitions=128 (2) spill/reload DMA 
  # (3) target=trn2+ to use HWDGE.
  ############################################################################
  # NKI_EXAMPLE_9_END
  ...
  out_tensor: tensor[128, 512] = nl.ndarray([128, 512], dtype=in_tensor.dtype,
                                            buffer=nl.shared_hbm)
  # NKI_EXAMPLE_9_BEGIN
  ...
  inp_tile: tensor[128, 512] = nl.load(in_tensor)
  out_tile: tensor[128, 512] = nl.zeros_like(inp_tile, buffer=nl.sbuf)
  nisa.dma_copy(dst=out_tile, src=inp_tile, dge_mode=nisa.dge_mode.hwdge)
  nl.store(out_tensor, value=out_tile)

  # NKI_EXAMPLE_9_END

  return out_tensor
      
class TestNkiIsaExamplesTensorCopy(unittest.TestCase):
  def test_tensor_copy(self):
    np.random.seed(0)
    src = np.random.random_sample([256, 1]).astype(np.float32) * 100
    dst_golden = np.copy(src)

    dst = nki_dma_copy(src)
    self.assertTrue(np.allclose(dst, dst_golden))


  def test_indirect_load_oob_err(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)

    b = nki_indirect_load_oob_err(a)
    
    b_golden = a[2 * np.arange(64, dtype=np.int32)]

    self.assertTrue(np.allclose(b, b_golden))


  def test_indirect_load_oob_err_negative(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)

    with self.assertRaises(IndexError) as cm:
      b = nki_indirect_load_oob_error_negative(a)
    exc = cm.exception
    self.assertEqual(type(exc), IndexError)
    self.assertIn(str(exc), 'index 66048 is out of bounds for axis 0 with size 65536')


  def test_indirect_load_oob_skip(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)

    b = nki_indirect_load_oob_skip(a)

    n, m = a.shape
    b_golden = np.full((n//2, m), -1, dtype=a.dtype)
    indices = 3 * np.arange((n//3) + 1)
    b_golden[0:len(indices)] = a[indices, :]

    self.assertTrue(np.allclose(b, b_golden))

    
  def test_indirect_store_rmw(self):
    np.random.seed(0)
    a = np.random.random_sample([64, 512]).astype(np.float32)

    b = nki_indirect_store_rmw(a)

    n, m = a.shape
    b_golden = np.full(shape=(2*n, m), fill_value=1, dtype=a.dtype)
    b_golden[2 * np.arange(n, dtype=np.int32)] += a

    self.assertTrue(np.allclose(b, b_golden))


  def test_indirect_store_oob_err(self):
    np.random.seed(0)
    a = np.random.random_sample([64, 512]).astype(np.float32)

    b = nki_indirect_store_oob_err(a)

    n, m = a.shape
    b_golden = np.full(shape=(2*n, m), fill_value=-1, dtype=a.dtype)
    b_golden[2 * np.arange(n, dtype=np.int32)] = a

    self.assertTrue(np.allclose(b, b_golden))


  def test_indirect_store_oob_err_negative(self):
    np.random.seed(0)
    a = np.random.random_sample([64, 512]).astype(np.float32)

    with self.assertRaises(IndexError) as cm:
      b = nki_indirect_store_oob_err_negative(a)
    exc = cm.exception
    self.assertEqual(type(exc), IndexError)
    self.assertIn(str(exc), 'index 66048 is out of bounds for axis 0 with size 65536')


  def test_indirect_store_oob_skip(self):
    np.random.seed(0)
    a = np.random.random_sample([64, 512]).astype(np.float32)

    b = nki_indirect_store_oob_skip(a)

    n, m = a.shape
    b_golden = np.full(shape=(2*n, m), fill_value=-1, dtype=a.dtype)
    indices = 3*np.arange(((2*n)//3) + 1)
    b_golden[indices, :] = a[0:len(indices), :]

    self.assertTrue(np.allclose(b, b_golden))

  def test_dma_copy_swdge(self):
    np.random.seed(0)
    a = np.random.random_sample([64, 512]).astype(np.float32)
    b = nki_dma_copy_swdge(a)
    self.assertTrue(np.allclose(b, a))

  def test_dma_copy_hwdge(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = nki_dma_copy_hwdge(a)
    self.assertTrue(np.allclose(b, a))
