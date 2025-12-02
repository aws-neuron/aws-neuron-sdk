"""
Copyright (C) 2025, Amazon.com. All Rights Reserved

"""
import unittest
import pytest

import numpy as np
import neuronxcc.nki as nki

# NKI_EXAMPLE_0_BEGIN NKI_EXAMPLE_1_BEGIN NKI_EXAMPLE_2_BEGIN NKI_EXAMPLE_3_BEGIN NKI_EXAMPLE_4_BEGIN NKI_EXAMPLE_5_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
# NKI_EXAMPLE_0_END NKI_EXAMPLE_1_END NKI_EXAMPLE_4_END NKI_EXAMPLE_5_END
from neuronxcc.nki.isa.constants import dge_mode
# NKI_EXAMPLE_2_END NKI_EXAMPLE_3_END

#############################################################################
# NOTE: if you modify this file, make sure to update neuron_isa.py file with
# NOTE: the correct line numbers under .. nki_example:: directive
#############################################################################

@nki.jit(mode="simulation")
# NKI_EXAMPLE_0_BEGIN
############################################################################
# Example 1: Simple 2D transpose (HBM->SB)
############################################################################
def nki_dma_transpose_2d_hbm2sb(a):
  b_sb = nisa.dma_transpose(a[:, :])
  b = nl.ndarray(shape=b_sb.shape, dtype=b_sb.dtype, buffer=nl.hbm)
  nl.store(dst=b, value=b_sb)
  return b
# NKI_EXAMPLE_0_END

@nki.jit(mode="simulation")
# NKI_EXAMPLE_1_BEGIN
############################################################################
# Example 2: Simple 2D transpose (SB->SB)
############################################################################
def nki_dma_transpose_2d_sb2sb(a):
  a_sb = nl.load(a)
  b_sb = nisa.dma_transpose(a_sb[:, :])
  b = nl.ndarray(shape=b_sb.shape, dtype=b_sb.dtype, buffer=nl.hbm)
  nl.store(dst=b, value=b_sb)
  return b
# NKI_EXAMPLE_1_END

@nki.jit(mode="simulation", platform_target="trn2")
# NKI_EXAMPLE_2_BEGIN
################################################################################
# Example 3: Simple 2D transpose (HBM->SB) using DGE xbar (NeuronCore-v3+ only)
################################################################################
def nki_dma_transpose_2d_hbm2sb_dge_xbar(a):
  b_sb = nisa.dma_transpose(a[:, :], dge_mode=dge_mode.hwdge)
  b = nl.ndarray(shape=b_sb.shape, dtype=b_sb.dtype, buffer=nl.hbm)
  nl.store(dst=b, value=b_sb)
  return b
# NKI_EXAMPLE_2_END

@nki.jit(mode="simulation", platform_target="trn2")
# NKI_EXAMPLE_3_BEGIN
###############################################################################
# Example 4: Simple 2D transpose (SB->SB) using DGE xbar (NeuronCore-v3+ only)
###############################################################################
def nki_dma_transpose_2d_sb2sb_dge_xbar(a):
  a_sb = nl.load(a)
  b_sb = nisa.dma_transpose(a_sb[:, :], dge_mode=dge_mode.hwdge)
  b = nl.ndarray(shape=b_sb.shape, dtype=b_sb.dtype, buffer=nl.hbm)
  nl.store(dst=b, value=b_sb)
  return b
# NKI_EXAMPLE_3_END

@nki.jit(mode="simulation", platform_target="trn2")
# NKI_EXAMPLE_4_BEGIN
############################################################################
# Example 5: 3D transpose (HBM->SB) w/Indirect Mem Access
############################################################################
def nki_dma_gather_transpose_3d_hbm2sb(src_tensor, idx_tensor):
  i_p = nl.arange(32)[:, None]
  idx = nl.load(idx_tensor)

  _, dim1, dim2 = src_tensor.shape

  iy = nl.arange(dim1)[None, :, None]
  iz = nl.arange(dim2)[None, None, :]

  dst = nisa.dma_transpose(src_tensor[idx[i_p, 0], iy, iz], axes=(2, 1, 0))
  dst_tensor = nl.ndarray(shape=(dim2, dim1, idx.shape[0]), dtype=src_tensor.dtype, buffer=nl.shared_hbm)
    
  nl.store(dst_tensor, dst)
  return dst_tensor
# NKI_EXAMPLE_4_END

@nki.jit(mode="simulation", platform_target="trn2")
# NKI_EXAMPLE_5_BEGIN
############################################################################
# Example 6: 3D transpose (SB->SB) w/Indirect Mem Access
############################################################################
def nki_dma_gather_transpose_3d_sb2sb(src_tensor, idx_tensor):
  src = nl.load(src_tensor)
  idx = nl.load(idx_tensor)

  dim0, dim1, dim2 = src.shape
  
  iy = nl.arange(dim1)[None, :, None]
  iz = nl.arange(dim2)[None, None, :]

  dst = nisa.dma_transpose(src[idx, iy, iz], axes=(2, 1, 0))
  dst_tensor = nl.ndarray(shape=(dim2, dim1, dim0), dtype=src.dtype, buffer=nl.shared_hbm)
  
  nl.store(dst_tensor, dst)
  return dst_tensor
# NKI_EXAMPLE_5_END

class TestNkiIsaExamplesDmaTranspose(unittest.TestCase):
  def test_dma_transpose_2d(self):
    np.random.seed(0)
    src = np.random.random_sample([16, 128]).astype(np.float16) * 100
    dst_golden = np.transpose(src)

    dst = nki_dma_transpose_2d_hbm2sb(src)
    self.assertTrue(np.allclose(dst, dst_golden))

    dst = nki_dma_transpose_2d_sb2sb(src)
    self.assertTrue(np.allclose(dst, dst_golden))

    dst = nki_dma_transpose_2d_hbm2sb_dge_xbar(src)
    self.assertTrue(np.allclose(dst, dst_golden))

    dst = nki_dma_transpose_2d_sb2sb_dge_xbar(src)
    self.assertTrue(np.allclose(dst, dst_golden))
  
  @pytest.mark.xfail(reason="PBE-63")
  def test_dma_transpose_indirect(self):
    np.random.seed(0)
    src_tensor = np.arange(64 * 4 * 128).reshape(64, 4, 128).astype(nl.uint16)
    idx_tensor = np.arange(32, dtype=nl.uint32).reshape(32, 1)
    
    nki_out = nki_dma_gather_transpose_3d_hbm2sb(src_tensor, idx_tensor)
    golden_out = np.transpose(src_tensor[idx_tensor.reshape(32)], axes=(2, 1, 0))

    assert np.allclose(nki_out, golden_out)
