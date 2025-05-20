"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np
import neuronxcc.nki as nki
# NKI_EXAMPLE_0_BEGIN NKI_EXAMPLE_1_BEGIN
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
...

# NKI_EXAMPLE_0_END NKI_EXAMPLE_1_END


@nki.jit(mode="simulation")
def example_tensor_copy_dynamic_src_0(src_tensor, offsets):
  out_tensor = nl.ndarray([128, 64], dtype=src_tensor.dtype,
                          buffer=nl.shared_hbm)
  # NKI_EXAMPLE_0_BEGIN
  #########################################################################################
  # TensorCopyDynamicSrc example 0:
  # - src_tensor in HBM of shape [128, 512]
  # - offsets in HBM of shape [1, 64] (with values [4, 5, 6, 7, ...])
  # - Gather tiles of shape [128, 1] from src_tensor into out_tensor using offsets
  #########################################################################################

  # Load src_tensor and offsets into SBUF
  src_tensor_sbuf: nt.tensor[128, 512] = nl.load(src_tensor)
  offsets_sbuf: nt.tensor[1, 64] = nl.load(offsets)

  # Copy into output tensor in SBUF
  out_sbuf: nt.tensor[128, 64] = nl.ndarray([128, 64], dtype=src_tensor.dtype,
                                            buffer=nl.sbuf)

  # Static indices to access a tile of shape [128, 1];
  # Add dynamic offsets to iy for tensor_copy_dynamic_src
  ix, iy = nl.mgrid[0:128, 0:1]

  for idx in nl.affine_range(offsets_sbuf.shape[1]):
    out_sbuf[ix, idx] = nisa.tensor_copy_dynamic_src(
        src_tensor_sbuf[ix, offsets_sbuf[0, idx] + iy])

  nl.store(out_tensor, value=out_sbuf)
  ...
  # NKI_EXAMPLE_0_END
  # write final output to HBM as buffered output
  return out_tensor


@nki.jit(mode="simulation")
def example_tensor_copy_dynamic_src_1(src_tensor, offsets):
  out_tensor = nl.ndarray([128, 8, 4], dtype=src_tensor.dtype,
                          buffer=nl.shared_hbm)
  # NKI_EXAMPLE_1_BEGIN
  #########################################################################################
  # TensorCopyDynamicSrc example 1:
  # - src_tensor in HBM of shape [128, 512, 4]
  # - offsets in HBM of shape [1 x 8] (with values [4, 5, 6, 7, ...]) to index into
  #   second axis of src_tensor
  # - Gather tiles of shape [128, 4] from src_tensor into out_tensor using offsets
  #########################################################################################

  # Load src_tensor and offsets into SBUF
  src_tensor_sbuf: nt.tensor[128, 512, 4] = nl.load(src_tensor)
  offsets_sbuf: nt.tensor[1, 8] = nl.load(offsets)

  # Copy into output tensor in SBUF
  out_sbuf: nt.tensor[128, 8, 4] = nl.ndarray([128, 8, 4], dtype=src_tensor.dtype,
                                              buffer=nl.sbuf)

  # Static indices to access a tile of shape [128, 1, 4];
  # Use dynamic offsets directly to index the second axis for tensor_copy_dynamic_src
  ix, _, iz = nl.mgrid[0:128, 0:1, 0:4]

  for idx in nl.affine_range(offsets.shape[1]):
    out_sbuf[ix, idx, iz] = nisa.tensor_copy_dynamic_src(
        src_tensor_sbuf[ix, offsets_sbuf[0, idx], iz])

  nl.store(out_tensor, value=out_sbuf)
  ...
  # NKI_EXAMPLE_1_END
  return out_tensor


class TestNkiExampleNisaTensorCopyDynSrc(unittest.TestCase):
  def test_tensor_copy_dyn_src_0(self):
    in_tensor = np.random.random_sample([128, 512]).astype(np.float32) * 100
    dynamic_offset = np.full([], fill_value=4, dtype=np.int32)
    idx_tensor = np.expand_dims(dynamic_offset + np.arange(64).astype(np.int32),
                                axis=0)
    golden = np.take(in_tensor, idx_tensor.squeeze(), axis=1)

    out_tensor = example_tensor_copy_dynamic_src_0(in_tensor, idx_tensor)
    self.assertTrue(np.allclose(out_tensor, golden))

  def test_tensor_copy_dyn_src_1(self):
    in_tensor = np.random.random_sample([128, 512, 4]).astype(np.float32) * 100
    dynamic_offset = np.full([], fill_value=4, dtype=np.int32)
    idx_tensor = np.expand_dims(dynamic_offset + np.arange(8).astype(np.int32),
                                axis=0)
    golden = np.take(in_tensor, idx_tensor.squeeze(), axis=1)

    out_tensor = example_tensor_copy_dynamic_src_1(in_tensor, idx_tensor)
    self.assertTrue(np.allclose(out_tensor, golden))
