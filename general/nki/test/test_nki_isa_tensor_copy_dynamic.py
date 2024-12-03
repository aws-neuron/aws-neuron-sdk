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
def example_tensor_copy_dynamic_src_0(in_tensor, idx_tensor):
  out_tensor = nl.ndarray([128, 64], dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # NKI_EXAMPLE_0_BEGIN
  ##############################################################################################
  # TensorCopyDynamicSrc example 1:
  # - ``in_tensor`` on HBM is of shape [128 x 512]
  # - ``in_tensor_sbuf`` on SBUF is of shape [128 x 512]
  # - ``idx_tensor`` on HBM is of shape [1 x 64] (with values [4, 5, 6, 7, ...])
  # - ``idx_tensor`` values are loaded into a SBUF tile, ``idx_tensor_sbuf``, from HBM
  # - ``in_tensor_sbuf`` is copied to ``out_sbuf``based on indices stored in ``idx_tensor_sbuf``
  # - ``out_tensor`` of shape [128 x 64] is finally written to HBM
  ##############################################################################################
  ix, iy = nl.mgrid[0:128, 0:1]

  in_tensor_sbuf = nl.load(in_tensor)

  # indices must be on SBUF
  idx_tensor_sbuf: nt.tensor[1, 64] = nl.load(idx_tensor)

  # write temporary output to SBUF
  out_sbuf: nt.tensor[128, 64] = nl.ndarray([128, 64], dtype=in_tensor.dtype,
                                            buffer=nl.sbuf)

  # in each iteration a 1 X 1 tensor offset is accessed in ``idx_tile``
  # in our example, we select the dynamic offset along axis=1.
  # ``idx_tensor`` is dynamically populated.
  for b_idx in nl.affine_range(idx_tensor_sbuf.shape[1]):
    out_sbuf[ix, b_idx] = nisa.tensor_copy_dynamic_src(
        in_tensor_sbuf[ix, idx_tensor_sbuf[0, b_idx] + iy])
  ...
  # NKI_EXAMPLE_0_END
  # write final output to HBM as buffered output
  nl.store(out_tensor, value=out_sbuf)
  return out_tensor


@nki.jit(mode="simulation")
def example_tensor_copy_dynamic_src_1(in_tensor, idx_tensor):
  out_tensor = nl.ndarray([128, 8, 4], dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # NKI_EXAMPLE_1_BEGIN
  ###############################################################################################
  # TensorCopyDynamicSrc example 1:
  # - ``in_tensor`` on HBM is of shape [128 x 512 x 4]
  # - ``in_tensor_sbuf`` on SBUF has shape [128 x 512 x 4]
  # - ``idx_tensor`` on HBM is of shape [1 x 8] (with values [4, 5, 6, 7, ...])
  # - ``idx_tensor`` values are loaded into a SBUF tile, ``idx_tensor_sbuf``, from HBM
  # - ``in_tensor_sbuf`` is copied to ``out_sbuf``based on indices stored in ``idx_tensor_sbuf``
  # - ``out_tensor`` of shape [128 x 8 x 4] is finally written to HBM
  ###############################################################################################
  ix, iy, iz = nl.mgrid[0:128, 0:1, 0:4]

  in_tensor_sbuf = nl.load(in_tensor)

  # indices must be on SBUF
  idx_tensor_sbuf: nt.tensor[1, 8] = nl.load(idx_tensor)

  # write temporary output to SBUF
  out_sbuf: nt.tensor[128, 8, 4] = nl.ndarray([128, 8, 4], dtype=in_tensor.dtype,
                                              buffer=nl.sbuf)

  # in each iteration a 1 X 1 tensor offset is accessed in ``idx_tile``
  # in our example, we select the dynamic offset along axis=1.
  # ``idx_tensor`` is dynamically populated.
  for b_idx in nl.affine_range(idx_tensor.shape[1]):
    out_sbuf[ix, b_idx, iz] = nisa.tensor_copy_dynamic_src(
        in_tensor_sbuf[ix, idx_tensor_sbuf[0, b_idx], iz])
  ...
  # NKI_EXAMPLE_1_END
  nl.store(out_tensor, value=out_sbuf)
  return out_tensor


class TestNkiExampleNisaTensorCopyDynSrc(unittest.TestCase):
  def test_tensor_copy_dyn_src_0(self):
    in_tensor = np.random.random_sample([128, 512]).astype(np.float32)
    dynamic_offset = np.full([], fill_value=4, dtype=np.int32)
    idx_tensor = np.expand_dims(dynamic_offset + np.arange(64).astype(np.int32),
                                axis=0)
    golden = np.take(in_tensor, idx_tensor.squeeze(), axis=1)

    out_tensor = example_tensor_copy_dynamic_src_0(in_tensor, idx_tensor)
    self.assertTrue(np.allclose(out_tensor, golden))

  def test_tensor_copy_dyn_src_1(self):
    in_tensor = np.random.random_sample([128, 512, 4]).astype(np.float32)
    dynamic_offset = np.full([], fill_value=4, dtype=np.int32)
    idx_tensor = np.expand_dims(dynamic_offset + np.arange(8).astype(np.int32),
                                axis=0)
    golden = np.take(in_tensor, idx_tensor.squeeze(), axis=1)

    out_tensor = example_tensor_copy_dynamic_src_1(in_tensor, idx_tensor)
    self.assertTrue(np.allclose(out_tensor, golden))
