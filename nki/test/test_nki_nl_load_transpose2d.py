"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np
import neuronxcc.nki as nki
# NKI_EXAMPLE_19_BEGIN
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
...

# NKI_EXAMPLE_19_END

########################################################################
# NOTE: if you modify this file, make sure to update the source .py with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################


@nki.jit(mode="simulation")
def example_kernel_0(in_tensor):
  out_tensor = nl.ndarray([in_tensor.shape[1], in_tensor.shape[0]], dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # NKI_EXAMPLE_19_BEGIN
  # load from in_tensor[F, P] that is on HBM
  # transpose and copy into local_tile[P, F] that is on SBUF
  N, M = in_tensor.shape
  local_tile: tensor[M, N] = nl.load_transpose2d(in_tensor)
  ...
  # NKI_EXAMPLE_19_END
  nl.store(out_tensor, value=local_tile)
  return out_tensor


@nki.jit(mode="simulation")
def example_kernel_1(in_tensor):
  out_tensor = nl.ndarray([in_tensor.shape[1], in_tensor.shape[0]], dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # NKI_EXAMPLE_20_BEGIN
  import neuronxcc.nki.isa as nisa
  ...

  # load from in_tensor[F, P] that is on HBM
  # transpose and copy into local_tile[P, F] that is on SBUF
  # always use the DMA engine
  N, M = in_tensor.shape
  local_tile: tensor[M, N] = nisa.dma_transpose(in_tensor)
  ...
  # NKI_EXAMPLE_20_END
  nl.store(out_tensor, value=local_tile)
  return out_tensor



class TestNkiExampleNlLoadTranspose2d(unittest.TestCase):
  def test_dma_transpose_load_0(self):
    np.random.seed(0)
    src = np.random.random_sample([2048, 128]).astype(np.float32) * 100

    dst = example_kernel_0(src)

    dst_golden = np.transpose(src)
    self.assertTrue(np.allclose(dst, dst_golden))

  def test_dma_transpose_load_1(self):
    np.random.seed(0)
    src = np.random.random_sample([2048, 128]).astype(np.float32) * 100

    dst = example_kernel_1(src)

    dst_golden = np.transpose(src)
    self.assertTrue(np.allclose(dst, dst_golden))
