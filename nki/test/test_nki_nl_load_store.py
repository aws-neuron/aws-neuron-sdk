"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np
import neuronxcc.nki as nki
# NKI_EXAMPLE_16_BEGIN NKI_EXAMPLE_15_BEGIN NKI_EXAMPLE_14_BEGIN NKI_EXAMPLE_11_BEGIN NKI_EXAMPLE_10_BEGIN
import neuronxcc.nki.language as nl
# NKI_EXAMPLE_16_END NKI_EXAMPLE_10_END NKI_EXAMPLE_11_END NKI_EXAMPLE_14_END NKI_EXAMPLE_15_END
...


########################################################################
# NOTE: if you modify this file, make sure to update the source .py with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################


@nki.jit(mode="simulation")
def example_kernel(in_tensor, use_scalar=False):
  out_tensor = nl.ndarray(in_tensor.shape, in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # NKI_EXAMPLE_10_BEGIN
  # load from in_tensor[P, F] that is on HBM
  # copy into data_tile[P, F] that is on SBUF
  data_tile = nl.load(in_tensor)
  ...
  # NKI_EXAMPLE_10_END
  if use_scalar:
    # NKI_EXAMPLE_16_BEGIN
    ...
    scalar = 100
    # store scalar into out_tensor on HBM (effectively a memset)
    nl.store(out_tensor, scalar)
    # NKI_EXAMPLE_16_END
  else:
    # NKI_EXAMPLE_14_BEGIN
    ...
    # store into out_tensor[P, F] that is on HBM
    # from data_tile[P, F] that is on SBUF
    nl.store(out_tensor, data_tile)
    # NKI_EXAMPLE_14_END
  return out_tensor


@nki.jit(mode="simulation")
def example_load_store_b(in_tensor):
  out_tensor = nl.ndarray(in_tensor.shape, in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # NKI_EXAMPLE_15_BEGIN NKI_EXAMPLE_11_BEGIN
  for i_b in nl.affine_range(4):
    data_tile = nl.zeros((128, 512), dtype=in_tensor.dtype) 
    # NKI_EXAMPLE_15_END
    # load from in_tensor[4, 128, 512] one batch at a time
    # copy into data_tile[128, 512]
    i_p, i_f = nl.mgrid[0:128, 0:512]
    data_tile[i_p, i_f] = nl.load(in_tensor[i_b, i_p, i_f])
    # NKI_EXAMPLE_15_BEGIN
    ...
    # NKI_EXAMPLE_11_END
    # store into out_tensor[4, 128, 512] one batch at a time
    # from data_tile[128, 512] 
    i_p, i_f = nl.mgrid[0:128, 0:512]
    nl.store(out_tensor[i_b, i_p, i_f], value=data_tile[i_p, i_f]) 
    # NKI_EXAMPLE_15_END
  return out_tensor


class TestNkiExampleNlLoad(unittest.TestCase):
  def test_nl_load(self):
    src = np.random.random_sample([128, 512]).astype(np.float32) * 100

    dst = example_kernel(src)
    self.assertTrue(np.allclose(src, dst))

  def test_nl_load_scalar(self):
    src = np.ones([128, 512]).astype(np.int32) * 100

    dst = example_kernel(src, use_scalar=True)
    self.assertTrue(np.allclose(src, dst))

  def test_load_store_3d(self):
    in_tensor = np.random.random_sample([4, 128, 512]).astype(np.float32) * 100

    out_tensor = example_load_store_b(in_tensor)
    self.assertTrue(np.allclose(out_tensor, in_tensor))
