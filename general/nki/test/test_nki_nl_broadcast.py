"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np
import neuronxcc.nki as nki
# NKI_EXAMPLE_5_BEGIN
import neuronxcc.nki.language as nl
# NKI_EXAMPLE_5_END
...


########################################################################
# NOTE: if you modify this file, make sure to update the source .py with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################
@nki.jit(mode="simulation")
def test_nl_broadcast(in_tensor):
  out_tensor = nl.ndarray([128, 64], in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # NKI_EXAMPLE_5_BEGIN
  ##################################################################
  # Example 1: Load from in_tensor[P, F] that is on HBM and
  # copy into out_tile[P, F] that is on SBUF by broadcasting
  ##################################################################
  ...
  # NKI_EXAMPLE_5_END
  # NKI_EXAMPLE_5_BEGIN
  ...
  # broadcast into out_tile[P, F] that is on SBUF
  # from data_tile[P, F] that is on SBUF
  in_tile = nl.load(in_tensor, dtype=in_tensor.dtype)
  out_tile = nl.broadcast_to(in_tile, shape=(128, in_tensor.shape[1]))

  # store output
  nl.store(out_tensor, out_tile)
  # NKI_EXAMPLE_5_END
  return out_tensor


class TestNkiExampleNlBroadcast(unittest.TestCase):
  def test_nl_broadcast_to(self):
    src = np.random.random_sample([1, 64]).astype(np.int32)

    dst = test_nl_broadcast(src)
    self.assertTrue(np.allclose(np.repeat(src, 128, axis=0), dst))
