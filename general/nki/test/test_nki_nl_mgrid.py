"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np
import neuronxcc.nki as nki
# NKI_EXAMPLE_9_BEGIN NKI_EXAMPLE_8_BEGIN
import neuronxcc.nki.language as nl
...

# NKI_EXAMPLE_8_END NKI_EXAMPLE_9_END

########################################################################
# NOTE: if you modify this file, make sure to update the source .py with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################


@nki.jit(mode="simulation")
def example_kernel(in_tensor):
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # NKI_EXAMPLE_8_BEGIN
  i_p, i_f = nl.mgrid[0:128, 0:512]
  tile = nl.load(in_tensor[i_p, i_f])
  ...
  nl.store(out_tensor[i_p, i_f], tile)

  # NKI_EXAMPLE_8_END
  return out_tensor


@nki.jit(mode="simulation")
def example_kernel_1(in_tensor):
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # NKI_EXAMPLE_9_BEGIN
  grid = nl.mgrid[0:128, 0:512]
  tile = nl.load(in_tensor[grid.p, grid.x])
  ...
  nl.store(out_tensor[grid.p, grid.x], tile)
  # NKI_EXAMPLE_9_END
  return out_tensor


class TestNkiExampleNlLoad(unittest.TestCase):
  def test_nl_load(self):
    a = np.random.random_sample([128, 512]).astype(np.float32) * 100
    b = np.ndarray(shape=(128, 512), dtype=np.float32)

    b = example_kernel(a)
    self.assertTrue(np.allclose(a, b))

  def test_nl_load_1(self):
    a = np.random.random_sample([128, 512]).astype(np.float32) * 100
    b = np.ndarray(shape=(128, 512), dtype=np.float32)

    b = example_kernel_1(a)
    self.assertTrue(np.allclose(a, b))