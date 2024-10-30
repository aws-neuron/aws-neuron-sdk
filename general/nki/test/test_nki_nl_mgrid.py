"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
...

nki_jit = nki.trace
simulate_kernel = nki.simulate_kernel

########################################################################
# NOTE: if you modify this file, make sure to update the source .py with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################

@nki_jit
def example_kernel(in_tensor, out_tensor):
  i_p, i_f = nl.mgrid[0:128, 0:512]
  tile = nl.load(in_tensor[i_p, i_f])
  ...
  nl.store(out_tensor[i_p, i_f], tile)


@nki_jit
def example_kernel_1(in_tensor, out_tensor):
  grid = nl.mgrid[0:128, 0:512]
  tile = nl.load(in_tensor[grid.p, grid.x])
  ...
  nl.store(out_tensor[grid.p, grid.x], tile)


class TestNkiExampleNlLoad(unittest.TestCase):
  def test_nl_load(self):
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.ndarray(shape=(128, 512), dtype=np.float32)

    simulate_kernel(example_kernel, a, b)
    self.assertTrue(np.allclose(a, b))

  def test_nl_load_1(self):
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.ndarray(shape=(128, 512), dtype=np.float32)

    simulate_kernel(example_kernel_1, a, b)
    self.assertTrue(np.allclose(a, b))