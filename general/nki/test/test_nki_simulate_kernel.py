"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np

nki_jit = nki.trace

@nki_jit
def print_kernel(a):
  i0 = nl.arange(4)[:, None]
  i1 = nl.arange(4)[None, :]

  # Create (4, 4) tensor in sbuf
  y = nl.zeros([4, 4], dtype=np.float32)

  # Print tensor y
  nl.device_print("value of y:", y)

  # Directly store tensor y as a single tile
  nl.store(a[i0, i1], value=y)

class TestNkiIsaExamplesSimulateKernel(unittest.TestCase):
  def test_simulate_kernel(self):
    np.random.seed(0)
    a = np.random.random_sample([4, 4]).astype(np.float32)

    nki.simulate_kernel(print_kernel, a)

    self.assertTrue(np.allclose(a, np.zeros([4, 4])))
