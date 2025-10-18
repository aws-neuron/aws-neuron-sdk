"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

# NKI_EXAMPLE_BEGIN
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np


@nki.jit
def print_kernel(a_tensor):
  b = nl.empty_like(a_tensor, buffer=nl.hbm)

  # Load tensor into sbuf
  a = nl.load(a_tensor)

  # Print tensor y
  nl.device_print("value of a:", a)

  # Directly store a into hbm
  nl.store(b, value=a)

  return b
# NKI_EXAMPLE_END


class TestNkiIsaExamplesSimulateKernel(unittest.TestCase):
  def test_simulate_kernel(self):
    # NKI_EXAMPLE_BEGIN
    np.random.seed(0)
    a = np.random.random_sample([3, 4]).astype(np.float32) * 10

    b = nki.simulate_kernel(print_kernel, a)

    assert np.allclose(a, b)
    # NKI_EXAMPLE_END