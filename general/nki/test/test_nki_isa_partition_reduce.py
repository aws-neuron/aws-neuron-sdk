"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
# NKI_EXAMPLE_1_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
...
# NKI_EXAMPLE_1_END
nki_jit = nki.trace
simulate_kernel = nki.simulate_kernel

########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################

@nki_jit
def nki_par_reduce(a_tensor, b_tensor):
  # NKI_EXAMPLE_1_BEGIN
  ##################################################################
  # Example 1: reduce add tile a of shape (128, 32, 4)
  # in the partition dimension and return
  # reduction result in tile b of shape (1, 32, 4)
  ##################################################################
  a = nl.load(a_tensor[0:128, 0:32, 0:4])  
  b = nisa.tensor_partition_reduce(np.add, a)
  nl.store(b_tensor[0:1, 0:32, 0:4], b)
  # NKI_EXAMPLE_1_END

@nki_jit
def nki_par_reduce_nd_b(a_tensor, b_tensor):
  # NKI_EXAMPLE_1_BEGIN
  ##################################################################
  # Example 2: reduce add tile a of shape (b, p, f1, ...)
  # in the partition dimension p and return
  # reduction result in tile b of shape (b, 1, f1, ...)
  ##################################################################
  for i in nl.affine_range(a_tensor.shape[0]):
    a = nl.load(a_tensor[i])
    b = nisa.tensor_partition_reduce(np.add, a)
    nl.store(b_tensor[i], b)
  # NKI_EXAMPLE_1_END


class TestNkiIsaExamplesPartitionReduce(unittest.TestCase):
  def test_par_reduce_nd(self):
    a = np.random.random_sample([128, 32, 4]).astype(np.float32)
    b = np.ndarray(shape=(1, 32, 4), dtype=np.float32)
    simulate_kernel(nki_par_reduce, a, b)

    self.assertTrue(np.allclose(b, np.sum(a, axis=0, keepdims=True)))

  def test_par_reduce_nd_b(self):
    a = np.random.random_sample([4, 128, 32, 8]).astype(np.float32)
    b = np.ndarray(shape=(4, 1, 32, 8), dtype=np.float32)
    simulate_kernel(nki_par_reduce_nd_b, a, b)

    self.assertTrue(np.allclose(b, np.sum(a, axis=1, keepdims=True)))