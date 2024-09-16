"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
...
nki_jit = nki.trace
simulate_kernel = nki.simulate_kernel

########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################

@nki_jit
def nki_reduce(a_tensor, b_tensor):
  ##################################################################
  # Example 1: reduce add tile a of shape (128, 512)
  # in the free dimension and return
  # reduction result in tile b of shape (128, 1)
  ##################################################################
  i_p_a = nl.arange(128)[:, None]
  i_f_a = nl.arange(512)[None, :]
  
  a = nl.load(a_tensor[i_p_a, i_f_a])  

  b = nisa.tensor_reduce(np.add, a[i_p_a, i_f_a], axis=[1])

  i_p_b, i_f_b = nl.mgrid[0:128, 0:1]
  nl.store(b_tensor[i_p_b, i_f_b], b)

      
class TestNkiIsaExamplesReduce(unittest.TestCase):
  def test_reduce(self):
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.ndarray(shape=(128, 1), dtype=np.float32)
    simulate_kernel(nki_reduce, a, b)

    self.assertTrue(np.allclose(b, np.sum(a, axis=1, keepdims=True)))
 