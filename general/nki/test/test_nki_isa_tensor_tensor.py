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
def nki_tensor_tensor(a_tensor, b_tensor, c_tensor):
  ##################################################################
  # Example 1: add two tiles, a and b, of the same
  # shape (128, 512) element-wise and get
  # the addition result in tile c
  ##################################################################
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(512)[None, :]
  a = nl.load(a_tensor[i_p, i_f])  
  b = nl.load(b_tensor[i_p, i_f])

  c = nisa.tensor_tensor(a[i_p, i_f], b[i_p, i_f], np.add)

  nl.store(c_tensor[i_p, i_f], c)
  
      
class TestNkiIsaExamplesTensorTensor(unittest.TestCase):
  def test_tensor_tensor(self):
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.random.random_sample([128, 512]).astype(np.float32)
    c = np.ndarray(shape=(128, 512), dtype=np.float32)
    simulate_kernel(nki_tensor_tensor, a, b, c)
    
    self.assertTrue(np.allclose(c, np.add(a, b)))
 