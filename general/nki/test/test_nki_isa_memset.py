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
def nki_memset(a_tensor):
  ##################################################################
  # Example 1: Initialize a float32 tile a of shape (128, 128)
  # with a value of 0.2
  ##################################################################
  a = nisa.memset(shape=(128, 128), value=0.2, dtype=np.float32)

  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(128)[None, :]
  nl.store(a_tensor[i_p, i_f], a)
  
      
class TestNkiIsaExamplesMemset(unittest.TestCase):
  def test_memset(self):
    a = np.zeros([128, 128]).astype(np.float32)
    
    simulate_kernel(nki_memset, a)

    a_golden = np.full([128, 128], 0.2).astype(np.float32)
    self.assertTrue(np.allclose(a, a_golden))
