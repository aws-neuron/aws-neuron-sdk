"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
# NKI_EXAMPLE_7_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
...
# NKI_EXAMPLE_7_END
import numpy as np

########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################


@nki.jit(mode="simulation")
def nki_memset():
  a_tensor = nl.ndarray([128, 128], dtype=nl.float32, buffer=nl.shared_hbm)
  # NKI_EXAMPLE_7_BEGIN
  ##################################################################
  # Example 1: Initialize a float32 tile a of shape (128, 128)
  # with a value of 0.2
  ##################################################################
  a = nisa.memset(shape=(128, 128), value=0.2, dtype=nl.float32)
  # NKI_EXAMPLE_7_END

  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(128)[None, :]
  nl.store(a_tensor[i_p, i_f], a)
  return a_tensor
  
      
class TestNkiIsaExamplesMemset(unittest.TestCase):
  def test_memset(self):
    a = nki_memset()

    a_golden = np.full([128, 128], 0.2).astype(np.float32)
    self.assertTrue(np.allclose(a, a_golden))
