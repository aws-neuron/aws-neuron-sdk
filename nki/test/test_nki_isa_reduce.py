"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
# NKI_EXAMPLE_2_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
...
# NKI_EXAMPLE_2_END

########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################


@nki.jit(mode="simulation")
def nki_reduce(a_tensor):
  b_tensor = nl.ndarray([a_tensor.shape[0], 1], dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  # NKI_EXAMPLE_2_BEGIN
  ##################################################################
  # Example 1: reduce add tile a of shape (128, 512)
  # in the free dimension and return
  # reduction result in tile b of shape (128, 1)
  ##################################################################
  i_p_a = nl.arange(128)[:, None]
  i_f_a = nl.arange(512)[None, :]
  # NKI_EXAMPLE_2_END
  
  a = nl.load(a_tensor[i_p_a, i_f_a])  

  # NKI_EXAMPLE_2_BEGIN
  b = nisa.tensor_reduce(np.add, a[i_p_a, i_f_a], axis=[1])
  # NKI_EXAMPLE_2_END

  i_p_b, i_f_b = nl.mgrid[0:128, 0:1]
  nl.store(b_tensor[i_p_b, i_f_b], b)
  return b_tensor

      
class TestNkiIsaExamplesReduce(unittest.TestCase):
  def test_reduce(self):
    a = np.random.random_sample([128, 512]).astype(np.float32) * 100
    b = np.ndarray(shape=(128, 1), dtype=np.float32)
    b = nki_reduce(a)

    self.assertTrue(np.allclose(b, np.sum(a, axis=1, keepdims=True)))
 