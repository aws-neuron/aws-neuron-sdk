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
def nki_iota(a_tensor, b_tensor, c_tensor, d_tensor, e_tensor):
  ##################################################################
  # Example 1: Generate tile a of 512 constant values in SBUF partition 0
  # that start at 0 and increment by 1:
  ##################################################################
  # a = [0, 1, ..., 511]
  expr_a = nl.arange(0, 512)[None, :]
  a = nisa.iota(expr_a, dtype=np.int32)

  ##################################################################
  # Example 2: Generate tile b of 128 constant values across SBUF partitions
  # that start at 0 and increment by 1, with one value per partition:
  # b = [[0],
  #      [1],
  #      ...,
  #      [127]]
  ##################################################################
  expr_b = nl.arange(0, 128)[:, None]
  b = nisa.iota(expr_b, dtype=np.int32)
  
  ##################################################################
  # Example 3: Generate tile c of 512 constant values in SBUF partition 0
  # that start at 0 and decrement by 1:
  # c = [0, -1, ..., -511]
  ##################################################################
  expr_c = expr_a * -1
  c = nisa.iota(expr_c, dtype=np.int32)

  ##################################################################
  # Example 4: Generate tile d of 128 constant values across SBUF
  # partitions that start at 5 and increment by 2
  ##################################################################
  # d = [[5],
  #      [7],
  #      ...,
  #      [259]]
  expr_d = 5 + expr_b * 2
  d = nisa.iota(expr_d, dtype=np.int32)

  ##################################################################
  # Example 5: Generate tile e of shape [128, 512] by
  # broadcast-add expr_a and expr_b
  # e = [[0, 1, ..., 511],
  #      [1, 2, ..., 512],
  #      ...
  #      [127, 2, ..., 638]]
  ##################################################################
  e = nisa.iota(expr_a + expr_b, dtype=np.int32)

  nl.store(a_tensor[0, expr_a], a)
  nl.store(b_tensor[expr_b, 0], b)
  nl.store(c_tensor[0, expr_a], c)  
  nl.store(d_tensor[expr_b, 0], d)
  nl.store(e_tensor[expr_b, expr_a], e)
  
      
class TestNkiIsaExamplesIota(unittest.TestCase):
  def test_iota(self):
    a = np.random.random_sample([1, 512]).astype(np.float32)
    b = np.random.random_sample([128, 1]).astype(np.float32)
    c = np.random.random_sample([1, 512]).astype(np.float32)
    d = np.random.random_sample([128, 1]).astype(np.float32)
    e = np.random.random_sample([128, 512]).astype(np.float32)
    
    simulate_kernel(nki_iota, a, b, c, d, e)

    a_golden = np.expand_dims(np.arange(0, 512), 0)
    b_golden = np.expand_dims(np.arange(0, 128), 1)
    c_golden = np.expand_dims(np.arange(0, 512)*-1, 0)
    d_golden = np.expand_dims(np.arange(5, 260, 2), 1)
    e_golden = a_golden + b_golden

    self.assertTrue(np.allclose(a, a_golden))
    self.assertTrue(np.allclose(b, b_golden))
    self.assertTrue(np.allclose(c, c_golden))
    self.assertTrue(np.allclose(d, d_golden))
    self.assertTrue(np.allclose(e, e_golden))

 