"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
# NKI_EXAMPLE_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
# NKI_EXAMPLE_END
import numpy as np


@nki.jit(mode="simulation")
def nki_iota():

  # NKI_EXAMPLE_BEGIN
  ##################################################################
  # Example 1: Generate tile a of 512 constant values in SBUF partition 0
  # that start at 0 and increment by 1:
  ##################################################################
  # a = [0, 1, ..., 511]
  expr_a = nl.arange(0, 512)[None, :]
  a: tensor[1, 512] = nisa.iota(expr_a, dtype=nl.int32)

  ##################################################################
  # Example 2: Generate tile b of 128 constant values across SBUF partitions
  # that start at 0 and increment by 1, with one value per partition:
  # b = [[0],
  #      [1],
  #      ...,
  #      [127]]
  ##################################################################
  expr_b = nl.arange(0, 128)[:, None]
  b: tensor[128, 1] = nisa.iota(expr_b, dtype=nl.int32)
  
  ##################################################################
  # Example 3: Generate tile c of 512 constant values in SBUF partition 0
  # that start at 0 and decrement by 1:
  # c = [0, -1, ..., -511]
  ##################################################################
  expr_c = expr_a * -1
  c: tensor[1, 512] = nisa.iota(expr_c, dtype=nl.int32)

  ##################################################################
  # Example 4: Generate tile d of 128 constant values across SBUF
  # partitions that start at 5 and increment by 2
  ##################################################################
  # d = [[5],
  #      [7],
  #      ...,
  #      [259]]
  expr_d = 5 + expr_b * 2
  d: tensor[128, 1] = nisa.iota(expr_d, dtype=nl.int32)

  ##################################################################
  # Example 5: Generate tile e of shape [128, 512] by
  # broadcast-add expr_a and expr_b
  # e = [[0, 1, ..., 511],
  #      [1, 2, ..., 512],
  #      ...
  #      [127, 2, ..., 638]]
  ##################################################################
  e: tensor[128, 512] = nisa.iota(expr_a + expr_b, dtype=nl.int32)
  # NKI_EXAMPLE_END

  a_tensor = nl.ndarray([1, 512], dtype=nl.float32, buffer=nl.shared_hbm)
  b_tensor = nl.ndarray([128, 1], dtype=nl.float32, buffer=nl.shared_hbm)
  c_tensor = nl.ndarray([1, 512], dtype=nl.float32, buffer=nl.shared_hbm)
  d_tensor = nl.ndarray([128, 1], dtype=nl.float32, buffer=nl.shared_hbm)
  e_tensor = nl.ndarray([128, 512], dtype=nl.float32, buffer=nl.shared_hbm)
  nl.store(a_tensor[0, expr_a], a)
  nl.store(b_tensor[expr_b, 0], b)
  nl.store(c_tensor[0, expr_a], c)  
  nl.store(d_tensor[expr_b, 0], d)
  nl.store(e_tensor[expr_b, expr_a], e)
  return a_tensor, b_tensor, c_tensor, d_tensor, e_tensor
  
      
class TestNkiIsaExamplesIota(unittest.TestCase):
  def test_iota(self):
    a, b, c, d, e = nki_iota()

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

 