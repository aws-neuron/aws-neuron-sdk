"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
# NKI_EXAMPLE_BEGIN
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
# NKI_EXAMPLE_END
import numpy as np


@nki.jit(mode="simulation")
def nki_activation(a_tensor, b_tensor, c_tensor):
  a_act_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
  b_act_tensor = nl.ndarray(b_tensor.shape, dtype=b_tensor.dtype, buffer=nl.shared_hbm)

  # NKI_EXAMPLE_BEGIN
  ##################################################################
  # Example 1: perform exponential function on matrix a of shape (128, 1024)
  ##################################################################
  a = nl.load(a_tensor)
  activated_a = nisa.activation(op=nl.exp, data=a)
  nl.store(a_act_tensor, activated_a)

  ##################################################################
  # Example 2: perform the following operations to matrix b of shape (128, 512)
  # using a single activation instruction: np.square(b * 2.0) + c
  # 1) compute `np.square(b * 2.0 + c)`
  # 2) cast 1) results into bfloat16
  ##################################################################
  b = nl.load(b_tensor)
  c = nl.load(c_tensor)
  activated_b = nisa.activation(op=np.square, data=b, bias=c, scale=2.0,
                                dtype=nl.bfloat16)
  nl.store(b_act_tensor, activated_b)
  # NKI_EXAMPLE_END

  return a_act_tensor, b_act_tensor

  
class TestNkiIsaExamplesActivation(unittest.TestCase):
  def test_activation(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 1024]).astype(np.float32)
    b = np.random.random_sample([128, 512]).astype(np.float32)
    c = np.random.random_sample([128, 1]).astype(np.float32)

    a_act, b_act = nki_activation(a, b, c)

    a_act_golden = np.exp(a)
    b_act_golden = np.square(b*2+c)

    self.assertTrue(np.allclose(a_act, a_act_golden))
    self.assertTrue(np.allclose(b_act, b_act_golden))
