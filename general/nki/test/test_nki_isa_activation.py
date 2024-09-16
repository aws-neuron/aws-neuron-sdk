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
def nki_activation(a_tensor, a_act_tensor, b_tensor, c_tensor, b_act_tensor):

  ##################################################################
  # Example 1: perform exponential function on matrix a of shape (128, 1024)
  ##################################################################
  i_p_a = nl.arange(128)[:, None]
  i_f_a = nl.arange(1024)[None, :]
  a = nl.load(a_tensor[i_p_a, i_f_a])

  activated_a = nisa.activation(op=nl.exp, data=a[i_p_a, i_f_a])

  nl.store(a_act_tensor[i_p_a, i_f_a], activated_a)

  ##################################################################
  # Example 2: perform the following operations to matrix b of shape (128, 512)
  # using a single activation instruction:
  # 1) multiply all elements in b by 2.0,
  # 2) add a user-input vector c to the 1) result,
  # 3) apply a square function on the 2) result
  # 4) cast 3) results into bfloat16
  ##################################################################

  i_p_b = i_p_c = nl.arange(128)[:, None]
  i_f_b = nl.arange(512)[None, :]
  i_f_c = nl.arange(1)[None, :]
  b = nl.load(b_tensor[i_p_b, i_f_b])
  c = nl.load(c_tensor[i_p_c, i_f_c])
  activated_b = nisa.activation(op=np.square, data=b[i_p_b, i_f_b],
                                bias=c[i_p_c, i_f_c], scale=2.0, dtype=nl.bfloat16)

  nl.store(b_act_tensor[i_p_b, i_f_b], activated_b)

  
class TestNkiIsaExamplesActivation(unittest.TestCase):
  def test_activation(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 1024]).astype(np.float32)
    a_act = np.ndarray(shape=[128, 1024], dtype=np.float32)
    b = np.random.random_sample([128, 512]).astype(np.float32)
    c = np.random.random_sample([128, 1]).astype(np.float32)
    b_act = np.ndarray(shape=[128, 512], dtype=np.float32)

    simulate_kernel(nki_activation, a, a_act, b, c, b_act)

    a_act_golden = np.exp(a)
    b_act_golden = np.square(b*2+c)

    self.assertTrue(np.allclose(a_act, a_act_golden))
    self.assertTrue(np.allclose(b_act, b_act_golden))
