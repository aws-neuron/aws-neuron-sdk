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
def nki_dropout(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

  # NKI_EXAMPLE_BEGIN
  ###########################################################################
  # Example 1: From an input tile a of shape [128, 512], dropout its values
  # with probabilities in tile b of shape [128, 1] and store the result in c.
  ###########################################################################
  a: tensor[128, 512] = nl.load(a_tensor)
  b: tensor[128, 1] = nl.load(b_tensor)

  c: tensor[128, 512] = nisa.dropout(a, prob=b)

  nl.store(c_tensor, c)
  # NKI_EXAMPLE_END

  return c_tensor


@nki.jit(mode="simulation")
def nki_dropout_scalar(in_tensor):
  import neuronxcc.nki.language as nl
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)

  # NKI_EXAMPLE_BEGIN
  ######################################################
  # Example 2: From an input tile a, dropout its values 
  # with probability of 0.2 and store the result in b.
  ######################################################
  a = nl.load(in_tensor)

  b = nisa.dropout(a, prob=0.2)

  nl.store(out_tensor, b)
  # NKI_EXAMPLE_END

  return out_tensor


class TestNkiIsaExamplesDropout(unittest.TestCase):
  def test_dropout(self):
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.random.random_sample([128, 1]).astype(np.float32) 
    c = np.zeros([128, 512]).astype(np.float32)
    c_zeros = np.copy(c)
    
    c = nki_dropout(a, b)

    self.assertFalse(np.allclose(c, c_zeros))
    # self.assertFalse(np.allclose(c, a)) # we don't have dropout simulation implementation
    
  def test_dropout_scalar(self):
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.zeros([128, 512]).astype(np.float32)
    b_zeros = np.copy(b)
    
    b = nki_dropout_scalar(a)

    self.assertFalse(np.allclose(b, b_zeros))
    # self.assertFalse(np.allclose(b, a)) # we don't have dropout simulation implementation