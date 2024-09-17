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
def nki_dropout(a_tensor, b_tensor, c_tensor):
  a = nl.load(a_tensor)
  b = nl.load(b_tensor)
  
  ...
  ###########################################################################
  # Example 1: From an input tile a of shape [128, 512], dropout its values
  # with probabilities in tile b of shape [128, 1] and store the result in c.
  ###########################################################################
  c = nisa.dropout(a[0:128, 0:512], prob=b[0:128, 0])

  nl.store(c_tensor, c)

@nki_jit
def nki_dropout_scalar(in_tensor, out_tensor):
  a = nl.load(in_tensor)
  
  ...
  ######################################################
  # Example 2: From an input tile a, dropout its values 
  # with probability of 0.2 and store the result in b.
  ######################################################
  b = nisa.dropout(a, prob=0.2)

  nl.store(out_tensor, b)


class TestNkiIsaExamplesDropout(unittest.TestCase):
  def test_dropout(self):
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.random.random_sample([128, 1]).astype(np.float32) 
    c = np.zeros([128, 512]).astype(np.float32)
    c_zeros = np.copy(c)
    
    simulate_kernel(nki_dropout, a, b, c)

    self.assertFalse(np.allclose(c, c_zeros))
    # self.assertFalse(np.allclose(c, a)) # we don't have dropout simulation implementation
    
  def test_dropout_scalar(self):
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.zeros([128, 512]).astype(np.float32)
    b_zeros = np.copy(b)
    
    simulate_kernel(nki_dropout_scalar, a, b)

    self.assertFalse(np.allclose(b, b_zeros))
    # self.assertFalse(np.allclose(b, a)) # we don't have dropout simulation implementation