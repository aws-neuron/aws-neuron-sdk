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
def nki_tensor_scalar(a_tensor, b_tensor, c_tensor, d_tensor,
                      e_tensor, f_tensor, g_tensor):
  ##################################################################
  # Example 1: subtract 1.0 from all elements of tile a of
  # shape (128, 512) and get the output tile in b
  ##################################################################
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(512)[None, :]
  a = nl.load(a_tensor[i_p, i_f])
  b = nisa.tensor_scalar(a[i_p, i_f], np.subtract, 1.0)

  nl.store(b_tensor[i_p, i_f], b)

  ##################################################################
  # Example 2: broadcast 1.0 into a shape of (128, 512) and subtract
  # it with tile c to get output tile d
  ##################################################################
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(512)[None, :]
  c = nl.load(c_tensor[i_p, i_f])
  d = nisa.tensor_scalar(c[i_p, i_f], np.subtract, 1.0, reverse0=True)

  nl.store(d_tensor[i_p, i_f], d)

  ##################################################################
  # Example 3: broadcast multiply tile e with vector f and
  # then broadcast add with scalar 2.5;
  # tile e has a shape of (64, 1024) and vector f has a shape of (64, 1)
  ##################################################################
  i_p_ef = nl.arange(64)[:, None]
  i_f_e = nl.arange(1024)[None, :]
  i_f_f = nl.arange(1)[None, :]
  e = nl.load(e_tensor[i_p_ef, i_f_e])
  f = nl.load(f_tensor[i_p_ef, i_f_f]) 
  g = nisa.tensor_scalar(e[i_p_ef, i_f_e], op0=np.multiply, operand0=f[i_p_ef, i_f_f], op1=np.add, operand1=2.5)  

  nl.store(g_tensor[i_p_ef, i_f_e], g)
  
      
class TestNkiIsaExamplesTensorScalar(unittest.TestCase):
  def test_tensor_scalar(self):
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.ndarray(shape=(128, 512), dtype=np.float32)

    c = np.random.random_sample([128, 512]).astype(np.float32)
    d = np.ndarray(shape=(128, 512), dtype=np.float32)

    e = np.random.random_sample([64, 1024]).astype(np.float32)
    f = np.random.random_sample([64, 1]).astype(np.float32)
    g = np.ndarray(shape=(64, 1024), dtype=np.float32) 
    
    simulate_kernel(nki_tensor_scalar, a, b, c, d, e, f, g)
    
    self.assertTrue(np.allclose(b, a-1))
    self.assertTrue(np.allclose(d, 1-c))
    self.assertTrue(np.allclose(g, e*f + 2.5))
 