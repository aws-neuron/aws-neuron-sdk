"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
# NKI_EXAMPLE_5_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
...
# NKI_EXAMPLE_5_END

########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################


@nki.jit(mode="simulation")
def nki_tensor_scalar(a_tensor, c_tensor, e_tensor, f_tensor):
  b_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  d_tensor = nl.ndarray(c_tensor.shape, dtype=c_tensor.dtype,
                        buffer=nl.shared_hbm)
  g_tensor = nl.ndarray(e_tensor.shape, dtype=e_tensor.dtype,
                        buffer=nl.shared_hbm)
  # NKI_EXAMPLE_5_BEGIN
  ##################################################################
  # Example 1: subtract 1.0 from all elements of tile a of
  # shape (128, 512) and get the output tile in b
  ##################################################################
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(512)[None, :]
  # NKI_EXAMPLE_5_END
  a = nl.load(a_tensor[i_p, i_f])
  # NKI_EXAMPLE_5_BEGIN
  b = nisa.tensor_scalar(a[i_p, i_f], np.subtract, 1.0)

  # NKI_EXAMPLE_5_END
  nl.store(b_tensor[i_p, i_f], b)

  # NKI_EXAMPLE_5_BEGIN
  ##################################################################
  # Example 2: broadcast 1.0 into a shape of (128, 512) and subtract
  # it with tile c to get output tile d
  ##################################################################
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(512)[None, :]
  # NKI_EXAMPLE_5_END
  c = nl.load(c_tensor[i_p, i_f])
  # NKI_EXAMPLE_5_BEGIN
  d = nisa.tensor_scalar(c[i_p, i_f], np.subtract, 1.0, reverse0=True)

  # NKI_EXAMPLE_5_END
  nl.store(d_tensor[i_p, i_f], d)

  # NKI_EXAMPLE_5_BEGIN
  ##################################################################
  # Example 3: broadcast multiply tile e with vector f and
  # then broadcast add with scalar 2.5;
  # tile e has a shape of (64, 1024) and vector f has a shape of (64, 1)
  ##################################################################
  i_p_ef = nl.arange(64)[:, None]
  i_f_e = nl.arange(1024)[None, :]
  i_f_f = nl.arange(1)[None, :]
  # NKI_EXAMPLE_5_END
  e = nl.load(e_tensor[i_p_ef, i_f_e])
  f = nl.load(f_tensor[i_p_ef, i_f_f]) 
  # NKI_EXAMPLE_5_BEGIN
  g = nisa.tensor_scalar(e[i_p_ef, i_f_e], op0=np.multiply, operand0=f[i_p_ef, i_f_f], op1=np.add, operand1=2.5)  
  # NKI_EXAMPLE_5_END

  nl.store(g_tensor[i_p_ef, i_f_e], g)
  return b_tensor, d_tensor, g_tensor
  
      
class TestNkiIsaExamplesTensorScalar(unittest.TestCase):
  def test_tensor_scalar(self):
    a = np.random.random_sample([128, 512]).astype(np.float32) * 100

    c = np.random.random_sample([128, 512]).astype(np.float32) * 100

    e = np.random.random_sample([64, 1024]).astype(np.float32) * 100
    f = np.random.random_sample([64, 1]).astype(np.float32) * 100
    
    b, d, g = nki_tensor_scalar(a, c, e, f)
    
    self.assertTrue(np.allclose(b, a-1))
    self.assertTrue(np.allclose(d, 1-c))
    self.assertTrue(np.allclose(g, e*f + 2.5))
 