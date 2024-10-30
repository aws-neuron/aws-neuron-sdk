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
def nki_tensor_tensor_scan(a_tensor, b_tensor, c_tensor):
  ##################################################################
  # Example 1: scan two tiles, a and b, of the same
  # shape (128, 1024) using multiply/add and get
  # the scan result in tile c
  ##################################################################
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(1024)[None, :]
  a = nl.load(a_tensor[i_p, i_f])
  b = nl.load(b_tensor[i_p, i_f])

  i_f_tile0 = nl.arange(512)[None, :]

  c = nl.ndarray(shape=(128, 1024), dtype=np.float32)

  c[i_p, i_f_tile0] = nisa.tensor_tensor_scan(a[i_p, i_f_tile0], b[i_p, i_f_tile0], 0,
                                              np.multiply, np.add)

  i_f_tile1 = 512 + i_f_tile0
  c[i_p, i_f_tile1] = nisa.tensor_tensor_scan(a[i_p, i_f_tile1], b[i_p, i_f_tile1], c[i_p, 511],
                                              np.multiply, np.add)

  nl.store(c_tensor[i_p, i_f], c)


class TestNkiIsaExamplesTensorTensorScan(unittest.TestCase):
  def test_tensor_tensor_scan(self):
    a = np.random.random_sample([128, 1024]).astype(np.float32)
    b = np.random.random_sample([128, 1024]).astype(np.float32)
    c = np.ndarray(shape=(128, 1024), dtype=np.float32)
    simulate_kernel(nki_tensor_tensor_scan, a, b, c)

    golden = np.zeros(c.shape)
    golden[:, 0] = a[:, 0] * 0 + b[:, 0]
    for i in range(1, c.shape[1]):
        golden[:, i] = a[:, i] * golden[:, i-1] + b[:, i]

    self.assertTrue(np.allclose(c, golden))

