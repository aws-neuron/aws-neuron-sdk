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
def nki_affine_select(a_tensor, b_tensor):
  ##################################################################
  # Example 1: Take tile a of shape [128, 128] and replace its
  # upper triangle with -9984.0;
  ##################################################################
  # return output in tile b
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(128)[None, :]
  a = nl.load(a_tensor[i_p, i_f])
  b = nisa.affine_select(pred=(i_f < i_p), on_true_tile=a[i_p, i_f], on_false_value=-9984.0)

  nl.store(b_tensor[i_p, i_f], b)

class TestNkiIsaExamplesAffineSelect(unittest.TestCase):
  def test_affine_select(self):
    a = np.random.random_sample([128, 128]).astype(np.float32)
    b = np.zeros([128, 128]).astype(np.float32)
    b_golden = np.copy(a)

    simulate_kernel(nki_affine_select, a, b)

    triui = np.triu_indices_from(b_golden) # upper triangle indicies
    b_golden[triui] = -9984.0

    self.assertTrue(np.allclose(b, b_golden))




 