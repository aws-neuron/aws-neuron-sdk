"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
...

nki_jit = nki.trace
simulate_kernel = nki.simulate_kernel

########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################

@nki_jit
def reciprocal_kernel(in_tensor, out_tensor):
  x = nl.load(in_tensor[nl.mgrid[0:128, 0:512]])
  
  y = nisa.reciprocal(x)

  nl.store(out_tensor[nl.mgrid[0:128, 0:512]], value=y)


class TestNkiExampleNisaReciprocal(unittest.TestCase):
  def test_nisa_reciprocal(self):
    np.random.seed(0)
    src = np.random.random_sample([128, 512]).astype(np.float32)
    dst = np.ndarray(shape=(128, 512), dtype=np.float32)
    dst_golden = np.reciprocal(src)

    simulate_kernel(reciprocal_kernel, src, dst)
    self.assertTrue(np.allclose(dst, dst_golden))
