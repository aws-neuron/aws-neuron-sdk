"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np
# NKI_EXAMPLE_6_BEGIN
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
...
# NKI_EXAMPLE_6_END

########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################


@nki.jit(mode="simulation")
def reciprocal_kernel(in_tensor):
  out_tensor = nl.ndarray([128, 512], dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # NKI_EXAMPLE_6_BEGIN
  x = nl.load(in_tensor[nl.mgrid[0:128, 0:512]])
  
  y = nisa.reciprocal(x)

  # NKI_EXAMPLE_6_END
  nl.store(out_tensor[nl.mgrid[0:128, 0:512]], value=y)
  return out_tensor


class TestNkiExampleNisaReciprocal(unittest.TestCase):
  def test_nisa_reciprocal(self):
    np.random.seed(0)
    src = np.random.random_sample([128, 512]).astype(np.float32) * 100
    dst_golden = np.reciprocal(src)

    dst = reciprocal_kernel(src)
    self.assertTrue(np.allclose(dst, dst_golden))
