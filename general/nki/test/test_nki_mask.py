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
# NOTE: if you modify this file, make sure to update nki.api.shared.rst with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################

@nki_jit
def nki_mask(in_tensor, out_tensor):
  ...
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(512)[None, :]
  in_tile = nl.load(in_tensor[i_p, i_f])
  out_tile = nl.square(in_tile, mask=((i_p<64) & (i_f<256)))

  nl.store(out_tensor[i_p, i_f], out_tile[i_p, i_f])


class TestNkiIsaExamplesMask(unittest.TestCase):
  def test_mask(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b_golden = b = a

    simulate_kernel(nki_mask, a, b)

    b_golden[:64, :256] = np.square(a[:64, :256])

    self.assertTrue(np.allclose(b, b_golden))
