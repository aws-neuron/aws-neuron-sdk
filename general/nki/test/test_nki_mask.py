"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
# NKI_EXAMPLE_15_BEGIN
import neuronxcc.nki.language as nl
# NKI_EXAMPLE_15_END
import numpy as np
...

########################################################################
# NOTE: if you modify this file, make sure to update nki.api.shared.rst with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################


@nki.jit(mode="simulation")
def nki_mask(in_tensor):
  ...
  out_tensor = nl.ndarray([64, 256], dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # NKI_EXAMPLE_15_BEGIN
  ...
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(512)[None, :]
  # NKI_EXAMPLE_15_END
  in_tile = nl.load(in_tensor[i_p, i_f])
  # NKI_EXAMPLE_15_BEGIN
  out_tile = nl.square(in_tile, mask=((i_p<64) & (i_f<256)))
  # NKI_EXAMPLE_15_END

  nl.store(out_tensor[i_p, i_f], out_tile[i_p, i_f],
           mask=((i_p < 64) & (i_f < 256)))
  return out_tensor


class TestNkiIsaExamplesMask(unittest.TestCase):
  def test_mask(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)

    b = nki_mask(a)

    b_golden = np.square(a[:64, :256])

    self.assertTrue(np.allclose(b, b_golden))
