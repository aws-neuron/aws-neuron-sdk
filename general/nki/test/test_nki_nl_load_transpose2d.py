"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
...

nki_jit = nki.trace
simulate_kernel = nki.simulate_kernel

########################################################################
# NOTE: if you modify this file, make sure to update the source .py with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################

@nki_jit
def example_kernel(in_tensor, out_tensor):
  # load from in_tensor[F, P] that is on HBM
  # transpose and copy into local_tile[P, F] that is on SBUF
  local_tile = nl.load_transpose2d(in_tensor)
  ...
  nl.store(out_tensor, value=local_tile)


class TestNkiExampleNlLoadTranspose2d(unittest.TestCase):
  def test_dma_transpose_load(self):
    np.random.seed(0)
    src = np.random.random_sample([2048, 128]).astype(np.float32)
    dst = np.ndarray(shape=[128, 2048], dtype=np.float32)

    simulate_kernel(example_kernel, src, dst)

    dst_golden = np.transpose(src)
    self.assertTrue(np.allclose(dst, dst_golden))
