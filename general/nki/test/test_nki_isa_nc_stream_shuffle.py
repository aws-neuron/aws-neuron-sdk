"""
Copyright (C) 2025, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
# NKI_EXAMPLE_0_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
# NKI_EXAMPLE_0_END
import numpy as np
nki_jit = nki.trace
simulate_kernel = nki.simulate_kernel


@nki_jit
def nki_stream_shuffle(in_tensor, out_tensor):
  # NKI_EXAMPLE_0_BEGIN
  #####################################################################
  # Example 1: 
  # Apply cross-partition data movement to a 32-partition tensor,
  # in-place shuffling the data in partition[i] to partition[(i+1)%32].
  #####################################################################
  a = nl.load(in_tensor)
  a_mgrid = nl.mgrid[0:32, 0:128]
  shuffle_mask = [(i - 1) % 32 for i in range(32)]
  nisa.nc_stream_shuffle(src=a[a_mgrid.p, a_mgrid.x], dst=a[a_mgrid.p, a_mgrid.x], shuffle_mask=shuffle_mask)
  
  nl.store(out_tensor, value=a)
  # NKI_EXAMPLE_0_END

      
class TestNkiIsaExamplesStreamShuffle(unittest.TestCase):
  def test_stream_shuffle(self):
    in_tensor = np.random.random_sample([32, 128]).astype(np.float32) * 100
    out_tensor = np.ndarray([32, 128], dtype=np.float32)

    simulate_kernel(nki_stream_shuffle, in_tensor, out_tensor)

    in_tensor[list(range(32))] = in_tensor[[(i - 1) % 32 for i in range(32)]]
    self.assertTrue(np.allclose(out_tensor, in_tensor))
