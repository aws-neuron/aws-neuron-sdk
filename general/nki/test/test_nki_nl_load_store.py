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
  # load from in_tensor[P, F] that is on HBM
  # copy into data_tile[P, F] that is on SBUF
  data_tile = nl.load(in_tensor)
  ...
  ...
  # store into out_tensor[P, F] that is on HBM
  # from data_tile[P, F] that is on SBUF
  nl.store(out_tensor, data_tile)



@nki_jit
def example_load_store_b(in_tensor, out_tensor):
  for i_b in nl.affine_range(4): 
    data_tile = nl.zeros((128, 512), dtype=in_tensor.dtype) 
    # load from in_tensor[4, 128, 512] one batch at a time
    # copy into data_tile[128, 512]
    i_p, i_f = nl.mgrid[0:128, 0:512]
    data_tile[i_p, i_f] = nl.load(in_tensor[i_b, i_p, i_f])
    ...
    # store into out_tensor[4, 128, 512] one batch at a time
    # from data_tile[128, 512] 
    i_p, i_f = nl.mgrid[0:128, 0:512]
    nl.store(out_tensor[i_b, i_p, i_f], value=data_tile[i_p, i_f]) 


class TestNkiExampleNlLoad(unittest.TestCase):
  def test_nl_load(self):
    src = np.random.random_sample([128, 512]).astype(np.float32)
    dst = np.ndarray(shape=(128, 512), dtype=np.float32)

    simulate_kernel(example_kernel, src, dst)
    self.assertTrue(np.allclose(src, dst))

  def test_load_store_3d(self):
    in_tensor = np.random.random_sample([4, 128, 512]).astype(np.float32)
    out_tensor = np.random.random_sample([4, 128, 512]).astype(np.float32)

    nki.simulate_kernel(example_load_store_b, in_tensor, out_tensor)
    self.assertTrue(np.allclose(out_tensor, in_tensor))
