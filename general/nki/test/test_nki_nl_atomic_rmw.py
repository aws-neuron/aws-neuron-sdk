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
def atomic_rmw_indirect_indices(in_tensor, rmw_tensor, indices_tensor, value_tensor):
  i_p = nl.arange(128)[:, None]
  i_f = nl.arange(512)[None, :]

  # Workaround to get simulation working for testing purposes.
  # reason: the IR builder marks in_out tensor as output only, hence the simulator ignores the input values of the in_out tensor.
  # workaround: load input value from another input tensor, write that value to our in_out tensor, so we can test atomic_rmw in simulation.
  in_tile = nl.load(in_tensor[i_p, i_f])
  nl.store(rmw_tensor[i_p, i_f], in_tile)

  # indices have to be in SBUF
  indices_tile = nl.load(indices_tensor[nl.mgrid[0:128, 0:1]])

  value = nl.load(value_tensor[i_p, i_f])

  ########################################################################
  # Atomic read-modify-write example:
  #   - read: values of rmw_tensor is indexed by values from indices_tile
  #   - modify: incremented by value
  #   - write: saved back into rmw_tensor
  # resulting in rmw_tensor = rmw_tensor + value
  ########################################################################
  nl.atomic_rmw(rmw_tensor[indices_tile[i_p, 0], i_f], value=value, op=np.add)


class TestNkiExampleNlLoad(unittest.TestCase):
  def test_atomic_rmw_indirect_indices(self):
    in_tensor = np.random.random_sample([128, 512]).astype(np.float32)
    rmw_tensor = np.random.random_sample([128, 512]).astype(np.float32)
    indices_tensor = np.arange(128, dtype=np.int32)
    indices_tensor = np.expand_dims(indices_tensor, axis=1)
    value_tensor = np.random.random_sample([128, 512]).astype(np.float32)
    golden = in_tensor + value_tensor

    nki.simulate_kernel(atomic_rmw_indirect_indices, in_tensor, rmw_tensor, indices_tensor, value_tensor)

    self.assertTrue(np.allclose(rmw_tensor, golden))
