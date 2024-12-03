"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np
import neuronxcc.nki as nki
# NKI_EXAMPLE_18_BEGIN
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
...
# NKI_EXAMPLE_18_END

########################################################################
# NOTE: if you modify this file, make sure to update the source .py with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################


@nki.jit(mode="simulation")
def atomic_rmw_indirect_indices(in_tensor, indices_tensor, value_tensor):
  rmw_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # Workaround to get simulation working for testing purposes.
  # reason: the IR builder marks in_out tensor as output only, hence the simulator ignores the input values of the in_out tensor.
  # workaround: load input value from another input tensor, write that value to our in_out tensor, so we can test atomic_rmw in simulation.
  in_tile = nl.load(in_tensor)
  nl.store(rmw_tensor, in_tile)

  N = 128
  M = 512

  # NKI_EXAMPLE_18_BEGIN
  value: tensor[N, M] = nl.load(value_tensor)

  # dynamic indices have to be in SBUF, with shape [N, 1]
  indices_tile: tensor[N, 1] = nl.load(indices_tensor)

  ix = nl.arange(M)[None, :]

  ########################################################################
  # Atomic read-modify-write example:
  #   - read: values of rmw_tensor is indexed by values from indices_tile
  #   - modify: incremented by value
  #   - write: saved back into rmw_tensor
  # resulting in rmw_tensor = rmw_tensor + value
  ########################################################################
  nl.atomic_rmw(rmw_tensor[indices_tile, ix], value=value, op=np.add)
  # NKI_EXAMPLE_18_END
  return rmw_tensor


class TestNkiExampleNlLoad(unittest.TestCase):
  def test_atomic_rmw_indirect_indices(self):
    in_tensor = np.random.random_sample([128, 512]).astype(np.float32)
    indices_tensor = np.arange(128, dtype=np.int32)
    indices_tensor = np.expand_dims(indices_tensor, axis=1)
    value_tensor = np.random.random_sample([128, 512]).astype(np.float32)
    golden = in_tensor + value_tensor

    rmw_tensor = atomic_rmw_indirect_indices(in_tensor, indices_tensor,
                                             value_tensor)

    self.assertTrue(np.allclose(rmw_tensor, golden))
