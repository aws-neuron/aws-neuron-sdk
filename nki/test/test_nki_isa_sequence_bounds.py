"""
Copyright (C) 2025, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
# NKI_EXAMPLE_0_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
# NKI_EXAMPLE_0_END
import numpy as np


@nki.jit(mode="simulation")
def nki_sequence_bounds(segment_ids):
  output = nl.ndarray([1, 2, 32], dtype=segment_ids.dtype, buffer=nl.shared_hbm)
  # NKI_EXAMPLE_0_BEGIN
  ######################################################################
  # Example 1: Generate tile of boundaries of sequence for each element:
  ######################################################################
  # Input example
  # segment_ids = np.array([[0, 1, 1, 2, 2, 2, 0, 3, 3]], dtype=np.int32)

  # Expected output for this example:
  # [[
  #   [9, 1, 1, 3, 3, 3, 9, 7, 7]       # start index
  #   [-1, 3, 3, 6, 6, 6, -1, 9, 9]     # end index
  #   ]]
  m, n = segment_ids.shape

  ix, iy, iz = nl.mgrid[0:m, 0:2, 0:n]

  out_tile = nl.ndarray([m, 2, n], dtype=segment_ids.dtype, buffer=nl.sbuf)
  seq_tile = nl.load(segment_ids)
  out_tile[ix, iy, iz] = nisa.sequence_bounds(segment_ids=seq_tile)
  # NKI_EXAMPLE_0_END
  nl.store(output, value=out_tile)
  return output



class TestNkiIsaExamplesSequenceBounds(unittest.TestCase):
  def test_sequence_bounds(self):
    m, n = 1, 32
    n_seq = m * n
    length = n

    np.random.seed(0)
    segment_ids = np.sort(np.random.randint(low=0, high=n_seq, size=length))
    segment_ids = segment_ids.reshape((m, n), order='F').astype(np.float32)
    reshaped_segment_ids = segment_ids.reshape(segment_ids.shape[0], -1)

    # NKI_EXAMPLE_1_BEGIN
    def compute_sequence_bounds(sequence):
      n = len(sequence)

      min_bounds = np.zeros(n, dtype=sequence.dtype)
      max_bounds = np.zeros(n, dtype=sequence.dtype)

      min_bound_pad = n
      max_bound_pad = -1

      min_bounds[0] = 0 if sequence[0] != 0 else min_bound_pad
      for i in range(1, n):
        if sequence[i] == 0:
          min_bounds[i] = min_bound_pad
        elif sequence[i] == sequence[i - 1]:
          min_bounds[i] = min_bounds[i - 1]
        else:
          min_bounds[i] = i

      max_bounds[-1] = n if sequence[-1] != 0 else max_bound_pad
      for i in range(n - 2, -1, -1):
        if sequence[i] == 0:
          max_bounds[i] = max_bound_pad
        elif sequence[i] == sequence[i + 1]:
          max_bounds[i] = max_bounds[i + 1]
        else:
          max_bounds[i] = i + 1

      return np.vstack((min_bounds, max_bounds))

    b = (
      np.apply_along_axis(
        compute_sequence_bounds, axis=1, arr=reshaped_segment_ids
      )
      .reshape(m, 2, n)
      .astype(np.float32)
    )
    # NKI_EXAMPLE_1_END

    a = nki_sequence_bounds(segment_ids=segment_ids)
    self.assertTrue(np.allclose(a, b))
