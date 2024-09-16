"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki

...
nki_jit = nki.trace
simulate_kernel = nki.simulate_kernel

########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################

@nki_jit
def nki_local_gather(src_buffer, index, num_elem_per_idx, num_valid_indices, output):
  import neuronxcc.nki.isa as nisa
  import neuronxcc.nki.language as nl

  ##################################################################
  # Example 1: gather src_buffer using index
  # Gather input: src_buffer_tile with shape (128, 512)
  # Gather indices: index_tile with shape (128, 4)
  # We use num_valid_indices indices per core, and read num_elem_per_idx
  # contiguous elements per partition.
  ##################################################################

  src_buffer_tile = nl.load(src_buffer)
  index_tile = nl.load(index)
  output_tile = nisa.local_gather(src_buffer_tile, index_tile,
                                  num_elem_per_idx, num_valid_indices)

  nl.store(output, output_tile)


class TestNkiIsaExamplesLocalGather(unittest.TestCase):
  def test_local_gather(self):
    import numpy as np

    # Engine constants
    num_gpsimd_cores = 8
    num_partitions_per_core = 16

    # example gather input: src_buffer = np.array((128, 512))
    # example gather indices: index = np.array((16, 4))
    # (optional, default=0) gather valid index count per core: num_valid_indices
    # (optional, default=1) gather element count per index: num_elem_per_idx

    src_buffer = np.random.random_sample([128, 512, 4]).astype(np.float32)
    index_per_core = np.random.randint(low=0, high=512, size=(16, 4), dtype=np.uint16)
    # replicate 8 times for 8 GpSimd cores
    index = np.tile(index_per_core, (num_gpsimd_cores, 1))
    num_elem_per_idx = 4
    index_hw = index * num_elem_per_idx
    num_valid_indices = 64
    output_shape = (128, 4, 16, 4)
    output_nki = np.ndarray(shape=output_shape,
                            dtype=src_buffer.dtype)

    # Run NKI
    simulate_kernel(nki_local_gather, src_buffer, index_hw, num_elem_per_idx,
                    num_valid_indices, output_nki)

    # NumPy reference
    num_active_cores = index.shape[0] / num_partitions_per_core
    num_valid_indices = num_valid_indices if num_valid_indices \
        else index.size / num_active_cores

    output_np = np.ndarray(shape=(128, num_valid_indices, num_elem_per_idx),
                           dtype=src_buffer.dtype)

    for i_core in range(num_gpsimd_cores):
      start_par = i_core * num_partitions_per_core
      end_par = (i_core+1) * num_partitions_per_core
      indices_1d = index[start_par:end_par].flatten(order='F')[0: num_valid_indices]

      output_np[start_par:end_par, :, :] = np.take(
        src_buffer[start_par:end_par],
        indices_1d, axis=1)
    
    output_np = output_np.reshape(output_shape)
    self.assertTrue(np.allclose(output_nki, output_np))