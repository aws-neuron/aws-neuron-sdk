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
simulate_kernel = nki.simulate_kernel


@nki.jit(mode="simulation")
def nki_nc_stream_shuffle(in_tensor):
  # NKI_EXAMPLE_0_BEGIN
  #####################################################################
  # Example 1: 
  # Apply cross-partition data movement to a 32-partition tensor,
  # in-place shuffling the data in partition[i] to partition[(i+1)%32].
  #####################################################################
  # NKI_EXAMPLE_0_END
  ...
  out_tensor = nl.ndarray(shape=(32, 128), dtype=np.float32, buffer=nl.shared_hbm)
  # NKI_EXAMPLE_0_BEGIN
  ...
  a: tensor[32, 128] = nl.load(in_tensor)
  a_mgrid = nl.mgrid[0:32, 0:128]
  shuffle_mask = [(i - 1) % 32 for i in range(32)]
  nisa.nc_stream_shuffle(src=a[a_mgrid.p, a_mgrid.x], dst=a[a_mgrid.p, a_mgrid.x], shuffle_mask=shuffle_mask)
  
  nl.store(out_tensor, value=a)
  # NKI_EXAMPLE_0_END
  return out_tensor

@nki.jit(mode="simulation")
def nki_nc_stream_shuffle_broadcast_partition(in_tensor):
  # NKI_EXAMPLE_1_BEGIN
  #####################################################################
  # Example 2: 
  # Broadcast data in 1 partition to 32 partitions.
  #####################################################################
  # NKI_EXAMPLE_1_END
  ...
  out_tensor = nl.ndarray(shape=(32, 128), dtype=np.float32, buffer=nl.shared_hbm)
  # NKI_EXAMPLE_1_BEGIN
  ...
  a: tensor[1, 128] = nl.load(in_tensor)
  b = nl.ndarray(shape=(32, 128), dtype=np.float32)
  dst_mgrid = nl.mgrid[0:32, 0:128]
  src_mgrid = nl.mgrid[0:1, 0:128]
  shuffle_mask = [0] * 32
  nisa.nc_stream_shuffle(src=a[0, src_mgrid.x], dst=b[dst_mgrid.p, dst_mgrid.x], shuffle_mask=shuffle_mask)
  
  nl.store(out_tensor, value=b)
  # NKI_EXAMPLE_1_END
  return out_tensor

@nki.jit(mode="simulation")
def nki_nc_stream_shuffle_broadcast_mask(in_tensor):
  # NKI_EXAMPLE_2_BEGIN
  #####################################################################
  # Example 3: 
  # In the case where src and dst access more than one quadrant (32 
  # partitions), the shuffle is applied to each quadrant independently, 
  # and the same shuffle_mask is used for each quadrant.
  #####################################################################
  # NKI_EXAMPLE_2_END
  ...
  out_tensor = nl.ndarray(shape=(128, 128), dtype=np.float32, buffer=nl.shared_hbm)
  # NKI_EXAMPLE_2_BEGIN
  ...
  a: tensor[128, 128] = nl.load(in_tensor)
  b = nl.ndarray(shape=(128, 128), dtype=np.float32)
  mgrid = nl.mgrid[0:128, 0:128]
  shuffle_mask = [(i - 1) % 32 for i in range(32)]
  nisa.nc_stream_shuffle(src=a[mgrid.p, mgrid.x], dst=b[mgrid.p, mgrid.x], shuffle_mask=shuffle_mask)
  
  nl.store(out_tensor, value=b)
  # NKI_EXAMPLE_2_END
  return out_tensor

      
class TestNkiIsaExamplesStreamShuffle(unittest.TestCase):
  def test_stream_shuffle(self):
    in_tensor = np.random.random_sample([32, 128]).astype(np.float32) * 100
    out_tensor = simulate_kernel(nki_nc_stream_shuffle, in_tensor)
    in_tensor[list(range(32))] = in_tensor[[(i - 1) % 32 for i in range(32)]]
    self.assertTrue(np.allclose(out_tensor, in_tensor))

  def test_broadcast_partition(self):
    in_tensor = np.random.random_sample([1, 128]).astype(np.float32) * 100
    out_tensor = simulate_kernel(nki_nc_stream_shuffle_broadcast_partition, in_tensor)
    out_tensor[0:32] = in_tensor[0]
    self.assertTrue(np.allclose(out_tensor, in_tensor))

  def test_broadcast_mask(self):
    in_tensor = np.random.random_sample([128, 128]).astype(np.float32) * 100
    out_tensor = simulate_kernel(nki_nc_stream_shuffle_broadcast_mask, in_tensor)
    for j in range(4):
      in_tensor[list(range(j * 32, (j + 1) * 32))] = in_tensor[[(i - 1) % 32 + j * 32 for i in range(32)]]
    self.assertTrue(np.allclose(out_tensor, in_tensor))
