"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import numpy as np

import neuronxcc.nki as nki
# NKI_EXAMPLE_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
# NKI_EXAMPLE_END


@nki.jit(mode="simulation")
def nki_bn_stats_bn_aggr_1(a_tensor):
  mean_a_tensor = nl.ndarray([a_tensor.shape[0], 1], dtype=a_tensor.dtype, buffer=nl.shared_hbm)
  var_a_tensor = nl.ndarray([a_tensor.shape[0], 1], dtype=a_tensor.dtype, buffer=nl.shared_hbm)

  # NKI_EXAMPLE_BEGIN
  ##################################################################
  # Example 1: Calculate the mean and variance for each partition
  # of tile a with shape (128, 128)
  ##################################################################
  a: tensor[128, 128] = nl.load(a_tensor)
  stats_a: tensor[128, 6] = nisa.bn_stats(a)
  mean_var_a: tensor[128, 2] = nisa.bn_aggr(stats_a)

  # Extract mean and variance
  mean_a = mean_var_a[:, 0]
  var_a = mean_var_a[:, 1]
  nl.store(mean_a_tensor, mean_a)
  nl.store(var_a_tensor, var_a)
  # NKI_EXAMPLE_END

  return mean_a_tensor, var_a_tensor


@nki.jit(mode="simulation")
def nki_bn_stats_bn_aggr_2(b_tensor):
  mean_b_tensor = nl.ndarray([b_tensor.shape[0], 1], dtype=b_tensor.dtype, buffer=nl.shared_hbm)
  var_b_tensor = nl.ndarray([b_tensor.shape[0], 1], dtype=b_tensor.dtype, buffer=nl.shared_hbm)

  # NKI_EXAMPLE_BEGIN
  # ##################################################################
  # # Example 2: Calculate the mean and variance for each partition of
  # # tile b with shape [128, 1024]
  # ##################################################################
  b: tensor[128, 1024] = nl.load(b_tensor)

  # Run bn_stats in two tiles because b has 1024 elements per partition,
  # but bn_stats has a limitation of nl.tile_size.bn_stats_fmax
  # Initialize a bn_stats output tile with shape of [128, 6*2] to
  # hold outputs of two bn_stats instructions
  stats_b = nl.ndarray((128, 6 * 2), dtype=nl.float32)
  bn_tile = nl.tile_size.bn_stats_fmax
  ix, iy = nl.mgrid[0:128, 0:bn_tile]
  iz, iw = nl.mgrid[0:128, 0:6]

  for i in range(1024 // bn_tile):
    stats_b[iz, i * 6 + iw] = nisa.bn_stats(b[ix, i * bn_tile + iy], dtype=nl.float32)

  mean_var_b = nisa.bn_aggr(stats_b)

  # Extract mean and variance
  mean_b = mean_var_b[:, 0]
  var_b = mean_var_b[:, 1]

  nl.store(mean_b_tensor, mean_b)
  nl.store(var_b_tensor, var_b)
  # NKI_EXAMPLE_END

  return mean_b_tensor, var_b_tensor


class TestNkiIsaExamplesBnStatsBnAggr(unittest.TestCase):
  def test_bn_stats_bn_aggr(self):
    a = np.random.random_sample([128, 128]).astype(np.float32) * 100
    b = np.random.random_sample([128, 1024]).astype(np.float32) * 100

    a_mean, a_var = nki_bn_stats_bn_aggr_1(a)
    b_mean, b_var = nki_bn_stats_bn_aggr_2(b)

    a_mean_golden = np.mean(a, axis=1, keepdims=True)
    b_mean_golden = np.mean(b, axis=1, keepdims=True)
    a_var_golden = np.var(a, axis=1, keepdims=True)
    b_var_golden = np.var(b, axis=1, keepdims=True)

    self.assertTrue(np.allclose(a_mean, a_mean_golden))
    self.assertTrue(np.allclose(a_var, a_var_golden))
    self.assertTrue(np.allclose(b_mean, b_mean_golden))
    self.assertTrue(np.allclose(b_var, b_var_golden))
