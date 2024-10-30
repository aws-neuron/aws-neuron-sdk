"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
...
nki_jit = nki.trace # or use torch_neuronx.nki_jit
simulate_kernel = nki.simulate_kernel

########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################

@nki_jit
def nki_bn_stats_bn_aggr_1(a_tensor, mean_a_tensor, var_a_tensor):
  ##################################################################
  # Example 1: Calculate the mean and variance for each partition
  # of tile a with shape (128, 128)
  ##################################################################
  i_p_a = nl.arange(128)[:, None]
  i_f_a = nl.arange(128)[None, :]
  a = nl.load(a_tensor[i_p_a, i_f_a])

  stats_a = nisa.bn_stats(a[i_p_a, i_f_a])
  assert stats_a.shape == (128, 6)

  mean_var_a = nisa.bn_aggr(stats_a)
  assert mean_var_a.shape == (128, 2)

  # Extract mean and variance
  mean_a = mean_var_a[:, 0]
  var_a = mean_var_a[:, 1]

  nl.store(mean_a_tensor[i_p_a, 0], mean_a)
  nl.store(var_a_tensor[i_p_a, 0], var_a)

@nki_jit
def nki_bn_stats_bn_aggr_2(b_tensor, mean_b_tensor, var_b_tensor):
  # ##################################################################
  # # Example 2: Calculate the mean and variance for each partition of
  # # tile b with shape [128, 1024]
  # ##################################################################
  i_p_b = nl.arange(128)[:, None]
  i_f_b = nl.arange(nl.tile_size.bn_stats_fmax)[None, :]
  i_f_b2 = nl.arange(1024)[None, :]
  b = nl.load(b_tensor[i_p_b, i_f_b2])

  # Run bn_stats in two tiles because b has 1024 elements per partition,
  # but bn_stats has a limitation of nl.tile_size.bn_stats_fmax
  # Initialize a bn_stats output tile with shape of [128, 6*2] to
  # hold outputs of two bn_stats instructions
  stats_b = nl.ndarray((128, 6*2), dtype=np.float32)

  i_p_stats_b = nl.arange(128)[:, None]
  i_f_stats_b = nl.arange(6)[None, :]
  stats_b[i_p_stats_b, i_f_stats_b] = nisa.bn_stats(b[i_p_b, i_f_b], dtype=np.float32)
  stats_b[i_p_stats_b, 6+i_f_stats_b] = nisa.bn_stats(b[i_p_b, nl.tile_size.bn_stats_fmax+i_f_b], dtype=np.float32)

  mean_var_b = nisa.bn_aggr(stats_b)

  # Extract mean and variance
  mean_b = mean_var_b[:, 0]
  var_b = mean_var_b[:, 1]

  nl.store(mean_b_tensor[i_p_b, 0], mean_b)
  nl.store(var_b_tensor[i_p_b, 0], var_b)

class TestNkiIsaExamplesBnStatsBnAggr(unittest.TestCase):
  def test_bn_stats_bn_aggr(self):
    a = np.random.random_sample([128, 128]).astype(np.float32)
    b = np.random.random_sample([128, 1024]).astype(np.float32)
    a_mean = np.zeros([128, 1], dtype=np.float32)
    b_mean = np.zeros([128, 1], dtype=np.float32)
    a_var = np.zeros([128, 1], dtype=np.float32)
    b_var = np.zeros([128, 1], dtype=np.float32)

    simulate_kernel(nki_bn_stats_bn_aggr_1, a, a_mean, a_var)
    simulate_kernel(nki_bn_stats_bn_aggr_2, b, b_mean, b_var)

    a_mean_golden = np.mean(a, axis=1, keepdims=True)
    b_mean_golden = np.mean(b, axis=1, keepdims=True)
    a_var_golden = np.var(a, axis=1, keepdims=True)
    b_var_golden = np.var(b, axis=1, keepdims=True)

    self.assertTrue(np.allclose(a_mean, a_mean_golden))
    self.assertTrue(np.allclose(a_var, a_var_golden))
    self.assertTrue(np.allclose(b_mean, b_mean_golden))
    self.assertTrue(np.allclose(b_var, b_var_golden))
  