"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np
...
nki_jit = nki.trace
simulate_kernel = nki.simulate_kernel

########################################################################
# NOTE: if you modify this file, make sure to update the source .py with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################

@nki_jit
def add_tensors(a_tensor, b_tensor, c_tensor):
  a = nl.load(a_tensor[0:128, 0:512])
  b = nl.load(b_tensor[0:128, 0:512])
  # add a and b element-wise and store in c[128, 512]
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)

@nki_jit
def add_tensor_scalar(a_tensor, c_tensor):
  a = nl.load(a_tensor[0:128, 0:512])
  b = 2.2
  # add constant b to each element in a
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)

@nki_jit
def add_broadcast_free_dim(a_tensor, b_tensor, c_tensor):
  a = nl.load(a_tensor[0:128, 0:512])
  b = nl.load(b_tensor[0:128, 0:1])
  # broadcast on free dimension -- [128, 1] is broadcasted to [128, 512]
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)

@nki_jit
def add_broadcast_par_dim(a_tensor, b_tensor, c_tensor):
  a = nl.load(a_tensor[0:128, 0:512])
  b = nl.load(b_tensor[0:1, 0:512])
  # broadcast on partition dimension -- [1, 512] is broadcasted to [128, 512]
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)

@nki_jit
def add_broadcast_both_dims(a_tensor, b_tensor, c_tensor):
  a = nl.load(a_tensor[0:128, 0:512])
  b = nl.load(b_tensor[0:1, 0:1])
  # broadcast on both dimensions -- [1, 1] is broadcasted to [128, 512]
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)

@nki_jit
def add_broadcast_each_dims(a_tensor, b_tensor, c_tensor):
  a = nl.load(a_tensor[0:128, 0:1])
  b = nl.load(b_tensor[0:1, 0:512])
  # broadcast on each dimensions -- [128, 1] and [1, 512] are broadcasted to [128, 512]
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)


class TestNkiNlExampleAdd(unittest.TestCase):
  def test_add(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.random.random_sample([128, 512]).astype(np.float32)
    c = np.zeros([128, 512]).astype(np.float32)
    c_golden = np.add(a, b)
    
    simulate_kernel(add_tensors, a, b, c)
    self.assertTrue(np.allclose(c, c_golden))

  def test_add_tensor_scalar(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = 2.2
    c = np.zeros([128, 512]).astype(np.float32)
    c_golden = np.add(a, b)

    simulate_kernel(add_tensor_scalar, a, c)
    self.assertTrue(np.allclose(c, c_golden))

  def test_add_broadcast_free_dim(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.random.random_sample([128, 1]).astype(np.float32)
    c = np.zeros([128, 512]).astype(np.float32)
    c_golden = np.add(a, b)

    simulate_kernel(add_broadcast_free_dim, a, b, c)
    self.assertTrue(np.allclose(c, c_golden))

  def test_add_broadcast_par_dim(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.random.random_sample([1, 512]).astype(np.float32)
    c = np.zeros([128, 512]).astype(np.float32)
    c_golden = np.add(a, b)

    simulate_kernel(add_broadcast_par_dim, a, b, c)
    self.assertTrue(np.allclose(c, c_golden))

  def test_add_broadcast_both_dims(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.random.random_sample([1, 1]).astype(np.float32)
    c = np.zeros([128, 512]).astype(np.float32)
    c_golden = np.add(a, b)

    simulate_kernel(add_broadcast_both_dims, a, b, c)
    self.assertTrue(np.allclose(c, c_golden))

  def test_add_broadcast_each_dims(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 1]).astype(np.float32)
    b = np.random.random_sample([1, 512]).astype(np.float32)
    c = np.zeros([128, 512]).astype(np.float32)
    c_golden = np.add(a, b)

    simulate_kernel(add_broadcast_each_dims, a, b, c)
    self.assertTrue(np.allclose(c, c_golden))