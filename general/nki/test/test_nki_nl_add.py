"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
# NKI_EXAMPLE_20_BEGIN
import neuronxcc.nki.language as nl
# NKI_EXAMPLE_20_END
import numpy as np

########################################################################
# NOTE: if you modify this file, make sure to update the source .py with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################


@nki.jit(mode="simulation")
def add_tensors(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  # NKI_EXAMPLE_20_BEGIN
  a = nl.load(a_tensor[0:128, 0:512])
  b = nl.load(b_tensor[0:128, 0:512])
  # add a and b element-wise and store in c[128, 512]
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  # NKI_EXAMPLE_20_END
  return c_tensor


@nki.jit(mode="simulation")
def add_tensor_scalar(a_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  # NKI_EXAMPLE_20_BEGIN
  a = nl.load(a_tensor[0:128, 0:512])
  b = 2.2
  # add constant b to each element in a
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  # NKI_EXAMPLE_20_END
  return c_tensor


@nki.jit(mode="simulation")
def add_broadcast_free_dim(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  # NKI_EXAMPLE_20_BEGIN
  a = nl.load(a_tensor[0:128, 0:512])
  b = nl.load(b_tensor[0:128, 0:1])
  # broadcast on free dimension -- [128, 1] is broadcasted to [128, 512]
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  # NKI_EXAMPLE_20_END
  return c_tensor


@nki.jit(mode="simulation")
def add_broadcast_par_dim(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  # NKI_EXAMPLE_20_BEGIN
  a = nl.load(a_tensor[0:128, 0:512])
  b = nl.load(b_tensor[0:1, 0:512])
  # broadcast on partition dimension -- [1, 512] is broadcasted to [128, 512]
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  # NKI_EXAMPLE_20_END
  return c_tensor


@nki.jit(mode="simulation")
def add_broadcast_both_dims(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  # NKI_EXAMPLE_20_BEGIN
  a = nl.load(a_tensor[0:128, 0:512])
  b = nl.load(b_tensor[0:1, 0:1])
  # broadcast on both dimensions -- [1, 1] is broadcasted to [128, 512]
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  # NKI_EXAMPLE_20_END
  return c_tensor


@nki.jit(mode="simulation")
def add_broadcast_each_dims(a_tensor, b_tensor):
  c_tensor = nl.ndarray([128, 512], dtype=a_tensor.dtype,
                        buffer=nl.shared_hbm)
  # NKI_EXAMPLE_20_BEGIN
  a = nl.load(a_tensor[0:128, 0:1])
  b = nl.load(b_tensor[0:1, 0:512])
  # broadcast on each dimensions -- [128, 1] and [1, 512] are broadcasted to [128, 512]
  c = nl.add(a, b)
  nl.store(c_tensor[0:128, 0:512], c)
  # NKI_EXAMPLE_20_END
  return c_tensor


class TestNkiNlExampleAdd(unittest.TestCase):
  def test_add(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.random.random_sample([128, 512]).astype(np.float32)
    c = np.zeros([128, 512]).astype(np.float32)
    c_golden = np.add(a, b)
    
    c = add_tensors(a, b)
    self.assertTrue(np.allclose(c, c_golden))

  def test_add_tensor_scalar(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = 2.2
    c = np.zeros([128, 512]).astype(np.float32)
    c_golden = np.add(a, b)

    c = add_tensor_scalar(a)
    self.assertTrue(np.allclose(c, c_golden))

  def test_add_broadcast_free_dim(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.random.random_sample([128, 1]).astype(np.float32)
    c = np.zeros([128, 512]).astype(np.float32)
    c_golden = np.add(a, b)

    c = add_broadcast_free_dim(a, b)
    self.assertTrue(np.allclose(c, c_golden))

  def test_add_broadcast_par_dim(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.random.random_sample([1, 512]).astype(np.float32)
    c = np.zeros([128, 512]).astype(np.float32)
    c_golden = np.add(a, b)

    c = add_broadcast_par_dim(a, b)
    self.assertTrue(np.allclose(c, c_golden))

  def test_add_broadcast_both_dims(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 512]).astype(np.float32)
    b = np.random.random_sample([1, 1]).astype(np.float32)
    c = np.zeros([128, 512]).astype(np.float32)
    c_golden = np.add(a, b)

    c = add_broadcast_both_dims(a, b)
    self.assertTrue(np.allclose(c, c_golden))

  def test_add_broadcast_each_dims(self):
    np.random.seed(0)
    a = np.random.random_sample([128, 1]).astype(np.float32)
    b = np.random.random_sample([1, 512]).astype(np.float32)
    c = np.zeros([128, 512]).astype(np.float32)
    c_golden = np.add(a, b)

    c = add_broadcast_each_dims(a, b)
    self.assertTrue(np.allclose(c, c_golden))