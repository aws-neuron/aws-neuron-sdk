"""
Copyright (C) 2025, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
# NKI_EXAMPLE_1_BEGIN
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
# NKI_EXAMPLE_1_END
import numpy as np


@nki.jit(mode="simulation")
def nki_select_reduce_basic(predicate_data, on_true_data):
  # NKI_EXAMPLE_1_BEGIN
  ##################################################################
  # Example 1: Basic usage of select_reduce
  # Create source data, predicate, and destination tensors
  ##################################################################
  # Create output tensor for result
  result_tensor = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.hbm)
  
  # Load input data to SBUF
  predicate = nl.load(predicate_data[...])
  on_true = nl.load(on_true_data[...])
  
  # Create destination tensor
  dst = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.sbuf)
  
  # Perform select operation - copy from on_true where predicate is true
  # and set to fp32.min where predicate is false
  nisa.select_reduce(
      dst=dst,
      predicate=predicate,
      on_true=on_true,
      on_false=nl.fp32.min,
  )
  
  # Store result to HBM
  nl.store(result_tensor, value=dst)
  # NKI_EXAMPLE_1_END

  return result_tensor


@nki.jit(mode="simulation")
def nki_select_reduce_with_reduction(predicate_data, on_true_data, on_false_data):
  # NKI_EXAMPLE_2_BEGIN
  ##################################################################
  # Example 2: Using select_reduce with reduction
  # Perform selection and compute max reduction per partition
  ##################################################################
  # Create output tensors for results
  result_tensor = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.hbm)
  reduce_tensor = nl.ndarray((on_true_data.shape[0], 1), dtype=nl.float32, buffer=nl.hbm)
  
  # Load input data to SBUF
  predicate = nl.load(predicate_data)
  on_true = nl.load(on_true_data)
  on_false = nl.load(on_false_data)

  # Create destination tensor
  dst = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.sbuf)
  
  # Create tensor for reduction results
  reduce_res = nl.ndarray((on_true_data.shape[0], 1), dtype=nl.float32, buffer=nl.sbuf)
  
  # Perform select operation with reduction
  nisa.select_reduce(
      dst=dst,
      predicate=predicate,
      on_true=on_true,
      on_false=on_false,
      reduce_cmd=nisa.reduce_cmd.reset_reduce,
      reduce_res=reduce_res,
      reduce_op=nl.max
  )
  
  # Store results to HBM
  nl.store(result_tensor, value=dst)
  nl.store(reduce_tensor, value=reduce_res)
  # NKI_EXAMPLE_2_END

  return result_tensor, reduce_tensor


@nki.jit(mode="simulation")
def nki_select_reduce_reverse_pred(predicate_data, on_true_data):
  # NKI_EXAMPLE_3_BEGIN
  ##################################################################
  # Example 3: Using select_reduce with reverse_pred option
  # Reverse the meaning of the predicate
  ##################################################################
  # Create output tensor for result
  result_tensor = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.hbm)
  
  # Load input data to SBUF
  predicate = nl.load(predicate_data[...])
  on_true = nl.load(on_true_data[...])
  
  # Create destination tensor
  dst = nl.ndarray(on_true_data.shape, dtype=nl.float32, buffer=nl.sbuf)
  
  # Perform select operation with reverse_pred=True
  # This will select on_true where predicate is FALSE
  nisa.select_reduce(
      dst=dst,
      predicate=predicate,
      on_true=on_true,
      on_false=nl.fp32.min,
      reverse_pred=True  # Reverse the meaning of the predicate
  )
  
  # Store result to HBM
  nl.store(result_tensor, value=dst)
  # NKI_EXAMPLE_3_END

  return result_tensor


class TestNkiIsaExamplesSelectReduce(unittest.TestCase):
  def test_select_reduce_basic(self):
    # Create input data
    on_true_data = np.ones((128, 64), dtype=np.float32)
    predicate_data = np.zeros((128, 64), dtype=np.uint8)
    predicate_data[0:64, :] = 1  # Set first half to 1 (true)
    
    # Run the test
    result = nki_select_reduce_basic(predicate_data, on_true_data)

    self.assertEqual(result.shape, (128, 64))
    
    # First half should be 1.0 (from on_true)
    self.assertTrue(np.all(result[0:64, :] == 1.0))
    
    # Second half should be fp32.min (from on_false)
    self.assertTrue(np.all(result[64:, :] == nl.fp32.min))

  def test_select_reduce_with_reduction(self):
    np.random.seed(0)
    on_true_data = np.random.random_sample([128, 512]).astype(np.float32) * 100
    on_false_data = np.random.random_sample([128, 1]).astype(np.float32) * 100
    predicate_data = np.random.randint(low=0, high=2, size=[128, 512], dtype=np.bool_)

    result, reduction = nki_select_reduce_with_reduction(predicate_data, on_true_data, on_false_data)

    self.assertEqual(result.shape, (128, 512))
    self.assertEqual(reduction.shape, (128, 1))

    golden_result = np.where(predicate_data, on_true_data, on_false_data)
    golden_reduce = np.max(golden_result, axis=1, keepdims=True)

    self.assertTrue(np.allclose(result, golden_result))
    self.assertTrue(np.allclose(reduction, golden_reduce))

  def test_select_reduce_reverse_pred(self):
    # Create input data
    on_true_data = np.ones((128, 64), dtype=np.float32)
    predicate_data = np.zeros((128, 64), dtype=np.uint8)
    predicate_data[0:64, :] = 1  # Set first half to 1 (true)
    
    # Run the test
    result = nki_select_reduce_reverse_pred(predicate_data, on_true_data)

    self.assertEqual(result.shape, (128, 64))
    
    # First half should be fp32.min (predicate is 1, but reversed)
    self.assertTrue(np.all(result[0:64, :] == nl.fp32.min))
    
    # Second half should be 1.0 (predicate is 0, but reversed)
    self.assertTrue(np.all(result[64:, :] == 1.0))
