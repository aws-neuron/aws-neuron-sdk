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
def nki_tensor_scalar_cumulative_scalar(
  src_data,
  op0,
  op1,
  imm0,
  imm1=None,
  reduce_cmd=nisa.reduce_cmd.reset_reduce):
  # NKI_EXAMPLE_1_BEGIN
  ##################################################################
  # Example 1: Basic usage of tensor scalar cumulative.
  # Using scalar as immeidate values.
  ##################################################################
  # Create output tensor for result.
  result_tensor = nl.ndarray(src_data.shape, dtype=nl.float32, buffer=nl.hbm)

  # Load data into SBUF.
  src = nl.load(src_data[...])

  # Create destination tensor with zeros.
  dst = nl.ndarray(src_data.shape, dtype=nl.float32, buffer=nl.sbuf)

  # Apply cumulative operation on tensor with scalar operations.
  nisa.tensor_scalar_cumulative(
    src=src,
    dst=dst,
    op0=op0,
    op1=op1,
    imm0=imm0,
    imm1=imm1,
    reduce_cmd=reduce_cmd
  )

  # Store result to HBM
  nl.store(result_tensor, value=dst)
  # NKI_EXAMPLE_1_END

  return result_tensor

@nki.jit(mode="simulation")
def nki_tensor_scalar_cumulative_vector(
  src_data,
  op0,
  op1,
  imm0,
  imm1=None,
  reduce_cmd=nisa.reduce_cmd.reset_reduce):
  # NKI_EXAMPLE_2_BEGIN
  ##################################################################
  # Example 2: Basic usage of tensor scalar cumulative.
  # Using vector as immediate values.
  ##################################################################
  # Create output tensor for result.
  result_tensor = nl.ndarray(src_data.shape, dtype=nl.float32, buffer=nl.hbm)

  # Load data into SBUF.
  src = nl.load(src_data[...])
  imm0 = nl.load(imm0[...])
  imm1 = nl.load(imm1[...]) if imm1 else None

  # Create destination tensor with zeros.
  dst = nl.ndarray(src_data.shape, dtype=nl.float32, buffer=nl.sbuf)

  # Apply cumulative operation on tensor with scalar operations.
  nisa.tensor_scalar_cumulative(
    src=src,
    dst=dst,
    op0=op0,
    op1=op1,
    imm0=imm0,
    imm1=imm1,
    reduce_cmd=reduce_cmd
  )

  # Store result to HBM
  nl.store(result_tensor, value=dst)
  # NKI_EXAMPLE_2_END

  return result_tensor

@nki.jit(mode="simulation")
def nki_tensor_scalar_cumulative_chain(
  src_data,
  op0,
  op1,
  imm0,
  imm1=None,
  reduce_cmd=nisa.reduce_cmd.reset_reduce):
  # NKI_EXAMPLE_3_BEGIN
  ##################################################################
  # Example 3: Chain two tensor scalar cumulative together.
  # Using scalar as immeidate values.
  ##################################################################
  # Create output tensor for result.
  result_tensor = nl.ndarray(src_data.shape, dtype=nl.float32, buffer=nl.hbm)

  # Load data into SBUF.
  src = nl.load(src_data[...])

  # Create destination tensor with zeros.
  dst = nl.ndarray(src_data.shape, dtype=nl.float32, buffer=nl.sbuf)

  # Apply cumulative operation on tensor with scalar operations.
  nisa.tensor_scalar_cumulative(
    src=src,
    dst=dst,
    op0=op0,
    op1=op1,
    imm0=imm0,
    imm1=imm1,
    reduce_cmd=reduce_cmd
  )

  # Apply cumulative operation with reduce as reduce_cmd.
  nisa.tensor_scalar_cumulative(
    src=src,
    dst=dst,
    op0=op0,
    op1=op1,
    imm0=imm0,
    imm1=imm1,
    reduce_cmd=nisa.reduce_cmd.reduce
  )

  # Store result to HBM
  nl.store(result_tensor, value=dst)
  # NKI_EXAMPLE_3_END

  return result_tensor

@nki.jit(mode="simulation")
def nki_tensor_scan(src_data, op, initial):
  # NKI_EXAMPLE_4_BEGIN
  ##################################################################
  # Example 4: Perform tensor scan using tensor scalar cumulative.
  ##################################################################
  # Create output tensor for result.
  result_tensor = nl.ndarray(src_data.shape, dtype=nl.float32, buffer=nl.hbm)

  # Load data into SBUF.
  src = nl.load(src_data[...])

  # Create destination tensor with zeros.
  dst = nl.ndarray(src_data.shape, dtype=nl.float32, buffer=nl.sbuf)

  # Apply cumulative operation on tensor with scalar operations.
  nisa.tensor_scalar_cumulative(
    src=src,
    dst=dst,
    op0=nl.add,
    op1=op,
    imm0=np.float32(0.0),
    imm1=initial,
    reduce_cmd=nisa.reduce_cmd.load_reduce
  )

  # Store result to HBM
  nl.store(result_tensor, value=dst)
  # NKI_EXAMPLE_4_END

  return result_tensor

class TestNkiIsaExamplesTensorScalarCumulative(unittest.TestCase):
  
  def test_tensor_scalar_cumulative_scalar1(self):
    """Test when op1 is nl.add with scalar imm0.
    """
    src = np.ones((128, 64), dtype=np.float32)

    result = nki_tensor_scalar_cumulative_scalar(
      src, op0=nl.add, op1=nl.add, imm0=np.float32(0.0))

    self.assertEqual(result.shape, (128, 64))

    golden = np.add.accumulate(src, axis=-1)

    self.assertTrue(np.allclose(result, golden))

  def test_tensor_scalar_cumulative_scalar2(self):
    """Test when op1 is nl.multiply with scalar imm0.
    """
    src = np.ones((128, 64), dtype=np.float32)

    result = nki_tensor_scalar_cumulative_scalar(
      src, op0=nl.add, op1=nl.multiply, imm0=np.float32(0.0))

    self.assertEqual(result.shape, (128, 64))

    golden = np.multiply.accumulate(src, axis=-1)

    self.assertTrue(np.allclose(result, golden))
  
  def test_tensor_scalar_cumulative_vector1(self):
    """Test when op1 is nl.add with vector imm0.
    """
    src = np.ones((128, 64), dtype=np.float32)

    imm0 = np.ones((128, 1), dtype=np.float32)
    result = nki_tensor_scalar_cumulative_vector(
      src, op0=nl.add, op1=nl.add, imm0=imm0)

    self.assertEqual(result.shape, (128, 64))

    golden = np.add.accumulate(np.add(src, imm0), axis=-1)

    self.assertTrue(np.allclose(result, golden))
  
  def test_tensor_scalar_cumulative_vector2(self):
    """Test when op1 is nl.multiply with vector imm0.
    """
    src = np.ones((128, 64), dtype=np.float32)

    imm0 = np.ones((128, 1), dtype=np.float32)
    result = nki_tensor_scalar_cumulative_vector(
      src, op0=nl.add, op1=nl.multiply, imm0=imm0)

    self.assertEqual(result.shape, (128, 64))

    golden = np.multiply.accumulate(np.add(src, imm0), axis=-1)

    self.assertTrue(np.allclose(result, golden))
  
  def test_tensor_scalar_cumulative_vector3(self):
    """Test when op1 is nl.max with vector imm0.
    """
    src = np.ones((128, 64), dtype=np.float32)

    imm0 = np.ones((128, 1), dtype=np.float32)
    result = nki_tensor_scalar_cumulative_vector(
      src, op0=nl.add, op1=nl.max, imm0=imm0)

    self.assertEqual(result.shape, (128, 64))

    golden = np.maximum.accumulate(np.add(src, imm0), axis=-1)

    self.assertTrue(np.allclose(result, golden))
  
  def test_tensor_scalar_cumulative_load_reduce1(self):
    """Test when op1 is nl.add with load_reduce.
    """
    src = np.ones((128, 64), dtype=np.float32)

    imm0 = np.ones((128, 1), dtype=np.float32)
    imm1 = np.ones((128, 1), dtype=np.float32)
    result = nki_tensor_scalar_cumulative_vector(
      src,
      op0=nl.add,
      op1=nl.add,
      imm0=imm0,
      imm1=imm1,
      reduce_cmd=nisa.reduce_cmd.load_reduce
    )

    self.assertEqual(result.shape, (128, 64))

    golden = np.add(np.add.accumulate(np.add(src, imm0), axis=-1), imm1)

    self.assertTrue(np.allclose(result, golden))

  def test_tensor_scalar_cumulative_load_reduce2(self):
    """Test when op1 is nl.multiply with load_reduce.
    """
    src = np.ones((128, 64), dtype=np.float32)

    imm0 = np.ones((128, 1), dtype=np.float32)
    imm1 = np.zeros((128, 1), dtype=np.float32)
    result = nki_tensor_scalar_cumulative_vector(
      src,
      op0=nl.add,
      op1=nl.multiply,
      imm0=imm0,
      imm1=imm1,
      reduce_cmd=nisa.reduce_cmd.load_reduce
    )

    self.assertEqual(result.shape, (128, 64))

    golden = np.zeros((128, 64), dtype=np.float32)

    self.assertTrue(np.allclose(result, golden))
  
  def test_tensor_scalar_cumulative_load_reduce3(self):
    """Test when op1 is nl.min with load_reduce.
    """
    src = np.ones((128, 64), dtype=np.float32)

    imm0 = np.ones((128, 1), dtype=np.float32)
    imm1 = np.zeros((128, 1), dtype=np.float32)
    result = nki_tensor_scalar_cumulative_vector(
      src,
      op0=nl.add,
      op1=nl.min,
      imm0=imm0,
      imm1=imm1,
      reduce_cmd=nisa.reduce_cmd.load_reduce
    )

    self.assertEqual(result.shape, (128, 64))

    golden = np.zeros((128, 1), dtype=np.float32)

    self.assertTrue(np.allclose(result, golden))

  def test_tensor_scalar_cumulative_chain1(self):
    """Test chaining two operations of reset_reduce followed by reduce.
    """
    src = np.ones((128, 64), dtype=np.float32)

    result = nki_tensor_scalar_cumulative_chain(
      src,
      op0=nl.add,
      op1=nl.add,
      imm0=np.float32(0.0),
      reduce_cmd=nisa.reduce_cmd.reset_reduce
    )

    self.assertEqual(result.shape, (128, 64))

    golden = np.add(np.add.accumulate(src, axis=-1), 64)

    self.assertTrue(np.allclose(result, golden))
  
  def test_tensor_scalar_cumulative_chain2(self):
    """Test chaining two operations of load_reduce followed by reduce.
    """
    src = np.ones((128, 64), dtype=np.float32)

    result = nki_tensor_scalar_cumulative_chain(
      src,
      op0=nl.add,
      op1=nl.add,
      imm0=np.float32(0.0),
      imm1=np.float32(1.0),
      reduce_cmd=nisa.reduce_cmd.load_reduce
    )

    self.assertEqual(result.shape, (128, 64))

    golden = np.add(np.add.accumulate(src, axis=-1), 65)

    self.assertTrue(np.allclose(result, golden))
  
  def test_tensor_scan(self):
    """Test tensor scan.
    """
    src = np.ones((128, 64), dtype=np.float32)

    result = nki_tensor_scan(src, op=nl.add, initial=np.float32(2.0))

    self.assertEqual(result.shape, (128, 64))

    golden = np.add(np.add.accumulate(src, axis=-1), np.float32(2.0))

    self.assertTrue(np.allclose(result, golden))