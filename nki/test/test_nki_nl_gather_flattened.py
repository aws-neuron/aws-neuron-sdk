"""
Copyright (C) 2025, Amazon.com. All Rights Reserved

"""
import unittest

import neuronxcc.nki as nki
# NKI_EXAMPLE_0_BEGIN
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
# NKI_EXAMPLE_0_END
import numpy as np


@nki.jit(mode="simulation")
def nki_gather_flattened():
    # NKI_EXAMPLE_0_BEGIN
    ##################################################################
    # Example 1: Gather values from a tensor using indices
    ##################################################################
    # Create source tensor
    N = 32
    M = 64
    data = nl.rand((N, M), dtype=nl.float32)

    # Create indices tensor - gather every 5th element
    indices = nl.zeros((N, 10), dtype=nl.uint32)
    for i in nl.static_range(N):
        for j in nl.static_range(10):
            indices[i, j] = j * 5

    # Gather values from data according to indices
    result = nl.gather_flattened(data=data, indices=indices)
    # NKI_EXAMPLE_0_END

    # Create output tensor and store result
    data_tensor = nl.ndarray([N, M], dtype=data.dtype, buffer=nl.shared_hbm)
    nl.store(data_tensor, value=data)
    indices_tensor = nl.ndarray([N, 10], dtype=nl.int32, buffer=nl.shared_hbm)
    nl.store(indices_tensor, value=indices)
    result_tensor = nl.ndarray([N, 10], dtype=data.dtype, buffer=nl.shared_hbm)
    nl.store(result_tensor, value=result)

    return data_tensor, indices_tensor, result_tensor


class TestNkiExamplesGather(unittest.TestCase):
    def test_gather_flattened(self):
        data, indices, result = nki_gather_flattened()

        self.assertEqual(result.shape, (32, 10))
        expected = np.take_along_axis(data, indices, axis=-1)
        self.assertTrue(np.allclose(result, expected))


TestNkiExamplesGather().test_gather_flattened()
