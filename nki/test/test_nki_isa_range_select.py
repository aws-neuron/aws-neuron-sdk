"""
Copyright (C) 2025, Amazon.com. All Rights Reserved
"""
import unittest
# NKI_EXAMPLE_0_BEGIN, NKI_EXAMPLE_1_BEGIN
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
...
# NKI_EXAMPLE_0_END, NKI_EXAMPLE_1_END

########################################################################
# NOTE: if you modify this file, make sure to update nki.isa .py file with
# NOTE: the correct line numbers under .. literalinclude:: directive
########################################################################

@nki.jit(mode="simulation", platform_target="trn2")
def nki_range_select_example(on_true, bound0, bound1, compare_op0, compare_op1, range_start):
    # Create output tensors
    select_res = nl.ndarray(on_true.shape, dtype=nl.float32, buffer=nl.hbm)
    reduce_result = nl.ndarray((on_true.shape[0], 1), dtype=nl.float32, buffer=nl.hbm)
    
    # NKI_EXAMPLE_0_BEGIN
    ##################################################################
    # Example 1: # Select elements where 
    # bound0 <= range_start + index < bound1 and compute max reduction
    # 
    # on_false_value must be nl.fp32.min
    ##################################################################
    on_true_tile = nl.load(on_true[...])
    bound0_tile = nl.load(bound0[...])
    bound1_tile = nl.load(bound1[...])

    reduce_res_tile = nl.ndarray((on_true.shape[0], 1), dtype=nl.float32, buffer=nl.sbuf)
    result = nl.ndarray(on_true.shape, dtype=nl.float32, buffer=nl.sbuf)
    
    result[...] = nisa.range_select(
        on_true_tile=on_true_tile,
        comp_op0=compare_op0,
        comp_op1=compare_op1,
        bound0=bound0_tile,
        bound1=bound1_tile,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
        reduce_res=reduce_res_tile,
        reduce_op=np.max,
        range_start=range_start,
        on_false_value=nl.fp32.min
    )

    nl.store(select_res[...], value=result[...])
    nl.store(reduce_result[...], value=reduce_res_tile[...])
    # NKI_EXAMPLE_0_END

    return result, reduce_result

@nki.jit(mode="simulation", platform_target="trn2")
def nki_range_select_chaining(on_true, bound0, bound1, compare_op0, compare_op1, range_start):
    # Create output tensors
    select_res = nl.ndarray(on_true.shape, dtype=np.float32, buffer=nl.hbm)
    reduce_result = nl.ndarray((on_true.shape[0], 1), dtype=np.float32, buffer=nl.hbm)
    
    # NKI_EXAMPLE_1_BEGIN
    ##################################################################
    # Example 2.a: Initialize reduction with first range_select
    # Notice we don't pass reduce_res since the accumulation
    # register keeps track of the accumulation until we're ready to 
    # read it. Also we use reset_reduce in order to "clobber" or zero
    # out the accumulation register before we start accumulating.
    #
    # Note: Since the type of these tensors are fp32, we use nl.fp32.min
    # for on_false_value due to HW constraints.
    ##################################################################
    on_true_tile = nl.load(on_true[...])
    bound0_tile = nl.load(bound0[...])
    bound1_tile = nl.load(bound1[...])

    reduce_res_sbuf = nl.ndarray((on_true.shape[0], 1), dtype=np.float32, buffer=nl.sbuf)
    result_sbuf = nl.ndarray(on_true.shape, dtype=np.float32, buffer=nl.sbuf)
    
    result_sbuf[...] = nisa.range_select(
        on_true_tile=on_true_tile,
        comp_op0=compare_op0,
        comp_op1=compare_op1,
        bound0=bound0_tile,
        bound1=bound1_tile,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
        reduce_op=np.max,
        range_start=range_start,
        on_false_value=nl.fp32.min
    )

    ##################################################################
    # Example 2.b: Chain multiple range_select operations 
    # with reduction in an affine loop. Adding ones just lets us ensure the reduction 
    # gets updated with new values.
    ##################################################################
    ones = nl.full(on_true.shape, fill_value=1, dtype=np.float32, buffer=nl.sbuf)
    # we are going to loop as if we're tiling on the partition dimension    
    iteration_step_size = on_true_tile.shape[0]
    
    # Perform chained operations using an affine loop index for range_start
    for i in range(1, 2):
        # Update input values
        on_true_tile[...] = nl.add(on_true_tile, ones)
        
        # Continue reduction with updated values
        # notice, we still don't have reduce_res specified
        result_sbuf[...] = nisa.range_select(
            on_true_tile=on_true_tile,
            comp_op0=compare_op0,
            comp_op1=compare_op1,
            bound0=bound0_tile,
            bound1=bound1_tile,
            reduce_cmd=nisa.reduce_cmd.reduce,
            reduce_op=np.max,
            # we can also use index expressions for setting the start of the range
            range_start=range_start + (i * iteration_step_size),
            on_false_value=nl.fp32.min
        )

    range_start = range_start + (2 * iteration_step_size)
    ##################################################################
    # Example 2.c: Final iteration, we actually want the results to 
    # return to the user so we pass reduce_res argument so the 
    # reduction  will be written from the accumulation 
    # register to reduce_res_tile
    ##################################################################
    on_true_tile[...] = nl.add(on_true_tile, ones)
    result_sbuf[...] = nisa.range_select(
        on_true_tile=on_true_tile,
        comp_op0=compare_op0,
        comp_op1=compare_op1,
        bound0=bound0_tile,
        bound1=bound1_tile,
        reduce_cmd=nisa.reduce_cmd.reduce,
        reduce_res=reduce_res_sbuf[...],
        reduce_op=np.max,
        range_start=range_start,
        on_false_value=nl.fp32.min
    )

    nl.store(select_res[...], value=result_sbuf[...])
    nl.store(reduce_result[...], value=reduce_res_sbuf[...])
    # NKI_EXAMPLE_1_END

    return select_res, reduce_result

class TestNkiIsaExamplesRangeSelect(unittest.TestCase):
    def test_range_select_example(self):
        on_true_data = np.random.random_sample((128, 512)).astype(np.float32)
        bound0 = np.zeros([128, 1], dtype=np.float32)
        bound1 = np.full([128, 1], 64, dtype=np.float32)
        range_start = 32
        result, reduction = nki_range_select_example(on_true_data, bound0, bound1, 
                                                     np.greater_equal, np.less, range_start)

        # The results should match the numpy equivalent from the docstring:
        # indices = np.zeros(on_true_data.shape)
        # indices[:] = range_start + np.arange(on_true_data[0].size)
        # mask = comp_op0(indices, bound0) & comp_op1(indices, bound1)
        # result = np.where(mask, on_true_data, on_false_value)
        # reduction = reduce_op(result, axis=1, keepdims=True)
        indices = np.zeros_like(on_true_data)
        indices[:] = range_start + np.arange(on_true_data.shape[1])
      
        mask = np.greater_equal(indices, bound0) & np.less(indices, bound1)
        golden = np.where(mask, on_true_data, nl.fp32.min)

        golden_reduce = np.max(golden, axis=1, keepdims=True)
  
        self.assertTrue(np.allclose(result, golden))
        self.assertTrue(np.allclose(reduction, golden_reduce))

    def test_range_select_chaining(self):
        on_true_data = np.random.random_sample((128, 512)).astype(np.float32)
        range_start = 32
        bound0 = np.zeros([128, 1], dtype=np.float32)
        bound1 = np.full([128, 1], 350, dtype=np.float32)
        
        result, reduction = nki_range_select_chaining(
            on_true_data, bound0, bound1,
            np.greater_equal, np.less, range_start
        )

        # Calculate golden reference
        indices = np.zeros_like(on_true_data)
               
        # Apply the same operations as in the kernel
        golden = on_true_data.copy()
        golden_max = np.zeros((on_true_data.shape[0], 1), dtype=on_true_data.dtype)
        selected = golden_max.copy()

        iteration_step_size = on_true_data.shape[0]
        for i in range(3):  # 3 iterations
            indices[:] = range_start + (i * iteration_step_size) + np.arange(on_true_data.shape[1])
            mask = np.greater_equal(indices, bound0) & np.less(indices, bound1)

            selected = np.where(mask, golden, nl.fp32.min)
            golden_max = np.maximum(golden_max, np.max(selected, axis=1, keepdims=True))
            golden = golden + 1

        self.assertTrue(np.allclose(result, selected))
        self.assertTrue(np.allclose(reduction, golden_max))
