"""Auto-generated stub file"""
from enum import Enum
import nki.language as nl
import ml_dtypes

def jit(func=None, mode="auto", **kwargs):
    r"""
    This decorator compiles a top-level NKI function to run on NeuronDevices.

    This decorator tries to automatically detect the current framework and compile
    the function as a custom operator. To bypass the framework detection logic, you
    can specify the ``mode`` parameter explicitly.

    You might need to explicitly set the target platform using the
    ``NEURON_PLATFORM_TARGET_OVERRIDE`` environment variable. Supported values are
    "trn1"/"gen2", "trn2"/"gen3", and "trn3"/"gen4".

    :param func: Function that defines the custom operation.
    :param mode: Compilation mode. Supported values are "jax", "torchxla",
                 and "auto". (Default: "auto".)

    .. code-block:: python
       :caption: Writing an addition kernel using ``@nki.jit``

        @nki.jit()
        def nki_tensor_add_kernel(a_input, b_input):
            # Check both input tensor shapes are the same for element-wise operation.
            assert a_input.shape == b_input.shape

            # Check the first dimension's size to ensure it does not exceed on-chip
            # memory tile size, since this simple kernel does not tile inputs.
            assert a_input.shape[0] <= nl.tile_size.pmax

            # Allocate space for the input tensors in SBUF and copy the inputs from HBM
            # to SBUF with DMA copy.
            a_tile = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.sbuf)
            nisa.dma_copy(dst=a_tile, src=a_input)

            b_tile = nl.ndarray(dtype=b_input.dtype, shape=b_input.shape, buffer=nl.sbuf)
            nisa.dma_copy(dst=b_tile, src=b_input)

            # Allocate space for the result and use tensor_tensor to perform
            # element-wise addition. Note: the first argument of 'tensor_tensor'
            # is the destination tensor.
            c_tile = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

            # Create a tensor in HBM and copy the result into HBM.
            c_output = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.hbm)
            nisa.dma_copy(dst=c_output, src=c_tile)

            # Return kernel output as function output.
            return c_output
    """
    ...

